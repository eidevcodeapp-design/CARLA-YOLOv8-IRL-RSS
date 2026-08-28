"""
================================================================================
 yolo_perception_test.py
 自動運転パイプライン Phase 2：認識モジュール（YOLOv8）独立実装・検証スクリプト

 【想定ワークフロー】
   1. VS Code（ローカル）でこのファイルを編集する
   2. Google Colab にアップロード、または GitHub 経由で pull して実行する
   3. Colab 上では GPU(T4) を使い、車載動画に対して YOLOv8 推論を行う
   4. 描画済み動画（.mp4）と検出ログ（.json）を Google Drive に自動保存する
   5. 検出ログは Phase 3 以降（IRLによる車線変更・追越し判断、RSS安全判定）で
      そのまま読み込めるデータ構造にしてある

 【Colab での実行手順（VS Code 上にコメントとして残しておく）】
   (1) Colab のランタイムを「GPU（T4）」に設定する
       メニュー：ランタイム > ランタイムのタイプを変更 > ハードウェアアクセラレータ = GPU

   (2) 最初のセルで依存パッケージをインストールする
       !pip install ultralytics opencv-python-headless

   (3) このファイルをそのまま1つのセルにコピー＆ペーストする、
       もしくは Google Drive にアップロードして
       !python /content/drive/MyDrive/AutoDrive_Project/yolo_perception_test.py
       のように実行する

   (4) main() 内の CONFIG（入力動画パスなど）を自分の環境に合わせて書き換える

 【ローカル(VS Code)での実行について】
   ローカルでは Google Drive マウント処理は自動的にスキップされ、
   カレントディレクトリ配下の ./AutoDrive_Project_local/ に出力される。
   （ロジックの確認・軽量デバッグ用。重い推論は Colab 側で行う想定）
================================================================================
"""

import os
import json
import time
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import cv2
import torch
from ultralytics import YOLO


# ==============================================================================
# 0. CONFIG（環境に合わせて書き換える設定値）
# ==============================================================================
class Config:
    # --- Google Drive 内の保存先ルート（Colab用）---
    DRIVE_ROOT = "/content/drive/MyDrive/AutoDrive_Project"

    # --- ローカル実行時（VS Code）の代替保存先 ---
    LOCAL_ROOT = "./AutoDrive_Project_local"

    # --- 入力動画（車載カメラ映像）のパス ---
    # Colab: Drive内にアップロードした動画のパスを指定する
    #        例）f"{DRIVE_ROOT}/inputs/dashcam_sample.mp4"
    # Local: とりあえずのサンプル動画パス
    INPUT_VIDEO_NAME = "dashcam_sample.mp4"

    # --- 使用モデル（軽量・高速重視でnano、精度重視ならsmall） ---
    MODEL_NAME = "yolov8n.pt"  # or "yolov8s.pt"

    # --- 検出対象クラス（COCOのクラス名。Vehicle / Pedestrian 系のみに絞る） ---
    TARGET_CLASSES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

    # --- 推論の信頼度しきい値 ---
    CONF_THRESHOLD = 0.35

    # --- 出力動画のフォルダ名・ログのフォルダ名（ルート配下に自動作成） ---
    OUTPUT_VIDEO_SUBDIR = "outputs/videos"
    OUTPUT_LOG_SUBDIR = "outputs/logs"

    # --- 進捗ログを何フレームごとに表示するか ---
    PROGRESS_LOG_INTERVAL = 30


# ==============================================================================
# 1. 検出結果のデータ構造
#    → Phase 3（IRL / RSS）にそのまま渡せるように、フレームごと・物体ごとに
#      「クラス名・確信度・中心座標(x,y)・幅高さ(w,h)」を保持する
# ==============================================================================
@dataclass
class Detection:
    frame_index: int          # 何フレーム目の検出か
    timestamp_sec: float      # 動画内の時刻（秒）
    class_name: str           # 検出クラス名（car, person 等）
    confidence: float         # 信頼度（0.0〜1.0）
    center_x: float           # バウンディングボックス中心のx座標（ピクセル）
    center_y: float           # バウンディングボックス中心のy座標（ピクセル）
    width: float               # バウンディングボックスの幅（ピクセル）
    height: float               # バウンディングボックスの高さ（ピクセル）

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerceptionLog:
    """1本の動画全体の検出結果をまとめるコンテナ。
    将来的にIRL側で「フレームごとの周辺車両の位置・速度」を復元する際、
    detections をフレーム単位でグループ化して利用する想定。"""
    source_video: str
    model_name: str
    device: str
    fps: float
    frame_width: int
    frame_height: int
    detections: List[Detection] = field(default_factory=list)

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)

    def save_json(self, path: str) -> None:
        payload = {
            "source_video": self.source_video,
            "model_name": self.model_name,
            "device": self.device,
            "fps": self.fps,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 2. 実行環境の判定 & Google Drive マウント
# ==============================================================================
def is_running_in_colab() -> bool:
    """Google Colab上で実行されているかどうかを判定する"""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup_output_paths(cfg: Config) -> Dict[str, str]:
    """
    実行環境を判定し、
      - Colabなら Google Drive をマウントして保存先を作成
      - ローカルならカレントディレクトリ配下に保存先を作成
    し、各種パスを辞書で返す。
    """
    if is_running_in_colab():
        from google.colab import drive  # Colab専用モジュール

        print("[INFO] Google Colab環境を検出。Google Driveをマウントします...")
        drive.mount("/content/drive")  # 初回はブラウザ経由の認証が必要
        root = cfg.DRIVE_ROOT
    else:
        print("[INFO] ローカル環境を検出。Google Driveマウントはスキップします。")
        root = cfg.LOCAL_ROOT

    input_dir = os.path.join(root, "inputs")
    video_dir = os.path.join(root, cfg.OUTPUT_VIDEO_SUBDIR)
    log_dir = os.path.join(root, cfg.OUTPUT_LOG_SUBDIR)

    # 保存先フォルダを一括作成（存在してもエラーにならないようexist_ok=True）
    for d in (input_dir, video_dir, log_dir):
        os.makedirs(d, exist_ok=True)

    paths = {
        "root": root,
        "input_dir": input_dir,
        "video_dir": video_dir,
        "log_dir": log_dir,
        "input_video": os.path.join(input_dir, cfg.INPUT_VIDEO_NAME),
    }
    print(f"[INFO] 保存先ルート: {root}")
    return paths


# ==============================================================================
# 3. YOLOv8モデルのロード（GPU/CPU自動判定）
# ==============================================================================
def load_model(model_name: str):
    """
    指定したYOLOv8モデル(.pt)をロードする。
    CUDA(GPU)が使える場合は自動でcudaに、使えない場合はcpuにフォールバックする。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用デバイス: {device}")

    if device == "cpu":
        print("[WARN] GPUが検出されませんでした。CPUで推論するため処理が遅くなります。")
        print("[WARN] Colabの場合は ランタイム > ランタイムのタイプを変更 で GPU(T4)を選択してください。")

    model = YOLO(model_name)  # 初回実行時は自動でweightsがダウンロードされる
    model.to(device)
    return model, device


# ==============================================================================
# 4. 動画への推論・可視化・Drive保存
# ==============================================================================
def process_video(
    model,
    device: str,
    input_video_path: str,
    output_video_path: str,
    target_classes: List[str],
    conf_threshold: float,
    progress_interval: int = 30,
) -> PerceptionLog:
    """
    車載動画を読み込み、フレームごとにYOLOv8推論を実行。
    - 対象クラス（Vehicle / Pedestrian系）のみバウンディングボックスを描画
    - 描画済みフレームを cv2.VideoWriter で mp4 として書き出す
      （cv2.imshowは使わない。Colabでの画面出力エラーを避けるため）
    - 検出結果（クラス名・確信度・中心座標・幅高さ）を PerceptionLog に蓄積する
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(
            f"入力動画が見つかりません: {input_video_path}\n"
            f"Config.INPUT_VIDEO_NAME、および inputs フォルダへの配置を確認してください。"
        )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] 入力動画: {input_video_path}")
    print(f"[INFO] 解像度: {frame_width}x{frame_height}, FPS: {fps:.2f}, 総フレーム数: {total_frames}")

    # mp4形式で書き出すためのVideoWriterを準備
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    perception_log = PerceptionLog(
        source_video=input_video_path,
        model_name=Config.MODEL_NAME,
        device=device,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # 色分け（歩行者は目立つ色、車両系は別の色にしておくとRSS判定時にも視認しやすい）
    COLOR_PERSON = (0, 0, 255)     # 赤系（BGR） - 歩行者
    COLOR_VEHICLE = (0, 200, 0)    # 緑系（BGR） - 車両
    COLOR_OTHER = (255, 200, 0)    # 青緑系（BGR） - その他対象クラス

    frame_index = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 動画終端

        # --- YOLOv8推論（1フレーム分）---
        # verbose=Falseでコンソールへの過剰なログ出力を抑制
        results = model.predict(source=frame, conf=conf_threshold, device=device, verbose=False)
        result = results[0]

        # --- 検出結果を1件ずつ処理 ---
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # 対象クラス（Vehicle / Pedestrian系）以外はスキップ
            if class_name not in target_classes:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上・右下座標

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            # --- 将来のIRL/RSS連携用に検出結果を記録 ---
            detection = Detection(
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                class_name=class_name,
                confidence=confidence,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )
            perception_log.add(detection)

            # --- 描画色をクラスに応じて切り替え ---
            if class_name == "person":
                color = COLOR_PERSON
            elif class_name in ("car", "truck", "bus"):
                color = COLOR_VEHICLE
            else:
                color = COLOR_OTHER

            # --- バウンディングボックスとラベルを描画 ---
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, color, thickness=2)

            label = f"{class_name} {confidence:.2f}"
            label_y = max(int(y1) - 8, 15)
            cv2.putText(
                frame, label, (int(x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness=2,
            )

        # --- 描画済みフレームをmp4に書き込み（画面表示はしない）---
        writer.write(frame)

        frame_index += 1
        if frame_index % progress_interval == 0:
            elapsed = time.time() - start_time
            processed_fps = frame_index / elapsed if elapsed > 0 else 0.0
            progress = f"{frame_index}/{total_frames}" if total_frames > 0 else f"{frame_index}"
            print(f"[PROGRESS] frame {progress} | 処理速度: {processed_fps:.1f} fps")

    cap.release()
    writer.release()

    elapsed_total = time.time() - start_time
    print(f"[INFO] 推論完了。総処理時間: {elapsed_total:.1f}秒 / 総検出数: {len(perception_log.detections)}")
    print(f"[INFO] 出力動画を保存しました: {output_video_path}")

    return perception_log


# ==============================================================================
# 5. メイン処理
# ==============================================================================
def main():
    cfg = Config()

    # --- 1. 保存先パスのセットアップ（Drive自動マウント含む） ---
    paths = setup_output_paths(cfg)

    # --- 2. モデルのロード（GPU/CPU自動判定） ---
    model, device = load_model(cfg.MODEL_NAME)

    # --- 3. 出力ファイル名を決定（入力ファイル名 + タイムスタンプ） ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(cfg.INPUT_VIDEO_NAME))[0]
    output_video_path = os.path.join(paths["video_dir"], f"{base_name}_yolo_{timestamp}.mp4")
    output_log_path = os.path.join(paths["log_dir"], f"{base_name}_detections_{timestamp}.json")

    # --- 4. 推論・可視化・Drive保存を実行 ---
    perception_log = process_video(
        model=model,
        device=device,
        input_video_path=paths["input_video"],
        output_video_path=output_video_path,
        target_classes=cfg.TARGET_CLASSES,
        conf_threshold=cfg.CONF_THRESHOLD,
        progress_interval=cfg.PROGRESS_LOG_INTERVAL,
    )

    # --- 5. 検出ログをJSONとして保存（Phase 3: IRL/RSSでそのまま読み込む想定） ---
    perception_log.save_json(output_log_path)
    print(f"[INFO] 検出ログを保存しました: {output_log_path}")

    print("\n[DONE] Phase 2（YOLOv8認識モジュール）の実行が完了しました。")
    print(f"       動画: {output_video_path}")
    print(f"       ログ: {output_log_path}")


if __name__ == "__main__":
    main()