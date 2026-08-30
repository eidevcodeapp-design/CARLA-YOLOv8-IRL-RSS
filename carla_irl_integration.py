"""
================================================================================
 carla_irl_integration.py
 自動運転パイプライン Phase 3：
 CARLA(シミュレータ) × YOLOv8(認識) × IRL特徴量抽出 リアルタイム統合スクリプト
 実行環境: Google Colab (GPU: T4等)

 【位置づけ】
   Phase 2  : YOLOv8による動画からの2D検出（別スクリプト yolo_perception_test.py）
   Phase 2.5: 検出ログ → IRL用時系列特徴量への変換（perception_to_irl_feature.py）
   Phase 3  : このスクリプト。CARLA上でEgo車両をAutopilot走行させながら、
              カメラ画像→YOLOv8推論→PerceptionFeatureExtractorをリアルタイムに繋ぎ、
              検証用デモ動画と状態ベクトルログを生成する
   Phase 4  : IRL方策学習 + RSS安全判定との統合（今後）

================================================================================
 【Colab セットアップ手順】
 （このスクリプトを実行する前に、Colabの別セルで上から順に実行しておくこと）

   # 0. ランタイム設定
   #    メニュー > ランタイムのタイプを変更 > ハードウェアアクセラレータ = GPU (T4)

   # 1. CARLAサーバー実行に必要な apt パッケージ
   !apt-get update -qq
   !apt-get install -y -qq libomp5 libsdl2-2.0-0 libsm6 libgl1-mesa-glx xdg-user-dirs libvulkan1

   # 2. CARLAサーバー本体（Prebuilt Linux Binary）のダウンロード・展開
   #    ★バージョンは手順3のpipパッケージと必ず一致させること★
   #    最新のダウンロードURLは公式リリースページで確認するのが確実です:
   #      https://github.com/carla-simulator/carla/releases
   #    例）CARLA 0.9.15 の場合:
   !wget -q https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz
   !mkdir -p /content/carla_server/CARLA_0.9.15
   !tar -xzf CARLA_0.9.15.tar.gz -C /content/carla_server/CARLA_0.9.15

   # 3. CARLA Pythonクライアント（サーバーと同一バージョンを指定）
   !pip install carla==0.9.15

   # 4. YOLOv8 / OpenCV（Phase2, 2.5と共通）
   !pip install ultralytics opencv-python-headless

   # 5. Phase2.5のモジュール(perception_to_irl_feature.py)を含むリポジトリをクローン
   !git clone https://github.com/<your-account>/CARLA-YOLOv8-IRL-RSS.git /content/CARLA-YOLOv8-IRL-RSS

   # 6. 本スクリプトを実行
   !python /content/CARLA-YOLOv8-IRL-RSS/carla_irl_integration.py

 【重要な注意：Google Colab（特に無料版）でのCARLA実行について】
   - CARLAはUnreal Engineベースの重量級シミュレータで、Colab上での実行は公式サポート外です。
     以下のような理由で起動・実行に失敗することがあります。
       ・GPUメモリ不足（YOLOv8推論とCARLAレンダリングでVRAMを取り合う）
       ・Vulkan/OpenGLドライバ関連のエラー
       ・ディスク容量不足（CARLAサーバーは展開後10GB超）
       ・Colabのセッション時間制限・切断
   - `-RenderOffScreen` 単体で起動しない場合は、下記 Config.USE_XVFB を True にして
     `xvfb-run` 経由の起動を試してください（仮想ディスプレイ経由でのオフスクリーン描画）。
   - どうしても安定しない場合は、CARLAサーバーをローカルPCや専用GPUクラウドVM上で起動し、
     本スクリプトの CARLA_HOST / CARLA_PORT をそちらに向けて「クライアントとしてのみ」
     Colabから接続する構成に切り替えることを推奨します（このスクリプトはクライアント側の
     ロジック自体は変更不要で、接続先を変えるだけで動きます）。
================================================================================
"""

import os
import sys
import time
import math
import socket
import queue
import random
import subprocess

import numpy as np
import cv2
import torch
from ultralytics import YOLO

# --- Phase2.5のモジュールをインポートできるようにリポジトリパスを追加 ---
REPO_PATH = "/content/CARLA-YOLOv8-IRL-RSS"  # git clone先。環境に応じて変更
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

try:
    from perception_to_irl_feature import (
        PerceptionFeatureExtractor,
        STATE_VECTOR_SCHEMA,
        save_feature_outputs,
    )
except ImportError as e:
    raise ImportError(
        "perception_to_irl_feature モジュールが見つかりません。\n"
        f"'{REPO_PATH}' に CARLA-YOLOv8-IRL-RSS リポジトリをclone済みか確認してください。\n"
        "  !git clone https://github.com/<your-account>/CARLA-YOLOv8-IRL-RSS.git " + REPO_PATH
    ) from e

try:
    import carla
except ImportError as e:
    raise ImportError(
        "`carla` パッケージが見つかりません。CARLAサーバーと同一バージョンで\n"
        "  pip install carla==<バージョン>\n"
        "を実行してください（スクリプト冒頭のセットアップ手順を参照）。"
    ) from e


# ==============================================================================
# 0. CONFIG
# ==============================================================================
class Config:
    # --- CARLAサーバー関連 ---
    CARLA_ROOT = "/content/carla_server/CARLA_0.9.15"  # CarlaUE4.shがある展開先
    CARLA_HOST = "localhost"
    CARLA_PORT = 2000
    CARLA_MAP = "Town01"                 # 軽量なマップを指定
    FIXED_DELTA_SECONDS = 1.0 / 30.0     # 同期モードでの1ステップの時間刻み
    USE_XVFB = False                     # -RenderOffScreen単体で失敗する場合 True に

    # --- カメラ（フロント車載カメラを模擬）---
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FOV = 90.0
    TARGET_FPS = 30

    # --- YOLOv8 ---
    MODEL_NAME = "yolov8n.pt"
    CONF_THRESHOLD = 0.35
    TARGET_CLASSES = ["person", "car", "truck", "bus", "motorcycle", "bicycle"]

    # --- IRL特徴量抽出 ---
    LANE_WIDTH_M = 3.5

    # --- 走行フレーム数（TARGET_FPS=30なら300フレーム≒10秒）---
    NUM_FRAMES = 300

    # --- 出力先 ---
    DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/AutoDrive_Project/outputs/carla_results"
    LOCAL_OUTPUT_DIR = "./AutoDrive_Project_local/outputs/carla_results"
    OUTPUT_NPY_NAME = "carla_irl_state_features.npy"
    OUTPUT_JSON_NAME = "carla_irl_state_features.json"
    OUTPUT_VIDEO_NAME = "carla_perception_demo.mp4"


# ==============================================================================
# 1. 環境ユーティリティ（Google Driveマウント判定）
# ==============================================================================
def is_running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def setup_output_dir(cfg: Config) -> str:
    if is_running_in_colab():
        from google.colab import drive
        print("[INFO] Google Colab環境を検出。Google Driveをマウントします...")
        drive.mount("/content/drive")
        output_dir = cfg.DRIVE_OUTPUT_DIR
    else:
        print("[INFO] ローカル環境を検出。Google Driveマウントはスキップします。")
        output_dir = cfg.LOCAL_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 出力先: {output_dir}")
    return output_dir


# ==============================================================================
# 2. CARLAサーバーの起動・接続
# ==============================================================================
def launch_carla_server(cfg: Config) -> subprocess.Popen:
    """CARLAサーバーをヘッドレス(-RenderOffScreen)でバックグラウンド起動する"""
    carla_sh_path = os.path.join(cfg.CARLA_ROOT, "CarlaUE4.sh")
    if not os.path.exists(carla_sh_path):
        raise FileNotFoundError(
            f"CARLAサーバーの実行ファイルが見つかりません: {carla_sh_path}\n"
            f"スクリプト冒頭のセットアップ手順に従い、CARLAサーバーをダウンロード・展開してください。"
        )
    os.chmod(carla_sh_path, 0o755)

    base_cmd = [
        carla_sh_path,
        "-RenderOffScreen",              # ディスプレイなしのヘッドレスレンダリング
        "-nosound",
        f"-carla-rpc-port={cfg.CARLA_PORT}",
        "-quality-level=Low",            # Colab無料版GPUの負荷軽減
    ]

    if cfg.USE_XVFB:
        # -RenderOffScreen だけで起動しない環境向けの代替手段（仮想ディスプレイ経由）
        cmd = ["xvfb-run", "-a"] + base_cmd
    else:
        cmd = base_cmd

    print(f"[INFO] CARLAサーバーを起動します: {' '.join(cmd)}")
    log_path = "/content/carla_server.log" if is_running_in_colab() else "./carla_server.log"
    log_file = open(log_path, "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"[INFO] CARLAサーバーのログ出力先: {log_path}")
    return process


def wait_for_carla_server(host: str, port: int, timeout: float = 120.0) -> None:
    """指定ポートが開くまでポーリングして待機する"""
    print(f"[INFO] CARLAサーバーの起動を待機しています（最大{timeout:.0f}秒）...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                print("[INFO] CARLAサーバーへの接続準備ができました。")
                return
        except OSError:
            time.sleep(2.0)
    raise TimeoutError(
        f"CARLAサーバーが{timeout:.0f}秒以内に起動しませんでした。"
        f"ログファイル(carla_server.log)を確認してください。"
    )


def connect_to_carla(cfg: Config):
    """CARLAクライアント接続 + 同期モード設定。(client, world, original_settings)を返す"""
    client = carla.Client(cfg.CARLA_HOST, cfg.CARLA_PORT)
    client.set_timeout(30.0)

    print(f"[INFO] マップ '{cfg.CARLA_MAP}' をロードします...")
    world = client.load_world(cfg.CARLA_MAP)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True           # world.tick()で明示的に1ステップ進める
    settings.fixed_delta_seconds = cfg.FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    # Traffic Manager（Autopilot車両の挙動制御）も同期モードに合わせる
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    print("[INFO] CARLAワールドへの接続・同期モード設定が完了しました。")
    return client, world, original_settings


# ==============================================================================
# 3. Ego車両・カメラセンサーのセットアップ
# ==============================================================================
def spawn_ego_vehicle(world):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("マップにスポーンポイントが見つかりません。")

    random.shuffle(spawn_points)
    vehicle = None
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("Ego車両のスポーンに失敗しました（他アクターとの衝突等）。")

    vehicle.set_autopilot(True)
    print(f"[INFO] Ego車両をスポーンしAutopilotを有効化しました（id={vehicle.id}）。")
    return vehicle


def attach_rgb_camera(world, vehicle, cfg: Config):
    """フロントバンパー付近にRGBカメラを取り付け、画像をqueue経由で受け取れるようにする"""
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(cfg.CAMERA_WIDTH))
    camera_bp.set_attribute("image_size_y", str(cfg.CAMERA_HEIGHT))
    camera_bp.set_attribute("fov", str(cfg.CAMERA_FOV))

    # 車載カメラを模擬: フロント寄り・やや高めの位置に前向き取り付け
    camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # センサーのコールバックは別スレッドで発火するため、queueでメインループに橋渡しする
    image_queue = queue.Queue(maxsize=1)

    def on_image(image):
        # 処理が追いつかず溜まっている場合は古い画像を捨てて最新のものに差し替える
        if image_queue.full():
            try:
                image_queue.get_nowait()
            except queue.Empty:
                pass
        image_queue.put(image)

    camera.listen(on_image)
    print("[INFO] フロントRGBカメラを取り付けました。")
    return camera, image_queue


def carla_image_to_bgr_array(carla_image, width: int, height: int) -> np.ndarray:
    """CARLAのraw_data(BGRA)をOpenCVで扱えるBGR配列に変換する"""
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((height, width, 4))
    return array[:, :, :3].copy()  # BGRAの先頭3ch=BGR（OpenCVとそのまま整合）


# ==============================================================================
# 4. YOLOv8推論
# ==============================================================================
def load_yolo_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] YOLOv8モデル({model_name})をロードします。使用デバイス: {device}")
    if device == "cpu":
        print("[WARN] GPUが検出されませんでした。CARLA+YOLOv8同時実行はCPUだと非常に遅くなります。")
    model = YOLO(model_name)
    model.to(device)
    return model, device


def run_yolo_inference(model, device: str, frame_bgr: np.ndarray, cfg: Config):
    """
    1フレーム分の推論結果を、Phase2/2.5と互換の辞書構造で返す。
    描画用に bbox の四隅座標(x1,y1,x2,y2)も併せて保持する。
    """
    results = model.predict(source=frame_bgr, conf=cfg.CONF_THRESHOLD, device=device, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name not in cfg.TARGET_CLASSES:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_name": class_name,
            "confidence": confidence,
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
            "width": x2 - x1,
            "height": y2 - y1,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return detections


# ==============================================================================
# 5. 可視化（デモ動画への描画）
# ==============================================================================
COLOR_PERSON = (0, 0, 255)      # 赤系(BGR) - 歩行者
COLOR_VEHICLE = (0, 200, 0)     # 緑系(BGR) - 車両
COLOR_OTHER = (255, 200, 0)     # 青緑系(BGR) - その他対象クラス
COLOR_HUD_BG = (30, 30, 30)
COLOR_HUD_TEXT = (255, 255, 255)

NO_OBJECT_THRESHOLD = 900.0  # PerceptionFeatureExtractorのNO_OBJECT_DISTANCE_M(999.0)に対応


def _fmt_distance(d: float) -> str:
    return "N/A" if d >= NO_OBJECT_THRESHOLD else f"{d:.1f}m"


def _find_matching_track(class_name: str, x_rel: float, y_rel: float, tracked_objects, tol: float = 1.0):
    """
    このフレームで検出したbboxの(x_rel,y_rel)推定値と、
    Extractorが返すtracked_objects内の値を突き合わせ、対応するトラック情報(速度含む)を探す。
    """
    best = None
    best_dist = tol
    for t in tracked_objects:
        if t["class_name"] != class_name:
            continue
        d = math.hypot(t["x_rel_m"] - x_rel, t["y_rel_m"] - y_rel)
        if d < best_dist:
            best_dist = d
            best = t
    return best


def _build_hud_lines(state_vector, ego_speed_mps: float):
    (
        _ego_speed, front_dist, front_vrel,
        left_dist, left_vrel, right_dist, right_vrel,
        ped_dist, _ped_lateral, num_vehicles, num_pedestrians,
    ) = state_vector

    return [
        f"Ego speed:  {ego_speed_mps * 3.6:5.1f} km/h",
        f"Front veh:  {_fmt_distance(front_dist):>7} ({front_vrel:+.1f} m/s)",
        f"Left lane:  {_fmt_distance(left_dist):>7} ({left_vrel:+.1f} m/s)",
        f"Right lane: {_fmt_distance(right_dist):>7} ({right_vrel:+.1f} m/s)",
        f"Pedestrian: {_fmt_distance(ped_dist):>7}",
        f"Visible: vehicles={int(num_vehicles)} pedestrians={int(num_pedestrians)}",
    ]


def draw_debug_overlay(frame_bgr, detections, extractor: PerceptionFeatureExtractor, record, ego_speed_mps: float, frame_index: int, total_frames: int):
    tracked_objects = record.tracked_objects

    # --- 検出物体ごとのバウンディングボックス + 距離/相対速度テキスト ---
    for det in detections:
        color = COLOR_PERSON if det["class_name"] == "person" else (
            COLOR_VEHICLE if det["class_name"] in ("car", "truck", "bus") else COLOR_OTHER
        )
        p1 = (int(det["x1"]), int(det["y1"]))
        p2 = (int(det["x2"]), int(det["y2"]))
        cv2.rectangle(frame_bgr, p1, p2, color, thickness=2)

        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.putText(frame_bgr, label, (p1[0], max(p1[1] - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)

        pos = extractor.estimate_relative_position(det)
        if pos is not None:
            x_rel, y_rel = pos
            matched = _find_matching_track(det["class_name"], x_rel, y_rel, tracked_objects)
            vrel_text = f'{matched["vy_rel_mps"]:+.1f}m/s' if matched else "N/A"
            dist_label = f'{y_rel:.1f}m  {vrel_text}'
            cv2.putText(frame_bgr, dist_label, (p1[0], min(p2[1] + 20, frame_bgr.shape[0] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)

    # --- 左上デバッグHUD（半透明パネル + 主要な状態ベクトルの数値）---
    hud_lines = [f"Frame: {frame_index}/{total_frames}"] + _build_hud_lines(record.state_vector, ego_speed_mps)
    hud_width = 360
    hud_height = 20 + 18 * len(hud_lines)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_width, 10 + hud_height), COLOR_HUD_BG, thickness=-1)
    frame_bgr = cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0)

    for i, line in enumerate(hud_lines):
        y = 30 + i * 18
        cv2.putText(frame_bgr, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_TEXT, thickness=1)

    return frame_bgr


# ==============================================================================
# 6. クリーンアップ（車両・センサー・CARLAサーバーの確実な破棄）
# ==============================================================================
def cleanup(world, vehicle, camera, original_settings, carla_process):
    print("[INFO] クリーンアップ処理を実行します...")

    try:
        if camera is not None:
            camera.stop()
            camera.destroy()
            print("[INFO] カメラセンサーを破棄しました。")
    except Exception as e:
        print(f"[WARN] カメラの破棄中にエラーが発生しました: {e}")

    try:
        if vehicle is not None:
            vehicle.destroy()
            print("[INFO] Ego車両を破棄しました。")
    except Exception as e:
        print(f"[WARN] 車両の破棄中にエラーが発生しました: {e}")

    try:
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)  # 同期モード等を元に戻す
            print("[INFO] ワールド設定を元に戻しました。")
    except Exception as e:
        print(f"[WARN] ワールド設定の復元中にエラーが発生しました: {e}")

    try:
        if carla_process is not None:
            carla_process.terminate()
            carla_process.wait(timeout=15)
            print("[INFO] CARLAサーバープロセスを終了しました。")
    except Exception as e:
        print(f"[WARN] CARLAサーバーの正常終了に失敗しました。強制終了します: {e}")
        try:
            carla_process.kill()
        except Exception:
            pass


# ==============================================================================
# 7. メイン処理
# ==============================================================================
def main():
    cfg = Config()
    output_dir = setup_output_dir(cfg)

    carla_process = None
    world = None
    vehicle = None
    camera = None
    original_settings = None

    try:
        # --- 1. CARLAサーバーの起動・接続 ---
        carla_process = launch_carla_server(cfg)
        wait_for_carla_server(cfg.CARLA_HOST, cfg.CARLA_PORT, timeout=120.0)
        client, world, original_settings = connect_to_carla(cfg)

        # --- 2. Ego車両のスポーン + Autopilot ---
        vehicle = spawn_ego_vehicle(world)

        # --- 3. フロントRGBカメラの取り付け ---
        camera, image_queue = attach_rgb_camera(world, vehicle, cfg)

        # --- 4. YOLOv8のロード ---
        model, device = load_yolo_model(cfg.MODEL_NAME)

        # --- 5. IRL特徴量抽出器の初期化 ---
        # ego_speed_kmhは初期値のダミー。各フレームでCARLAの実測速度に上書きする。
        extractor = PerceptionFeatureExtractor(
            frame_width=cfg.CAMERA_WIDTH,
            frame_height=cfg.CAMERA_HEIGHT,
            assumed_hfov_deg=cfg.CAMERA_FOV,
            ego_speed_kmh=0.0,
            lane_width_m=cfg.LANE_WIDTH_M,
        )

        # --- 6. デモ動画のVideoWriter準備 ---
        video_path = os.path.join(output_dir, cfg.OUTPUT_VIDEO_NAME)
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"),
            cfg.TARGET_FPS, (cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT),
        )

        records = []
        frame_index = 0
        start_time = time.time()

        print(f"[INFO] シミュレーションを開始します（{cfg.NUM_FRAMES}フレーム走行）...")

        while frame_index < cfg.NUM_FRAMES:
            # --- 同期モードで1ステップ進め、対応する画像を取得 ---
            expected_frame = world.tick()
            carla_image = image_queue.get(timeout=10.0)
            # フレームがずれている場合、対応する画像が来るまで読み捨てる（同期保証）
            while carla_image.frame != expected_frame:
                carla_image = image_queue.get(timeout=10.0)

            frame_bgr = carla_image_to_bgr_array(carla_image, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT)
            timestamp_sec = carla_image.timestamp  # CARLAシミュレーション内時刻[s]

            # --- Ego車両の実測速度を取得し、Extractorに反映(Phase2.5の仮定値を実測に置換) ---
            velocity = vehicle.get_velocity()
            ego_speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            extractor.ego_speed_mps = ego_speed_mps

            # --- YOLOv8推論 ---
            detections = run_yolo_inference(model, device, frame_bgr, cfg)

            # --- IRL特徴量抽出（トラッキング・状態ベクトル構築） ---
            record = extractor.process_frame(frame_index, timestamp_sec, detections)
            records.append(record)

            # --- 検証用デモ動画への描画・書き込み ---
            annotated_frame = draw_debug_overlay(
                frame_bgr, detections, extractor, record, ego_speed_mps,
                frame_index, cfg.NUM_FRAMES,
            )
            writer.write(annotated_frame)

            if frame_index % 30 == 0:
                elapsed = time.time() - start_time
                proc_fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"[PROGRESS] frame {frame_index}/{cfg.NUM_FRAMES} | "
                    f"ego={ego_speed_mps * 3.6:.1f}km/h | 検出数={len(detections)} | "
                    f"処理速度={proc_fps:.1f}fps"
                )

            frame_index += 1

        writer.release()
        print(f"[INFO] 検証用デモ動画を保存しました: {video_path}")

        # --- 7. 状態ベクトル・詳細ログの保存 ---
        npy_path, json_path = save_feature_outputs(
            records, output_dir, cfg.OUTPUT_NPY_NAME, cfg.OUTPUT_JSON_NAME,
        )
        print(f"[INFO] 状態ベクトル行列(.npy)を保存しました: {npy_path}")
        print(f"[INFO] 詳細ログ(.json)を保存しました: {json_path}")

        print("\n[DONE] Phase 3（CARLA×YOLOv8×IRL統合）の実行が完了しました。")
        print(f"       動画: {video_path}")
        print(f"       特徴量: {npy_path}")
        print(f"       詳細ログ: {json_path}")

    finally:
        # --- 8. クリーンアップ（例外発生時・正常終了時とも必ず実行） ---
        cleanup(world, vehicle, camera, original_settings, carla_process)


if __name__ == "__main__":
    main()