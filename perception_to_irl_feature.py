"""
================================================================================
 perception_to_irl_feature.py
 自動運転パイプライン Phase 2.5：
 YOLOv8検出ログ(JSON) → IRL(逆強化学習)用 時系列特徴量ベクトル 変換モジュール

 【位置づけ】
   Phase 2  : YOLOv8 で動画から2D検出（class, confidence, center_x/y, w/h）を抽出しJSON保存
   Phase 2.5: このスクリプト。2D検出 → 簡易3D相対位置 → トラッキング → 相対速度
              → IRL状態ベクトル（固定長）に変換する（今回の担当範囲）
   Phase 3  : CARLA連携（オンラインでこのExtractorを毎フレーム呼び出す想定）
   Phase 4  : IRL（車線変更・追越し方策学習）+ RSS（安全判定）統合

 【重要な前提・精度に関する注意】
   ここでの「奥行き推定」は、単眼カメラ映像から得られる2Dバウンディングボックスの
   サイズのみを使った簡易的なピンホールカメラモデル近似である。
     - 対象物体のクラスごとの「平均的な実サイズ（車幅/歩行者身長など）」を仮定
     - レンズ歪み補正・カメラキャリブレーション・路面勾配などは考慮していない
   そのため、ここで得られる x_rel, y_rel はあくまで「IRLの学習用の近似特徴量」であり、
   実車の安全判定（RSS）にそのまま使う場合は、実カメラのキャリブレーション値
   （焦点距離・光学中心・歪み係数）に置き換える、あるいはステレオ/LiDAR等で
   精度を検証してから使うことを推奨。

 【Colab / VS Code 双方での実行について】
   このスクリプトは環境非依存（Google Drive等の特別なマウント処理は不要）。
   Phase 2で保存したJSONログのパスを指定するだけで、ローカルでもColab上でも動作する。
   （Colab上でDriveをマウント済みなら、Drive内のパスをそのまま渡せば動く）

================================================================================
"""

import os
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np


# ==============================================================================
# 0. CONFIG
# ==============================================================================
class Config:
    # --- Phase 2で生成されたJSONログのパス（実行時に書き換える） ---
    INPUT_LOG_PATH = "./AutoDrive_Project_local/outputs/logs/sample_detections.json"

    # --- このスクリプトの出力先 ---
    OUTPUT_DIR = "./AutoDrive_Project_local/outputs/irl_features"
    OUTPUT_NPY_NAME = "irl_state_features.npy"
    OUTPUT_JSON_NAME = "irl_state_features.json"

    # --- カメラの仮定パラメータ（実カメラのキャリブレーション値があれば置き換える） ---
    ASSUMED_HORIZONTAL_FOV_DEG = 90.0  # 一般的なドラレコ/車載広角カメラの想定値

    # --- 自車速度（仮定値。Phase3でCARLA側の実速度に差し替え可能） ---
    ASSUMED_EGO_SPEED_KMH = 60.0

    # --- クラスごとの実世界サイズ仮定値 [m]（幅, 高さ） ---
    KNOWN_OBJECT_SIZE_M = {
        "car":        {"width": 1.8, "height": 1.5},
        "truck":      {"width": 2.4, "height": 3.0},
        "bus":        {"width": 2.5, "height": 3.2},
        "motorcycle": {"width": 0.8, "height": 1.3},
        "bicycle":    {"width": 0.6, "height": 1.4},
        "person":     {"width": 0.5, "height": 1.7},
    }

    # --- 車線幅の仮定値 [m]（前方/左隣/右隣の判定に使用） ---
    LANE_WIDTH_M = 3.5

    # --- トラッキング関連 ---
    MAX_TRACK_MATCH_DISTANCE_M = 4.0  # このメートル数以内なら同一物体とみなす
    MAX_MISSED_FRAMES = 5             # これ以上連続で検出されなければトラック消去

    # --- 状態ベクトルで「物体なし」を表すデフォルト距離 [m] ---
    NO_OBJECT_DISTANCE_M = 999.0


# 状態ベクトルの各次元の意味（ドキュメント兼テスト用）
STATE_VECTOR_SCHEMA = [
    "ego_speed_mps",
    "front_vehicle_distance_m",
    "front_vehicle_rel_velocity_mps",
    "left_lane_vehicle_distance_m",
    "left_lane_vehicle_rel_velocity_mps",
    "right_lane_vehicle_distance_m",
    "right_lane_vehicle_rel_velocity_mps",
    "nearest_pedestrian_distance_m",
    "nearest_pedestrian_lateral_offset_m",
    "num_vehicles_visible",
    "num_pedestrians_visible",
]
STATE_VECTOR_DIM = len(STATE_VECTOR_SCHEMA)


# ==============================================================================
# 1. データ構造
# ==============================================================================
@dataclass
class TrackedObject:
    """1つの物体（車両 or 歩行者）を跨フレームで追跡するための状態"""
    track_id: int
    class_name: str
    x_rel: float            # 自車に対する横方向相対位置 [m]（右+）
    y_rel: float             # 自車に対する前方相対距離 [m]（前方+）
    vx_rel: float = 0.0      # 横方向の相対速度 [m/s]
    vy_rel: float = 0.0      # 前後方向の相対速度 [m/s]
    last_frame_index: int = -1
    last_timestamp: float = 0.0
    missed_frames: int = 0   # 何フレーム連続で未検出か


@dataclass
class FrameFeatureRecord:
    """1フレーム分の変換結果（デバッグ・Phase3検証用の詳細情報つき）"""
    frame_index: int
    timestamp_sec: float
    state_vector: List[float]
    tracked_objects: List[Dict] = field(default_factory=list)  # このフレームで見えている全物体の詳細

    def to_dict(self) -> Dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "state_vector": self.state_vector,
            "tracked_objects": self.tracked_objects,
        }


# ==============================================================================
# 2. PerceptionFeatureExtractor
#    バッチ処理（JSONログ一括変換）・オンライン処理（フレーム逐次入力）の
#    どちらからも同じロジックで呼び出せるように設計。
# ==============================================================================
class PerceptionFeatureExtractor:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        assumed_hfov_deg: float = Config.ASSUMED_HORIZONTAL_FOV_DEG,
        ego_speed_kmh: float = Config.ASSUMED_EGO_SPEED_KMH,
        lane_width_m: float = Config.LANE_WIDTH_M,
        max_track_match_distance_m: float = Config.MAX_TRACK_MATCH_DISTANCE_M,
        max_missed_frames: int = Config.MAX_MISSED_FRAMES,
        no_object_distance_m: float = Config.NO_OBJECT_DISTANCE_M,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.principal_x = frame_width / 2.0

        # --- ピンホールカメラモデル: 水平画角から焦点距離[px]を逆算 ---
        # focal_length_px = (frame_width / 2) / tan(HFOV / 2)
        hfov_rad = math.radians(assumed_hfov_deg)
        self.focal_length_px = (frame_width / 2.0) / math.tan(hfov_rad / 2.0)

        self.ego_speed_mps = ego_speed_kmh / 3.6
        self.lane_width_m = lane_width_m
        self.max_track_match_distance_m = max_track_match_distance_m
        self.max_missed_frames = max_missed_frames
        self.no_object_distance_m = no_object_distance_m

        # --- トラッキング状態（オンライン呼び出し時はインスタンスをまたいで保持される） ---
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 0

    # --------------------------------------------------------------------
    # 2-1. 単眼奥行き推定（ピンホールカメラモデルによる簡易3D位置推定）
    # --------------------------------------------------------------------
    def estimate_relative_position(self, detection: Dict) -> Optional[Tuple[float, float]]:
        """
        2Dバウンディングボックス（中心座標・幅・高さ）から、
        自車に対する相対位置 (x_rel, y_rel) [m] を簡易推定する。

        原理：
          実物体のサイズ(既知と仮定) : 画像上のピクセルサイズ
          = 焦点距離(px) : 奥行き距離(m)
          という相似関係（ピンホールカメラモデル）を利用する。

          depth = (real_size_m * focal_length_px) / pixel_size

        幅・高さ両方から推定できる場合は平均を取り、ノイズを軽減する。
        """
        class_name = detection["class_name"]
        size_info = Config.KNOWN_OBJECT_SIZE_M.get(class_name)
        if size_info is None:
            return None  # 既知サイズが未定義のクラスは推定不可

        bbox_w = detection.get("width", 0.0)
        bbox_h = detection.get("height", 0.0)

        depth_estimates = []
        if bbox_w > 1.0:  # 1px以下は分母近すぎでノイズ大なので除外
            depth_from_width = (size_info["width"] * self.focal_length_px) / bbox_w
            depth_estimates.append(depth_from_width)
        if bbox_h > 1.0:
            depth_from_height = (size_info["height"] * self.focal_length_px) / bbox_h
            depth_estimates.append(depth_from_height)

        if not depth_estimates:
            return None

        depth_m = sum(depth_estimates) / len(depth_estimates)  # y_rel（前方距離）

        # --- 横方向オフセット: 画像中心からのピクセルオフセット×相似関係で算出 ---
        center_x = detection["center_x"]
        pixel_offset_x = center_x - self.principal_x
        x_rel = depth_m * (pixel_offset_x / self.focal_length_px)

        y_rel = depth_m
        return x_rel, y_rel

    # --------------------------------------------------------------------
    # 2-2. フレーム間トラッキング（簡易 Nearest-Neighbor マッチング）
    # --------------------------------------------------------------------
    def _match_and_update_tracks(
        self,
        frame_index: int,
        timestamp_sec: float,
        observations: List[Dict],
    ) -> List[TrackedObject]:
        """
        observations: [{"class_name":..., "x_rel":..., "y_rel":..., "confidence":...}, ...]
        同一クラス・近距離のトラックにマッチングし、位置更新と相対速度算出を行う。
        マッチしなかった観測は新規トラックとして登録する。
        戻り値: このフレームで実際に観測された（更新された）トラックのリスト
        """
        # --- 候補ペア（観測 × 既存トラック）を距離付きで列挙し、距離昇順でgreedyに割当 ---
        candidate_pairs = []
        for obs_idx, obs in enumerate(observations):
            for track_id, track in self.tracks.items():
                if track.class_name != obs["class_name"]:
                    continue
                dist = math.hypot(obs["x_rel"] - track.x_rel, obs["y_rel"] - track.y_rel)
                if dist <= self.max_track_match_distance_m:
                    candidate_pairs.append((dist, obs_idx, track_id))

        candidate_pairs.sort(key=lambda p: p[0])  # 距離が近いペアから確定させる

        matched_obs = set()
        matched_tracks = set()
        assignment = {}  # obs_idx -> track_id
        for dist, obs_idx, track_id in candidate_pairs:
            if obs_idx in matched_obs or track_id in matched_tracks:
                continue
            assignment[obs_idx] = track_id
            matched_obs.add(obs_idx)
            matched_tracks.add(track_id)

        updated_tracks: List[TrackedObject] = []

        # --- マッチした観測 → 既存トラックを更新（相対速度を差分から算出） ---
        for obs_idx, obs in enumerate(observations):
            if obs_idx in assignment:
                track = self.tracks[assignment[obs_idx]]
                dt = timestamp_sec - track.last_timestamp
                if dt > 1e-6:
                    track.vx_rel = (obs["x_rel"] - track.x_rel) / dt
                    track.vy_rel = (obs["y_rel"] - track.y_rel) / dt
                track.x_rel = obs["x_rel"]
                track.y_rel = obs["y_rel"]
                track.last_frame_index = frame_index
                track.last_timestamp = timestamp_sec
                track.missed_frames = 0
                updated_tracks.append(track)
            else:
                # --- マッチしなかった観測 → 新規トラックとして登録 ---
                new_track = TrackedObject(
                    track_id=self._next_track_id,
                    class_name=obs["class_name"],
                    x_rel=obs["x_rel"],
                    y_rel=obs["y_rel"],
                    vx_rel=0.0,  # 初回検出時は速度不明のため0で初期化
                    vy_rel=0.0,
                    last_frame_index=frame_index,
                    last_timestamp=timestamp_sec,
                    missed_frames=0,
                )
                self.tracks[self._next_track_id] = new_track
                self._next_track_id += 1
                updated_tracks.append(new_track)

        # --- このフレームで観測されなかった既存トラックは missed_frames を加算 ---
        for track_id, track in list(self.tracks.items()):
            if track_id not in assignment.values():
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    del self.tracks[track_id]  # 一定フレーム見失ったら削除

        return updated_tracks

    # --------------------------------------------------------------------
    # 2-3. IRL状態ベクトルの構築（固定長）
    # --------------------------------------------------------------------
    def _build_state_vector(self, visible_tracks: List[TrackedObject]) -> List[float]:
        """
        STATE_VECTOR_SCHEMA の順序に従って固定長ベクトルを構築する。
        自車に最も影響を与える物体（前方最近接車両・左右隣接車線車両・最近接歩行者）
        のみを抽出し、他は無視する設計（IRLの状態空間を過大にしないため）。
        """
        vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}

        front_candidates = []
        left_candidates = []
        right_candidates = []
        pedestrian_candidates = []

        half_lane = self.lane_width_m / 2.0

        for track in visible_tracks:
            if track.class_name in vehicle_classes:
                # 自車と同一車線（前方）: 横オフセットが車線幅の半分以内、かつ前方
                if abs(track.x_rel) <= half_lane and track.y_rel > 0:
                    front_candidates.append(track)
                # 左隣車線: 左寄り(x_rel が負)で1〜1.5車線分オフセット
                elif -1.5 * self.lane_width_m <= track.x_rel < -half_lane:
                    left_candidates.append(track)
                # 右隣車線: 右寄り(x_rel が正)で1〜1.5車線分オフセット
                elif half_lane < track.x_rel <= 1.5 * self.lane_width_m:
                    right_candidates.append(track)
            elif track.class_name == "person":
                pedestrian_candidates.append(track)

        def nearest(candidates: List[TrackedObject]) -> Optional[TrackedObject]:
            if not candidates:
                return None
            return min(candidates, key=lambda t: t.y_rel if t.y_rel > 0 else float("inf"))

        def nearest_by_euclidean(candidates: List[TrackedObject]) -> Optional[TrackedObject]:
            if not candidates:
                return None
            return min(candidates, key=lambda t: math.hypot(t.x_rel, t.y_rel))

        front = nearest(front_candidates)
        left = nearest(left_candidates)
        right = nearest(right_candidates)
        pedestrian = nearest_by_euclidean(pedestrian_candidates)

        vector = [
            self.ego_speed_mps,
            front.y_rel if front else self.no_object_distance_m,
            front.vy_rel if front else 0.0,
            left.y_rel if left else self.no_object_distance_m,
            left.vy_rel if left else 0.0,
            right.y_rel if right else self.no_object_distance_m,
            right.vy_rel if right else 0.0,
            math.hypot(pedestrian.x_rel, pedestrian.y_rel) if pedestrian else self.no_object_distance_m,
            pedestrian.x_rel if pedestrian else 0.0,
            float(len(front_candidates) + len(left_candidates) + len(right_candidates)),
            float(len(pedestrian_candidates)),
        ]
        return vector

    # --------------------------------------------------------------------
    # 2-4. 1フレーム処理（★オンライン/リアルタイム呼び出しの入口はここ★）
    #      Phase3のCARLAループから毎フレーム呼び出す場合は、この関数だけ使えばよい。
    # --------------------------------------------------------------------
    def process_frame(
        self,
        frame_index: int,
        timestamp_sec: float,
        detections: List[Dict],
    ) -> FrameFeatureRecord:
        """
        1フレーム分のYOLOv8検出結果（Phase2と同じ辞書構造のリスト）を受け取り、
        - 相対位置推定
        - トラッキング更新
        - 固定長状態ベクトル構築
        を行い、FrameFeatureRecordとして返す。
        """
        observations = []
        for det in detections:
            pos = self.estimate_relative_position(det)
            if pos is None:
                continue
            x_rel, y_rel = pos
            observations.append({
                "class_name": det["class_name"],
                "confidence": det.get("confidence", 0.0),
                "x_rel": x_rel,
                "y_rel": y_rel,
            })

        visible_tracks = self._match_and_update_tracks(frame_index, timestamp_sec, observations)
        state_vector = self._build_state_vector(visible_tracks)

        tracked_objects_detail = [
            {
                "track_id": t.track_id,
                "class_name": t.class_name,
                "x_rel_m": round(t.x_rel, 3),
                "y_rel_m": round(t.y_rel, 3),
                "vx_rel_mps": round(t.vx_rel, 3),
                "vy_rel_mps": round(t.vy_rel, 3),
            }
            for t in visible_tracks
        ]

        return FrameFeatureRecord(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            state_vector=state_vector,
            tracked_objects=tracked_objects_detail,
        )

    # --------------------------------------------------------------------
    # 2-5. バッチ処理（Phase2のJSONログ全体を一括変換）
    # --------------------------------------------------------------------
    def process_log(self, log_data: Dict) -> List[FrameFeatureRecord]:
        """
        Phase2で保存されたJSONログ全体（dict）を読み込み、
        フレームごとに process_frame() を順番に呼び出して時系列変換する。
        検出が1件もないフレームも空リストとして処理し、トラックの missed_frames を進める。
        """
        detections_by_frame: Dict[int, List[Dict]] = {}
        for det in log_data.get("detections", []):
            detections_by_frame.setdefault(det["frame_index"], []).append(det)

        if not detections_by_frame:
            return []

        fps = log_data.get("fps", 30.0)
        min_frame = min(detections_by_frame.keys())
        max_frame = max(detections_by_frame.keys())

        records: List[FrameFeatureRecord] = []
        for frame_index in range(min_frame, max_frame + 1):
            dets = detections_by_frame.get(frame_index, [])
            # 検出ログにtimestampが無いフレーム(検出0件)は fps から逆算する
            if dets:
                timestamp_sec = dets[0]["timestamp_sec"]
            else:
                timestamp_sec = frame_index / fps
            record = self.process_frame(frame_index, timestamp_sec, dets)
            records.append(record)

        return records


# ==============================================================================
# 3. 入出力ユーティリティ
# ==============================================================================
def load_detection_log(path: str) -> Dict:
    """Phase2で保存されたJSON検出ログを読み込む"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"検出ログが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_feature_outputs(
    records: List[FrameFeatureRecord],
    output_dir: str,
    npy_name: str,
    json_name: str,
) -> Tuple[str, str]:
    """
    - .npy : 状態ベクトルのみを (フレーム数, STATE_VECTOR_DIM) の numpy配列として保存
             → そのままIRL学習ループ(state = feature_matrix[t])に読み込める形式
    - .json: トラック詳細つきの人間可読なログ（デバッグ・Phase3検証用）
    """
    os.makedirs(output_dir, exist_ok=True)

    feature_matrix = np.array([r.state_vector for r in records], dtype=np.float32)
    npy_path = os.path.join(output_dir, npy_name)
    np.save(npy_path, feature_matrix)

    json_path = os.path.join(output_dir, json_name)
    payload = {
        "state_vector_schema": STATE_VECTOR_SCHEMA,
        "num_frames": len(records),
        "frames": [r.to_dict() for r in records],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return npy_path, json_path


def create_sample_log_if_missing(path: str) -> None:
    """
    動作確認用：Phase2の実ログがまだ無い場合に、簡単なダミー検出ログを生成する。
    （前方車両がゆっくり近づき、隣接車線をバイクが追い越していく想定の合成データ）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fps = 10.0
    frame_width, frame_height = 1280, 720
    detections = []

    for frame_index in range(30):
        t = frame_index / fps

        # 前方の車：徐々に近づく(バウンディングボックスが大きくなる)想定
        front_car_width = 80 + frame_index * 1.5
        detections.append({
            "frame_index": frame_index,
            "timestamp_sec": t,
            "class_name": "car",
            "confidence": 0.9,
            "center_x": frame_width / 2.0,
            "center_y": frame_height * 0.55,
            "width": front_car_width,
            "height": front_car_width * 0.8,
        })

        # 右車線をバイクが追い越していく想定（横方向に移動）
        moto_center_x = frame_width * 0.75 + frame_index * 4.0
        if moto_center_x < frame_width:
            detections.append({
                "frame_index": frame_index,
                "timestamp_sec": t,
                "class_name": "motorcycle",
                "confidence": 0.8,
                "center_x": moto_center_x,
                "center_y": frame_height * 0.6,
                "width": 40,
                "height": 60,
            })

    payload = {
        "source_video": "SAMPLE_SYNTHETIC_DATA",
        "model_name": "yolov8n.pt",
        "device": "cpu",
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "num_detections": len(detections),
        "detections": detections,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Phase2の実ログが見つからなかったため、動作確認用のサンプルログを生成しました: {path}")


# ==============================================================================
# 4. メイン処理（テストコード）
# ==============================================================================
def main():
    cfg = Config()

    # --- 1. Phase2ログの読み込み（無ければ動作確認用サンプルを自動生成） ---
    if not os.path.exists(cfg.INPUT_LOG_PATH):
        print(f"[WARN] 指定パスにログが見つかりません: {cfg.INPUT_LOG_PATH}")
        create_sample_log_if_missing(cfg.INPUT_LOG_PATH)

    log_data = load_detection_log(cfg.INPUT_LOG_PATH)
    print(f"[INFO] 検出ログを読み込みました: {cfg.INPUT_LOG_PATH}")
    print(f"[INFO] 総検出数: {log_data.get('num_detections')}, FPS: {log_data.get('fps')}")

    # --- 2. Extractorを初期化（カメラ解像度はログのメタ情報から取得） ---
    extractor = PerceptionFeatureExtractor(
        frame_width=log_data["frame_width"],
        frame_height=log_data["frame_height"],
        assumed_hfov_deg=cfg.ASSUMED_HORIZONTAL_FOV_DEG,
        ego_speed_kmh=cfg.ASSUMED_EGO_SPEED_KMH,
        lane_width_m=cfg.LANE_WIDTH_M,
    )

    # --- 3. バッチ変換の実行 ---
    start = time.time()
    records = extractor.process_log(log_data)
    elapsed = time.time() - start
    print(f"[INFO] {len(records)}フレーム分の特徴量変換が完了しました（{elapsed:.3f}秒）")

    # --- 4. コンソールに冒頭数フレームを表示（動作確認用） ---
    print("\n[STATE VECTOR SCHEMA]")
    print("  " + ", ".join(STATE_VECTOR_SCHEMA))

    print("\n[SAMPLE OUTPUT] 先頭5フレーム分の状態ベクトル:")
    for record in records[:5]:
        vec_str = ", ".join(f"{v:.2f}" for v in record.state_vector)
        print(f"  frame={record.frame_index:04d} t={record.timestamp_sec:.2f}s -> [{vec_str}]")

    # --- 5. 出力保存(.npy / .json) ---
    npy_path, json_path = save_feature_outputs(
        records,
        output_dir=cfg.OUTPUT_DIR,
        npy_name=cfg.OUTPUT_NPY_NAME,
        json_name=cfg.OUTPUT_JSON_NAME,
    )
    print(f"\n[INFO] 状態ベクトル行列(.npy)を保存しました: {npy_path}")
    print(f"[INFO] 詳細ログ(.json)を保存しました: {json_path}")

    print("\n[DONE] Phase 2.5（IRL用時系列特徴量変換）の実行が完了しました。")


if __name__ == "__main__":
    main()