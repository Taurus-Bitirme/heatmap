"""
heatmap_pipeline.py
===================
Yangın Risk Isı Haritası Oluşturma Pipeline'ı

Akış:
  1. Drone kalkış + 40 kare çekimi → PID tabanlı optimal BB tespiti
  2. Depth kamera ile tespit edilen BB'lerin gerçek alanını (m²) hesaplama + GPS koordinatı
  3. OpenCV ile renk analizi (yangın renk skoru)
  4. Risk skoru hesaplama: renk×0.2 + alan×0.3 + model×0.5
  5. 0-1 arası skor → 5 risk kategorisine bölme
  6. Folium harita üzerinde ısı haritası oluşturma
"""

import os
import time
import math
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import airsim
import cv2
import numpy as np
from ultralytics import YOLO

from static_heatmap import generate_static_osm_heatmap

# ============================================================
#  AYARLAR
# ============================================================
MODEL_PATH = "./modelimiz.pt"
CAMERA_NAME = "0"
TARGET_ALTITUDE = 10.0

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

CAMERA_FOV_H = 90  # Yatay FOV (derece)
FOV_H_RAD = math.radians(CAMERA_FOV_H)
FOV_V_RAD = 2 * math.atan(math.tan(FOV_H_RAD / 2) * (IMAGE_HEIGHT / IMAGE_WIDTH))

# BB Optimizasyonu
WARMUP_FRAMES = 4
CAPTURE_COUNT = 40
FRAME_INTERVAL_SEC = 0.15

EARLY_STOP_MIN_FRAMES = 40
EARLY_STOP_STD_THRESHOLD = 0.13
EARLY_STOP_WINDOW = 6

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

FIRE_KEYWORDS = {"fire", "smoke"}

TRACK_IOU_MATCH_THRESHOLD = 0.20
MAX_TRACK_MISSED_FRAMES = 5
MIN_TRACK_OBSERVATIONS = 4
FINAL_MERGE_IOU_THRESHOLD = 0.15
FINAL_MERGE_CENTER_DIST_PX = 150.0

# Risk katsayıları
WEIGHT_COLOR = 0.2
WEIGHT_AREA = 0.3
WEIGHT_MODEL = 0.5

OUTPUT_DIR = "scan_results"
FRAMES_RAW_DIR = os.path.join(OUTPUT_DIR, "pid_frames_raw")
FRAMES_ANN_DIR = os.path.join(OUTPUT_DIR, "pid_frames_annotated")
HEATMAP_REPORT = os.path.join(OUTPUT_DIR, "heatmap_report.json")
HEATMAP_PNG = os.path.join(OUTPUT_DIR, "heatmap_osm.png")
FINAL_IMG_PNG = os.path.join(OUTPUT_DIR, "heatmap_result.png")

SHOW_LIVE_PREVIEW = True
SAVE_EVERY_FRAME = True


# ============================================================
#  YARDIMCI SINIFLAR
# ============================================================
@dataclass
class TrackDetection:
    frame_idx: int
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    conf: float
    cls_name: str
    area_px: float
    pid_est_area_px: float
    quality: float


@dataclass
class FireTrack:
    track_id: int
    pid: "PIDController"
    detections: List[TrackDetection]
    last_bbox: np.ndarray
    est_area: float
    missed: int = 0


class PIDController:
    """Gürültülü tespitlerde bbox alanını stabilize eden basit PID."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: Tuple[float, float]):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint: float, measured: float, dt: float) -> float:
        error = setpoint - measured
        self.integral += error * dt
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        out = max(self.min_out, min(self.max_out, out))
        self.prev_error = error
        return out


# ============================================================
#  GEOMETRİK YARDIMCILAR
# ============================================================
def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_center(box: np.ndarray) -> Tuple[float, float]:
    return (float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5))


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0 else float(inter / union)


def clamp_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1 = float(max(0, min(w - 1, box[0])))
    y1 = float(max(0, min(h - 1, box[1])))
    x2 = float(max(0, min(w - 1, box[2])))
    y2 = float(max(0, min(h - 1, box[3])))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1.0)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def scale_box_around_center(box: np.ndarray, target_area: float,
                             w: int, h: int) -> np.ndarray:
    curr_area = max(1.0, box_area(box))
    scale = math.sqrt(max(1e-6, target_area / curr_area))
    cx, cy = box_center(box)
    bw = (box[2] - box[0]) * scale
    bh = (box[3] - box[1]) * scale
    scaled = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                      dtype=np.float32)
    return clamp_box(scaled, w, h)


def robust_median(values: List[float]) -> float:
    return 0.0 if not values else float(np.median(np.array(values, dtype=np.float32)))


def robust_inlier_mask(values: List[float], z_k: float = 2.8) -> np.ndarray:
    if not values:
        return np.array([], dtype=bool)
    arr = np.array(values, dtype=np.float32)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-6:
        return np.ones_like(arr, dtype=bool)
    return np.abs(0.6745 * (arr - med) / mad) <= z_k


# ============================================================
#  TESPİT YARDIMCILARI
# ============================================================
def extract_fire_candidates(result, names: dict) -> List[Tuple[np.ndarray, float, str]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    all_cands, kw_cands = [], []
    for box in boxes:
        xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
        conf = float(box.conf[0])
        cls_name = str(names.get(int(box.cls[0]), str(int(box.cls[0])))).lower()
        if conf < CONF_THRESHOLD:
            continue
        cand = (xyxy, conf, cls_name)
        all_cands.append(cand)
        if any(k in cls_name for k in FIRE_KEYWORDS):
            kw_cands.append(cand)
    return kw_cands if kw_cands else all_cands


def match_tracks_with_candidates(tracks, candidates, iou_threshold):
    if not tracks:
        return [], [], list(range(len(candidates)))
    if not candidates:
        return [], list(range(len(tracks))), []
    used_tr, used_ca = set(), set()
    scores = sorted(
        [(iou_xyxy(tr.last_bbox, cb), ti, ci)
         for ti, tr in enumerate(tracks)
         for ci, (cb, _, _) in enumerate(candidates)],
        reverse=True
    )
    matches = []
    for iou_val, ti, ci in scores:
        if iou_val < iou_threshold:
            break
        if ti in used_tr or ci in used_ca:
            continue
        used_tr.add(ti)
        used_ca.add(ci)
        matches.append((ti, ci, float(iou_val)))
    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tr]
    unmatched_cands = [i for i in range(len(candidates)) if i not in used_ca]
    return matches, unmatched_tracks, unmatched_cands


def compute_quality(conf, area_px, est_area_px, iou_pid):
    rel_err = abs(area_px - est_area_px) / max(est_area_px, 1.0)
    area_score = max(0.0, 1.0 - min(1.0, rel_err))
    return 0.40 * conf + 0.40 * area_score + 0.20 * iou_pid


def merge_fused_results(fused_results: List[dict]) -> List[dict]:
    if not fused_results:
        return []
    sorted_items = sorted(fused_results, key=lambda x: x["fused_area_px"], reverse=True)
    merged = []
    for item in sorted_items:
        box_i = np.array(item["fused_bbox_xyxy"], dtype=np.float32)
        cx_i, cy_i = box_center(box_i)
        into = False
        for m in merged:
            box_m = np.array(m["fused_bbox_xyxy"], dtype=np.float32)
            cx_m, cy_m = box_center(box_m)
            if (iou_xyxy(box_i, box_m) >= FINAL_MERGE_IOU_THRESHOLD or
                    math.hypot(cx_i - cx_m, cy_i - cy_m) <= FINAL_MERGE_CENTER_DIST_PX):
                w_i = max(1.0, float(item["observations"]))
                w_m = max(1.0, float(m["observations"]))
                w_sum = w_i + w_m
                new_box = clamp_box(
                    (box_i * w_i + box_m * w_m) / w_sum, IMAGE_WIDTH, IMAGE_HEIGHT)
                m["fused_bbox_xyxy"] = [float(v) for v in new_box.tolist()]
                m["fused_area_px"] = float(box_area(new_box))
                m["observations"] += item["observations"]
                m["inliers"] += item["inliers"]
                if item["representative_quality"] > m["representative_quality"]:
                    m["representative_frame"] = item["representative_frame"]
                    m["representative_conf"] = item["representative_conf"]
                    m["representative_quality"] = item["representative_quality"]
                ids = set(m.get("merged_track_ids", []))
                ids.update(item.get("merged_track_ids", [item["track_id"]]))
                m["merged_track_ids"] = sorted(ids)
                into = True
                break
        if not into:
            ni = dict(item)
            ni["merged_track_ids"] = [item["track_id"]]
            merged.append(ni)
    for idx, m in enumerate(merged, 1):
        m["track_id"] = idx
    return merged


# ============================================================
#  DEPTH & GPS YARDIMCILARI
# ============================================================
def bbox_area_from_depth(depth_m: float, bb_w_px: int, bb_h_px: int) -> float:
    """
    Bounding box'ın gerçek alanını derinlik kullanarak hesaplar (m²).
    depth_m  : bbox merkezindeki derinlik kamera ölçümü (metre)
    bb_w_px  : bbox genişliği (piksel)
    bb_h_px  : bbox yüksekliği (piksel)
    """
    m_per_px_w = (2.0 * depth_m * math.tan(FOV_H_RAD / 2.0)) / IMAGE_WIDTH
    m_per_px_h = (2.0 * depth_m * math.tan(FOV_V_RAD / 2.0)) / IMAGE_HEIGHT
    return (bb_w_px * m_per_px_w) * (bb_h_px * m_per_px_h)


def pixel_to_gps(drone_lat: float, drone_lon: float, yaw_rad: float,
                 depth_m: float, u_c: float, v_c: float) -> Tuple[float, float]:
    """
    Aşağı bakan kamera için piksel koordinatından yerdeki GPS konumunu hesaplar.
    Dönüş: (fire_lat, fire_lon) — (derece, derece)
    """
    du = u_c - IMAGE_WIDTH / 2.0
    dv = v_c - IMAGE_HEIGHT / 2.0

    m_per_px_w = (2.0 * depth_m * math.tan(FOV_H_RAD / 2.0)) / IMAGE_WIDTH
    m_per_px_h = (2.0 * depth_m * math.tan(FOV_V_RAD / 2.0)) / IMAGE_HEIGHT

    right_offset = du * m_per_px_w
    forward_offset = dv * m_per_px_h

    offset_north = forward_offset * math.cos(yaw_rad) - right_offset * math.sin(yaw_rad)
    offset_east = forward_offset * math.sin(yaw_rad) + right_offset * math.cos(yaw_rad)

    dlat = offset_north / 111320.0
    dlon = offset_east / (111320.0 * math.cos(math.radians(drone_lat)))

    return drone_lat + dlat, drone_lon + dlon


# ============================================================
#  FAZ 3 – RENK ANALİZİ
# ============================================================
def analyze_fire_color(img_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> float:
    """
    Yangın BB'si içindeki renkleri analiz ederek 0-1 arası bir renk skoru döndürür.
    Kırmızı/turuncu → yüksek skor (aktif alev)
    Sarı → orta skor
    Koyu/gri/beyaz (duman) → düşük skor
    """
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    # Sınır kontrolü
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_bgr.shape[1], x2)
    y2 = min(img_bgr.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = max(1, roi.shape[0] * roi.shape[1])

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Kırmızı maske (H: 0-10 ve 170-180, S > 80, V > 80)
    red_mask_low = (h <= 10) & (s >= 80) & (v >= 80)
    red_mask_high = (h >= 170) & (s >= 80) & (v >= 80)
    red_mask = red_mask_low | red_mask_high

    # Turuncu maske (H: 10-25, S > 80, V > 80)
    orange_mask = (h > 10) & (h <= 25) & (s >= 80) & (v >= 80)

    # Sarı maske (H: 25-35, S > 60, V > 80)
    yellow_mask = (h > 25) & (h <= 35) & (s >= 60) & (v >= 80)

    # Beyaz/parlak maske (duman veya parıltı — S < 50, V > 200)
    bright_mask = (s < 50) & (v > 200)

    red_ratio = float(np.sum(red_mask)) / total_pixels
    orange_ratio = float(np.sum(orange_mask)) / total_pixels
    yellow_ratio = float(np.sum(yellow_mask)) / total_pixels
    bright_ratio = float(np.sum(bright_mask)) / total_pixels

    # Ağırlıklı renk skoru
    raw_score = (
        red_ratio * 1.0 +
        orange_ratio * 0.85 +
        yellow_ratio * 0.5 +
        bright_ratio * 0.2
    )

    # 0-1 arasına kısıtla (teorik max = 1.0 tüm pikseller kırmızıysa)
    color_score = min(1.0, max(0.0, raw_score))
    return color_score


# ============================================================
#  FAZ 4 & 5 – RİSK HESAPLAMA
# ============================================================
def compute_risk_score(color_score: float, area_score: float,
                       model_conf: float) -> float:
    """
    risk_score = color_score × 0.2 + area_score × 0.3 + model_conf × 0.5
    """
    return (WEIGHT_COLOR * color_score +
            WEIGHT_AREA * area_score +
            WEIGHT_MODEL * model_conf)


def risk_category(score: float) -> int:
    """0-1 arası skoru 1-5 risk kategorisine çevirir."""
    if score <= 0.2:
        return 1
    elif score <= 0.4:
        return 2
    elif score <= 0.6:
        return 3
    elif score <= 0.8:
        return 4
    else:
        return 5


def risk_color(category: int) -> str:
    """Risk kategorisine göre harita rengi (hex)."""
    colors = {
        1: "#FFD700",   # Sarı — Düşük risk
        2: "#FFA500",   # Turuncu açık — Orta-düşük risk
        3: "#FF6600",   # Turuncu koyu — Orta risk
        4: "#FF2200",   # Kırmızı açık — Yüksek risk
        5: "#8B0000",   # Koyu kırmızı — Çok yüksek risk
    }
    return colors.get(category, "#808080")


def risk_label(category: int) -> str:
    """Risk kategorisi açıklaması."""
    labels = {
        1: "Düşük Risk",
        2: "Orta-Düşük Risk",
        3: "Orta Risk",
        4: "Yüksek Risk",
        5: "Çok Yüksek Risk",
    }
    return labels.get(category, "Bilinmeyen")




# ============================================================
#  ANA PIPELINE
# ============================================================

def run_heatmap_scan(client: airsim.MultirotorClient,
                     parcel_out_dir: str,
                     vehicle_name: str = "",
                     show_preview: bool = None,
                     model=None) -> dict:
    """
    Performs one full scan cycle at the current drone position.
    Receives an already-connected, armed, airborne AirSim client.
    All output is written to parcel_out_dir (one subdirectory per parcel).
    Returns the report dict (same structure as heatmap_report.json).
    Does NOT connect, arm, takeoff, land, or disarm.

    Parameters (multi-drone support — backward compatible)
    ----------
    vehicle_name : str
        AirSim vehicle name (e.g. "Drone0"). Empty string = default vehicle.
    show_preview : bool or None
        True/False to force preview on/off. None = use global SHOW_LIVE_PREVIEW.
    model : YOLO or None
        Pre-loaded YOLO model. None = load fresh (original behavior).
    """
    # Multi-drone uyumlu önizleme ayarı
    _show = show_preview if show_preview is not None else SHOW_LIVE_PREVIEW
    _vn = vehicle_name  # kısa alias
    _frames_raw = os.path.join(parcel_out_dir, "pid_frames_raw")
    _frames_ann = os.path.join(parcel_out_dir, "pid_frames_annotated")
    _report_json = os.path.join(parcel_out_dir, "heatmap_report.json")
    _heatmap_png = os.path.join(parcel_out_dir, "heatmap_osm.png")
    _result_png  = os.path.join(parcel_out_dir, "heatmap_result.png")

    os.makedirs(parcel_out_dir, exist_ok=True)
    os.makedirs(_frames_raw, exist_ok=True)
    os.makedirs(_frames_ann, exist_ok=True)

    # ---- Model yükle ----
    if model is None:
        print(f"YOLO modeli yükleniyor: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("✓ Model yüklendi")
    else:
        print(f"✓ Önceden yüklenmiş YOLO modeli kullanılıyor")

    # Kamera aşağı (uçuş sırasında güvenlidir)
    client.simSetCameraPose(
        CAMERA_NAME,
        airsim.Pose(airsim.Vector3r(0, 0, 0),
                    airsim.to_quaternion(math.radians(-90), 0, 0)),
        vehicle_name=_vn,
    )

    # Mevcut irtifayı drone pozisyonundan oku
    position = client.simGetVehiclePose(vehicle_name=_vn).position
    actual_altitude = -position.z_val
    print(f"✓ Mevcut irtifa: {actual_altitude:.2f} m")

    if _show:
        _win_title = f"Heatmap Pipeline Preview [{_vn}]" if _vn else "Heatmap Pipeline Preview"
        cv2.namedWindow(_win_title, cv2.WINDOW_NORMAL)

    def _empty_report() -> dict:
        return {
            "altitude_m": actual_altitude,
            "drone_gps": {"lat": 0.0, "lon": 0.0},
            "captured_frames": CAPTURE_COUNT,
            "total_tracks": 0,
            "detected_fires": 0,
            "risk_weights": {
                "color": WEIGHT_COLOR,
                "area": WEIGHT_AREA,
                "model": WEIGHT_MODEL,
            },
            "fires": [],
        }

    # ==============================================================
    #  FAZ 1: PID TABANLI OPTİMAL BB TESPİTİ (40 KARE)
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"  FAZ 1: OPTİMAL BB TESPİTİ ({CAPTURE_COUNT} kare)")
    print(f"{'='*60}")

    tracks: List[FireTrack] = []
    next_track_id = 1
    best_vis = None
    best_raw_img = None

    # Isınma
    for _ in range(WARMUP_FRAMES):
        client.simGetImages(
            [airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)],
            vehicle_name=_vn)
        time.sleep(FRAME_INTERVAL_SEC)

    for i in range(1, CAPTURE_COUNT + 1):
        resp = client.simGetImages(
            [airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)],
            vehicle_name=_vn)
        if not resp or resp[0].height == 0:
            time.sleep(FRAME_INTERVAL_SEC)
            continue

        img = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8).reshape(
            resp[0].height, resp[0].width, 3).copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        preds = model.predict(rgb, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        candidates = extract_fire_candidates(preds[0], model.names)

        vis = img.copy()

        for cb, cc, cn in candidates:
            bx1, by1, bx2, by2 = map(int, cb)
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 120, 0), 1)
            cv2.putText(vis, f"{cn} {cc:.2f}", (bx1, max(16, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)

        active_idx = [idx for idx, t in enumerate(tracks)
                      if t.missed <= MAX_TRACK_MISSED_FRAMES]
        active_tr = [tracks[idx] for idx in active_idx]
        matches, unmatched_tr_l, unmatched_ca_idx = match_tracks_with_candidates(
            active_tr, candidates, TRACK_IOU_MATCH_THRESHOLD)

        matched_global = [(active_idx[ti], ci, iv) for ti, ci, iv in matches]
        unmatched_tr_global = [active_idx[ti] for ti in unmatched_tr_l]

        frame_track_logs = []
        for track_idx, cand_idx, _ in matched_global:
            track = tracks[track_idx]
            bbox, conf, cls_name = candidates[cand_idx]
            area_px = box_area(bbox)

            area_history = [d.area_px for d in track.detections] + [area_px]
            inlier_mask = robust_inlier_mask(area_history)
            inlier_areas = np.array(area_history, dtype=np.float32)[inlier_mask]
            setpoint = (float(np.median(inlier_areas)) if len(inlier_areas)
                        else robust_median(area_history))

            dt = max(FRAME_INTERVAL_SEC, 1e-3)
            correction = track.pid.update(setpoint, track.est_area, dt)
            pred_area = max(1.0, track.est_area + correction)
            track.est_area = pred_area + 0.22 * (area_px - pred_area)

            pid_box = scale_box_around_center(bbox, track.est_area,
                                               IMAGE_WIDTH, IMAGE_HEIGHT)
            iou_pid = iou_xyxy(bbox, pid_box)
            q = compute_quality(conf, area_px, track.est_area, iou_pid)

            det = TrackDetection(
                frame_idx=i, track_id=track.track_id,
                bbox=bbox.astype(np.float32), conf=conf, cls_name=cls_name,
                area_px=area_px, pid_est_area_px=float(track.est_area),
                quality=float(q))
            track.detections.append(det)
            track.last_bbox = bbox.astype(np.float32)
            track.missed = 0

            x1, y1, x2, y2 = map(int, bbox)
            px1, py1, px2, py2 = map(int, pid_box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 255), 2)
            cv2.putText(vis, f"T{track.track_id} conf={conf:.2f} q={q:.2f}",
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255), 2)
            frame_track_logs.append({
                "track_id": track.track_id, "conf": float(conf),
                "quality": float(q)
            })

        for cand_idx in unmatched_ca_idx:
            bbox, conf, cls_name = candidates[cand_idx]
            area_px = box_area(bbox)
            pid = PIDController(kp=0.35, ki=0.08, kd=0.10,
                                 output_limits=(-0.30, 0.30))
            new_track = FireTrack(
                track_id=next_track_id, pid=pid, detections=[],
                last_bbox=bbox.astype(np.float32),
                est_area=float(area_px), missed=0)
            det = TrackDetection(
                frame_idx=i, track_id=next_track_id,
                bbox=bbox.astype(np.float32), conf=float(conf),
                cls_name=cls_name, area_px=float(area_px),
                pid_est_area_px=float(area_px), quality=float(conf))
            new_track.detections.append(det)
            tracks.append(new_track)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.putText(vis, f"T{next_track_id} yeni conf={conf:.2f}",
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 200), 2)
            next_track_id += 1

        for track_idx in unmatched_tr_global:
            tracks[track_idx].missed += 1
        inactive_idx = [idx for idx in range(len(tracks)) if idx not in active_idx]
        for idx_i in inactive_idx:
            tracks[idx_i].missed += 1

        cv2.putText(vis, f"FAZ 1 | Kare {i}/{CAPTURE_COUNT}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if frame_track_logs:
            best_vis = vis
            best_raw_img = img.copy()

        if SAVE_EVERY_FRAME:
            cv2.imwrite(os.path.join(_frames_raw, f"frame_{i:03d}.png"), img)
            cv2.imwrite(os.path.join(_frames_ann, f"frame_{i:03d}.png"), vis)

        if _show:
            cv2.imshow(_win_title, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        print(f"  Kare {i}: tespit={len(frame_track_logs)}, toplam track={len(tracks)}")

        # Erken durdurma
        stable = [t for t in tracks if len(t.detections) >= EARLY_STOP_WINDOW]
        if i >= EARLY_STOP_MIN_FRAMES and stable:
            stds = [
                float(np.std([d.area_px for d in t.detections[-EARLY_STOP_WINDOW:]]) /
                      max(np.mean([d.area_px for d in t.detections[-EARLY_STOP_WINDOW:]]), 1.0))
                for t in stable
            ]
            if float(np.mean(stds)) <= EARLY_STOP_STD_THRESHOLD:
                print(f"  Erken durdurma: kare {i} stabilite sağlandı")
                break

        time.sleep(FRAME_INTERVAL_SEC)

    # ---- BB Füzyon ----
    all_dets = [d for t in tracks for d in t.detections]
    if not all_dets:
        print("⚠  Faz 1: Hiç yangın tespiti bulunamadı.")
        if _show:
            cv2.destroyAllWindows()
        return _empty_report()

    valid_tracks = [t for t in tracks if len(t.detections) >= MIN_TRACK_OBSERVATIONS]
    if not valid_tracks:
        valid_tracks = [t for t in tracks if t.detections]

    fused_results = []
    for tr in valid_tracks:
        tr_areas = [d.area_px for d in tr.detections]
        tr_mask = robust_inlier_mask(tr_areas)
        inliers = [d for d, m in zip(tr.detections, tr_mask) if bool(m)] or tr.detections
        inliers_sorted = sorted(inliers, key=lambda d: d.quality, reverse=True)
        top_k = max(2, math.ceil(len(inliers_sorted) * 0.40))
        best_g = inliers_sorted[:top_k]
        weights = np.array([max(1e-4, d.quality) for d in best_g], dtype=np.float32)
        weights /= np.sum(weights)
        fused_box = clamp_box(np.array([
            float(np.sum([d.bbox[0] for d in best_g] * weights)),
            float(np.sum([d.bbox[1] for d in best_g] * weights)),
            float(np.sum([d.bbox[2] for d in best_g] * weights)),
            float(np.sum([d.bbox[3] for d in best_g] * weights)),
        ], dtype=np.float32), IMAGE_WIDTH, IMAGE_HEIGHT)

        rep = max(inliers, key=lambda d: 0.65 * iou_xyxy(d.bbox, fused_box) + 0.35 * d.quality)
        fused_results.append({
            "track_id": tr.track_id,
            "observations": len(tr.detections),
            "inliers": len(inliers),
            "fused_bbox_xyxy": [float(v) for v in fused_box.tolist()],
            "fused_area_px": float(box_area(fused_box)),
            "representative_frame": int(rep.frame_idx),
            "representative_conf": float(rep.conf),
            "representative_quality": float(rep.quality),
        })

    merged_fires = merge_fused_results(fused_results)
    print(f"\n✓ Faz 1 tamamlandı: {len(merged_fires)} yangın tespit edildi")
    for item in merged_fires:
        bb = [round(v, 1) for v in item["fused_bbox_xyxy"]]
        print(f"  T{item['track_id']}: bbox={bb} alan={item['fused_area_px']:.1f}px "
              f"conf={item['representative_conf']:.2f}")

    # ==============================================================
    #  FAZ 2: DEPTH KAMERA İLE GERÇEK ALAN + GPS KOORDİNAT
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"  FAZ 2: DEPTH KAMERA + GPS KOORDİNAT HESAPLAMA")
    print(f"{'='*60}")

    # Depth + RGB görüntüsü al
    responses = client.simGetImages([
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPerspective, True),
    ], vehicle_name=_vn)

    if not responses or responses[0].height == 0:
        print("⚠  Depth görüntüsü alınamadı!")
        if _show:
            cv2.destroyAllWindows()
        return _empty_report()

    depth_img_bgr = np.frombuffer(
        responses[0].image_data_uint8, dtype=np.uint8
    ).reshape(responses[0].height, responses[0].width, 3).copy()

    # Depth haritası
    depth_raw = None
    if len(responses) > 1 and responses[1].width > 0:
        depth_raw = np.array(responses[1].image_data_float, dtype=np.float32)
        depth_raw = depth_raw.reshape(responses[1].height, responses[1].width)
        if depth_raw.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            depth_raw = cv2.resize(depth_raw, (IMAGE_WIDTH, IMAGE_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

    # Drone GPS ve yaw bilgisi
    try:
        gps_state = client.getMultirotorState(vehicle_name=_vn).gps_location
        drone_lat = gps_state.latitude
        drone_lon = gps_state.longitude
        drone_pose = client.simGetVehiclePose(vehicle_name=_vn)
        angles = airsim.utils.to_eularian_angles(drone_pose.orientation)
        drone_yaw_rad = angles[2]
    except Exception:
        drone_lat, drone_lon, drone_yaw_rad = 0.0, 0.0, 0.0

    print(f"  Drone GPS: Lat={drone_lat:.7f}, Lon={drone_lon:.7f}")
    print(f"  Drone Yaw: {math.degrees(drone_yaw_rad):.1f}°")

    # Yerdeki piksel boyutu (flat ground fallback)
    fov_h_rad = math.radians(CAMERA_FOV_H)
    fov_v_rad = 2 * math.atan(math.tan(fov_h_rad / 2) * (IMAGE_HEIGHT / IMAGE_WIDTH))
    gw = 2 * actual_altitude * math.tan(fov_h_rad / 2)
    gh = 2 * actual_altitude * math.tan(fov_v_rad / 2)
    m2_per_pixel = (gw * gh) / (IMAGE_WIDTH * IMAGE_HEIGHT)

    fire_entries = []

    for item in merged_fires:
        fbox = np.array(item["fused_bbox_xyxy"], dtype=np.float32)
        x1i, y1i, x2i, y2i = map(int, fbox)
        bb_w = x2i - x1i
        bb_h = y2i - y1i
        u_c = (x1i + x2i) // 2
        v_c = (y1i + y2i) // 2

        # Depth bazlı alan
        use_depth = False
        d_val = actual_altitude  # fallback
        if depth_raw is not None:
            d_val_raw = float(depth_raw[
                min(v_c, IMAGE_HEIGHT - 1),
                min(u_c, IMAGE_WIDTH - 1)
            ])
            if not (math.isinf(d_val_raw) or math.isnan(d_val_raw) or d_val_raw > 300):
                d_val = d_val_raw
                use_depth = True

        if use_depth:
            area_m2 = bbox_area_from_depth(d_val, bb_w, bb_h)
        else:
            area_m2 = bb_w * bb_h * m2_per_pixel

        # GPS koordinatı
        fire_lat, fire_lon = pixel_to_gps(
            drone_lat, drone_lon, drone_yaw_rad, d_val, u_c, v_c)

        depth_tag = f"{d_val:.1f}m" if use_depth else "tahmin"
        print(f"  T{item['track_id']}: alan={area_m2:.2f}m² "
              f"GPS=({fire_lat:.7f}, {fire_lon:.7f}) derinlik={depth_tag}")

        fire_entries.append({
            "track_id": item["track_id"],
            "fused_bbox_xyxy": item["fused_bbox_xyxy"],
            "fused_area_px": item["fused_area_px"],
            "representative_conf": item["representative_conf"],
            "area_m2": area_m2,
            "depth_m": d_val,
            "use_depth": use_depth,
            "lat": fire_lat,
            "lon": fire_lon,
        })

    print(f"✓ Faz 2 tamamlandı: {len(fire_entries)} yangın için alan ve GPS hesaplandı")

    # ==============================================================
    #  FAZ 3: RENK ANALİZİ
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"  FAZ 3: RENK ANALİZİ")
    print(f"{'='*60}")

    # Renk analizi için en son çekilen RGB görüntüsünü (depth_img_bgr) kullan
    for entry in fire_entries:
        fbox = np.array(entry["fused_bbox_xyxy"], dtype=np.float32)
        cs = analyze_fire_color(depth_img_bgr, fbox)
        entry["color_score"] = cs
        print(f"  T{entry['track_id']}: renk skoru = {cs:.3f}")

    print(f"✓ Faz 3 tamamlandı")

    # ==============================================================
    #  FAZ 4 & 5: RİSK SKORU + KATEGORİLENDİRME
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"  FAZ 4 & 5: RİSK SKORU HESAPLAMA + KATEGORİLENDİRME")
    print(f"{'='*60}")

    # Alan normalizasyonu: tüm yangınlar arasında en büyüğe göre normalize
    all_areas = [e["area_m2"] for e in fire_entries]
    max_area = max(all_areas) if all_areas else 1.0
    if max_area <= 0:
        max_area = 1.0

    for entry in fire_entries:
        # Alan skoru: 0-1 arası normalize
        area_score = min(1.0, entry["area_m2"] / max_area) if max_area > 0 else 0.0

        # Model güven skoru (zaten 0-1)
        model_conf = entry["representative_conf"]

        # Renk skoru (zaten 0-1)
        color_score = entry["color_score"]

        # Risk skoru
        rs = compute_risk_score(color_score, area_score, model_conf)
        cat = risk_category(rs)

        entry["area_score"] = area_score
        entry["risk_score"] = rs
        entry["risk_cat"] = cat
        entry["model_conf"] = model_conf

        cat_label = risk_label(cat)
        cat_color = risk_color(cat)
        print(f"  T{entry['track_id']}: "
              f"renk={color_score:.2f}×{WEIGHT_COLOR} + "
              f"alan={area_score:.2f}×{WEIGHT_AREA} + "
              f"model={model_conf:.2f}×{WEIGHT_MODEL} "
              f"→ skor={rs:.3f} → Kategori {cat} ({cat_label})")

    print(f"✓ Faz 4 & 5 tamamlandı")

    # ==============================================================
    #  FAZ 6: FOLİUM HARİTA OLUŞTURMA
    # ==============================================================
    print(f"\n{'='*60}")
    print(f"  FAZ 6: ISI HARİTASI OLUŞTURMA")
    print(f"{'='*60}")

    generate_static_osm_heatmap(fire_entries, _heatmap_png)

    # ---- Özet görsel kaydet ----
    if best_raw_img is not None:
        vis_final = best_raw_img.copy()
    else:
        vis_final = depth_img_bgr.copy()

    for entry in fire_entries:
        fbox = np.array(entry["fused_bbox_xyxy"], dtype=np.float32)
        fx1, fy1, fx2, fy2 = map(int, fbox)
        cat = entry["risk_cat"]
        # Renk: BGR olarak
        rc_hex = risk_color(cat)
        rc_bgr = (int(rc_hex[5:7], 16), int(rc_hex[3:5], 16), int(rc_hex[1:3], 16))
        cv2.rectangle(vis_final, (fx1, fy1), (fx2, fy2), rc_bgr, 3)
        cv2.putText(
            vis_final,
            f"T{entry['track_id']} Risk:{cat} ({entry['risk_score']:.2f})",
            (fx1, max(20, fy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, rc_bgr, 2,
        )

    cv2.imwrite(_result_png, vis_final)
    print(f"✓ Sonuç görseli kaydedildi: {_result_png}")

    # ---- JSON Rapor ----
    report = {
        "altitude_m": actual_altitude,
        "drone_gps": {"lat": drone_lat, "lon": drone_lon},
        "captured_frames": CAPTURE_COUNT,
        "total_tracks": len(tracks),
        "detected_fires": len(fire_entries),
        "risk_weights": {
            "color": WEIGHT_COLOR,
            "area": WEIGHT_AREA,
            "model": WEIGHT_MODEL,
        },
        "fires": [],
    }
    for entry in fire_entries:
        report["fires"].append({
            "track_id": entry["track_id"],
            "lat": entry["lat"],
            "lon": entry["lon"],
            "area_m2": round(entry["area_m2"], 4),
            "depth_m": round(entry["depth_m"], 2),
            "use_depth": entry["use_depth"],
            "model_conf": round(entry["model_conf"], 4),
            "color_score": round(entry["color_score"], 4),
            "area_score": round(entry["area_score"], 4),
            "risk_score": round(entry["risk_score"], 4),
            "risk_category": entry["risk_cat"],
            "risk_label": risk_label(entry["risk_cat"]),
            "fused_bbox_xyxy": entry["fused_bbox_xyxy"],
        })

    report["parcel_out_dir"] = parcel_out_dir

    with open(_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Rapor kaydedildi: {_report_json}")

    # ---- Konsol Özet ----
    print(f"\n{'='*60}")
    print(f"  PIPELINE SONUÇ ÖZETİ")
    print(f"{'='*60}")
    print(f"  İrtifa          : {actual_altitude:.2f} m")
    print(f"  Tespit edilen   : {len(fire_entries)} yangın")
    for entry in fire_entries:
        print(f"    T{entry['track_id']}: "
              f"Risk {entry['risk_cat']}/5 ({risk_label(entry['risk_cat'])}) "
              f"skor={entry['risk_score']:.3f} "
              f"alan={entry['area_m2']:.2f}m²")
    print(f"  Isı haritası (OSM) : {_heatmap_png}")
    print(f"  Sonuç görseli (BB) : {_result_png}")
    print(f"{'='*60}")

    if _show:
        cv2.destroyAllWindows()

    return report


def main_standalone():
    """
    Standalone çalıştırma modu — bağımsız test için kendi
    connect/arm/takeoff/land döngüsünü içerir.
    Görev denetleyicisinden çağrılmaz; yalnızca doğrudan
    `python heatmap_OPtimUS.py` ile test amacıyla kullanılır.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FRAMES_RAW_DIR, exist_ok=True)
    os.makedirs(FRAMES_ANN_DIR, exist_ok=True)

    # ---- AirSim bağlantısı ----
    print("AirSim'e bağlanılıyor...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✓ Bağlantı başarılı, API aktif")

    # ---- Kalkış ----
    print(f"\nKalkış → {TARGET_ALTITUDE:.1f} m irtifa")
    client.takeoffAsync().join()
    client.moveToZAsync(-TARGET_ALTITUDE, 2.5).join()
    time.sleep(1.0)

    try:
        run_heatmap_scan(client)
    finally:
        # Güvenli iniş
        try:
            print("\nEve dönülüyor ve iniş yapılıyor...")
            client.moveToPositionAsync(0.0, 0.0, -TARGET_ALTITUDE, 2.5).join()
            client.landAsync().join()
        except Exception:
            pass

        client.simSetCameraPose(
            CAMERA_NAME,
            airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)),
        )
        client.armDisarm(False)
        client.enableApiControl(False)

        print("✓ Pipeline tamamlandı!")


if __name__ == "__main__":
    main_standalone()
