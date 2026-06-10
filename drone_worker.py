"""
drone_worker.py
===============
Tek drone thread worker — zigzag tarama + YOLO bbox konum dogrulugu.

Her waypoint'te:
  1. moveToPositionAsync → hedefe uc
  2. Kameradan TEK kare al
  3. YOLO inference (yerelde, BGR)
  4. bbox merkezi → kamera geometrisi → gercek zemin NED koordinati
  5. Queue.put( (Drone_ID, fire_X, fire_Y, Skor) )
"""

import math
import time
import threading
from typing import List, Tuple, Optional, Dict
import queue
from dataclasses import dataclass, field

import cv2
import numpy as np
import airsim
from ultralytics import YOLO

from multi_drone_config import (
    ALTITUDE_M, CRUISE_SPEED_MPS, HOVER_SEC,
    CAMERA_NAME, MODEL_PATH, IMAGE_SIZE,
    ALTITUDE_OFFSETS, YOLO_CONF, YOLO_IOU,
    DETECTION_FRAME_COUNT, DETECTION_FRAME_INTERVAL_SEC,
    TRACK_IOU_MATCH_THRESHOLD, MIN_TRACK_OBSERVATIONS, FIRE_KEYWORDS,
    bbox_to_ground_ned, bbox_area_from_depth, pixel_to_gps,
    WEIGHT_COLOR, WEIGHT_AREA, WEIGHT_MODEL,
)

class PIDController:
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
        self.prev_error = error
        return max(self.min_out, min(self.max_out, out))


@dataclass
class TrackDetection:
    bbox: np.ndarray
    conf: float
    area_px: float
    pid_area_px: float
    quality: float


@dataclass
class FireTrack:
    track_id: int
    pid: PIDController
    detections: List[TrackDetection] = field(default_factory=list)
    images: List[np.ndarray] = field(default_factory=list)   # Her frame'in goruntusunu sakla
    last_bbox: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    est_area: float = 0.0
    missed: int = 0


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return 0.0 if union <= 0 else float(inter / union)


def _clamp_box(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1 = float(max(0, min(w - 1, box[0])))
    y1 = float(max(0, min(h - 1, box[1])))
    x2 = float(max(0, min(w - 1, box[2])))
    y2 = float(max(0, min(h - 1, box[3])))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1.0)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _scale_box_around_center(box: np.ndarray, target_area: float,
                              w: int, h: int) -> np.ndarray:
    curr_area = max(1.0, _box_area(box))
    scale = math.sqrt(max(1e-6, target_area / curr_area))
    cx = float((box[0] + box[2]) * 0.5)
    cy = float((box[1] + box[3]) * 0.5)
    bw = (box[2] - box[0]) * scale
    bh = (box[3] - box[1]) * scale
    return _clamp_box(
        np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                 dtype=np.float32),
        w, h,
    )


def _robust_inlier_mask(values: List[float], z_k: float = 2.8) -> np.ndarray:
    if not values:
        return np.array([], dtype=bool)
    arr = np.array(values, dtype=np.float32)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-6:
        return np.ones_like(arr, dtype=bool)
    return np.abs(0.6745 * (arr - med) / mad) <= z_k


def _compute_quality(conf: float, area_px: float, est_area_px: float,
                     iou_pid: float) -> float:
    rel_err = abs(area_px - est_area_px) / max(est_area_px, 1.0)
    area_score = max(0.0, 1.0 - min(1.0, rel_err))
    return 0.40 * conf + 0.40 * area_score + 0.20 * iou_pid


def _edge_clip_ratio(box: np.ndarray, img_w: int, img_h: int,
                     margin: float = 4.0) -> float:
    """
    Bir bbox'in goruntu kenarlarina ne kadar yapisik oldugunu olcer.
    0.0 = tamamen icerde, 1.0 = ciddi sekilde kirilmis.

    Yuksek kirilma orani, alevin goruntu disina tastigini gosterir.
    Bu durumda _fire_color_centroid yanlis merkez verir ve NED hesabi
    sistematik hata uretir. Bu deger quality'den cikararak cezalandirilir.
    """
    bw = max(box[2] - box[0], 1.0)
    bh = max(box[3] - box[1], 1.0)
    clip_x1 = max(0.0, margin - box[0])            / bw
    clip_y1 = max(0.0, margin - box[1])            / bh
    clip_x2 = max(0.0, box[2] - (img_w - margin)) / bw
    clip_y2 = max(0.0, box[3] - (img_h - margin)) / bh
    return float(min(1.0, clip_x1 + clip_y1 + clip_x2 + clip_y2))


def _fire_color_centroid(img_bgr: np.ndarray, bbox: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)

    roi = img_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    red = ((hue <= 10) | (hue >= 170)) & (sat >= 80) & (val >= 90)
    orange = (hue > 10) & (hue <= 25) & (sat >= 80) & (val >= 90)
    yellow = (hue > 25) & (hue <= 38) & (sat >= 60) & (val >= 110)
    mask = red | orange | yellow

    if int(np.count_nonzero(mask)) < 8:
        return float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)

    weights = val.astype(np.float32) * mask.astype(np.float32)
    ys, xs = np.indices(mask.shape)
    total = float(weights.sum())
    if total <= 1e-6:
        return float((bbox[0] + bbox[2]) * 0.5), float((bbox[1] + bbox[3]) * 0.5)

    cx = x1 + float((xs * weights).sum() / total)
    cy = y1 + float((ys * weights).sum() / total)
    return cx, cy

# ============================================================
# RENK ANALİZİ
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

class DroneWorker(threading.Thread):
    """
    Tek drone thread'i — zigzag waypoint takibi + YOLO yangin tespiti.
    """

    def __init__(
        self,
        drone_id: str,
        waypoints: List[Tuple[float, float]],
        result_queue: queue.Queue,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(name=f"Worker-{drone_id}", daemon=True)
        self.drone_id = drone_id
        self.waypoints = waypoints
        self.result_queue = result_queue
        self.stop_event = stop_event or threading.Event()

        self.altitude = ALTITUDE_M + ALTITUDE_OFFSETS.get(drone_id, 0.0)
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.scanned_count: int = 0
        self.fire_count: int = 0
        self.status: str = "BEKLEMEDE"

    def _extract_fire_candidates(self, result, names: dict) -> List[Tuple[np.ndarray, float]]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        all_candidates: List[Tuple[np.ndarray, float]] = []
        keyword_candidates: List[Tuple[np.ndarray, float]] = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONF:
                continue
            xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
            cls_name = str(names.get(int(box.cls[0]), str(int(box.cls[0])))).lower()
            candidate = (xyxy, conf)
            all_candidates.append(candidate)
            if any(keyword in cls_name for keyword in FIRE_KEYWORDS):
                keyword_candidates.append(candidate)

        return keyword_candidates if keyword_candidates else all_candidates

    def _capture_stable_detections(
        self,
        client: airsim.MultirotorClient,
        model: YOLO,
    ) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        tracks: List[FireTrack] = []
        next_track_id = 1
        last_img: Optional[np.ndarray] = None

        for _ in range(DETECTION_FRAME_COUNT):
            resp = client.simGetImages(
                [airsim.ImageRequest(
                    CAMERA_NAME, airsim.ImageType.Scene, False, False
                )],
                vehicle_name=self.drone_id,
            )
            if not resp or resp[0].height == 0:
                time.sleep(DETECTION_FRAME_INTERVAL_SEC)
                continue

            img = np.frombuffer(
                resp[0].image_data_uint8, dtype=np.uint8
            ).reshape(resp[0].height, resp[0].width, 3).copy()
            last_img = img

            results = model.predict(
                img,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                verbose=False,
            )
            candidates = self._extract_fire_candidates(results[0], model.names) if results else []

            used_tracks = set()
            used_candidates = set()
            matches = []
            scores = sorted(
                [
                    (_iou_xyxy(track.last_bbox, bbox), ti, ci)
                    for ti, track in enumerate(tracks)
                    if track.missed <= 5
                    for ci, (bbox, _) in enumerate(candidates)
                ],
                reverse=True,
            )

            for iou_val, track_idx, cand_idx in scores:
                if iou_val < TRACK_IOU_MATCH_THRESHOLD:
                    break
                if track_idx in used_tracks or cand_idx in used_candidates:
                    continue
                used_tracks.add(track_idx)
                used_candidates.add(cand_idx)
                matches.append((track_idx, cand_idx))

            for track_idx, cand_idx in matches:
                track = tracks[track_idx]
                bbox, conf = candidates[cand_idx]
                area_px = _box_area(bbox)
                area_history = [d.area_px for d in track.detections] + [area_px]
                mask = _robust_inlier_mask(area_history)
                inlier_areas = np.array(area_history, dtype=np.float32)[mask]
                setpoint = float(np.median(inlier_areas)) if len(inlier_areas) else area_px

                correction = track.pid.update(
                    setpoint, track.est_area, DETECTION_FRAME_INTERVAL_SEC
                )
                pred_area = max(1.0, track.est_area + correction)
                track.est_area = pred_area + 0.22 * (area_px - pred_area)

                pid_box = _scale_box_around_center(
                    bbox, track.est_area, img.shape[1], img.shape[0]
                )
                quality = _compute_quality(
                    conf, area_px, track.est_area, _iou_xyxy(bbox, pid_box)
                )

                track.detections.append(TrackDetection(
                    bbox=bbox,
                    conf=conf,
                    area_px=area_px,
                    pid_area_px=float(track.est_area),
                    quality=float(quality),
                ))
                track.images.append(img)   # Bu frame'in goruntusunu sakla
                track.last_bbox = bbox
                track.missed = 0

            for cand_idx, (bbox, conf) in enumerate(candidates):
                if cand_idx in used_candidates:
                    continue
                area_px = _box_area(bbox)
                track = FireTrack(
                    track_id=next_track_id,
                    pid=PIDController(0.35, 0.08, 0.10, (-0.30, 0.30)),
                    last_bbox=bbox,
                    est_area=area_px,
                )
                track.detections.append(TrackDetection(
                    bbox=bbox,
                    conf=conf,
                    area_px=area_px,
                    pid_area_px=area_px,
                    quality=conf,
                ))
                track.images.append(img)
                tracks.append(track)
                next_track_id += 1

            for track_idx, track in enumerate(tracks):
                if track_idx not in used_tracks:
                    track.missed += 1

            time.sleep(DETECTION_FRAME_INTERVAL_SEC)

        # -----------------------------------------------------------------
        #  Stable detection ciktisi
        # -----------------------------------------------------------------
        stable_detections: List[Tuple[np.ndarray, float, np.ndarray]] = []
        for track in tracks:
            if len(track.detections) < MIN_TRACK_OBSERVATIONS:
                continue

            areas = [d.area_px for d in track.detections]
            mask = _robust_inlier_mask(areas)
            inliers_idx = [i for i, keep in enumerate(mask) if bool(keep)]
            if not inliers_idx:
                inliers_idx = list(range(len(track.detections)))

            inliers = [track.detections[i] for i in inliers_idx]

            # -------------------------------------------------------
            # Kenar kirpma cezasi: bbox goruntu kenarina yapismissa,
            # _fire_color_centroid yanlis merkez verir. Bu frame'leri
            # dusuk agirlikla degerlendiriyoruz.
            # -------------------------------------------------------
            def _effective_quality(det: TrackDetection) -> float:
                clip = _edge_clip_ratio(det.bbox, IMAGE_SIZE, IMAGE_SIZE)
                # %50+ kirpma ciddi ceza: max 0.60 puan duser
                clip_penalty = clip * 0.60
                return max(1e-4, det.quality - clip_penalty)

            top = sorted(inliers, key=_effective_quality, reverse=True)
            top = top[:max(2, math.ceil(len(top) * 0.40))]
            weights = np.array(
                [max(1e-4, _effective_quality(d)) for d in top],
                dtype=np.float32
            )
            weights /= np.sum(weights)
            fused_box = _clamp_box(
                np.array([
                    float(np.sum([d.bbox[0] for d in top] * weights)),
                    float(np.sum([d.bbox[1] for d in top] * weights)),
                    float(np.sum([d.bbox[2] for d in top] * weights)),
                    float(np.sum([d.bbox[3] for d in top] * weights)),
                ], dtype=np.float32),
                IMAGE_SIZE, IMAGE_SIZE,
            )
            best_conf = max(d.conf for d in inliers)

            # ---------------------------------------------------------
            # En az kirpilmis frame'i renk centroidu icin sec.
            # Son frame yerine, alev en iyi gorunen frame kullanilir.
            # Bu, sistematik NED hatasini azaltir.
            # ---------------------------------------------------------
            best_img = last_img
            best_eq = -1.0
            for det, frame_img in zip(
                [track.detections[i] for i in inliers_idx],
                [track.images[i] for i in inliers_idx if i < len(track.images)],
            ):
                eq = _effective_quality(det)
                if eq > best_eq:
                    best_eq = eq
                    best_img = frame_img

            if best_img is not None:
                stable_detections.append((fused_box, best_conf, best_img))

        return stable_detections

    def _capture_depth_frame(
        self,
        client: airsim.MultirotorClient,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Eş zamanlı Scene + DepthPerspective çifti alır.
        Dönüş: (depth_raw, depth_img_bgr)
          depth_raw      : float32 derinlik matrisi (IMAGE_SIZE × IMAGE_SIZE)  veya None
          depth_img_bgr  : RGB görüntü (BGR, uint8)  veya None
        """
        try:
            resp = client.simGetImages([
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False),
                airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.DepthPerspective, True),
            ], vehicle_name=self.drone_id)
        except Exception as e:
            self._log(f"  ⚠ Depth çerçeve hatası: {e}")
            return None, None

        depth_img_bgr = None
        depth_raw = None

        if resp and resp[0].height > 0:
            depth_img_bgr = np.frombuffer(
                resp[0].image_data_uint8, dtype=np.uint8
            ).reshape(resp[0].height, resp[0].width, 3).copy()

        if len(resp) > 1 and resp[1].width > 0:
            raw = np.array(resp[1].image_data_float, dtype=np.float32)
            raw = raw.reshape(resp[1].height, resp[1].width)
            if raw.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
            depth_raw = raw

        return depth_raw, depth_img_bgr

    def run(self) -> None:
        self.status = "BASLATILIYOR"
        self._log(f"Gorev basliyor — {len(self.waypoints)} waypoint")

        try:
            # 1. AirSim baglantisi (her thread ayri client)
            client = airsim.MultirotorClient()
            client.confirmConnection()
            client.enableApiControl(True, vehicle_name=self.drone_id)
            client.armDisarm(True, vehicle_name=self.drone_id)
            self._log("AirSim baglantisi kuruldu")

            # 2. Kamerayi asagi bak
            client.simSetCameraPose(
                CAMERA_NAME,
                airsim.Pose(
                    airsim.Vector3r(0, 0, 0),
                    airsim.to_quaternion(math.radians(-90), 0, 0),
                ),
                vehicle_name=self.drone_id,
            )

            # 3. YOLO modelini yukle
            self._log("YOLO modeli yukleniyor...")
            model = YOLO(MODEL_PATH)
            self._log("YOLO modeli yuklendi")

            # 4. Kalkis
            self._log(f"Kalkis -> {self.altitude:.1f} m")
            client.takeoffAsync(vehicle_name=self.drone_id).join()
            client.moveToZAsync(
                -self.altitude, 2.5, vehicle_name=self.drone_id
            ).join()
            time.sleep(1.0)

            # 5. Ilk waypoint'e git
            first_x, first_y = self.waypoints[0]
            self._log(f"Baslangic noktasina gidiliyor: ({first_x:.1f}, {first_y:.1f})")
            client.moveToPositionAsync(
                first_x, first_y, -self.altitude,
                CRUISE_SPEED_MPS, vehicle_name=self.drone_id,
            ).join()
            time.sleep(1.0)

            self.status = "TARANIYOR"

            # 6. Zigzag waypoint tarama dongusu
            for idx, (wx, wy) in enumerate(self.waypoints, 1):
                if self.stop_event.is_set():
                    self._log("Gorev iptal sinyali alindi")
                    break

                self._log(
                    f"[{idx:02d}/{len(self.waypoints):02d}] "
                    f"-> ({wx:.1f}, {wy:.1f})"
                )

                client.moveToPositionAsync(
                    wx, wy, -self.altitude,
                    CRUISE_SPEED_MPS,
                    vehicle_name=self.drone_id,
                ).join()
                time.sleep(HOVER_SEC)

                # Drone gerçek konumu + GPS + yaw
                pose = client.simGetVehiclePose(vehicle_name=self.drone_id)
                self.current_x = pose.position.x_val
                self.current_y = pose.position.y_val
                _, _, drone_yaw = airsim.to_eularian_angles(pose.orientation)
                if self.scanned_count < 2:
                    self._log(f"  [DEBUG] AirSim pose: X={self.current_x:.2f}, Y={self.current_y:.2f}")
                    self._log(f"  [DEBUG] Hedef:       X={wx:.2f}, Y={wy:.2f}")

                # GPS (lat/lon) — pixel_to_gps için gerekli
                try:
                    gps_state = client.getMultirotorState(
                        vehicle_name=self.drone_id
                    ).gps_location
                    drone_lat = gps_state.latitude
                    drone_lon = gps_state.longitude
                except Exception:
                    drone_lat, drone_lon = 0.0, 0.0

                # FAZ 1: PID tabanlı stabil tespit (mevcut mekanizma korunuyor)
                stable_detections = self._capture_stable_detections(client, model)

                # FAZ 2: Derinlik kamerası (tek çerçeve — tüm bbox'lar için ortak)
                depth_raw, depth_img_bgr = self._capture_depth_frame(client)

                # Alan normalizasyonu — waypointteki tüm yangınlar arasında
                raw_areas: List[float] = []
                for fused_box, model_conf, img in stable_detections:
                    bbox_cx, bbox_cy = _fire_color_centroid(img, fused_box)
                    bb_w = int(fused_box[2] - fused_box[0])
                    bb_h = int(fused_box[3] - fused_box[1])
                    uc = int(min(max(bbox_cx, 0), img.shape[1] - 1))
                    vc = int(min(max(bbox_cy, 0), img.shape[0] - 1))
                    use_depth = False
                    d_val = self.altitude
                    if depth_raw is not None:
                        d_raw = float(depth_raw[
                            min(vc, IMAGE_SIZE - 1),
                            min(uc, IMAGE_SIZE - 1)
                        ])
                        if not (math.isinf(d_raw) or math.isnan(d_raw) or d_raw > 300):
                            d_val = d_raw
                            use_depth = True
                    area_m2 = bbox_area_from_depth(d_val, bb_w, bb_h)
                    raw_areas.append(area_m2)
                max_area = max(raw_areas) if raw_areas else 1.0
                if max_area <= 0:
                    max_area = 1.0

                # FAZ 3-5: Risk skoru + koordinatlar
                for det_idx, (fused_box, model_conf, img) in enumerate(stable_detections):
                    bbox_cx, bbox_cy = _fire_color_centroid(img, fused_box)

                    # NED koordinatı (edge-clip cezalı centroid ile)
                    fire_x, fire_y = bbox_to_ground_ned(
                        self.current_x, self.current_y,
                        bbox_cx, bbox_cy,
                        img.shape[1], img.shape[0],
                        self.altitude,
                        drone_yaw,
                    )

                    # Depth bazlı alan (m²)
                    bb_w = int(fused_box[2] - fused_box[0])
                    bb_h = int(fused_box[3] - fused_box[1])
                    uc = int(min(max(bbox_cx, 0), img.shape[1] - 1))
                    vc = int(min(max(bbox_cy, 0), img.shape[0] - 1))
                    use_depth = False
                    d_val = self.altitude
                    if depth_raw is not None:
                        d_raw = float(depth_raw[
                            min(vc, IMAGE_SIZE - 1),
                            min(uc, IMAGE_SIZE - 1)
                        ])
                        if not (math.isinf(d_raw) or math.isnan(d_raw) or d_raw > 300):
                            d_val = d_raw
                            use_depth = True
                    area_m2 = raw_areas[det_idx]
                    area_score = min(1.0, area_m2 / max_area)

                    # GPS koordinatı (lat/lon)
                    fire_lat, fire_lon = pixel_to_gps(
                        drone_lat, drone_lon, drone_yaw,
                        d_val, bbox_cx, bbox_cy,
                        img.shape[1], img.shape[0],
                    )

                    # Renk skoru (HSV analizi — analyze_fire_color())
                    color_src = depth_img_bgr if depth_img_bgr is not None else img
                    color_score = analyze_fire_color(color_src, fused_box)

                    # Risk skoru: renk×0.2 + alan×0.3 + model×0.5
                    risk_score = (
                        WEIGHT_COLOR * color_score +
                        WEIGHT_AREA  * area_score  +
                        WEIGHT_MODEL * model_conf
                    )

                    edge_clip = _edge_clip_ratio(fused_box, img.shape[1], img.shape[0])

                    calibration = {
                        "drone_x":       round(self.current_x, 4),
                        "drone_y":       round(self.current_y, 4),
                        "altitude_m":    round(self.altitude, 4),
                        "yaw_rad":       round(drone_yaw, 6),
                        "yaw_deg":       round(math.degrees(drone_yaw), 3),
                        "image_w":       int(img.shape[1]),
                        "image_h":       int(img.shape[0]),
                        "flame_px":      [round(bbox_cx, 3), round(bbox_cy, 3)],
                        "fused_bbox_xyxy": [round(float(v), 3) for v in fused_box.tolist()],
                        "edge_clip":     round(edge_clip, 3),
                        # Yeni alanlar
                        "lat":           fire_lat,
                        "lon":           fire_lon,
                        "area_m2":       round(area_m2, 4),
                        "depth_m":       round(d_val, 3),
                        "use_depth":     use_depth,
                        "color_score":   round(color_score, 4),
                        "area_score":    round(area_score, 4),
                        "model_conf":    round(model_conf, 4),
                        "risk_score":    round(risk_score, 4),
                    }

                    self.result_queue.put(
                        (self.drone_id, fire_x, fire_y, risk_score, calibration)
                    )
                    self.fire_count += 1

                    depth_tag = f"d={d_val:.1f}m" if use_depth else "d~irtifa"
                    self._log(
                        f"  YANGIN! risk={risk_score:.3f} "
                        f"(renk={color_score:.2f} alan={area_score:.2f} model={model_conf:.2f}) "
                        f"alan={area_m2:.2f}m² [{depth_tag}] "
                        f"clip={edge_clip:.2f} "
                        f"GPS=({fire_lat:.6f},{fire_lon:.6f}) "
                        f"NED=({fire_x:.1f},{fire_y:.1f})"
                    )

                self.scanned_count += 1

            # 7. Eve don ve inis
            self._log("Baslangica donuluyor...")
            client.moveToPositionAsync(
                0, 0, -self.altitude, CRUISE_SPEED_MPS,
                vehicle_name=self.drone_id,
            ).join()

            self._log("Inis yapiliyor...")
            client.landAsync(vehicle_name=self.drone_id).join()
            client.armDisarm(False, vehicle_name=self.drone_id)
            client.enableApiControl(False, vehicle_name=self.drone_id)

            self.status = "TAMAMLANDI"
            self._log(
                f"Gorev tamamlandi — "
                f"{self.scanned_count} waypoint, {self.fire_count} yangin"
            )

        except Exception as exc:
            self.status = "HATA"
            self._log(f"HATA: {exc}")
            import traceback
            traceback.print_exc()
            try:
                client.landAsync(vehicle_name=self.drone_id).join()
                client.armDisarm(False, vehicle_name=self.drone_id)
                client.enableApiControl(False, vehicle_name=self.drone_id)
            except Exception:
                pass

    def _log(self, msg: str) -> None:
        print(f"  [{self.drone_id}] {msg}")
