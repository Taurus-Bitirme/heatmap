"""
multi_drone_config.py
=====================
3 Drone'lu Federe Yangın Tespiti — Ortak Konfigürasyon

Referans: tarama_algoritmasi_cok_onceki.py zigzag mantığı
Alan: 5039×5039 UE birim = 50.39×50.39 metre (AirSim NED)
"""

import math
import numpy as np
from typing import List, Tuple, Dict

# ============================================================
#  DRONE TEMEL KONFİGÜRASYONU (Diğer hesaplamalar için gerekli)
# ============================================================

ALTITUDE_M: float        = 16.0
CAMERA_FOV_DEG: float    = 90.0

# ============================================================
#  ALAN TANIMLARI
# ============================================================

AREA_UE_X_MIN: float = 0.0
AREA_UE_Y_MIN: float = 0.0
AREA_UE_X_MAX: float = 10000.0
AREA_UE_Y_MAX: float = 10000.0

# PlayerStart UE konumu (cm)
PLAYER_START_X: float = 0.0
PLAYER_START_Y: float = 0.0

# UE → AirSim dönüşümü (cm → metre)
def ue_to_airsim(ue_x: float, ue_y: float) -> Tuple[float, float]:
    return (
        (ue_x - PLAYER_START_X) / 100.0,
        (ue_y - PLAYER_START_Y) / 100.0,
    )

# Landscape bounds in AirSim NED meters. These are the final map bounds.
LANDSCAPE_MIN_X, LANDSCAPE_MIN_Y = ue_to_airsim(AREA_UE_X_MIN, AREA_UE_Y_MIN)
LANDSCAPE_MAX_X, LANDSCAPE_MAX_Y = ue_to_airsim(AREA_UE_X_MAX, AREA_UE_Y_MAX)
AREA_WIDTH_M: float = LANDSCAPE_MAX_Y - LANDSCAPE_MIN_Y
AREA_HEIGHT_M: float = LANDSCAPE_MAX_X - LANDSCAPE_MIN_X
AREA_SIZE_M: float = max(AREA_WIDTH_M, AREA_HEIGHT_M)

# Camera footprint at the mission altitude. This helps coverage planning, but
# final heatmap bounds stay equal to the real landscape bounds.
CAMERA_FOOTPRINT_M: float = 2.0 * ALTITUDE_M * math.tan(
    math.radians(CAMERA_FOV_DEG / 2.0)
)
FOV_MARGIN_M: float = CAMERA_FOOTPRINT_M / 2.0

# Backward-compatible names: scan limits are the real landscape limits.
SCAN_MIN_X: float = LANDSCAPE_MIN_X
SCAN_MIN_Y: float = LANDSCAPE_MIN_Y
SCAN_MAX_X: float = LANDSCAPE_MAX_X
SCAN_MAX_Y: float = LANDSCAPE_MAX_Y
# ============================================================
#  DRONE KONFİGÜRASYONU (devam)
# ============================================================

DRONE_NAMES: List[str] = ["Drone1", "Drone2", "Drone3"]
NUM_DRONES: int = 3

CRUISE_SPEED_MPS: float  = 5.0
HOVER_SEC: float         = 1.5
CAMERA_NAME: str         = "0"
MODEL_PATH: str          = "./modelimiz.pt"

# ── Footprint tabanlı tarama (scan_algorithm_parcel.py mekanizması) ──────────
# Sabit aralık yerine irtifa + FOV'dan otomatik hesap:
#   footprint = 2 × altitude × tan(FOV/2)
#   step      = footprint × (1 − overlap)
# 16m irtifa, 90° FOV → footprint = 32m, step = 25.6m (%20 overlap)

OVERLAP_RATIO: float = 0.75


def compute_ground_footprint() -> Tuple[float, float]:
    """
    Kameranın yerde kapladığı alanı hesaplar (scan_algorithm_parcel.py ile aynı).
    Dönüş: (footprint_genislik_m, footprint_yukseklik_m)
      gw = 2 × irtifa × tan(FOV_H / 2)
      gh = 2 × irtifa × tan(FOV_V / 2)
    IMAGE_SIZE kare olduğundan FOV_V = FOV_H → gw = gh = 32m.
    """
    fov_h = math.radians(CAMERA_FOV_DEG)
    gw = 2.0 * ALTITUDE_M * math.tan(fov_h / 2.0)
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0))   # kare görüntü → eşit
    gh = 2.0 * ALTITUDE_M * math.tan(fov_v / 2.0)
    return gw, gh


# Modül yüklenirken bir kez hesaplanır — sonra sabit olarak kullanılır
GROUND_FOOTPRINT_W, GROUND_FOOTPRINT_H = compute_ground_footprint()
STEP_X: float = GROUND_FOOTPRINT_W * (1.0 - OVERLAP_RATIO)   # X adımı (m)
STEP_Y: float = GROUND_FOOTPRINT_H * (1.0 - OVERLAP_RATIO)   # Y adımı (m)

# Geriye dönük uyumluluk (eski kodda SCAN_ROW_SPACING / SCAN_COL_SPACING geçiyorsa)
SCAN_ROW_SPACING: float = STEP_Y
SCAN_COL_SPACING: float = STEP_X

# ============================================================
#  DRONE BAŞLANGIÇ KONUMLARı (UE koordinatlarında cm cinsinden)
#  Z=100 cm = 1 metre irtifası (harita seviyesi)
# ============================================================

DRONE_START_POSITIONS_UE: Dict[str, Tuple[float, float, float]] = {
    "Drone1": (0.0, 0.0, 100.0),      # X=0, Y=0, Z=100 cm
    "Drone2": (0.0, 800.0, 100.0),    # X=0, Y=8 m, Z=100 cm
    "Drone3": (0.0, 1600.0, 100.0),   # X=0, Y=16 m, Z=100 cm
}

# UE cinsinden drone konumlarını AirSim NED'ye çevir
def drone_start_pos_ned(drone_id: str) -> Tuple[float, float]:
    """Drone başlangıç konumunu UE'den NED'ye çevirir."""
    if drone_id not in DRONE_START_POSITIONS_UE:
        raise ValueError(f"Bilinmeyen drone: {drone_id}")
    ue_x, ue_y, _ = DRONE_START_POSITIONS_UE[drone_id]
    ned_x, ned_y = ue_to_airsim(ue_x, ue_y)
    return (ned_x, ned_y)

# Çarpışma koruması
ALTITUDE_OFFSETS: Dict[str, float] = {
    "Drone1": 0.0,
    "Drone2": 0.5,
    "Drone3": 1.0,
}

DRONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Drone1": (0, 255, 0),
    "Drone2": (255, 200, 0),
    "Drone3": (0, 0, 255),
}

# ============================================================
#  ISI HARİTASI (1000×1000 piksel)
# ============================================================

HEATMAP_SIZE_PX: int = 1000
HEATMAP_SCALE: float = min(
    HEATMAP_SIZE_PX / max(AREA_WIDTH_M, 1e-6),
    HEATMAP_SIZE_PX / max(AREA_HEIGHT_M, 1e-6),
)
HEATMAP_CONTENT_W_PX: int = int(round(AREA_WIDTH_M * HEATMAP_SCALE))
HEATMAP_CONTENT_H_PX: int = int(round(AREA_HEIGHT_M * HEATMAP_SCALE))
HEATMAP_OFFSET_COL: int = max(0, (HEATMAP_SIZE_PX - HEATMAP_CONTENT_W_PX) // 2)
HEATMAP_OFFSET_ROW: int = max(0, (HEATMAP_SIZE_PX - HEATMAP_CONTENT_H_PX) // 2)
# ============================================================
#  YOLO AYARLARI
# ============================================================

YOLO_CONF: float = 0.25
YOLO_IOU: float  = 0.45
IMAGE_SIZE: int   = 360   # Kamera çözünürlüğü (settings.json ile aynı)

# Multi-drone tespitte tek kare yerine kısa bir pencere üzerinden bbox stabilize edilir.
DETECTION_FRAME_COUNT: int = 12
DETECTION_FRAME_INTERVAL_SEC: float = 0.08
TRACK_IOU_MATCH_THRESHOLD: float = 0.20
MIN_TRACK_OBSERVATIONS: int = 3
FIRE_KEYWORDS = {"fire", "smoke"}

# Optional ground-truth points for calibration logs. Values are UE centimeters.
# Leave CALIBRATION_ENABLED as False for normal missions.
CALIBRATION_ENABLED: bool = True
CALIBRATION_FIRE_POINTS_UE: Dict[str, Tuple[float, float]] = {
    "Yangin1": (2500.0, 2500.0),
    "Yangin2": (7500.0, 7500.0),
}
CALIBRATION_FIRE_POINTS_NED: Dict[str, Tuple[float, float]] = {
    name: ue_to_airsim(ue_x, ue_y)
    for name, (ue_x, ue_y) in CALIBRATION_FIRE_POINTS_UE.items()
}

# ============================================================
#  ZİGZAG TARAMA YOLU OLUŞTURMA
#  (tarama_algoritmasi_cok_onceki.py'den uyarlanmıştır)
# ============================================================

def generate_parcel_waypoints(
    x_min: float, y_min: float,
    x_max: float, y_max: float,
    step_x: float = None,
    step_y: float = None,
) -> List[Tuple[float, float]]:
    """
    scan_algorithm_parcel.py mekanizması — MERKEZLENMİŞ parsel waypoint'leri.

    Kenardan başlamak yerine her parselin tam ortasını hesaplar:
        center_x = x_min + strip_w * (col + 0.5) / n_cols
        center_y = y_min + strip_h * (row + 0.5) / n_rows

    Boustrophedon (zigzag) sıralaması:
        Çift satırlarda sütun sırası tersine çevrilir.

    Parametreler
    ------------
    step_x, step_y : Adım boyutları (metre). None → STEP_X / STEP_Y kullanılır.
    """
    if step_x is None:
        step_x = STEP_X
    if step_y is None:
        step_y = STEP_Y

    strip_w = x_max - x_min   # X yönündeki genişlik
    strip_h = y_max - y_min   # Y yönündeki derinlik

    n_cols = max(1, math.ceil(strip_w / step_x))   # X sütun sayısı
    n_rows = max(1, math.ceil(strip_h / step_y))   # Y satır sayısı

    # Eşit aralıklı parsel merkezleri — şerit sınırları içinde kalır
    x_centers = [x_min + strip_w * (col + 0.5) / n_cols for col in range(n_cols)]
    y_centers = [y_min + strip_h * (row + 0.5) / n_rows for row in range(n_rows)]

    waypoints: List[Tuple[float, float]] = []
    for row_idx, yc in enumerate(y_centers):
        cols = list(reversed(x_centers)) if row_idx % 2 == 1 else list(x_centers)
        for xc in cols:
            waypoints.append((xc, yc))

    return waypoints


def alani_dronelara_bol() -> Dict[str, List[Tuple[float, float]]]:
    """
    100×100 m alanı X ekseninde NUM_DRONES eşit şeride böler.
    Her drone kendi şeridinde footprint tabanlı boustrophedon tarama yapar.

    Footprint hesabı (scan_algorithm_parcel.py ile aynı mantık):
        footprint = 2 × ALTITUDE_M × tan(FOV/2) = 32 m
        step      = footprint × (1 − OVERLAP_RATIO) = 25.6 m

    100 m × 100 m harita, 3 drone, 25.6 m adım:
        Drone başına X şeridi : 33.33 m  →  n_cols = 2
        Y boyutu              : 100 m    →  n_rows = 4
        Drone başına waypoint : 2 × 4   = 8
        Toplam                : 3 × 8   = 24
    """
    x_min = LANDSCAPE_MIN_X
    x_max = LANDSCAPE_MAX_X
    y_min = LANDSCAPE_MIN_Y
    y_max = LANDSCAPE_MAX_Y

    serit_genisligi = (x_max - x_min) / NUM_DRONES
    assignments: Dict[str, List[Tuple[float, float]]] = {}

    for i, drone_id in enumerate(DRONE_NAMES):
        bolum_x_min = x_min + i * serit_genisligi
        bolum_x_max = x_min + (i + 1) * serit_genisligi

        wps = generate_parcel_waypoints(
            bolum_x_min, y_min,
            bolum_x_max, y_max,
        )
        assignments[drone_id] = wps

    return assignments


def zigzag_yolu_olustur(
    x_min: float, y_min: float,
    x_max: float, y_max: float,
    satir_araligi: float,
    sutun_araligi: float = None,
) -> List[Tuple[float, float]]:
    """Geriye dönük uyumluluk — generate_parcel_waypoints'e yönlendirir."""
    sx = sutun_araligi if sutun_araligi is not None else STEP_X
    return generate_parcel_waypoints(x_min, y_min, x_max, y_max,
                                     step_x=sx, step_y=satir_araligi)

# ============================================================
#  YANGIN KONUM HESAPLAMA (YOLO bbox → gerçek zemin koordinatı)
# ============================================================

def bbox_to_ground_ned(
    drone_x: float, drone_y: float,
    bbox_cx_px: float, bbox_cy_px: float,
    img_w: int = IMAGE_SIZE, img_h: int = IMAGE_SIZE,
    altitude: float = ALTITUDE_M,
    yaw_rad: float = 0.0,
) -> Tuple[float, float]:
    """
    YOLO bounding box merkezini kamera geometrisi ile
    gercek zemin NED koordinatina cevirir.

    Kamera asagi bakiyor (nadir), perspektif izdusuum:
      - Goruntu merkezi (img_w/2, img_h/2) = drone'un tam alti
      - Piksel ofseti x metre/piksel = zemin ofseti

    AirSim Nadir Koordinat Eslesimi:
      - Goruntu X ekseni (sutun, sol→sag)   → NED Y ekseni (east/sag)
      - Goruntu Y ekseni (satir, yukari→asagi) → NED X ekseni (north/ileri, TERS)

    Parameters
    ----------
    drone_x, drone_y : Drone'un NED konumu (metre)
    bbox_cx_px, bbox_cy_px : YOLO bbox merkezi (piksel)
    img_w, img_h : Goruntu boyutu (piksel)
    altitude : Ucus yuksekligi (metre)

    Returns
    -------
    (fire_x, fire_y) : Yangin zemini NED koordinati (metre)
    """
    fov_rad = math.radians(CAMERA_FOV_DEG)
    # Yatay FOV goruntu genisligi (NED Y / right ekseni) icin kullanilir.
    # Dikey FOV goruntu yuksekligi (NED X / forward ekseni) icin ayri hesaplanir.
    ground_half_w = altitude * math.tan(fov_rad / 2.0)
    ground_half_h = altitude * math.tan(fov_rad / 2.0) * (img_h / img_w)

    mpp_x = (2.0 * ground_half_w) / img_w   # metre/piksel, X sutun ekseni (NED Y)
    mpp_y = (2.0 * ground_half_h) / img_h   # metre/piksel, Y satir ekseni (NED X)

    # Piksel ofseti (goruntu merkezine gore)
    dx_px = bbox_cx_px - img_w / 2.0   # sag pozitif → NED Y pozitif
    dy_px = bbox_cy_px - img_h / 2.0   # asagi pozitif → NED X negatif

    # Drone govde cercevesindeki ofsetler
    right_m   =  dx_px * mpp_x   # NED Y yonu (sag)
    forward_m = -dy_px * mpp_y   # NED X yonu (ileri, goruntu Y tersi)

    # Drone yaw'ina gore global NED'e don
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    offset_x = forward_m * cos_y - right_m * sin_y
    offset_y = forward_m * sin_y + right_m * cos_y

    fire_x = drone_x + offset_x
    fire_y = drone_y + offset_y

    return fire_x, fire_y


# ============================================================
#  DERINLIK KAMERA — ALAN VE GPS HESAPLAMA
# ============================================================

# Risk ağırlıkları (heatmap_OPtimUS.py ile aynı)
WEIGHT_COLOR: float = 0.2
WEIGHT_AREA:  float = 0.3
WEIGHT_MODEL: float = 0.5

def bbox_area_from_depth(
    depth_m: float,
    bb_w_px: int,
    bb_h_px: int,
    img_w: int = IMAGE_SIZE,
    img_h: int = IMAGE_SIZE,
) -> float:
    """
    Bounding box'ın gerçek alanını derinlik kamera ölçümü ile hesaplar (m²).
    depth_m  : bbox merkezindeki DepthPerspective değeri (metre)
    bb_w_px  : bbox genişliği (piksel)
    bb_h_px  : bbox yüksekliği (piksel)
    """
    fov_h_rad = math.radians(CAMERA_FOV_DEG)
    fov_v_rad = 2.0 * math.atan(math.tan(fov_h_rad / 2.0) * (img_h / img_w))
    m_per_px_w = (2.0 * depth_m * math.tan(fov_h_rad / 2.0)) / img_w
    m_per_px_h = (2.0 * depth_m * math.tan(fov_v_rad / 2.0)) / img_h
    return (bb_w_px * m_per_px_w) * (bb_h_px * m_per_px_h)


def pixel_to_gps(
    drone_lat: float,
    drone_lon: float,
    yaw_rad: float,
    depth_m: float,
    u_c: float,
    v_c: float,
    img_w: int = IMAGE_SIZE,
    img_h: int = IMAGE_SIZE,
) -> Tuple[float, float]:
    """
    Aşağı bakan kamera piksel koordinatından yerdeki GPS konumunu hesaplar.
    Drone yaw açısı dahil tam rotasyon uygulanır.
    Dönüş: (fire_lat, fire_lon) derece cinsinden.
    """
    fov_h_rad = math.radians(CAMERA_FOV_DEG)
    fov_v_rad = 2.0 * math.atan(math.tan(fov_h_rad / 2.0) * (img_h / img_w))
    du = u_c - img_w / 2.0
    dv = v_c - img_h / 2.0
    m_per_px_w = (2.0 * depth_m * math.tan(fov_h_rad / 2.0)) / img_w
    m_per_px_h = (2.0 * depth_m * math.tan(fov_v_rad / 2.0)) / img_h
    right_offset   =  du * m_per_px_w
    forward_offset =  dv * m_per_px_h
    offset_north = forward_offset * math.cos(yaw_rad) - right_offset * math.sin(yaw_rad)
    offset_east  = forward_offset * math.sin(yaw_rad) + right_offset * math.cos(yaw_rad)
    dlat = offset_north / 111320.0
    dlon = offset_east / (111320.0 * math.cos(math.radians(drone_lat)))
    return drone_lat + dlat, drone_lon + dlon


# ============================================================
#  HEATMAP KOORDİNAT DÖNÜŞÜMÜ
# ============================================================

def ned_to_heatmap_px(ned_x: float, ned_y: float) -> Tuple[int, int]:
    """Convert landscape NED meters to a scaled mini-map pixel."""
    col = int(HEATMAP_OFFSET_COL + (ned_y - LANDSCAPE_MIN_Y) * HEATMAP_SCALE)
    row = int(HEATMAP_OFFSET_ROW + (LANDSCAPE_MAX_X - ned_x) * HEATMAP_SCALE)

    col = max(0, min(HEATMAP_SIZE_PX - 1, col))
    row = max(0, min(HEATMAP_SIZE_PX - 1, row))
    return col, row


def is_inside_landscape(ned_x: float, ned_y: float) -> bool:
    """Return True if the point is inside the configured landscape bounds."""
    return (
        LANDSCAPE_MIN_X <= ned_x <= LANDSCAPE_MAX_X and
        LANDSCAPE_MIN_Y <= ned_y <= LANDSCAPE_MAX_Y
    )
