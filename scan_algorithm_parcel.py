"""
scan_algorithm_parcel.py
========================
Systematic Grid Survey Mission Controller — Single Continuous Flight

Architecture
------------
  connect_and_arm()      (ONCE at mission start)
  takeoffAsync()         (ONCE)
  moveToZAsync(-10m)     (ONCE)

  FOR each parcel in waypoints:
    moveToPositionAsync(…)   ← stays at 10 m, never lands between parcels
    time.sleep(HOVER)
    run_heatmap_scan(client) ← inline function call, no subprocess
    log result

  moveToPositionAsync(0, 0, -10)  ← return to spawn at altitude
  landAsync()                      (ONCE at mission end)
  armDisarm(False)                 (ONCE)
  enableApiControl(False)          (ONCE)
"""

import os
import math
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import airsim

from heatmap_OPtimUS import run_heatmap_scan
from static_heatmap import generate_static_osm_heatmap

# ============================================================
#  MISSION PARAMETERS
# ============================================================

# Geographic reference — spawn = NED origin = SW corner of survey area
SPAWN_LAT: float = 47.641472
SPAWN_LON: float = -122.140167

# Survey-area corners  (lat, lon)  decimal degrees
CORNERS: Dict[str, Tuple[float, float]] = {
    "bottom_left":  (47.641472, -122.140167),   # Spawn — SW
    "bottom_right": (47.641972, -122.140165),   # NW (user label kept)
    "top_right":    (47.641972, -122.139419),   # NE
    "top_left":     (47.641468, -122.139419),   # SE (user label kept)
}

# Drone / camera
ALTITUDE_M:       float = 10.0    # AGL flight altitude (m)
IMAGE_WIDTH:      int   = 640     # Camera resolution (px)
IMAGE_HEIGHT:     int   = 640
CAMERA_FOV_H_DEG: float = 90.0   # Horizontal field-of-view (°)
OVERLAP_RATIO:    float = 0.20    # Image overlap between adjacent parcels

# Navigation
CRUISE_SPEED_MPS:    float = 5.0    # Cruising speed (m/s)
HOVER_STABILIZE_SEC: float = 2.0    # Stabilisation pause after arrival

# Camera (must match heatmap_OPtimUS.py CAMERA_NAME)
CAMERA_NAME: str = "0"

# Paths
_SCRIPT_DIR:         str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR:          str = os.path.join(_SCRIPT_DIR, "scan_results")
MISSION_REPORT_JSON: str = os.path.join(OUTPUT_DIR, "mission_report.json")
PARCEL_LOG_JSON:     str = os.path.join(OUTPUT_DIR, "parcel_log.json")


# ============================================================
#  GRID MATHEMATICS
# ============================================================

def compute_ground_footprint() -> Tuple[float, float]:
    """
    Returns (footprint_width_m, footprint_height_m)
    — the ground rectangle covered by one image at ALTITUDE_M.

    Maths
    -----
      gw = 2 * h * tan(FoV_h / 2)
      FoV_v derived from FoV_h assuming square pixels
    """
    fov_h = math.radians(CAMERA_FOV_H_DEG)
    gw = 2.0 * ALTITUDE_M * math.tan(fov_h / 2.0)
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * (IMAGE_HEIGHT / IMAGE_WIDTH))
    gh = 2.0 * ALTITUDE_M * math.tan(fov_v / 2.0)
    return gw, gh


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two GPS points (metres)."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2)
    return 2.0 * R * math.asin(math.sqrt(min(1.0, a)))


def gps_to_ned(lat: float, lon: float) -> Tuple[float, float]:
    """
    Convert a GPS coordinate to AirSim NED offsets (north_m, east_m)
    relative to SPAWN_LAT / SPAWN_LON  (the NED origin).
    """
    north = (lat - SPAWN_LAT) * 111_320.0
    east  = (lon - SPAWN_LON) * 111_320.0 * math.cos(math.radians(SPAWN_LAT))
    return north, east


def ned_to_gps(north_m: float, east_m: float) -> Tuple[float, float]:
    """Inverse of gps_to_ned — for reference / logging."""
    lat = SPAWN_LAT + north_m / 111_320.0
    lon = SPAWN_LON + east_m / (111_320.0 * math.cos(math.radians(SPAWN_LAT)))
    return lat, lon


def generate_grid_waypoints() -> List[Dict]:
    """
    Build ordered parcel-center waypoints in boustrophedon (snake) order.

    Pattern
    -------
      Row 1  →  R1C1  R1C2  R1C3  R1C4   (west → east)
      Row 2  ←  R2C4  R2C3  R2C2  R2C1   (east → west)
      Row 3  →  R3C1  …                   (west → east)

    Parcel IDs denote *physical* position; traversal order may differ.
    """
    gw, gh = compute_ground_footprint()
    step_e = gw * (1.0 - OVERLAP_RATIO)   # east-ward step (m)
    step_n = gh * (1.0 - OVERLAP_RATIO)   # north-ward step (m)

    # Derive bounding box from the four corners
    all_lats = [c[0] for c in CORNERS.values()]
    all_lons = [c[1] for c in CORNERS.values()]
    lat_sw = min(all_lats)
    lon_sw = min(all_lons)
    lat_ne = max(all_lats)
    lon_ne = max(all_lons)

    area_e_m = haversine_m(lat_sw, lon_sw, lat_sw, lon_ne)   # E–W extent
    area_n_m = haversine_m(lat_sw, lon_sw, lat_ne, lon_sw)   # N–S extent

    n_cols = max(1, math.ceil(area_e_m / step_e))
    n_rows = max(1, math.ceil(area_n_m / step_n))

    print(f"\n  Tarama alanı       : {area_e_m:.1f} m (D) × {area_n_m:.1f} m (K)")
    print(f"  Görüntü ayak izi   : {gw:.2f} m (genişlik) × {gh:.2f} m (yükseklik)")
    print(f"  Adım (üstüste %{int(OVERLAP_RATIO * 100):02d}) : "
          f"{step_e:.2f} m (D) × {step_n:.2f} m (K)")
    print(f"  Izgara             : {n_rows} satır × {n_cols} sütun "
          f"= {n_rows * n_cols} parsel")

    waypoints: List[Dict] = []

    for row in range(n_rows):
        # Physical centre of this row band (metres north of SW corner)
        north_m = step_n * row + step_n / 2.0
        lat_c   = lat_sw + north_m / 111_320.0

        row_wps: List[Dict] = []
        for col in range(n_cols):
            east_m = step_e * col + step_e / 2.0
            lon_c  = lon_sw + east_m / (111_320.0 * math.cos(math.radians(lat_sw)))
            ned_n, ned_e = gps_to_ned(lat_c, lon_c)

            row_wps.append({
                "parcel_id":  f"R{row + 1}C{col + 1}",
                "row":        row + 1,
                "col":        col + 1,
                "center_lat": round(lat_c, 8),
                "center_lon": round(lon_c, 8),
                "ned_north":  round(ned_n, 3),
                "ned_east":   round(ned_e, 3),
                "altitude_m": ALTITUDE_M,
            })

        # Boustrophedon: reverse every second row (right → left)
        if row % 2 == 1:
            row_wps = list(reversed(row_wps))

        waypoints.extend(row_wps)

    return waypoints


# ============================================================
#  AIRSIM HELPERS
# ============================================================

def navigate_to_waypoint(client: airsim.MultirotorClient, wp: Dict) -> None:
    """
    Fly to the waypoint NED position at ALTITUDE_M.
    The drone is always airborne during the mission loop — no takeoff guard.
    Blocks until arrival, then waits HOVER_STABILIZE_SEC seconds.
    """
    print(f"      Hedefe uçuluyor: NED=({wp['ned_north']:.1f}, "
          f"{wp['ned_east']:.1f}, {-ALTITUDE_M:.0f}) @ {CRUISE_SPEED_MPS} m/s")
    client.moveToPositionAsync(
        wp["ned_north"], wp["ned_east"], -ALTITUDE_M,
        CRUISE_SPEED_MPS,
    ).join()

    print(f"      Stabilizasyon ({HOVER_STABILIZE_SEC:.0f}s)...")
    time.sleep(HOVER_STABILIZE_SEC)

    pose = client.simGetVehiclePose()
    print(f"      ✓ Varıldı — AGL ≈ {-pose.position.z_val:.2f} m")


# ============================================================
#  PARCEL SCAN  —  one call per grid cell
# ============================================================

def scan_parcel(client: airsim.MultirotorClient, parcel: Dict) -> Dict:
    """
    Full scan cycle for a single parcel using the shared AirSim client.
      a. Build per-parcel output directory: scan_results/<parcel_id>/
      b. navigate_to_waypoint — moves drone to parcel centre
      c. run_heatmap_scan(client, parcel_out_dir) — all 6 phases, isolated output
      d. returns structured log dict
    """
    ts = datetime.now(timezone.utc).isoformat()
    print(f"  Timestamp   : {ts}")

    parcel_out_dir = os.path.join(OUTPUT_DIR, parcel["parcel_id"])
    os.makedirs(parcel_out_dir, exist_ok=True)

    # ── 1. Navigate to parcel centre ─────────────────────────
    nav_ok = True
    try:
        navigate_to_waypoint(client, parcel)
    except Exception as exc:
        print(f"    ⚠ Navigasyon hatası: {exc}")
        nav_ok = False

    # ── 2. Execute scan — output written to parcel_out_dir ─────
    result = run_heatmap_scan(client, parcel_out_dir)

    return {
        "parcel_id":      parcel["parcel_id"],
        "center_lat":     parcel["center_lat"],
        "center_lon":     parcel["center_lon"],
        "altitude_m":     ALTITUDE_M,
        "timestamp":      ts,
        "nav_success":    nav_ok,
        "parcel_out_dir": parcel_out_dir,
        "heatmap_data":   result,
        "detected_fires": len(result.get("fires", [])),
    }


# ============================================================
#  GLOBAL HEATMAP
# ============================================================

def build_global_heatmap(parcel_logs: List[Dict]) -> str:
    """
    Reads heatmap_report.json from every parcel directory,
    merges all fire entries into one list,
    writes scan_results/global/all_fires.json,
    and generates scan_results/global/global_heatmap_osm.png.
    Returns the path to the global output directory.
    """
    global_dir = os.path.join(OUTPUT_DIR, "global")
    os.makedirs(global_dir, exist_ok=True)

    all_fires: List[Dict] = []
    for log in parcel_logs:
        parcel_id   = log["parcel_id"]
        report_path = os.path.join(OUTPUT_DIR, parcel_id, "heatmap_report.json")
        if not os.path.exists(report_path):
            print(f"  ⚠ {parcel_id}: heatmap_report.json bulunamadı, atlanıyor")
            continue
        try:
            with open(report_path, "r", encoding="utf-8") as fh:
                parcel_report = json.load(fh)
            fires = parcel_report.get("fires", [])
            for fire in fires:
                fire["source_parcel"] = parcel_id
            all_fires.extend(fires)
            print(f"  {parcel_id}: {len(fires)} yangın okundu")
        except Exception as exc:
            print(f"  ⚠ {parcel_id}: okuma hatası — {exc}")

    # Write merged JSON
    all_fires_path = os.path.join(global_dir, "all_fires.json")
    with open(all_fires_path, "w", encoding="utf-8") as fh:
        json.dump({
            "total_fires":            len(all_fires),
            "total_parcels_scanned":  len(parcel_logs),
            "fires":                  all_fires,
        }, fh, indent=2, ensure_ascii=False)
    print(f"  ✓ Birleşik yangın verisi: {all_fires_path}  ({len(all_fires)} yangın)")

    # Generate global OSM heatmap
    global_heatmap_path = os.path.join(global_dir, "global_heatmap_osm.png")
    if all_fires:
        generate_static_osm_heatmap(all_fires, global_heatmap_path)
        print(f"  ✓ Global ısı haritası: {global_heatmap_path}")
    else:
        print("  ⚠ Hiç yangın tespiti yok — global harita oluşturulmadı")

    return global_dir


# ============================================================
#  MAIN MISSION
# ============================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    SEP  = "=" * 65
    DASH = "─" * 65

    print(SEP)
    print("  IZGARA TARAMA GÖREVİ  —  BAŞLATILIYOR")
    print(SEP)
    print("  Tarama alanı köşe koordinatları:")
    for label, (lat, lon) in CORNERS.items():
        print(f"    {label:<14s}: ({lat:.6f}, {lon:.6f})")
    print(f"  Spawn (NED origin)  : ({SPAWN_LAT:.6f}, {SPAWN_LON:.6f})")

    # ── ADIM 1: Grid hesaplama ──────────────────────────────
    print(f"\n{DASH}")
    print("  ADIM 1: Izgara hesaplanıyor")
    print(DASH)

    waypoints = generate_grid_waypoints()
    gw, gh = compute_ground_footprint()

    print(f"\n  Boustrophedon parsel sırası:")
    for wp in waypoints:
        arrow = "→" if wp["row"] % 2 == 1 else "←"
        print(f"    {arrow} {wp['parcel_id']:6s} | "
              f"GPS ({wp['center_lat']:.7f}, {wp['center_lon']:.7f}) | "
              f"NED ({wp['ned_north']:+.1f} m K, {wp['ned_east']:+.1f} m D)")

    # ── Tek uçuş oturumu başlat ─────────────────────────────
    print(f"\n{DASH}")
    print("  AirSim bağlantısı kuruluyor (tek oturum)...")
    print(DASH)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Kamera aşağı — bir kez ayarlanır
    client.simSetCameraPose(
        CAMERA_NAME,
        airsim.Pose(airsim.Vector3r(0, 0, 0),
                    airsim.to_quaternion(math.radians(-90), 0, 0)),
    )

    print("Kalkış → 10 m")
    client.takeoffAsync().join()
    client.moveToZAsync(-ALTITUDE_M, 2.5).join()
    time.sleep(1.0)

    # ── ADIM 2: Parsel taramaları ───────────────────────────
    print(f"\n{DASH}")
    print(f"  ADIM 2: {len(waypoints)} parsel sırayla taranıyor")
    print(DASH)

    parcel_logs: List[Dict] = []
    mission_t0 = time.time()

    try:
        for idx, wp in enumerate(waypoints, 1):
            print(f"\n{DASH}")
            print(f"  [{idx:02d}/{len(waypoints):02d}]  Parsel: {wp['parcel_id']}  "
                  f"({wp['center_lat']:.7f}, {wp['center_lon']:.7f})")
            print(DASH)

            log = scan_parcel(client, wp)
            parcel_logs.append(log)

            # Incremental persistence — survives a mid-mission crash
            with open(PARCEL_LOG_JSON, "w", encoding="utf-8") as fh:
                json.dump(parcel_logs, fh, indent=2, ensure_ascii=False)

            print(f"  Tespit: {log['detected_fires']} yangın  |  Log: {PARCEL_LOG_JSON}")

    finally:
        # ── ADIM 3: Her durumda spawn noktasına dön ve in ──────
        print(f"\n{DASH}")
        print("  ADIM 3: Spawn noktasına dönülüyor ve iniş yapılıyor...")
        print(DASH)
        try:
            client.moveToPositionAsync(0.0, 0.0, -ALTITUDE_M, CRUISE_SPEED_MPS).join()
            client.landAsync().join()
        except Exception as exc:
            print(f"  ⚠ Eve dönüş hatası: {exc}")

        client.armDisarm(False)
        client.enableApiControl(False)
        print("✓ Görev tamamlandı, indi.")
    # ── ADIM 4: Global ısı haritası ──────────────────────────
    print(f"\n{DASH}")
    print("  ADIM 4: Global ısı haritası oluşturuluyor")
    print(DASH)
    global_dir = build_global_heatmap(parcel_logs)
    # ── Final mission report ────────────────────────────────
    mission_elapsed = time.time() - mission_t0
    n_rows = max(wp["row"] for wp in waypoints) if waypoints else 0
    n_cols = max(wp["col"] for wp in waypoints) if waypoints else 0

    report = {
        "mission_end":      datetime.now(timezone.utc).isoformat(),
        "total_elapsed_s":  round(mission_elapsed, 2),
        "total_parcels":    len(waypoints),
        "successful_scans": sum(1 for lg in parcel_logs if lg["nav_success"]),
        "failed_nav":       sum(1 for lg in parcel_logs if not lg["nav_success"]),
        "spawn_point":      {"lat": SPAWN_LAT, "lon": SPAWN_LON},
        "global_output_dir": global_dir,
        "grid_params": {
            "footprint_width_m":  round(gw, 3),
            "footprint_height_m": round(gh, 3),
            "step_east_m":        round(gw * (1.0 - OVERLAP_RATIO), 3),
            "step_north_m":       round(gh * (1.0 - OVERLAP_RATIO), 3),
            "overlap_pct":        int(OVERLAP_RATIO * 100),
            "altitude_m":         ALTITUDE_M,
            "total_rows":         n_rows,
            "total_cols":         n_cols,
        },
        "parcels": parcel_logs,
    }
    with open(MISSION_REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ── Console summary table ───────────────────────────────
    print(f"\n{SEP}")
    print("  GÖREV ÖZETİ")
    print(SEP)
    print(f"  Toplam parsel    : {report['total_parcels']}  "
          f"({n_rows} satır × {n_cols} sütun)")
    print(f"  Başarılı tarama  : {report['successful_scans']}")
    print(f"  Başarısız nav    : {report['failed_nav']}")
    print(f"  Toplam süre      : {mission_elapsed / 60:.1f} dk  "
          f"({mission_elapsed:.0f} s)")
    print(f"  Görev raporu     : {MISSION_REPORT_JSON}")
    print(f"  Parsel logu      : {PARCEL_LOG_JSON}")
    print(f"  Global ısı haritası  : {os.path.join(global_dir, 'global_heatmap_osm.png')}")
    print(f"  Birleşik yangın JSON : {os.path.join(global_dir, 'all_fires.json')}")
    print()

    col_parcel = 8
    col_gps    = 38
    col_fire   = 7
    col_nav    = 7

    hdr = (f"  {'Parsel':<{col_parcel}} {'GPS (lat, lon)':<{col_gps}} "
           f"{'Yangın':>{col_fire}} {'Nav':>{col_nav}}")
    print(hdr)
    print(f"  {'─' * col_parcel} {'─' * col_gps} {'─' * col_fire} {'─' * col_nav}")

    for lg in parcel_logs:
        n_f  = str(lg["detected_fires"])
        sym  = "✓ OK " if lg["nav_success"] else "✗ ERR"
        gps  = f"({lg['center_lat']:.6f}, {lg['center_lon']:.6f})"
        print(f"  {lg['parcel_id']:<{col_parcel}} {gps:<{col_gps}} "
              f"{n_f:>{col_fire}} {sym:>{col_nav}}")

    print(SEP)
    print("  ✓ GÖREV TAMAMLANDI")
    print(SEP)


if __name__ == "__main__":
    main()
