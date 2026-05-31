"""
central_server.py
=================
Merkezi Isı Haritası Sunucusu — 1000×1000 piksel

3 drone'dan gelen (Drone_ID, X, Y, Skor) verilerini
siyah bir tuval üzerine Gaussian blob olarak işler.
OpenCV COLORMAP_JET ile renkli ısı haritası üretir.
"""

import os
import time
import json
import threading
import queue
from typing import List, Dict, Tuple
import math
from datetime import datetime, timezone

import cv2
import numpy as np

from multi_drone_config import (
    DRONE_NAMES, NUM_DRONES,
    AREA_WIDTH_M, AREA_HEIGHT_M, HEATMAP_SIZE_PX, HEATMAP_SCALE,
    HEATMAP_OFFSET_COL, HEATMAP_OFFSET_ROW,
    HEATMAP_CONTENT_W_PX, HEATMAP_CONTENT_H_PX,
    DRONE_COLORS, ned_to_heatmap_px, is_inside_landscape,
    CALIBRATION_ENABLED, CALIBRATION_FIRE_POINTS_NED,
)

# Minimum güven skoru — bu altındaki tespitler haritaya eklenmez
MIN_SCORE_THRESHOLD: float = 0.20

# Spatial clustering: bu mesafedeki (metre) tespitler aynı yangın sayılır
CLUSTER_RADIUS_M: float = 8.0
CLUSTER_MERGE_RADIUS_M: float = 12.0

# A cluster becomes confirmed when repeated observations or multi-drone
# agreement make it reliable enough for the final heatmap.
CONFIRMED_MIN_COUNT: int = 3
CONFIRMED_MIN_DRONES: int = 2
CONFIRMED_MIN_MAX_SCORE: float = 0.75


class CentralHeatmapServer(threading.Thread):
    """
    Merkezi ısı haritası birleştirici.

    Queue'dan gelen (drone_id, ned_x, ned_y, score) tuple'larını dinler,
    1000×1000 siyah tuval üzerine Gaussian blob ekler.
    """

    def __init__(
        self,
        result_queue: queue.Queue,
        output_dir: str = "scan_results/global",
    ):
        super().__init__(name="CentralServer", daemon=True)
        self.result_queue = result_queue
        self.output_dir = output_dir

        # 1000×1000 siyah tuval (float32, birikimli)
        self.heatmap_raw = np.zeros(
            (HEATMAP_SIZE_PX, HEATMAP_SIZE_PX), dtype=np.float32
        )

        # Thread güvenliği
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Drone konumları
        self.drone_positions: Dict[str, Tuple[float, float]] = {
            name: (0.0, 0.0) for name in DRONE_NAMES
        }

        # Tüm tespitler
        self.all_fires: List[Dict] = []
        # Kümeler: {"cx": float, "cy": float, "max_score": float, "count": int, "total_score": float}
        self.clusters: List[Dict] = []
        self.finished_drones: int = 0

        os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    #  Triangulasyon ile Kume Merkezi Hesaplama
    # ----------------------------------------------------------------

    def _update_cluster_center(self, cluster: Dict) -> None:
        """
        Kume merkezini guncelle: skor agirlikli ortalama.

        Yuksek guven skorlu tespitler daha fazla agirlik alir.
        """
        xs = cluster["xs"]
        ys = cluster["ys"]
        scores = cluster.get("scores", [1.0] * len(xs))

        n = min(len(xs), len(ys), len(scores))
        if n == 0:
            return

        w_arr = [max(s, 0.01) for s in scores[:n]]
        w_sum = sum(w_arr)
        cluster["cx"] = sum(x * w for x, w in zip(xs, w_arr)) / w_sum
        cluster["cy"] = sum(y * w for y, w in zip(ys, w_arr)) / w_sum
        cluster["center_method"] = "score_weighted"

    def _draw_cluster_blob(self, heatmap: np.ndarray, cluster: Dict) -> None:
        """Draw one cluster onto the supplied heatmap canvas."""
        col, row = ned_to_heatmap_px(cluster["cx"], cluster["cy"])
        confidence = float(cluster.get("confidence", cluster["max_score"]))
        radius = max(10, int(16 + 12 * confidence))
        tmp = np.zeros_like(heatmap)
        cv2.circle(tmp, (col, row), radius, confidence, -1)
        ksize = radius * 4 + 1
        if ksize % 2 == 0:
            ksize += 1
        tmp = cv2.GaussianBlur(tmp, (ksize, ksize), radius / 2.0)
        heatmap += tmp

    def _rebuild_heatmap_locked(self) -> None:
        """Rebuild heatmap from clusters to avoid overlap/subtraction artifacts."""
        self.heatmap_raw.fill(0.0)
        for cluster in self.clusters:
            if cluster.get("status") == "confirmed":
                self._draw_cluster_blob(self.heatmap_raw, cluster)

    def _refresh_cluster_quality(self, cluster: Dict) -> None:
        count = max(1, int(cluster["count"]))
        avg_score = float(cluster["total_score"]) / count
        drone_count = len(cluster.get("drone_ids", []))
        xs = cluster.get("xs", [cluster["cx"]])
        ys = cluster.get("ys", [cluster["cy"]])
        spread_m = float(max(np.std(xs), np.std(ys))) if len(xs) > 1 else 0.0

        count_score = min(1.0, count / CONFIRMED_MIN_COUNT)
        drone_score = min(1.0, drone_count / CONFIRMED_MIN_DRONES)
        spread_score = max(0.0, 1.0 - spread_m / max(CLUSTER_RADIUS_M, 1e-6))
        confidence = (
            0.35 * float(cluster["max_score"]) +
            0.25 * avg_score +
            0.20 * count_score +
            0.10 * drone_score +
            0.10 * spread_score
        )

        is_confirmed = (
            count >= CONFIRMED_MIN_COUNT or drone_count >= CONFIRMED_MIN_DRONES
        ) and float(cluster["max_score"]) >= CONFIRMED_MIN_MAX_SCORE

        cluster["avg_score"] = avg_score
        cluster["drone_count"] = drone_count
        cluster["spread_m"] = spread_m
        cluster["confidence"] = float(max(0.0, min(1.0, confidence)))
        cluster["status"] = "confirmed" if is_confirmed else "candidate"

    def _merge_cluster_pair(self, target: Dict, source: Dict) -> None:
        target["count"] += source["count"]
        target["total_score"] += source["total_score"]
        target["max_score"] = max(target["max_score"], source["max_score"])

        target["xs"].extend(source.get("xs", [source["cx"]]))
        target["ys"].extend(source.get("ys", [source["cy"]]))
        for drone_id in source.get("drone_ids", []):
            if drone_id not in target["drone_ids"]:
                target["drone_ids"].append(drone_id)

        target["cx"] = float(np.mean(target["xs"]))
        target["cy"] = float(np.mean(target["ys"]))
        self._refresh_cluster_quality(target)

    def _merge_close_clusters_locked(self) -> None:
        changed = True
        while changed:
            changed = False
            for i in range(len(self.clusters)):
                if changed:
                    break
                for j in range(i + 1, len(self.clusters)):
                    a = self.clusters[i]
                    b = self.clusters[j]
                    dist = math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])
                    if dist <= CLUSTER_MERGE_RADIUS_M:
                        self._merge_cluster_pair(a, b)
                        del self.clusters[j]
                        changed = True
                        break

    def _build_calibration_record(
        self,
        ned_x: float,
        ned_y: float,
        calibration: Dict = None,
    ) -> Dict:
        record = dict(calibration or {})
        if CALIBRATION_ENABLED and CALIBRATION_FIRE_POINTS_NED:
            nearest_name = None
            nearest_x = 0.0
            nearest_y = 0.0
            nearest_err = float("inf")
            for name, (gt_x, gt_y) in CALIBRATION_FIRE_POINTS_NED.items():
                err = math.hypot(ned_x - gt_x, ned_y - gt_y)
                if err < nearest_err:
                    nearest_name = name
                    nearest_x = gt_x
                    nearest_y = gt_y
                    nearest_err = err

            record["nearest_gt_name"] = nearest_name
            record["nearest_gt_ned"] = [round(nearest_x, 4), round(nearest_y, 4)]
            record["error_m"] = round(nearest_err, 4)
            record["error_dx_m"] = round(ned_x - nearest_x, 4)
            record["error_dy_m"] = round(ned_y - nearest_y, 4)

        return record

    def run(self) -> None:
        print("[MERKEZ] Isi haritasi sunucusu baslatildi (1000x1000 px)")

        while not self._stop_event.is_set():
            try:
                msg = self.result_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            if isinstance(msg, tuple) and len(msg) in (4, 5):
                drone_id, ned_x, ned_y, score = msg[:4]
                calibration = msg[4] if len(msg) == 5 else None
                self._add_fire(drone_id, ned_x, ned_y, score, calibration)

    def stop(self) -> None:
        self._stop_event.set()

    def _add_fire(
        self,
        drone_id: str,
        ned_x: float,
        ned_y: float,
        score: float,
        calibration: Dict = None,
    ) -> None:
        """Yangın noktasını 1000×1000 grid'e Gaussian blob olarak ekle."""
        with self._lock:
            # ── 1. Minimum skor filtresi ──────────────────────────────
            if score < MIN_SCORE_THRESHOLD:
                print(
                    f"[MERKEZ] ATLANDI (düşük skor) — {drone_id} | "
                    f"NED({ned_x:.1f}, {ned_y:.1f}) | Skor={score:.2f}"
                )
                return

            # ── 2. Alan sınırı filtresi ───────────────────────────────
            if not is_inside_landscape(ned_x, ned_y):
                print(
                    f"[MERKEZ] ATLANDI (alan dışı) — {drone_id} | "
                    f"NED({ned_x:.1f}, {ned_y:.1f})"
                )
                return

            # ── 3. Spatial clustering ─────────────────────────────────
            # Yakın bir küme var mı?
            matched_cluster = None
            is_new_cluster = False
            
            for cl in self.clusters:
                dist = math.hypot(ned_x - cl["cx"], ned_y - cl["cy"])
                if dist <= CLUSTER_RADIUS_M:
                    matched_cluster = cl
                    break

            if matched_cluster is None:
                # Yeni kume olustur
                matched_cluster = {
                    "cx": ned_x, "cy": ned_y,
                    "max_score": score, "count": 0, "total_score": 0.0,
                    "drone_ids": [],
                    "xs": [], "ys": [],
                    "drone_xs": [], "drone_ys": [],
                    "scores": [],
                    "status": "candidate",
                    "confidence": score,
                    # GPS koordinatları (lat/lon — OSM haritası için)
                    "lats": [], "lons": [],
                }
                self.clusters.append(matched_cluster)
                is_new_cluster = True
                action = "YENI KUME"
            else:
                is_new_cluster = False
                action = "KUME GUNCELLEME"

            # --- Kume istatistiklerini guncelle ---
            matched_cluster["count"] += 1
            matched_cluster["total_score"] += score
            if drone_id not in matched_cluster["drone_ids"]:
                matched_cluster["drone_ids"].append(drone_id)

            # Tespit ve drone konumlarini sakla
            matched_cluster["xs"].append(float(ned_x))
            matched_cluster["ys"].append(float(ned_y))
            matched_cluster["scores"].append(float(score))

            # GPS lat/lon — OSM haritası için kümeye ekle
            if "lats" not in matched_cluster:
                matched_cluster["lats"] = []
                matched_cluster["lons"] = []
            if calibration:
                lat_val = calibration.get("lat", 0.0)
                lon_val = calibration.get("lon", 0.0)
                if lat_val and lon_val and lat_val != 0.0 and lon_val != 0.0:
                    matched_cluster["lats"].append(float(lat_val))
                    matched_cluster["lons"].append(float(lon_val))

            # Drone konumunu calibration'dan al
            drone_x_pos = float(calibration.get("drone_x", ned_x)) if calibration else ned_x
            drone_y_pos = float(calibration.get("drone_y", ned_y)) if calibration else ned_y
            if "drone_xs" not in matched_cluster:
                matched_cluster["drone_xs"] = []
                matched_cluster["drone_ys"] = []
            matched_cluster["drone_xs"].append(drone_x_pos)
            matched_cluster["drone_ys"].append(drone_y_pos)

            if score > matched_cluster["max_score"]:
                matched_cluster["max_score"] = score

            # --- Kume merkezini hesapla ---
            self._update_cluster_center(matched_cluster)

            self._refresh_cluster_quality(matched_cluster)
            self._merge_close_clusters_locked()

            self._rebuild_heatmap_locked()

            # Logu kaydet
            fire_entry = {
                "drone_id": drone_id,
                "ned_x": round(ned_x, 2),
                "ned_y": round(ned_y, 2),
                "score": round(score, 4),
                "timestamp": time.time(),
            }
            calibration_record = self._build_calibration_record(
                ned_x, ned_y, calibration
            )
            if calibration_record:
                fire_entry["calibration"] = calibration_record
            self.all_fires.append(fire_entry)

            cm = matched_cluster.get('center_method', '?')
            print(
                f"[MERKEZ] YANGIN ({action}) — {drone_id} | "
                f"NED({ned_x:.1f}, {ned_y:.1f}) | "
                f"Skor={score:.2f} | "
                f"Merkez=({matched_cluster['cx']:.1f}, {matched_cluster['cy']:.1f}) "
                f"[{cm}] | "
                f"Durum={matched_cluster['status']} | "
                f"Kume sayisi: {len(self.clusters)}"
            )


    def update_drone_pos(
        self, drone_id: str, ned_x: float, ned_y: float
    ) -> None:
        """Drone konumunu dışarıdan güncelle (harita gösterimi için)."""
        with self._lock:
            self.drone_positions[drone_id] = (ned_x, ned_y)

    # ----------------------------------------------------------------
    #  Görselleştirme
    # ----------------------------------------------------------------
    def get_heatmap_image(self) -> np.ndarray:
        """
        1000×1000 heatmap'i renkli BGR görüntüye dönüştür.
        Drone konumlarını ve bilgi panelini ekle.
        """
        with self._lock:
            raw_copy = self.heatmap_raw.copy()
            positions = dict(self.drone_positions)
            n_fires = len(self.all_fires)
            n_clusters = len(self.clusters)
            n_confirmed = sum(1 for c in self.clusters if c.get("status") == "confirmed")
            n_candidates = n_clusters - n_confirmed

        # Normalize → 0-255
        max_val = raw_copy.max()
        if max_val > 1e-6:
            norm = np.clip(raw_copy / max_val, 0, 1)
        else:
            norm = raw_copy
        gray = (norm * 255).astype(np.uint8)

        # COLORMAP_JET uygula
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Yangın olmayan bölgeler → koyu gri (siyah tuval etkisi)
        no_fire = gray < 3
        colored[no_fire] = [30, 30, 30]

        # Draw the actual landscape frame inside the square mini-map canvas.
        x0 = HEATMAP_OFFSET_COL
        y0 = HEATMAP_OFFSET_ROW
        x1 = HEATMAP_OFFSET_COL + HEATMAP_CONTENT_W_PX
        y1 = HEATMAP_OFFSET_ROW + HEATMAP_CONTENT_H_PX
        cv2.rectangle(colored, (x0, y0), (x1, y1), (95, 95, 95), 1)
        # Drone konumlarını çiz
        for drone_id, (nx, ny) in positions.items():
            col, row = ned_to_heatmap_px(nx, ny)
            color = DRONE_COLORS.get(drone_id, (255, 255, 255))

            cv2.drawMarker(colored, (col, row), color,
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(colored, drone_id, (col + 12, row - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Bilgi paneli
        info = [
            f"GLOBAL ISI HARITASI — 3 Drone",
            f"Landscape: {AREA_HEIGHT_M:.0f}x{AREA_WIDTH_M:.0f} m ({HEATMAP_SIZE_PX}x{HEATMAP_SIZE_PX} px)",
            f"Confirmed: {n_confirmed} | Candidate: {n_candidates} | Ham: {n_fires}",
        ]
        for i, txt in enumerate(info):
            cv2.putText(colored, txt, (10, 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

        # 10m ölçek çubuğu
        scale_px = int(10 * HEATMAP_SCALE)
        x1 = HEATMAP_SIZE_PX - 20 - scale_px
        cv2.line(colored, (x1, HEATMAP_SIZE_PX - 20),
                 (x1 + scale_px, HEATMAP_SIZE_PX - 20), (200, 200, 200), 2)
        cv2.putText(colored, "10m", (x1, HEATMAP_SIZE_PX - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return colored

    # ----------------------------------------------------------------
    #  Final Rapor
    # ----------------------------------------------------------------
    def save_report(self) -> None:
        print(f"\n{'='*60}")
        print("  FINAL RAPOR")
        print(f"{'='*60}")

        # JSON
        report = {
            "mission_end": datetime.now(timezone.utc).isoformat(),
            "area_m": {
                "height_x": AREA_HEIGHT_M,
                "width_y": AREA_WIDTH_M,
            },
            "heatmap_px": HEATMAP_SIZE_PX,
            "total_fires": len(self.all_fires),
            "unique_fire_clusters": len(self.clusters),
            "confirmed_fire_clusters": sum(
                1 for c in self.clusters if c.get("status") == "confirmed"
            ),
            "calibration_enabled": CALIBRATION_ENABLED,
            "calibration_ground_truth_ned": {
                name: [round(x, 4), round(y, 4)]
                for name, (x, y) in CALIBRATION_FIRE_POINTS_NED.items()
            },
            "clusters": self.clusters,
            "fires": self.all_fires,
        }
        json_path = os.path.join(self.output_dir, "all_fires.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  JSON: {json_path}")

        # Heatmap PNG
        img = self.get_heatmap_image()
        png_path = os.path.join(self.output_dir, "global_heatmap.png")
        cv2.imwrite(png_path, img)
        print(f"  PNG : {png_path}")

        # Raw heatmap (numpy)
        npy_path = os.path.join(self.output_dir, "heatmap_raw.npy")
        np.save(npy_path, self.heatmap_raw)
        print(f"  NPY : {npy_path}")

        # OSM / Folium tabanlı statik harita
        # Ham tespitler yerine KÜME MERKEZLERİ kullanılır:
        # aynı yangının farklı waypoint'lerden defalarca görülmesi
        # tek bir blob olarak gösterilir.
        try:
            from static_heatmap import generate_static_osm_heatmap

            osm_entries = []
            for cl in self.clusters:
                lats = cl.get("lats", [])
                lons = cl.get("lons", [])
                if not lats or not lons:
                    continue
                # Küme üyelerinin ortalama GPS koordinatı
                avg_lat = sum(lats) / len(lats)
                avg_lon = sum(lons) / len(lons)
                # Küme için ortalama alan (calibration'dan)
                area_vals = []
                for fire in self.all_fires:
                    cal = fire.get("calibration", {})
                    fx = fire.get("ned_x", None)
                    fy = fire.get("ned_y", None)
                    if fx is None or fy is None:
                        continue
                    dist = math.hypot(fx - cl["cx"], fy - cl["cy"])
                    if dist <= CLUSTER_RADIUS_M:
                        a = cal.get("area_m2")
                        if a:
                            area_vals.append(float(a))
                avg_area = sum(area_vals) / len(area_vals) if area_vals else 1.0

                osm_entries.append({
                    "lat":        avg_lat,
                    "lon":        avg_lon,
                    "area_m2":    avg_area,
                    "risk_score": float(cl.get("confidence", cl["max_score"])),
                    "track_id":   f"cluster_{len(osm_entries)}",
                })

            if osm_entries:
                osm_path = os.path.join(self.output_dir, "osm_heatmap.png")
                generate_static_osm_heatmap(osm_entries, osm_path)
                print(f"  OSM : {osm_path}  ({len(osm_entries)} küme)")
            else:
                print("  OSM : Geçerli GPS koordinatı bulunamadı, atlandı")
        except Exception as e:
            print(f"  OSM : Oluşturulamadı — {e}")

        print(f"\n  Toplam yangin: {len(self.all_fires)}")
        for d in DRONE_NAMES:
            cnt = sum(1 for f in self.all_fires if f["drone_id"] == d)
            print(f"    {d}: {cnt}")
        print(f"{'='*60}")
