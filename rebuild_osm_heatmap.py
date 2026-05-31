"""
rebuild_osm_heatmap.py
======================
Mevcut all_fires.json dosyasından OSM ısı haritasını yeniden çizer.

Kullanım:
    python rebuild_osm_heatmap.py
    python rebuild_osm_heatmap.py scan_results/global/all_fires.json

Her küme için üye tespit noktalarının ortalama GPS koordinatı alınır.
Böylece aynı yangının farklı waypoint'lerden tekrarlanan tespitleri
tek bir blob olarak gösterilir.
"""

import os
import sys
import json
import math
from typing import List, Dict

# ── Varsayılan JSON yolu ────────────────────────────────────────
DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scan_results", "global", "all_fires.json",
)

CLUSTER_RADIUS_M = 8.0   # küme üyeliği için mesafe eşiği


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assign_fires_to_clusters(
    fires: List[dict],
    clusters: List[dict],
) -> Dict[int, List[dict]]:
    """
    Her tespit noktasını en yakın küme merkezine atar.
    Dönüş: {küme_indeksi: [fire_dict, ...]}
    """
    assignments: Dict[int, List[dict]] = {i: [] for i in range(len(clusters))}

    for fire in fires:
        fx = fire.get("ned_x", 0.0)
        fy = fire.get("ned_y", 0.0)
        best_idx = -1
        best_dist = float("inf")
        for i, cl in enumerate(clusters):
            d = math.hypot(fx - cl["cx"], fy - cl["cy"])
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0 and best_dist <= CLUSTER_RADIUS_M * 2:
            assignments[best_idx].append(fire)

    return assignments


def rebuild(json_path: str) -> str:
    """
    json_path : all_fires.json konumu
    Dönüş     : kaydedilen OSM PNG yolu
    """
    print(f"Veri okunuyor: {json_path}")
    data = _load_json(json_path)

    clusters = data.get("clusters", [])
    fires    = data.get("fires",    [])

    print(f"  Küme sayısı   : {len(clusters)}")
    print(f"  Tespit sayısı : {len(fires)}")

    if not clusters:
        print("Küme verisi yok, çıkılıyor.")
        return ""

    # Her tespit noktasını kümeye ata
    assignments = _assign_fires_to_clusters(fires, clusters)

    # Her küme için ortalama GPS hesapla
    osm_entries = []
    for i, cl in enumerate(clusters):
        member_fires = assignments.get(i, [])

        lats, lons, areas = [], [], []
        for fire in member_fires:
            cal = fire.get("calibration", {})
            lat = cal.get("lat", 0.0)
            lon = cal.get("lon", 0.0)
            area = cal.get("area_m2", 0.0)
            if lat and lon and lat != 0.0 and lon != 0.0:
                lats.append(lat)
                lons.append(lon)
            if area:
                areas.append(area)

        if not lats:
            print(f"  Küme {i}: GPS koordinatı yok, atlandı")
            continue

        avg_lat  = sum(lats)  / len(lats)
        avg_lon  = sum(lons)  / len(lons)
        avg_area = sum(areas) / len(areas) if areas else 1.0
        confidence = float(cl.get("confidence", cl.get("max_score", 0.5)))
        status     = cl.get("status", "candidate")

        osm_entries.append({
            "lat":        avg_lat,
            "lon":        avg_lon,
            "area_m2":    avg_area,
            "risk_score": confidence,
            "track_id":   f"Küme-{i} ({status})",
        })

        print(
            f"  Küme {i}: lat={avg_lat:.6f} lon={avg_lon:.6f} "
            f"alan={avg_area:.2f}m² güven={confidence:.3f} "
            f"[{status}] ({len(lats)} GPS noktası)"
        )

    if not osm_entries:
        print("Hiçbir kümede GPS koordinatı bulunamadı.")
        return ""

    # OSM haritasını çiz
    out_dir  = os.path.dirname(json_path)
    osm_path = os.path.join(out_dir, "osm_heatmap.png")

    from static_heatmap import generate_static_osm_heatmap
    generate_static_osm_heatmap(osm_entries, osm_path)

    print(f"\n✓ OSM haritası kaydedildi: {osm_path}")
    print(f"  Gösterilen küme sayısı  : {len(osm_entries)}")
    return osm_path


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON

    if not os.path.exists(json_path):
        print(f"HATA: Dosya bulunamadı: {json_path}")
        sys.exit(1)

    rebuild(json_path)
