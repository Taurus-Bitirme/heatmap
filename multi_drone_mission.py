"""
multi_drone_mission.py
======================
3 Drone'lu Federe Yangin Tespiti — Ana Orkestrator

Kullanim:
    py -3.12 multi_drone_mission.py
"""

import os
import time
import queue
import threading

import cv2

from multi_drone_config import (
    DRONE_NAMES, NUM_DRONES, AREA_WIDTH_M, AREA_HEIGHT_M, CAMERA_FOOTPRINT_M,
    SCAN_MIN_X, SCAN_MIN_Y, SCAN_MAX_X, SCAN_MAX_Y,
    HEATMAP_SIZE_PX, SCAN_ROW_SPACING,
    alani_dronelara_bol,
)
from drone_worker import DroneWorker
from central_server import CentralHeatmapServer

OUTPUT_DIR = "scan_results"


def main() -> None:
    SEP = "=" * 60

    print(SEP)
    print("  3 DRONE FEDERE YANGIN TESPIT SISTEMI")
    print(SEP)
    print(f"  Alan     : ({SCAN_MIN_X:.0f},{SCAN_MIN_Y:.0f}) → "
          f"({SCAN_MAX_X:.1f},{SCAN_MAX_Y:.1f}) m")
    print(f"  Boyut    : X={AREA_HEIGHT_M:.1f} m | Y={AREA_WIDTH_M:.1f} m")
    print(f"  Kamera   : {CAMERA_FOOTPRINT_M:.1f} x {CAMERA_FOOTPRINT_M:.1f} m footprint")
    print(f"  Heatmap  : {HEATMAP_SIZE_PX} x {HEATMAP_SIZE_PX} px")
    print(f"  Zigzag   : {SCAN_ROW_SPACING} m aralik")
    print(SEP)

    # ── ADIM 1: Alan bölme + Zigzag waypoint ──
    print("\n  ADIM 1: Alan bolme + zigzag waypoint hesaplama...")
    assignments = alani_dronelara_bol()

    for drone_id, wps in assignments.items():
        if wps:
            x_vals = [w[0] for w in wps]
            y_vals = [w[1] for w in wps]
            print(f"  {drone_id}: {len(wps)} waypoint | "
                  f"X=[{min(x_vals):.1f}, {max(x_vals):.1f}] "
                  f"Y=[{min(y_vals):.1f}, {max(y_vals):.1f}]")
        else:
            print(f"  {drone_id}: waypoint yok!")

    total_wp = sum(len(w) for w in assignments.values())
    print(f"  Toplam: {total_wp} waypoint")

    # ── ADIM 2: Merkezi sunucu ──
    print(f"\n  ADIM 2: Merkezi sunucu baslatiliyor...")
    result_queue = queue.Queue()
    global_out = os.path.join(OUTPUT_DIR, "global")

    server = CentralHeatmapServer(
        result_queue=result_queue,
        output_dir=global_out,
    )
    server.start()
    print("  Merkezi sunucu aktif (1000x1000 px tuval)")

    # ── ADIM 3: Drone worker'lar ──
    print(f"\n  ADIM 3: Drone worker'lar baslatiliyor...")
    stop_event = threading.Event()
    workers = []

    for drone_id, wps in assignments.items():
        if not wps:
            print(f"  {drone_id}: Waypoint yok, atlaniyor")
            continue

        w = DroneWorker(
            drone_id=drone_id,
            waypoints=wps,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        workers.append(w)
        w.start()
        print(f"  {drone_id} baslatildi — {len(wps)} waypoint zigzag")
        time.sleep(1.0)

    if not workers:
        print("  HATA: Hic aktif drone yok!")
        server.stop()
        return

    # ── ADIM 4: Canlı ısı haritası ──
    print(f"\n  ADIM 4: Canli gosterim")
    print("  ESC/Q=Cikis | S=Snapshot")
    print(SEP)

    cv2.namedWindow("Global Isi Haritasi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Global Isi Haritasi", 800, 800)

    t0 = time.time()

    try:
        while True:
            # Drone konumlarını güncelle
            for w in workers:
                server.update_drone_pos(
                    w.drone_id, w.current_x, w.current_y
                )

            img = server.get_heatmap_image()
            cv2.imshow("Global Isi Haritasi", img)
            key = cv2.waitKey(200) & 0xFF

            if key in (27, ord('q')):
                print("\n  Kullanici cikis yapti")
                stop_event.set()
                break

            if key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                sp = os.path.join(OUTPUT_DIR, f"snapshot_{ts}.png")
                cv2.imwrite(sp, img)
                print(f"  Snapshot: {sp}")

            if not any(w.is_alive() for w in workers):
                print("\n  Tum drone'lar gorevi tamamladi!")
                time.sleep(1.0)
                break

    except KeyboardInterrupt:
        print("\n  Ctrl+C — durduruluyor")
        stop_event.set()

    finally:
        for w in workers:
            w.join(timeout=30)

        time.sleep(0.5)
        server.stop()
        server.join(timeout=10)

        server.save_report()
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        print(f"\n{SEP}")
        print("  GOREV OZETI")
        print(SEP)
        print(f"  Sure    : {elapsed/60:.1f} dk ({elapsed:.0f} s)")
        print(f"  Drone   : {len(workers)}")
        print(f"  Waypoint: {total_wp}")
        for w in workers:
            print(f"    {w.drone_id}: {w.scanned_count} tarama, "
                  f"{w.fire_count} yangin | {w.status}")
        print(f"  Yangin  : {len(server.all_fires)}")
        print(f"  Rapor   : {global_out}")
        print(SEP)


if __name__ == "__main__":
    main()
