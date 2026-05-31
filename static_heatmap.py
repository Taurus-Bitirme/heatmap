"""
static_heatmap.py
=================
OpenStreetMap tabanlı statik termal ısı haritası üreteci.

Folium (HTML) yerine, OSM tile'larından statik PNG görseli üretir.
Yangın tespitlerini Gaussian termal katman ile harita üzerine bindirir.

Bağımlılıklar:
    pip install staticmap Pillow scipy numpy
"""

import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from staticmap import StaticMap, CircleMarker


# ============================================================
#  YARDIMCI FONKSİYONLAR
# ============================================================

def _lat_lon_to_tile_coords(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Lat/Lon → Slippy Map tile koordinatları (kesirli)."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _meters_per_pixel(lat: float, zoom: int) -> float:
    """Belirli bir enlem ve zoom seviyesinde piksel başına metre."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _build_custom_colormap(n: int = 256) -> np.ndarray:
    """
    Özel renk haritası:
    Ateşin çekirdeğinde ve risk alanlarında orijinal risk renkleri korunur.
    Çevresinde kademeli olarak Sarı -> Yeşil -> Mavi şeklinde soğuma/yayılım uygulanır.
    """
    cmap = np.zeros((n, 4), dtype=np.uint8)

    # Kontrol noktaları: (pozisyon 0-1, R, G, B)
    # 0.0 - 0.50 arası: Çevreye yayılan ısı (Mavi -> Yeşil -> Sarı geçişi)
    # 0.50 - 1.0 arası: Kendi yangın risk seviyelerinizdeki renkler (Sarı -> Koyu Kırmızı)
    control_points = [
        (0.00, 0,   0,   255),   # koyu mavi (en dış soğuk kenar)
        (0.15, 0,   150, 255),   # açık mavi
        (0.35, 0,   255, 0),     # yeşil
        
        # Orijinal Risk Renkleri (Merkeze/Yüksek riske yaklaştıkça kırmızılaşır)
        (0.50, 255, 215, 0),     # Sarı           (0.0-0.2 Risk eşdeğeri / #FFD700)
        (0.65, 255, 165, 0),     # Turuncu Açık   (0.2-0.4 Risk eşdeğeri / #FFA500)
        (0.80, 255, 102, 0),     # Turuncu Koyu   (0.4-0.6 Risk eşdeğeri / #FF6600)
        (0.92, 255, 34,  0),     # Kırmızı Açık   (0.6-0.8 Risk eşdeğeri / #FF2200)
        (1.00, 139, 0,   0),     # Koyu Kırmızı   (0.8-1.0 Risk eşdeğeri / #8B0000)
    ]

    for i in range(n):
        t = i / max(1, n - 1)
        # Hangi segment?
        seg = 0
        for s in range(len(control_points) - 1):
            if t >= control_points[s][0]:
                seg = s
        p0 = control_points[seg]
        p1 = control_points[min(seg + 1, len(control_points) - 1)]
        seg_len = p1[0] - p0[0]
        local_t = (t - p0[0]) / seg_len if seg_len > 0 else 0.0
        local_t = max(0.0, min(1.0, local_t))

        r = int(p0[1] + (p1[1] - p0[1]) * local_t)
        g = int(p0[2] + (p1[2] - p0[2]) * local_t)
        b = int(p0[3] + (p1[3] - p0[3]) * local_t)
        cmap[i] = [r, g, b, 255]

    return cmap


def _draw_legend(overlay: Image.Image) -> None:
    """Sağ alt köşeye risk seviyesi lejantı çizer."""
    draw = ImageDraw.Draw(overlay)
    W, H = overlay.size

    # Orijinal heatmap_pipeline.py içerisindeki risk_color renk değerleri
    legend_items = [
        ("Çok Yüksek (0.8-1.0)", (139, 0,   0)),   # #8B0000
        ("Yüksek Risk (0.6-0.8)", (255, 34,  0)),   # #FF2200
        ("Orta Risk   (0.4-0.6)", (255, 102, 0)),   # #FF6600
        ("Orta-Düşük  (0.2-0.4)", (255, 165, 0)),   # #FFA500
        ("Düşük Risk  (0.0-0.2)", (255, 215, 0)),   # #FFD700
    ]

    line_h = 18
    pad = 10
    box_w = 210
    box_h = pad * 2 + len(legend_items) * line_h + 22
    x0 = W - box_w - 12
    y0 = H - box_h - 12

    # Yarı şeffaf arka plan
    legend_bg = Image.new("RGBA", (box_w, box_h), (255, 255, 255, 200))
    overlay.paste(legend_bg, (x0, y0), legend_bg)

    # Başlık
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    draw.text((x0 + pad, y0 + pad), "Yangın Risk Seviyeleri", fill=(0, 0, 0, 255), font=font)

    cy = y0 + pad + 20
    for label, color in legend_items:
        # Renk kutusu
        draw.rectangle([x0 + pad, cy + 2, x0 + pad + 12, cy + 14], fill=color + (255,))
        draw.text((x0 + pad + 18, cy), label, fill=(0, 0, 0, 255), font=font_small)
        cy += line_h


# ============================================================
#  ANA FONKSİYON
# ============================================================

def generate_static_osm_heatmap(
    fire_entries: List[dict],
    output_path: str,
    map_width: int = 800,
    map_height: int = 600,
    heat_alpha: float = 0.40,
    map_alpha: float = 0.60,
    zoom: int = None,
    padding_factor: float = 1.5,
) -> str:
    """
    Yangın tespitlerinden OpenStreetMap tabanlı statik termal ısı haritası PNG üretir.

    Parametreler
    ----------
    fire_entries : list[dict]
        Her eleman şu anahtarları içermelidir:
            - lat (float): Yangın enlem koordinatı
            - lon (float): Yangın boylam koordinatı
            - area_m2 (float): Yangının gerçek dünya alanı (m²)
            - risk_score (float): 0-1 arası risk skoru
        Opsiyonel:
            - track_id (int): Yangın takip ID'si

    output_path : str
        Çıktı PNG dosyasının yolu

    map_width, map_height : int
        Çıktı görselinin piksel boyutları (varsayılan: 800×600)

    heat_alpha : float
        Termal katmanın opaklığı (varsayılan: 0.40 → %40)

    map_alpha : float
        OSM harita katmanının opaklığı (varsayılan: 0.60 → %60)

    zoom : int veya None
        Harita zoom seviyesi (None → otomatik hesaplama)

    padding_factor : float
        Harita kenar boşluğu çarpanı (varsayılan: 1.5)

    Dönüş
    ------
    str : Çıktı dosyasının yolu
    """
    if not fire_entries:
        print("⚠  Yangın tespiti yok, harita oluşturulmadı.")
        return output_path

    # ----------------------------------------------------------
    # 1) Bounding box hesapla (tüm yangınları kapsayan)
    # ----------------------------------------------------------
    lats = [e["lat"] for e in fire_entries]
    lons = [e["lon"] for e in fire_entries]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # ----------------------------------------------------------
    # 2) Zoom seviyesini otomatik belirle
    # ----------------------------------------------------------
    if zoom is None:
        if len(fire_entries) == 1:
            # Tek yangın: sabit yakın zoom
            zoom = 18
        else:
            lat_span = max(lats) - min(lats)
            lon_span = max(lons) - min(lons)
            span = max(lat_span, lon_span) * padding_factor
            if span <= 0:
                zoom = 18
            else:
                # Her zoom seviyesinde görünen derece miktarı ≈ 360/2^zoom
                zoom = max(1, min(19, int(math.log2(360.0 / span)) - 1))

    # ----------------------------------------------------------
    # 3) StaticMap ile OSM tile'larını çek
    # ----------------------------------------------------------
    m = StaticMap(map_width, map_height,
                  url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png")

    # Tüm yangın noktalarını marker olarak ekle (haritanın sınırlarını belirlemek için)
    for e in fire_entries:
        # Görünmez (çok küçük) marker — sadece extent hesabı için
        m.add_marker(CircleMarker((e["lon"], e["lat"]), color="#00000000", width=1))

    try:
        pil_map = m.render(zoom=zoom, center=(center_lon, center_lat))
    except Exception as exc:
        print(f"⚠  StaticMap render hatası: {exc}")
        print("    İnternet bağlantısını kontrol edin.")
        return output_path

    # RGBA'ya çevir
    pil_map = pil_map.convert("RGBA")
    actual_w, actual_h = pil_map.size

    # ----------------------------------------------------------
    # 4) Koordinat dönüşümü: Lat/Lon → piksel
    # ----------------------------------------------------------
    # StaticMap iç hesaplamaları: tile koordinat sistemi
    # Merkez tile koordinatlarını hesapla
    cx_tile, cy_tile = _lat_lon_to_tile_coords(center_lat, center_lon, zoom)

    # Tile piksel boyutu (varsayılan 256)
    tile_size = 256
    # Merkez piksel (harita ortası)
    center_px_x = actual_w / 2.0
    center_px_y = actual_h / 2.0

    def latlon_to_pixel(lat: float, lon: float) -> Tuple[float, float]:
        """Lat/Lon → harita pikseli."""
        tx, ty = _lat_lon_to_tile_coords(lat, lon, zoom)
        px = center_px_x + (tx - cx_tile) * tile_size
        py = center_px_y + (ty - cy_tile) * tile_size
        return px, py

    # Metre/piksel oranı
    mpp = _meters_per_pixel(center_lat, zoom)

    # ----------------------------------------------------------
    # 5) Termal ısı haritası katmanı oluştur
    # ----------------------------------------------------------
    heat_layer = np.zeros((actual_h, actual_w), dtype=np.float64)

    for e in fire_entries:
        px, py = latlon_to_pixel(e["lat"], e["lon"])
        px_i = int(round(px))
        py_i = int(round(py))

        # Fiziksel yarıçap: sqrt(area / pi) metre → piksele çevir
        area_m2 = max(0.01, e.get("area_m2", 1.0))
        radius_m = math.sqrt(area_m2 / math.pi)
        radius_px = max(8, int(round(radius_m / mpp)))

        # Risk skoru → yoğunluk
        risk = max(0.05, min(1.0, e.get("risk_score", 0.5)))

        # Gaussian blob oluştur
        r = radius_px * 3  # etki alanı (3 sigma)
        y_min = max(0, py_i - r)
        y_max = min(actual_h, py_i + r + 1)
        x_min = max(0, px_i - r)
        x_max = min(actual_w, px_i + r + 1)

        if y_min >= y_max or x_min >= x_max:
            continue

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        dist_sq = (xx - px_i) ** 2 + (yy - py_i) ** 2
        sigma = max(3, radius_px)
        gauss = np.exp(-dist_sq / (2.0 * sigma ** 2))

        # Risk skoru ile yoğunluğu ölçekle
        gauss *= risk

        # Mevcut katmana ekle (birden fazla yangın üst üste binebilir)
        heat_layer[y_min:y_max, x_min:x_max] = np.maximum(
            heat_layer[y_min:y_max, x_min:x_max], gauss
        )

    # Ek Gaussian blur ile pürüzsüzleştirme
    if np.max(heat_layer) > 0:
        heat_layer = gaussian_filter(heat_layer, sigma=3.0)

    # ----------------------------------------------------------
    # 6) Colormap uygula
    # ----------------------------------------------------------
    cmap = _build_custom_colormap(256)

    # 0-1 arası normalize
    max_val = np.max(heat_layer)
    if max_val > 0:
        heat_norm = heat_layer / max_val
    else:
        heat_norm = heat_layer

    # 0-255 indeks
    indices = (heat_norm * 255).astype(np.uint8)
    heat_rgba = cmap[indices]  # (H, W, 4)
    heat_img = Image.fromarray(heat_rgba, "RGBA")

    # ----------------------------------------------------------
    # 7) Alpha blending: ısı katmanı + OSM haritası
    # ----------------------------------------------------------
    # Isı katmanının alpha kanalını ayarla
    heat_arr = np.array(heat_img, dtype=np.float64)
    map_arr = np.array(pil_map, dtype=np.float64)

    blended = (map_alpha * map_arr + heat_alpha * heat_arr)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended[:, :, 3] = 255  # Tam opak

    result = Image.fromarray(blended, "RGBA")

    # ----------------------------------------------------------
    # 8) Yangın konumlarını işaretle
    # ----------------------------------------------------------
    draw = ImageDraw.Draw(result)
    try:
        label_font = ImageFont.truetype("arial.ttf", 12)
    except (OSError, IOError):
        label_font = ImageFont.load_default()

    for e in fire_entries:
        px, py = latlon_to_pixel(e["lat"], e["lon"])
        px_i, py_i = int(round(px)), int(round(py))
        # Küçük çarpı işareti
        cross_size = 6
        draw.line([(px_i - cross_size, py_i), (px_i + cross_size, py_i)],
                  fill=(255, 255, 255, 220), width=2)
        draw.line([(px_i, py_i - cross_size), (px_i, py_i + cross_size)],
                  fill=(255, 255, 255, 220), width=2)

        # Track ID etiketi
        tid = e.get("track_id", "?")
        risk_s = e.get("risk_score", 0)
        label = f"T{tid} ({risk_s:.2f})"
        draw.text((px_i + 8, py_i - 8), label,
                  fill=(255, 255, 255, 240), font=label_font)

    # ----------------------------------------------------------
    # 9) Lejant ekle
    # ----------------------------------------------------------
    _draw_legend(result)

    # ----------------------------------------------------------
    # 10) Kaydet
    # ----------------------------------------------------------
    result_rgb = result.convert("RGB")
    result_rgb.save(output_path, "PNG")
    print(f"✓ Statik ısı haritası kaydedildi: {output_path}")

    return output_path


# ============================================================
#  BAĞIMSIZ TEST
# ============================================================
if __name__ == "__main__":
    # Örnek yangın verileri (İstanbul civarı)
    test_fires = [
        {
            "lat": 41.0082,
            "lon": 28.9784,
            "area_m2": 120.0,
            "risk_score": 0.85,
            "track_id": 1,
        },
        {
            "lat": 41.0079,
            "lon": 28.9790,
            "area_m2": 45.0,
            "risk_score": 0.55,
            "track_id": 2,
        },
        {
            "lat": 41.0085,
            "lon": 28.9778,
            "area_m2": 8.0,
            "risk_score": 0.20,
            "track_id": 3,
        },
    ]

    out = generate_static_osm_heatmap(test_fires, "test_heatmap_output.png")
    print(f"Test çıktısı: {out}")
