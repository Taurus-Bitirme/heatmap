# Çok Drone'lu Otonom Yangın Tespit ve Risk Haritalama Sistemi

> **Bitirme Projesi** — Dağıtık Algılama + Merkezi Füzyon Mimarisi  
> Microsoft AirSim + Unreal Engine simülasyon ortamında geliştirilmiştir.

---

## Projeye Genel Bakış

Bu sistem, 100×100 metrelik bir simülasyon haritasında **3 drone'u eş zamanlı** çalıştırarak yangın tespiti yapar ve GPS koordinatlarına dayalı ısı haritası üretir.

Her drone kendi bölgesini **bağımsız** tarar, YOLO modelini **yerel** olarak çalıştırır ve tespit sonuçlarını merkezi bir sunucuya iletir. Merkezi sunucu gelen verileri birleştirir, sahte pozitifleri filtreler ve nihai haritayı üretir.

```
Drone1 (thread) ──┐
Drone2 (thread) ──┼──► Merkezi Sunucu ──► OSM Haritası + JET Isı Haritası
Drone3 (thread) ──┘
```

---

## Kurulum Gereksinimleri

### Python Paketleri

```bash
pip install airsim ultralytics opencv-python numpy scipy Pillow staticmap
```

### AirSim Ayarları

`settings_3drone.json` dosyasını AirSim ayarlar klasörüne kopyala:

**Windows:**
```
C:\Users\<kullanici_adin>\Documents\AirSim\settings.json
```

Bu dosya 3 drone'u (Drone1, Drone2, Drone3) tanımlar. Her drone için ayrı kamera ve başlangıç konumu ayarlanmıştır.

### YOLO Modeli

`modelimiz.pt` dosyası zaten klasörde mevcut. Model, simülasyon ortamından toplanan yangın görüntüleriyle eğitilmiştir.

---

## Dosya Yapısı ve Görevleri

```
final_son_kod/
│
├── multi_drone_mission.py      ← BURADAN BAŞLA — Ana giriş noktası
├── drone_worker.py             ← Her drone'un görev döngüsü (PID, YOLO, HSV renk analizi, risk skoru)
├── central_server.py           ← Merkezi füzyon sunucusu
├── multi_drone_config.py       ← Tüm parametreler ve waypoint üretimi
│
├── static_heatmap.py           ← OSM tabanlı gerçek harita üretimi
│
├── settings_3drone.json        ← AirSim 3 drone konfigürasyonu
├── modelimiz.pt                ← Eğitilmiş YOLO modeli (~20 MB)
└── README.md                   ← Bu dosya
```

---

## Nasıl Çalıştırılır?

### 1. AirSim'i Başlat

Unreal Engine projesini aç ve oynat (Play). AirSim ortamın hazır olduğundan emin ol.

### 2. Görevi Başlat

```bash
python multi_drone_mission.py
```

Sistem otomatik olarak:
- 3 drone thread'ini başlatır
- Her drone'u kalkışa hazırlar
- Tarama görevini başlatır
- Merkezi sunucuyu dinlemeye alır
- Canlı ısı haritasını ekranda gösterir

### 3. Sonuçları Gör

Görev tamamlandığında `scan_results/global/` klasöründe şunlar oluşur:

```
scan_results/global/
├── all_fires.json          ← Tüm tespitler, kümeler, GPS koordinatları
├── global_heatmap.png      ← JET renk haritası (OpenCV)
├── osm_heatmap.png         ← Gerçek harita üzerinde ısı katmanı (OSM)
└── heatmap_raw.npy         ← Ham numpy verisi
```

---

## Sistem Nasıl Çalışır? (Adım Adım)

### Adım 1 — Alan Bölümleme

100×100 m harita X ekseninde 3 eşit şeride bölünür:

```
Drone1: X = [  0.0 – 33.3 m ]
Drone2: X = [ 33.3 – 66.7 m ]
Drone3: X = [ 66.7 – 100.0 m ]
```

### Adım 2 — Waypoint Üretimi (Footprint Tabanlı)

Her drone'un tarama adımı irtifa ve FOV'dan hesaplanır:

```
Footprint = 2 × 16m × tan(45°) = 32 m
Adım      = 32 × (1 − 0.75)   =  8 m  (%75 overlap)
```

Her drone şeridine 65 waypoint düşer, toplam 195 waypoint.  
Waypoint'ler boustrophedon (zigzag) sırasıyla ziyaret edilir.

### Adım 3 — Her Waypoint'te Tespit Pipeline'ı

Drone waypoint'e gelince 5 aşamalı pipeline çalışır:

**Aşama 1 — PID Tabanlı Bounding-Box Stabilizasyonu**
- 12 kare boyunca YOLO çalıştırılır
- Her yangın adayı için PID kontrolcüsü bbox alanını stabilize eder
- MAD yöntemiyle aykırı kareler elenir
- En iyi %40 karenin ağırlıklı ortalaması alınır → fused bbox

**Aşama 2 — Kenar Kirpma Cezası**
- Bbox görüntü kenarına yakınsa koordinat hesabı bozulur
- `edge_clip` değeri hesaplanır (0=tamamen içeride, 1=kırpılmış)
- Kenar yakını tespitler düşük ağırlık alır

**Aşama 3 — HSV Renk Ağırlıklı Alev Centroid**
- Bbox geometrik merkezi yerine gerçek alev merkezini bulur
- Kırmızı/turuncu/sarı pikseller HSV maskesiyle ayrılır
- Parlaklık ağırlıklı centroid hesabı → daha doğru NED koordinatı

**Aşama 4 — Derinlik Kamerası ile Alan ve GPS**
- `DepthPerspective` ile yüzey mesafesi ölçülür
- Gerçek alan (m²) hesaplanır: `alan = bb_genişlik × m_per_px × bb_yükseklik × m_per_px`
- Drone yaw açısı dahil tam rotasyonla GPS lat/lon hesaplanır

**Aşama 5 — Risk Skoru**

```
risk_score = 0.20 × renk_skoru
           + 0.30 × alan_skoru
           + 0.50 × model_güveni
```

### Adım 4 — Merkezi Füzyon Sunucusu

- Her tespit `result_queue`'ya gönderilir
- Minimum skor filtresi: `score < 0.20` olanlar atılır
- **Uzamsal kümeleme:** 8 m yarıçap içindeki tespitler birleştirilir
- **Küme birleştirme:** 12 m içindeki kümeler merge edilir
- **Güven skoru:**
  ```
  confidence = 0.35×max_skor + 0.25×ort_skor + 0.20×sayı_skoru
             + 0.10×drone_skoru + 0.10×yayılım_skoru
  ```
- **Onay kriteri:** `(count ≥ 3 VEYA drone ≥ 2) VE max_score ≥ 0.75`

### Adım 5 — Görselleştirme

- **Canlı:** 200 ms'de bir güncellenen JET ısı haritası (1000×1000 px)
- **Görev sonu:** Küme merkezlerinden OSM tabanlı gerçek harita

---

## Önemli Parametreler

`multi_drone_config.py` içindeki değerleri değiştirerek sistemi ayarlayabilirsin:

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `ALTITUDE_M` | 16.0 m | Tarama irtifası |
| `CAMERA_FOV_DEG` | 90° | Kamera açısı |
| `OVERLAP_RATIO` | 0.75 | Görüntü örtüşme oranı (%75) |
| `CRUISE_SPEED_MPS` | 5.0 m/s | Drone uçuş hızı |
| `HOVER_SEC` | 1.5 s | Waypoint'te bekleme süresi |
| `DETECTION_FRAME_COUNT` | 12 | PID tracking kare sayısı |
| `MIN_SCORE_THRESHOLD` | 0.20 | Minimum kabul edilen skor |
| `CLUSTER_RADIUS_M` | 8.0 m | Kümeleme yarıçapı |
| `CLUSTER_MERGE_RADIUS_M` | 12.0 m | Küme birleştirme yarıçapı |
| `YOLO_CONF` | 0.25 | YOLO güven eşiği |

---

## Kalibrasyon Sistemi

Bilinen koordinatlardaki yangınların tespit hassasiyetini ölçmek için `multi_drone_config.py` içinde kalibrasyon noktaları tanımlanabilir:

```python
CALIBRATION_ENABLED = True
CALIBRATION_FIRE_POINTS_UE = {
    "Yangin1": (2500.0, 2500.0),   # UE koordinatları (cm)
    "Yangin2": (7500.0, 7500.0),
}
```

Her tespit için `all_fires.json` dosyasına şu metrikler otomatik kaydedilir:
- `error_m` — gerçek konuma olan toplam hata (metre)
- `error_dx_m` — X ekseninde sapma
- `error_dy_m` — Y ekseninde sapma

---

## Deneysel Sonuçlar

Bilinen iki yangın noktasıyla yapılan deneyde:

| | Yangın 1 | Yangın 2 |
|---|---|---|
| Gerçek Konum | (25.0, 25.0) m | (75.0, 75.0) m |
| Tespit Merkezi | (25.1, 22.6) m | (74.1, 61.9) m |
| Koordinat Hatası | **~2.4 m** ✅ | ~13 m ⚠️ |
| Güven Skoru | 0.87 | 0.88 |
| Tespit Durumu | Onaylı ✅ | Onaylı ✅ |
| Katkı Yapan Drone | Drone1 + Drone2 | Drone2 + Drone3 |

Yangın 2'deki Y ekseni hatası, drone'un o yangını hep açılı perspektiften görmesinden kaynaklanmaktadır. Overlap oranı artırıldığında bu hata azalmaktadır.

---

## Sık Karşılaşılan Sorunlar

**AirSim bağlanamıyor:**
- Unreal Engine'in açık ve Play modunda olduğundan emin ol
- `settings_3drone.json` dosyasının doğru konumda olduğunu kontrol et

**YOLO modeli bulunamıyor:**
- `modelimiz.pt` dosyasının `final_son_kod/` klasöründe olduğundan emin ol
- `multi_drone_config.py` içindeki `MODEL_PATH` değerini kontrol et

**OSM haritası oluşturulamıyor:**
- İnternet bağlantısı gereklidir (OpenStreetMap tile'ları indirilir)
- `staticmap` paketi kurulu olmalı: `pip install staticmap`

**Drone'lar çakışıyor:**
- Her drone farklı irtifada uçar (16.0 / 16.5 / 17.0 m)
- Bu değerler `ALTITUDE_OFFSETS` ile ayarlanır

---

## Mimari Özeti

```
multi_drone_mission.py
    │
    ├── DroneWorker × 3  (drone_worker.py)
    │     ├── AirSim bağlantısı
    │     ├── Boustrophedon tarama (multi_drone_config.py)
    │     ├── YOLO çıkarımı + analyze_fire_color (HSV renk analizi)
    │     ├── PID tracking + edge clip + HSV centroid
    │     ├── Depth kamera → alan (m²) + GPS koordinatı
    │     └── risk_score → result_queue.put(...)
    │
    └── CentralHeatmapServer  (central_server.py)
          ├── result_queue dinle
          ├── Skor filtresi (≥ 0.20)
          ├── Uzamsal kümeleme (8 m)
          ├── Küme birleştirme (12 m)
          ├── Güven skoru + onay
          ├── Canlı JET haritası (OpenCV)
          └── Görev sonu → OSM haritası (static_heatmap.py)

## Demo Video Linki
-https://drive.google.com/file/d/19ZWTUCpW2XAu3FnGJR5mnLQ_m3DU4vn6/view?usp=sharing
```
