"""
batch_evaluation.py — Standart YOLOv8 vs SAHI'yi 50 test görseli üzerinde
karşılaştırır, ground truth ile hata oranını ölçer.

Çıktı:
  - Görüntü başına ortalama tespit sayısı
  - Mean Absolute Error (MAE) — gerçeğe yakınlık
  - Inference süresi karşılaştırması
"""

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
import time

# ============================================================
# Ayarlar
# ============================================================
MODEL_PATH = "runs/detect/wheat_yolov8n_v1-3/weights/best.pt"
CONFIDENCE = 0.25
NUM_IMAGES = 50  # ilk 50 test görseli (tüm test 1360, 50 yeterli istatistik)
SLICE_SIZE = 320

# ============================================================
# Modelleri yükle
# ============================================================
print("🔄 Modeller yükleniyor...")
std_model = YOLO(MODEL_PATH)

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device='cuda:0',
)
print("✅ Modeller hazır\n")

# ============================================================
# Test verisini hazırla
# ============================================================
test_img_dir = Path("Wheat-Head-1/test/images")
test_label_dir = Path("Wheat-Head-1/test/labels")
test_images = sorted(test_img_dir.glob("*.jpg"))[:NUM_IMAGES]

print(f"📁 {len(test_images)} görsel işlenecek...\n")

# ============================================================
# Sonuçları topla
# ============================================================
results = []
std_time_total = 0
sahi_time_total = 0

for i, img_path in enumerate(test_images, 1):
    # --- Ground truth: etiket dosyasındaki satır sayısı = gerçek başak sayısı ---
    label_path = test_label_dir / (img_path.stem + ".txt")
    if label_path.exists():
        with open(label_path) as f:
            gt_count = sum(1 for line in f if line.strip())
    else:
        gt_count = 0
    
    # --- Standart YOLO ---
    t0 = time.time()
    std_res = std_model(str(img_path), conf=CONFIDENCE, device=0, verbose=False)
    std_time = time.time() - t0
    std_count = len(std_res[0].boxes)
    std_time_total += std_time
    
    # --- SAHI Sliced ---
    t0 = time.time()
    sahi_res = get_sliced_prediction(
        str(img_path), sahi_model,
        slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )
    sahi_time = time.time() - t0
    sahi_count = len(sahi_res.object_prediction_list)
    sahi_time_total += sahi_time
    
    results.append((gt_count, std_count, sahi_count))
    
    if i % 10 == 0:
        print(f"  İşlendi: {i}/{len(test_images)}")

# ============================================================
# İstatistikleri hesapla
# ============================================================
gt_total = sum(r[0] for r in results)
std_total = sum(r[1] for r in results)
sahi_total = sum(r[2] for r in results)
n = len(results)

# MAE = ortalama mutlak hata = |tahmin - gerçek|'in ortalaması
std_mae = sum(abs(r[1] - r[0]) for r in results) / n
sahi_mae = sum(abs(r[2] - r[0]) for r in results) / n

# RMSE de hesaplayalım (büyük hatalara daha duyarlı)
std_rmse = (sum((r[1] - r[0])**2 for r in results) / n) ** 0.5
sahi_rmse = (sum((r[2] - r[0])**2 for r in results) / n) ** 0.5

# ============================================================
# Raporu yazdır
# ============================================================
print("\n" + "=" * 70)
print(f"AGGREGATE SONUÇLAR ({n} test görseli)")
print("=" * 70)
print(f"\n📊 TOPLAM SAYIM")
print(f"  Ground Truth:        {gt_total:>6,} buğday başı")
print(f"  Standart YOLOv8:     {std_total:>6,}  ({(std_total/gt_total-1)*100:+.1f}% sapma)")
print(f"  YOLOv8 + SAHI:       {sahi_total:>6,}  ({(sahi_total/gt_total-1)*100:+.1f}% sapma)")

print(f"\n🎯 GÖRÜNTÜ BAŞINA ORTALAMA")
print(f"  Ground Truth:        {gt_total/n:.1f}")
print(f"  Standart YOLOv8:     {std_total/n:.1f}")
print(f"  YOLOv8 + SAHI:       {sahi_total/n:.1f}")

print(f"\n📈 DOĞRULUK (Mean Absolute Error — düşük = iyi)")
print(f"  Standart YOLOv8 MAE: {std_mae:.2f} başak/görüntü")
print(f"  YOLOv8 + SAHI MAE:   {sahi_mae:.2f} başak/görüntü")

if sahi_mae < std_mae:
    iyilesme = (std_mae - sahi_mae) / std_mae * 100
    print(f"  → SAHI {iyilesme:.1f}% daha doğru sayım yapıyor ✅")
elif sahi_mae > std_mae:
    print(f"  → Bu image set'te standart YOLO daha doğru")
else:
    print(f"  → Aynı doğruluk")

print(f"\n⏱️  HIZ")
print(f"  Standart YOLOv8: {std_time_total*1000/n:.1f} ms/görüntü")
print(f"  YOLOv8 + SAHI:   {sahi_time_total*1000/n:.1f} ms/görüntü")
print(f"  → SAHI {sahi_time_total/std_time_total:.1f}x daha yavaş "
      f"(slice sayısı kadar fazla inference yapıyor)")

print("\n" + "=" * 70)
print("YORUM: SAHI doğruluk için trade-off, hız feda eder.")
print("Üretim drone görüntüleri (2K+) için bu trade-off değer ifade eder.")
print("=" * 70)