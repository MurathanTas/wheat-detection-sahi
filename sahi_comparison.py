"""
sahi_comparison.py — Aynı görüntü üzerinde:
  1) Standart YOLOv8 inference
  2) SAHI (Sliced Aided Hyper Inference)
yapıp tespit sayısı ve görsel farkını ortaya koyar.

Asıl amaç: SAHI'nin slicing yaklaşımının küçük nesne tespitinde
nasıl katkı sağladığını somut rakamlarla göstermek.
"""

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path

MODEL_PATH = "runs/detect/wheat_yolov8n_v1-3/weights/best.pt"
CONFIDENCE = 0.25  # confidence threshold (her iki yöntem için aynı)

# ============================================================
# 1) Test görseli seç
# ============================================================
test_dir = Path("Wheat-Head-1/test/images")
test_images = sorted(test_dir.glob("*.jpg"))
img_path = str(test_images[0])  # ilk test görseli
print(f"🎯 İnceleniyor: {Path(img_path).name}\n")

# ============================================================
# 2) STANDART YOLO INFERENCE (baseline)
# ============================================================
print("=" * 60)
print("YÖNTEM 1: Standart YOLOv8 Inference")
print("=" * 60)

model = YOLO(MODEL_PATH)
std_result = model(img_path, conf=CONFIDENCE, device=0, verbose=False)
std_count = len(std_result[0].boxes)
std_result[0].save(filename="standard_result.jpg")

print(f"📊 Tespit sayısı: {std_count}")
print(f"💾 Görsel: standard_result.jpg")

# ============================================================
# 3) SAHI SLICED INFERENCE
# ============================================================
print("\n" + "=" * 60)
print("YÖNTEM 2: YOLOv8 + SAHI (Sliced Inference)")
print("=" * 60)

# SAHI kendi model wrapper'ı — Ultralytics modelini import ediyor
# AutoDetectionModel = SAHI'nin "agnostic" arayüzü (YOLO/Detectron/MMDet hepsini destekler)
sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',           # YOLOv8 için bu
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device='cuda:0',
)

# Sliced inference parametreleri:
# - slice_height/width: her dilimin boyutu (orijinal görüntüden kesilecek)
# - overlap_ratio: dilimler arası %20 örtüşme (kenar nesneleri kaçırmamak için)
sahi_result = get_sliced_prediction(
    img_path,
    sahi_model,
    slice_height=320,        # küçük slice = daha çok dilim = daha hassas
    slice_width=320,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

sahi_count = len(sahi_result.object_prediction_list)

# SAHI'nin kendi görselleştirme metodu — annotated görseli kaydeder
sahi_result.export_visuals(
    export_dir="./",
    file_name="sahi_result",
)

print(f"📊 Tespit sayısı: {sahi_count}")
print(f"💾 Görsel: sahi_result.png")

# ============================================================
# 4) KARŞILAŞTIRMA
# ============================================================
print("\n" + "=" * 60)
print("KARŞILAŞTIRMA SONUÇLARI")
print("=" * 60)
print(f"Standart YOLOv8:     {std_count} tespit")
print(f"YOLOv8 + SAHI:       {sahi_count} tespit")

fark = sahi_count - std_count
if fark > 0:
    yuzde = fark / std_count * 100
    print(f"📈 SAHI ekstra tespit: +{fark} ({yuzde:.1f}% artış)")
elif fark == 0:
    print(f"➖ Aynı sonuç (640x640'da slicing'in faydası sınırlı)")
else:
    print(f"📉 SAHI daha az: {fark}")

print("\n✅ İki görseli karşılaştır: standard_result.jpg vs sahi_result.png")