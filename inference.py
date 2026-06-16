"""
inference.py — Eğitilmiş YOLOv8 modelimizle bir test görseli üzerinde
buğday başı sayımı yapar. Sonucu hem konsola yazdırır, hem de annotated
görseli diske kaydeder.
"""

from ultralytics import YOLO
from pathlib import Path

# ============================================================
# 1) Eğitilmiş modeli yükle
# ============================================================
# "best.pt" = eğitim boyunca en yüksek val mAP'i veren ağırlıklar
# Ultralytics her epoch sonunda val'i kontrol eder, en iyisini saklar
MODEL_PATH = "runs/detect/wheat_yolov8n_v1-3/weights/best.pt"
model = YOLO(MODEL_PATH)

print(f"✅ Model yüklendi: {MODEL_PATH}")
print(f"   Sınıflar: {model.names}")  # {0: 'wheat'}

# ============================================================
# 2) Test görsellerinden birini seç
# ============================================================
# Modelin daha önce GÖRMEDİĞİ test set'inden bir foto seçelim
# (train ve val'i eğitim sürecinde görmüştü, test ise el değmemiş)
test_dir = Path("Wheat-Head-1/test/images")
test_images = sorted(test_dir.glob("*.jpg"))

print(f"\n📁 Test set: {len(test_images)} görsel mevcut")

# İlk test görselini seçelim
test_img_path = str(test_images[0])
print(f"🎯 İnceleniyor: {test_img_path}")

# ============================================================
# 3) Inference (tahmin) yap
# ============================================================
# device=0 → GPU 0'ı kullan
# conf=0.25 → confidence threshold (0.25'in altındaki tahminleri at)
results = model(test_img_path, conf=0.25, device=0)
result = results[0]  # tek görüntü → results listesinde tek eleman

# ============================================================
# 4) Sonuçları analiz et
# ============================================================
detected_count = len(result.boxes)
print(f"\n📊 Tespit edilen buğday başı sayısı: {detected_count}")

# İlk 5 tespitin detayını göster
print(f"\n🔍 İlk 5 tespit:")
for i, box in enumerate(result.boxes[:5]):
    conf = float(box.conf[0])         # güven skoru (0-1)
    coords = box.xyxy[0].tolist()     # [x1, y1, x2, y2] piksel cinsinden
    print(f"   #{i+1}: güven={conf:.3f}, kutu={[round(c) for c in coords]}")

# ============================================================
# 5) Annotated görseli kaydet
# ============================================================
output_path = "inference_result.jpg"
result.save(filename=output_path)
print(f"\n✅ Görsel kaydedildi: {output_path}")
print(f"   → VS Code'dan açıp inceleyebilirsin")