"""
visualize_demo.py — Standart YOLOv8 ve YOLOv8 + SAHI tespitlerini
aynı görüntüde yan yana karşılaştırır. Mülakat sunumu için.

Çıktı: comparison_demo.jpg
  [ Standart YOLO panel | SAHI panel ]
"""

import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path

# ============================================================
# Ayarlar
# ============================================================
MODEL_PATH = "runs/detect/wheat_yolov8n_v1-3/weights/best.pt"
CONFIDENCE = 0.25
SLICE_SIZE = 320

# Renkler (BGR — OpenCV formatı!)
COLOR_STANDARD = (255, 100, 50)    # turuncu-mavi tonlu
COLOR_SAHI = (50, 200, 50)         # canlı yeşil

# ============================================================
# 1) Test görseli seç ve oku
# ============================================================
test_images = sorted(Path("Wheat-Head-1/test/images").glob("*.jpg"))
img_path = str(test_images[0])
print(f"🎯 Görsel: {Path(img_path).name}")

original = cv2.imread(img_path)
h, w = original.shape[:2]
print(f"📐 Boyut: {w}x{h}")

# ============================================================
# 2) İki yöntemle inference
# ============================================================
print("\n🔍 Standart YOLOv8 çalışıyor...")
std_model = YOLO(MODEL_PATH)
std_result = std_model(img_path, conf=CONFIDENCE, device=0, verbose=False)[0]
# .xyxy formatı: [x1, y1, x2, y2] (sol-üst ve sağ-alt köşe pikselleri)
std_boxes = std_result.boxes.xyxy.cpu().numpy()
std_count = len(std_boxes)

print("🔍 SAHI sliced inference çalışıyor...")
sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE,
    device='cuda:0',
)
sahi_result = get_sliced_prediction(
    img_path, sahi_model,
    slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    verbose=0,
)
# SAHI farklı obje formatı kullanıyor, biz [x1,y1,x2,y2] listesine çeviriyoruz
sahi_boxes = [
    [p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy]
    for p in sahi_result.object_prediction_list
]
sahi_count = len(sahi_boxes)

print(f"\n📊 Standart: {std_count} | SAHI: {sahi_count}")

# ============================================================
# 3) Yardımcı fonksiyon — Görüntü üstüne kutular çiz
# ============================================================
def draw_boxes(img, boxes, color, thickness=2):
    """
    img: orijinal görüntü (numpy array)
    boxes: liste [[x1, y1, x2, y2], ...]
    color: BGR tuple
    Geri döner: kutular çizilmiş YENİ görüntü (orijinal değişmez)
    """
    out = img.copy()  # 🔑 orijinali bozma!
    for box in boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out

img_standard = draw_boxes(original, std_boxes, COLOR_STANDARD)
img_sahi = draw_boxes(original, sahi_boxes, COLOR_SAHI)

# ============================================================
# 4) Yardımcı fonksiyon — Başlık şeridi ekle
# ============================================================
def add_title_bar(img, title, count, color):
    """Görüntünün üstüne yöntem ismi ve tespit sayısı içeren şerit ekler."""
    bar_height = 60
    h, w = img.shape[:2]
    # Koyu gri arkaplanlı şerit
    bar = np.full((bar_height, w, 3), 30, dtype=np.uint8)
    # Sol kenarda renk işaretçisi
    cv2.rectangle(bar, (0, 0), (8, bar_height), color, -1)
    # Başlık metni
    cv2.putText(bar, title, (25, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    # Tespit sayısı
    cv2.putText(bar, f"Tespit: {count} bugday basi", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Şerit + görüntü → tek görüntü
    return np.vstack([bar, img])

img_standard_titled = add_title_bar(img_standard, "Standart YOLOv8", 
                                     std_count, COLOR_STANDARD)
img_sahi_titled = add_title_bar(img_sahi, "YOLOv8 + SAHI", 
                                 sahi_count, COLOR_SAHI)

# ============================================================
# 5) İki paneli yan yana birleştir
# ============================================================
# Aralarına 5 piksellik siyah ayraç koyalım, gözü dinlendirir
panel_height = img_standard_titled.shape[0]
separator = np.full((panel_height, 5, 3), 0, dtype=np.uint8)

comparison = np.hstack([img_standard_titled, separator, img_sahi_titled])

# ============================================================
# 6) Kaydet
# ============================================================
output_path = "comparison_demo.jpg"
cv2.imwrite(output_path, comparison)

print(f"\n✅ Demo görseli kaydedildi: {output_path}")
print(f"   Final boyut: {comparison.shape[1]}x{comparison.shape[0]}")
print(f"\n🎯 Bu görsel mülakatta GitHub repo'nun README'sinde gösterilecek.")