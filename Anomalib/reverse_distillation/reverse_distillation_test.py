import os
import numpy as np
import torch
import cv2
from sklearn.metrics import jaccard_score
from torchvision.io import read_image

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.reverse_distillation import ReverseDistillation

# ---------------- IoU Fonksiyonu ----------------

def compute_iou(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        return 1.0
    elif np.sum(gt_flat) == 0 or np.sum(pred_flat) == 0:
        return 0.0
    else:
        return jaccard_score(gt_flat, pred_flat)

# ---------------- IoU Değerlendirme ----------------

def evaluate_iou_adaptive(results, gt_dir, percentile=93, min_area=10):
    ious = []
    skipped = 0

    for result in results:
        # Sadece 'defect' klasöründen gelen görselleri dahil et
        if "defect" not in result.image_path[0]:
            continue

        anomaly_map = result.anomaly_map[0].detach().cpu().numpy()
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_map = cv2.GaussianBlur(anomaly_map, (3, 3), 0)

        threshold = np.percentile(anomaly_map, percentile)
        binary = (anomaly_map > threshold).astype(np.uint8)

        # Maske genişletme
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)

        # Küçük alanları temizle
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = 0

        # GT maskesini al
        filename = os.path.basename(result.image_path[0])
        gt_path = os.path.join(gt_dir, filename)
        if not os.path.exists(gt_path):
            print(f"Uyarı: GT bulunamadı → {filename}")
            skipped += 1
            continue

        gt_mask = read_image(gt_path)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[0]
        gt_mask = (gt_mask.numpy() > 0).astype(np.uint8)

        iou = compute_iou(binary, gt_mask)
        ious.append(iou)

    if len(ious) == 0:
        mean_iou = 0.0
        print("Hiç IoU hesaplanamadı (GT maskesi eksik olabilir).")
    else:
        mean_iou = np.mean(ious)

    print(f"\n== IoU Sonucu (Percentile={percentile}, Min Area={min_area}) ==")
    print(f"Ortalama IoU: {mean_iou:.4f} (Toplam: {len(ious)} örnek)")
    if skipped > 0:
        print(f"{skipped} görselde GT maskesi bulunamadı.")

# ---------------- Modeli Yükle ----------------

model = ReverseDistillation.load_from_checkpoint(
    "/content/anomaly_detection/checkpoints/ReverseDistillation/wood/latest/weights/lightning/model.ckpt"
)

# ---------------- Veri Seti ----------------

datamodule = Folder(
    name="wood",
    root="/content/anomaly_detection/preprocessed_wood_dataset/wood",
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    mask_dir="ground_truth/defect",
    train_batch_size=1,
    eval_batch_size=16,
    num_workers=4,
)

# ---------------- Test Süreci ----------------

engine = Engine(default_root_dir="/content/anomaly_detection/results")
engine.test(model=model, datamodule=datamodule)
results = engine.predict(model=model, datamodule=datamodule)

# ---------------- IoU Hesapla ----------------

evaluate_iou_adaptive(
    results,
    gt_dir="/content/anomaly_detection/preprocessed_wood_dataset/wood/ground_truth/defect",
    percentile=94,
    min_area=8
)
