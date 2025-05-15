import os
import numpy as np
import torch
import cv2
from sklearn.metrics import jaccard_score
from torchvision.io import read_image

from anomalib.data import Folder
from anomalib.models import EfficientAd
from anomalib.engine import Engine

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

# -------------- IoU Hesapla ----------------------

def evaluate_iou_simple(results, gt_dir, threshold=0.30):
    ious = []
    for result in results:
        anomaly_map = result.anomaly_map[0].detach().cpu().numpy()
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        binary = (anomaly_map > threshold).astype(np.uint8)

        filename = os.path.basename(result.image_path[0])
        gt_path = os.path.join(gt_dir, filename)
        if not os.path.exists(gt_path):
            print(f"Uyarı: GT bulunamadı → {filename}")
            continue

        gt_mask = read_image(gt_path)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[0]
        gt_mask = (gt_mask.numpy() > 0).astype(np.uint8)

        iou = compute_iou(binary, gt_mask)
        ious.append(iou)

    print(f"\n== IoU Sonucu (Threshold={threshold}) ==")
    print(f"Ortalama IoU: {np.mean(ious):.4f}")

# ------------------- TEST ----------------------

# Modeli checkpoint'ten yükle
model = EfficientAd.load_from_checkpoint("/content/anomaly_detection/checkpoints/EfficientAd/wood/latest/weights/lightning/model.ckpt")

# Veri setini yükle
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

# Test sürecini yöneten Engine sınıfını oluştur
engine = Engine(default_root_dir="/content/anomaly_detection/results")

# Test ve tahmin sürecini başlat    
engine.test(model=model, datamodule=datamodule)
results = engine.predict(model=model, datamodule=datamodule)

# IoU hesapla
evaluate_iou_simple(
    results,
    gt_dir="/content/anomaly_detection/preprocessed_wood_dataset/wood/ground_truth/defect/defect",
    threshold=0.30
)