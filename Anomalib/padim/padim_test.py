import os
import numpy as np
import torch
import cv2
from sklearn.metrics import jaccard_score
from torchvision.io import read_image

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.padim import Padim

# -------------------------- IoU Fonksiyonu --------------------------

def compute_iou(pred_mask, gt_mask):
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()

    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        return 1.0
    elif np.sum(gt_flat) == 0 or np.sum(pred_flat) == 0:
        return 0.0
    else:
        return jaccard_score(gt_flat, pred_flat)

# -------------------------- Model ve Veri Yükleme --------------------------

model = Padim.load_from_checkpoint(
    "/content/checkpoints/Padim/wood/latest/weights/lightning/model.ckpt"
)

datamodule = Folder(
    name="wood",
    root="/content/drive/MyDrive/preprocessed_wood_dataset/wood",
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    mask_dir="ground_truth/defect",
    train_batch_size=1,
    eval_batch_size=16,
    num_workers=4,
)

engine = Engine()

# -------------------------- Test ve Tahmin --------------------------

engine.test(model=model, datamodule=datamodule)
results = engine.predict(model=model, datamodule=datamodule)

# -------------------------- IoU Hesaplama --------------------------

threshold = 0.3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
ious = []

for result in results:
    pred_mask = result.anomaly_map[0].detach().cpu().numpy()
    filename = os.path.basename(result.image_path[0])
    gt_path = f"/content/drive/MyDrive/preprocessed_wood_dataset/wood/ground_truth/defect/{filename}"

    if not os.path.exists(gt_path):
        continue

    gt_mask = read_image(gt_path)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]
    gt_mask = (gt_mask.numpy() > 0).astype(np.uint8)

    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    post_mask = (dilated_mask > 0).astype(np.uint8)

    iou = compute_iou(post_mask, gt_mask)
    ious.append(iou)

print(f"Threshold = {threshold:.2f} → Ortalama IoU: {np.mean(ious):.4f}")
