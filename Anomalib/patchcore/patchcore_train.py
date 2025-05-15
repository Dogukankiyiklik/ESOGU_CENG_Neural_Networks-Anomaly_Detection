import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.patchcore import Patchcore

# -------------------------- Model Kurulumu --------------------------

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=9,
    coreset_sampling_ratio=0.3,
)

# -------------------------- Veri Modülü --------------------------

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

# -------------------------- Engine --------------------------

engine = Engine(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    default_root_dir="/content/anomaly_detection/checkpoints",
)

# -------------------------- Eğitim --------------------------

engine.fit(model=model, datamodule=datamodule)