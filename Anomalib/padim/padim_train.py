import torch
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.padim import Padim

# -------------------------- Model Kurulumu --------------------------

model = Padim(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    n_features=512,
    pre_trained=True,
    post_processor=True,
    evaluator=True,
    visualizer=True,
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

engine = Engine(
    max_epochs=1,
    accelerator="auto",
    devices=1,
    default_root_dir="/content/checkpoints",
)

# -------------------------- Eğitim --------------------------

engine.fit(model=model, datamodule=datamodule)