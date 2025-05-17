from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.reverse_distillation import ReverseDistillation

# --------------------- Model Tanımı ---------------------
model = ReverseDistillation(
    backbone="wide_resnet50_2",
    layers=["layer1", "layer2", "layer3"],
    anomaly_map_mode="multiply",
    pre_trained=True,
    pre_processor=True
)

# --------------------- Veri Seti ---------------------
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

# --------------------- Eğitim ---------------------
engine = Engine(
    max_epochs=35,
    accelerator="auto",
    devices=1,
    default_root_dir="/content/anomaly_detection/checkpoints",
)

engine.fit(model=model, datamodule=datamodule)
