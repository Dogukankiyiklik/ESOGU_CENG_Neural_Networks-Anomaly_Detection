from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.reverse_distillation import ReverseDistillation

# --------------------- Model Tanımı ---------------------
model = ReverseDistillation(
    backbone="wide_resnet50_2",                # CNN omurgası
    layers=["layer1", "layer2", "layer3"],               # Özellik çıkarılacak katmanlar
    anomaly_map_mode="add",                    # Anomali haritası oluşturma modu
    pre_trained=True,                          # Hazır ağrılıkları kullan
    pre_processor=True                         # Girişler için ön işleme
)

# --------------------- Veri Seti ---------------------
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

# --------------------- Eğitim ---------------------
engine = Engine(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    default_root_dir="/content/checkpoints",
)

engine.fit(model=model, datamodule=datamodule)
