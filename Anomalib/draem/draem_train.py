from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.draem import Draem

# --------------------- Model Tanımı ---------------------
model = Draem(
    enable_sspcab=False,           # SSPCAB eğitimi varsayılan olarak kapalı
    sspcab_lambda=0.1,             # SSPCAB loss ağırlığı
    anomaly_source_path=None,      # Rastgele noise kullan
    beta=(0.1, 1.0),               # Anomali karıştırma oranı
    pre_processor=True,
    post_processor=True,
    evaluator=True,
    visualizer=True,
)

# --------------------- Veri Seti Tanımı ---------------------
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

# --------------------- Eğitim Motoru ---------------------
engine = Engine(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    default_root_dir="/content/checkpoints",
)

# --------------------- Eğitimi Başlat ---------------------
engine.fit(model=model, datamodule=datamodule)
