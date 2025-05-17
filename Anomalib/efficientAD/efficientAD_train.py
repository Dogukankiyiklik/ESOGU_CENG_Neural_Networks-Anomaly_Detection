from anomalib.data import Folder # Kendi veri setimi kullandığım için Folder kullandım.
from anomalib.engine import Engine # Eğitim sürecini yöneten ana sınıf
from anomalib.models import EfficientAd # Model sınıfı
from pytorch_lightning.loggers import TensorBoardLogger

# Model sınıfını oluşturuyoruz
model = EfficientAd(
    teacher_out_channels=384, 
    model_size="small",
    lr=1e-4,
)

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

# Eğitim sürecini yöneten Engine sınıfını oluşturuyoruz
engine = Engine(
    max_epochs=35,
    accelerator="auto",
    devices=1,
    default_root_dir="/content/anomaly_detection/checkpoints",
)

# Eğitim sürecini başlatıyoruz
engine.fit(model=model, datamodule=datamodule)
