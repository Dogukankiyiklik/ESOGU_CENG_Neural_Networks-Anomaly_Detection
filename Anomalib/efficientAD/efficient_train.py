from anomalib.data import Folder # Kendi veri setimi kullandığım için Folder kullandım.
from anomalib.engine import Engine # Eğitim sürecini yöneten ana sınıf
from anomalib.models import EfficientAd # Model sınıfı
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="/content/logs", name="efficientAD_logs")

# Model sınıfını oluşturuyoruz
model = EfficientAd(
    teacher_out_channels=384, # Öğretmen modelin çıkış kanal sayısı (384'ün üstünde hata veriyor. O yüzden 384 olarak bıraktım.)
    model_size="small", # Modelin boyutu (small ve large seçenekleri var)
    lr=1e-4, # Öğrenme oranı
)

# Veri setini yükleyeceğimiz klasörü belirtiyoruz
datamodule = Folder(
    name="wood_folder", # Veri setinin adı
    root="/content/drive/MyDrive/preprocessed_wood_dataset/wood", # Veri setinin konumu
    normal_dir="train/good", # Normal verilerin konumu
    abnormal_dir="test/defect", # Anormal verilerin konumu
    normal_test_dir="test/good", # Normal test verilerinin konumu
    mask_dir="ground_truth/defect", # Maske verilerinin konumu
    train_batch_size=1, # Eğitim batch boyutu
    eval_batch_size=16, # Değerlendirme batch boyutu
    num_workers=4, # Çalışan iş parçacıkları sayısı
)

# Eğitim sürecini yöneten Engine sınıfını oluşturuyoruz
engine = Engine(
    max_epochs=5, # Eğitim süresi
    accelerator="auto", # Hızlandırıcı (GPU, TPU, vs.)
    devices=1, # Kullanılacak cihaz sayısı
    default_root_dir="/content/checkpoints", # Modelin ağırlıklarının kaydedileceği klasör
    logger=logger,
)

# Eğitim sürecini başlatıyoruz
engine.fit(model=model, datamodule=datamodule)