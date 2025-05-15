from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image.draem import Draem

# ------------------ Eğitilmiş Modeli Yükle ------------------
model = Draem.load_from_checkpoint(
    "/content/checkpoints/Draem/wood/latest/weights/lightning/model.ckpt"
)

# ------------------ Veri Setini Hazırla ------------------
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

# ------------------ Test ve Tahmin ------------------
engine = Engine()
engine.test(model=model, datamodule=datamodule)
results = engine.predict(model=model, datamodule=datamodule)

# ------------------ (Opsiyonel) IoU ------------------
# evaluate_iou_simple(results, "/path/to/ground_truth", threshold=0.30)
