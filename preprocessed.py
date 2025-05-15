import os
import cv2
import numpy as np
from tqdm import tqdm

# =================== Fonksiyonlar ===================

def crop_and_resize(image_path, mask_path=None, save_image_path=None, save_mask_path=None, pad=50):

    """
    Verilen görüntü dosyasını ve varsa maskesini:
    - En büyük konturu temel alarak kırptım,
    - Kırpılan alanı 256x256 boyutuna getirdim ve kaydettim.
    Ayrıca "train/good" verileri için yatay ve dikey flip augmentasyonları da uyguladım.
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"{os.path.basename(image_path)} okunamadı.")
        return False

    # Görüntüyü griye çevirip eşikleme yaparak kontur tespiti
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"{os.path.basename(image_path)} için kontur bulunamadı.")
        return False

    # En büyük konturu al
    biggest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest_contour)

    # Pad değeriyle birlikte kırpma koordinatlarını belirleme
    x1 = max(x + pad, 0)
    y1 = max(y + pad, 0)
    x2 = min(x + w - pad, img.shape[1])
    y2 = min(y + h - pad, img.shape[0])

    # Geçersiz kırpma durumu kontrolü
    if x2 <= x1 or y2 <= y1:
        print(f"{os.path.basename(image_path)} geçersiz kırpma boyutları.")
        return False

    # Görüntüyü kırp ve yeniden boyutlandırma
    cropped_img = img[y1:y2, x1:x2]
    resized_img = cv2.resize(cropped_img, (256, 256))
    cv2.imwrite(save_image_path, resized_img)

    # Sadece "train/good" klasörü için augmentasyon uyguladım
    if "train/good" in save_image_path:
        base, ext = os.path.splitext(save_image_path)

        # Yatay flip (code=1)
        flip_h = cv2.flip(resized_img, 1)
        cv2.imwrite(base + "_flipH" + ext, flip_h)

        # Dikey flip (code=0)
        flip_v = cv2.flip(resized_img, 0)
        cv2.imwrite(base + "_flipV" + ext, flip_v)

    # Maske varsa aynı şekilde işlem yap
    if mask_path and save_mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"{os.path.basename(mask_path)} maskesi okunamadı.")
            return False

        cropped_mask = mask[y1:y2, x1:x2]
        resized_mask = cv2.resize(cropped_mask, (256, 256))
        cv2.imwrite(save_mask_path, resized_mask)

    return True

# =================== Dataset Ayarları ===================

# Orijinal ve işlenmiş dataset yolları
original_base_path = "/content/drive/MyDrive/wood_dataset/wood"
preprocessed_base_path = "/content/drive/MyDrive/preprocessed_wood_dataset/wood"

# Her bir klasör için işlenecek veri yolları ve ayarları
process_list = [
    {
        "img_dir": os.path.join(original_base_path, "train/good"),
        "mask_dir": None,
        "save_img_dir": os.path.join(preprocessed_base_path, "train/good"),
        "save_mask_dir": None,
    },
    {
        "img_dir": os.path.join(original_base_path, "test/good"),
        "mask_dir": None,
        "save_img_dir": os.path.join(preprocessed_base_path, "test/good"),
        "save_mask_dir": None,
    },
    {
        "img_dir": os.path.join(original_base_path, "test/defect"),
        "mask_dir": os.path.join(original_base_path, "ground_truth/defect"),
        "save_img_dir": os.path.join(preprocessed_base_path, "test/defect"),
        "save_mask_dir": os.path.join(preprocessed_base_path, "ground_truth/defect"),
    }
]

# =================== İşlem Akışı ===================

for item in process_list:
    os.makedirs(item["save_img_dir"], exist_ok=True)
    if item.get("save_mask_dir"):
        os.makedirs(item["save_mask_dir"], exist_ok=True)

    # Sadece .jpg uzantılı dosyaları al
    filenames = [f for f in os.listdir(item["img_dir"]) if f.endswith('.jpg')]

    for filename in tqdm(filenames, desc=f"{os.path.basename(item['img_dir'])} kırpılıyor"):
        image_path = os.path.join(item["img_dir"], filename)
        save_image_path = os.path.join(item["save_img_dir"], filename)

        # Eğer maskesi varsa, maskeyi de eşleştirerek işle
        if item.get("mask_dir"):
            mask_filename = filename.replace('.jpg', '_mask.jpg')
            mask_path = os.path.join(item["mask_dir"], mask_filename)

            # _mask kısmını kaldırarak maske dosyasını kaydet
            clean_mask_filename = mask_filename.replace('_mask.jpg', '.jpg')
            save_mask_path = os.path.join(item["save_mask_dir"], clean_mask_filename)

            crop_and_resize(image_path, mask_path, save_image_path, save_mask_path)
        else:
            crop_and_resize(image_path, save_image_path=save_image_path)

print("\nTÜM İŞLEMLER TAMAMLANDI!")