# Neural Network Ahşap Yüzeylerde Anomali Tespiti ve Segmentasyonu Projesi

Bu proje, ahşap yüzey verileri üzerinde denetimsiz öğrenme yöntemleri kullanılarak anomali tespiti ve segmentasyonu gerçekleştirmek amacıyla geliştirilmiştir. Amaç, kusurlu bölgelerin maske (segmentation mask) ve anomali skorları aracılığıyla yakalanmasıdır.

Projeyi çalıştırmak için öncelikle `Kullanım_Kılavuzu.ipynb` dosyası Google Colab üzerinde açılmalı ve hücreler adım adım çalıştırılmalıdır. Preprocess işlemi tekrardan yapılacak ise Kılavuzda belirtilen dosya dizini ayarlarına dikkat edilerek işlem yapılmalıdır. Eğitim ve test süreçleri bu notebook üzerinden yürütülmektedir.

Projenin ayrıca bir kullanıcı arayüzü bulunmaktadır. `gradio_arayuz.py` dosyası çalıştırıldığında Gradio otomatik olarak bir bağlantı üretir. Bu bağlantı üzerinden tarayıcıda arayüz açılarak modeller eğitilebilir ve test edilebilir, test görsellerinin maske çıktıları incelenebilir, farklı modellerin metriklere göre karşılaştırılması sağlanabilir ve eğitim kaybı (loss) grafikleri görüntülenebilir.

Kullanılan veri seti MVTecAD veri setinin yalnızca Wood alt kümesidir. Veri seti yapısı `train/good`, `test/good`, `test/defect` ve `ground_truth/defect` klasörlerinden oluşmaktadır.
