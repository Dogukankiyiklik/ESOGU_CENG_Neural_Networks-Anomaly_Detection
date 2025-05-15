import gradio as gr
import os
import subprocess
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
from glob import glob

# =================== MODEL KARŞILAŞTIRMA İÇİN BELLEK ===================
comparison_data = {}

# =================== EĞİTİM / TEST + GÖRSEL VE METRİK ÇEKME ===================
def run_and_parse(model_name, operation):
    if not model_name or not operation:
        return "❗️ Lütfen önce model veya işlem türünü seçin.", pd.DataFrame(), []

    if model_name == "EfficientAD":
        script = "/content/anomaly_detection/Anomalib/efficientAD/efficientAD_train.py" if operation == "Train" else "/content/anomaly_detection/Anomalib/efficientAD/efficientAD_test.py"
        checkpoint_path = "/content/anomaly_detection/checkpoints/EfficientAd/wood/latest/weights/lightning/model.ckpt"
        image_dir = "/content/anomaly_detection/results/EfficientAd/wood/latest/images/test"
    elif model_name == "PaDiM":
        script = "/content/anomaly_detection/Anomalib/padim/padim_train.py" if operation == "Train" else "/content/anomaly_detection/Anomalib/padim/padim_test.py"
        checkpoint_path = "/content/anomaly_detection/checkpoints/Padim/wood/latest/weights/lightning/model.ckpt"
        image_dir = "/content/anomaly_detection/results/Padim/wood/latest/images/test"
    elif model_name == "PatchCore":
        script = "/content/anomaly_detection/Anomalib/patchcore/patchcore_train.py" if operation == "Train" else "/content/anomaly_detection/Anomalib/patchcore/patchcore_test.py"
        checkpoint_path = "/content/anomaly_detection/checkpoints/Patchcore/wood/latest/weights/lightning/model.ckpt"
        image_dir = "/content/anomaly_detection/results/Patchcore/wood/latest/images/test"
    elif model_name == "STFPM":
        script = "/content/anomaly_detection/Anomalib/stfpm/stfpm_train.py" if operation == "Train" else "/content/anomaly_detection/Anomalib/stfpm/stfpm_test.py"
        checkpoint_path = "/content/anomaly_detection/checkpoints/Stfpm/wood/latest/weights/lightning/model.ckpt"
        image_dir = "/content/anomaly_detection/results/Stfpm/wood/latest/images/test"
    elif model_name == "DRAEM":
        script = "/content/anomaly_detection/Anomalib/draem/draem_train.py" if operation == "Train" else "/content/anomaly_detection/Anomalib/draem/draem_test.py"
        checkpoint_path = "/content/anomaly_detection/checkpoints/Draem/wood/latest/weights/lightning/model.ckpt"
        image_dir = "/content/anomaly_detection/results/Draem/wood/latest/images/test"
    elif model_name == "REVERSE DISTILLATION":
        script = "/content/anomaly_detection/Anomalib/reverse_distillation/reverse_distillation_train.py" if operation == "Train" else "/content/anomaly_detection/Anomalib/reverse_distillation/reverse_distillation_test.py"
        checkpoint_path = "/content/anomaly_detection/checkpoints/ReverseDistillation/wood/latest/weights/lightning/model.ckpt"
        image_dir = "/content/anomaly_detection/results/ReverseDistillation/wood/latest/images/test"
    else:
        return "❗️ Model seçimi hatalı.", pd.DataFrame(), []

    if operation == "Test" and not os.path.exists(checkpoint_path):
        return f"⚠️ {model_name} için model eğitilmemiş. Lütfen önce eğitim yapın.", pd.DataFrame(), []

    log_file_path = f"/content/anomaly_detection/logs/{model_name}_{operation.lower()}_log.txt"
    os.makedirs("/content/anomaly_detection/logs", exist_ok=True)

    if operation == "Train":
        with open(log_file_path, "w") as log_file:
            subprocess.run(["python", script], stdout=log_file, stderr=subprocess.STDOUT, text=True)
        output = f"✅ {model_name} için eğitim tamamlandı. Log dosyası kaydedildi."
    else:
        with open(log_file_path, "w") as log_file:
            subprocess.run(["python", script], stdout=log_file, stderr=subprocess.STDOUT, text=True)
        output = f"✅ {model_name} için Test işlemi tamamlandı."

    metrics = {}
    patterns = {
        "image_AUROC": r"image_AUROC\s*\│\s*([\d.]+)",
        "image_F1Score": r"image_F1Score\s*\│\s*([\d.]+)",
        "pixel_AUROC": r"pixel_AUROC\s*\│\s*([\d.]+)",
        "pixel_F1Score": r"pixel_F1Score\s*\│\s*([\d.]+)",
        "IoU": r"Ortalama IoU:\s*([\d.]+)"
    }

    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as f:
            log_contents = f.read()

        for key, pattern in patterns.items():
            match = re.search(pattern, log_contents)
            metrics[key] = float(match.group(1)) if match else "Yok"
    else:
        metrics = {key: "Yok" for key in patterns}

    df = pd.DataFrame(metrics.items(), columns=["📈 Metrik", "🎯 Değer"])
    if operation == "Test":
        comparison_data[model_name] = metrics.copy()

    # Görseller
    image_paths = []
    if operation == "Test":
        for sub in ["defect", "good"]:
            sub_path = os.path.join(image_dir, sub)
            if os.path.isdir(sub_path):
                image_paths += glob(f"{sub_path}/*.png") + glob(f"{sub_path}/*.jpg")
        image_paths = sorted(image_paths)
        if len(image_paths) > 15:
            image_paths = random.sample(image_paths, 15)

    return output, df, image_paths

# =================== LOSS GRAFİĞİ ÇİZME ===================
def draw_loss_plot(model_name):
    log_file_path = f"/content/anomaly_detection/logs/{model_name}_train_log.txt"
    if not os.path.exists(log_file_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"❗️ {model_name} log dosyası bulunamadı.", ha='center', va='center')
        ax.axis("off")
        return fig

    epochs = []
    losses = []
    last_loss_per_epoch = {}

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = re.search(r"Epoch (\d+):.*?train_loss_epoch=([\d.]+)", line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    last_loss_per_epoch[epoch] = loss

        if not last_loss_per_epoch:
            raise ValueError("Log dosyasında geçerli train_loss_epoch verisi yok.")

        for epoch in sorted(last_loss_per_epoch.keys()):
            epochs.append(epoch)
            losses.append(last_loss_per_epoch[epoch])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, losses, marker='o', color='blue', label='Train Loss')
        ax.set_title(f"{model_name} - Eğitim Loss Grafiği")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"⚠️ Grafik oluşturulamadı: {str(e)}", ha='center', va='center')
        ax.axis("off")
        return fig

# =================== MODEL KARŞILAŞTIRMA ===================
def compare_models():
    if not comparison_data:
        return pd.DataFrame([["-", "-", "-", "-", "-", "-"]],
                            columns=["Model", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "IoU"])

    df = pd.DataFrame.from_dict(comparison_data, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Model"}, inplace=True)
    return df[["Model", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "IoU"]]

# =================== CSS ===================
custom_css = """
body {
    font-family: 'Segoe UI', Roboto, sans-serif;
    background-color: #f9fafb;
}
.gr-box {
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}
.gr-button {
    font-size: 16px !important;
    padding: 12px 24px !important;
    background-color: #0d9488 !important;
    color: white !important;
}
"""

# =================== GRADIO ARAYÜZÜ ===================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("""
    <h1 style="text-align: center;">🪵 Ahşap Yüzey Anomali Tespiti Paneli</h1>
    <p style="text-align: center; max-width: 800px; margin: auto;">
    Eğitim, test, segmentasyon görselleştirme ve model karşılaştırması yapabilir; eğitim loglarından loss grafiğini çizdirebilirsiniz.
    </p>
    <hr>
    """)

    with gr.Tabs():
        with gr.Tab("⚙️ Eğitim / Test Paneli"):
            with gr.Row():
                model = gr.Dropdown(["EfficientAD", "PaDiM", "PatchCore", "STFPM", "DRAEM", "REVERSE DISTILLATION"], label="🔍 Model Seçimi", interactive=True)
                operation = gr.Radio(["Train", "Test"], label="⚙️ İşlem Türü", interactive=True)

            run_button = gr.Button("🚀 Başlat", size="lg")
            output_text = gr.Textbox(label="📤 Durum Mesajı", lines=2)

        with gr.Tab("📊 Test Sonuçları"):
            table = gr.Dataframe(headers=["📈 Metrik", "🎯 Değer"], label="Skor Tablosu", wrap=True)

        with gr.Tab("🖼️ Segmentasyon Maskeleri"):
            image_gallery = gr.Gallery(label="🖼️ Test Görselleri", columns=[4], object_fit="contain", height="auto", show_label=False)

        with gr.Tab("📈 Model Karşılaştırması"):
            comparison_table = gr.Dataframe(headers=["Model", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "IoU"],
                                            label="📈 Model Skorları (Karşılaştırmalı)",
                                            wrap=True, interactive=False)
            compare_button = gr.Button("📊 Karşılaştırmayı Göster")
            compare_button.click(fn=compare_models, outputs=comparison_table)

        with gr.Tab("📉 Eğitim Kaybı (Loss) Grafiği"):
            selected_model_for_plot = gr.Dropdown(["EfficientAD", "PaDiM", "PatchCore", "STFPM", "DRAEM", "REVERSE DISTILLATION"], label="Model Seçin")
            plot_button = gr.Button("📈 Loss Grafiğini Göster")
            loss_plot_output = gr.Plot()
            plot_button.click(fn=draw_loss_plot, inputs=selected_model_for_plot, outputs=loss_plot_output)

    run_button.click(fn=run_and_parse, inputs=[model, operation], outputs=[output_text, table, image_gallery])

demo.launch(share=True)
