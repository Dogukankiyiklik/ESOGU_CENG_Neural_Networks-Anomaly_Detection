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

    base_path = "/content/anomaly_detection"

    script_map = {
        "EfficientAD": "efficientAD",
        "PaDiM": "padim",
        "PatchCore": "patchcore",
        "STFPM": "stfpm",
        "DRAEM": "draem",
        "REVERSE DISTILLATION": "reverse_distillation"
    }

    model_folder = script_map.get(model_name)
    if not model_folder:
        return "❗️ Model seçimi hatalı.", pd.DataFrame(), []

    script = f"{base_path}/Anomalib/{model_folder}/{model_folder}_{operation.lower()}.py"
    checkpoint_path = f"{base_path}/checkpoints/{model_name.replace(' ', '')}/wood/latest/weights/lightning/model.ckpt"
    image_dir = f"{base_path}/results/{model_name.replace(' ', '')}/wood/latest/images/test"

    if operation == "Test" and not os.path.exists(checkpoint_path):
        return f"⚠️ {model_name} için model eğitilmemiş. Lütfen önce eğitim yapın.", pd.DataFrame(), []

    os.makedirs(f"{base_path}/logs", exist_ok=True)
    log_file_path = f"{base_path}/logs/{model_name}_{operation.lower()}.txt"

    with open(log_file_path, "w") as log_file:
        subprocess.run(["python", script], stdout=log_file, stderr=subprocess.STDOUT, text=True)

    output = f"✅ {model_name} için {operation} işlemi tamamlandı."

    metrics = {}
    patterns = {
        "image_AUROC": r"image_AUROC\s*\|​\s*([\d.]+)",
        "image_F1Score": r"image_F1Score\s*\|​\s*([\d.]+)",
        "pixel_AUROC": r"pixel_AUROC\s*\|​\s*([\d.]+)",
        "pixel_F1Score": r"pixel_F1Score\s*\|​\s*([\d.]+)",
        "IoU": r"Ortalama IoU:\s*([\d.]+)"
    }

    if operation == "Test" and os.path.exists(log_file_path):
        with open(log_file_path, "r") as f:
            content = f.read()
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                metrics[key] = float(match.group(1)) if match else "Yok"
    else:
        metrics = {key: "-" for key in patterns}

    df = pd.DataFrame(metrics.items(), columns=["📈 Metrik", "🎯 Değer"])

    if operation == "Test":
        comparison_data[model_name] = metrics.copy()

    image_paths = []
    if operation == "Test" and os.path.exists(image_dir):
        for sub in ["defect", "good"]:
            sub_path = os.path.join(image_dir, sub)
            if os.path.isdir(sub_path):
                image_paths += glob(f"{sub_path}/*.png") + glob(f"{sub_path}/*.jpg")
        image_paths = sorted(image_paths)
        image_paths = random.sample(image_paths, min(len(image_paths), 15))

    return output, df, image_paths

# =================== LOSS GRAFİĞİ ÇİZME ===================
def draw_loss_plot(model_name):
    log_file_path = f"/content/anomaly_detection/logs/{model_name}_train.txt"
    if not os.path.exists(log_file_path):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"❗️ {model_name} eğitim log'u yok.", ha='center', va='center')
        ax.axis("off")
        return fig

    epochs, losses, last_loss_per_epoch = [], [], {}
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                match = re.search(r"Epoch (\d+):.*?train_loss_epoch=([\d.]+)", line)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    last_loss_per_epoch[epoch] = loss

        if not last_loss_per_epoch:
            raise ValueError("Eğitim log'unda loss verisi yok.")

        for epoch in sorted(last_loss_per_epoch):
            epochs.append(epoch)
            losses.append(last_loss_per_epoch[epoch])

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, losses, marker='o', color='blue', label='Train Loss')
        ax.set_title(f"{model_name} - Eğitim Loss Grafiği")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.grid(True)
        ax.legend()
        return fig

    except Exception as e:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"⚠️ Grafik hatası: {str(e)}", ha='center', va='center')
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

# =================== ARAYÜZ ===================
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

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="teal")) as demo:
    gr.Markdown("""
    <h1 style="text-align: center;">🪝 Ahşap Yüzey Anomali Tespiti Paneli</h1>
    <p style="text-align: center; max-width: 800px; margin: auto;">
    Modelleri eğitip test edebilir, sonuçları görüntüleyebilir ve eğitim kaybı grafiğini çizebilirsiniz.
    </p><hr>
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
            comparison_table = gr.Dataframe(headers=["Model", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score", "IoU"], wrap=True, interactive=False)
            compare_button = gr.Button("📊 Karşılaştırmayı Göster")
            compare_button.click(fn=compare_models, outputs=comparison_table)

        with gr.Tab("📉 Eğitim Kaybı (Loss) Grafiği"):
            selected_model_for_plot = gr.Dropdown(["EfficientAD", "PaDiM", "PatchCore", "STFPM", "DRAEM", "REVERSE DISTILLATION"], label="Model Seçin")
            plot_button = gr.Button("📁 Loss Grafiğini Göster")
            loss_plot_output = gr.Plot()
            plot_button.click(fn=draw_loss_plot, inputs=selected_model_for_plot, outputs=loss_plot_output)

    run_button.click(fn=run_and_parse, inputs=[model, operation], outputs=[output_text, table, image_gallery])

demo.launch(share=True)