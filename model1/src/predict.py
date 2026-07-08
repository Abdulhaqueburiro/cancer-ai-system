import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Abdulhaque/breast-cancer-detection", filename="best_model_ft.keras")
model = tf.keras.models.load_model(model_path)
IMG_SIZE = 96
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

def predict(image_path, threshold=0.3):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = (np.array(img, dtype=np.float32)/255.0 - mean) / std
    prob = float(model.predict(arr[None], verbose=0)[0][0])
    return {
        "prediction": "CANCER DETECTED" if prob >= threshold else "NO CANCER",
        "confidence": f"{prob*100:.1f}%",
        "risk_level": "HIGH RISK" if prob >= threshold else "LOW RISK"
    }
