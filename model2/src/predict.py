import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Abdulhaque/cancer-type-classification", filename="best_model_ft.keras")
model = tf.keras.models.load_model(model_path)
IMG_SIZE = 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
CLASS_NAMES = ["Lung Adenocarcinoma","Lung Squamous Cell Carcinoma",
               "Colon Adenocarcinoma","Lung Normal","Colon Normal"]

def predict(image_path):
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = (np.array(img, dtype=np.float32)/255.0 - mean) / std
    preds = model.predict(arr[None], verbose=0)[0]
    pred_class = int(np.argmax(preds))
    return {
        "prediction": CLASS_NAMES[pred_class],
        "confidence": f"{preds[pred_class]*100:.1f}%",
        "all_probabilities": {CLASS_NAMES[i]: f"{preds[i]*100:.1f}%" for i in range(5)}
    }
