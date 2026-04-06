import time
import random
import torch
import torch.nn.functional as F
import streamlit as st
from pathlib import Path
from PIL import Image

from utils.config import CNN_CLASSES, HF_MODEL_REPO, MODEL_FILENAME
from modules.cnn.preprocessor import preprocess_image


@st.cache_resource(show_spinner="Chargement du modèle CNN…")
def _load_model():
    import timm

    local = Path("models") / MODEL_FILENAME
    model_path = None

    if local.exists():
        model_path = str(local)
    else:
        # on essaie de télécharger depuis HuggingFace
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME)
        except Exception:
            pass

    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=len(CNN_CLASSES))

    if model_path:
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception:
            pass

    return None 


def predict_image(image):
    model = _load_model()

    if model is None:
        # pas de modèle probabilités aléatoires pour la démo
        raw = [random.random() for _ in CNN_CLASSES]
        total = sum(raw)
        probas = {c: round(v / total, 4) for c, v in zip(CNN_CLASSES, raw)}
        pred = max(probas, key=probas.get)
        return {
            "predicted_class": pred,
            "confidence":      probas[pred],
            "all_probas":      probas,
            "inference_ms":    0.0,
            "demo_mode":       True,
        }

    tensor = preprocess_image(image)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    ms = round((time.perf_counter() - t0) * 1000, 1)

    probas_list = probs.tolist()
    probas = dict(zip(CNN_CLASSES, [round(p, 4) for p in probas_list]))
    idx = int(probs.argmax())

    return {
        "predicted_class": CNN_CLASSES[idx],
        "confidence":      probas_list[idx],
        "all_probas":      probas,
        "inference_ms":    ms,
        "demo_mode":       False,
    }
