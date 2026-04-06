import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# on utilise Groq qui est gratuit et rapide
XAI_BASE_URL = "https://api.groq.com/openai/v1"
XAI_MODEL    = "llama-3.3-70b-versatile"

# classes du dataset CT-Kidney (ordre alphabétique = ordre ImageFolder)
CNN_CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]

CNN_CLASS_INFO = {
    "Cyst":   {"color": "#60A5FA", "desc": "Kyste rénal"},
    "Normal": {"color": "#34D399", "desc": "Rein sain"},
    "Stone":  {"color": "#FCD34D", "desc": "Calcul rénal"},
    "Tumor":  {"color": "#F87171", "desc": "Tumeur rénale"},
}

HF_MODEL_REPO  = "Calypso-MB/kidney-cnn"
MODEL_FILENAME = "kidney_efficientnet_b4.pt"

COMPANIES = {
    "NVDA": {"name": "NVIDIA",  "sector": "Semi-conducteurs"},
    "ORCL": {"name": "Oracle",  "sector": "Cloud & Logiciels"},
    "IBM":  {"name": "IBM",     "sector": "Services IT"},
    "CSCO": {"name": "Cisco",   "sector": "Réseaux"},
}

HORIZON_MIN = 3
HORIZON_MAX = 12

LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "medstock-ai")


def get_xai_key():
    return os.getenv("GROQ_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("XAI_API_KEY", "")))


def get_langsmith_key():
    return os.getenv("LANGCHAIN_API_KEY", "")
