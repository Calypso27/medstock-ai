---
title: MedStock AI - Groupe 2
emoji: 🏥
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
license: mit
---

# MedStock AI — Groupe 2

Application Streamlit multipage combinant classification d'images médicales et prédiction boursière.

## Modules

### Module 1 — Classification CNN (Maladies rénales)

- **EfficientNet-B4** fine-tuné sur le dataset CT-Kidney (Cyst / Normal / Stone / Tumor)
- Chatbot médical **Grok** avec contexte CNN injecté automatiquement
- Traduction automatique de la réponse en **allemand**
- **Résumé IA** distinct de la réponse chatbot
- Synthèse vocale **gTTS** de la traduction allemande → `st.audio()`
- Monitoring **LangSmith**

### Module 2 — Prédiction boursière

- Données historiques via **yfinance** (NVIDIA · Oracle · IBM · Cisco)
- Trois modèles : **LSTM** (PyTorch) · **Facebook Prophet** · **NeuralProphet**
- Sélecteur de période : 3 à 12 mois
- Graphique interactif **Plotly** (historique + prédiction)
- Chatbot d'analyse financière **Grok** avec contexte prédiction injecté
- **Résumé IA** de la prédiction

---

## Installation locale

```bash
git clone https://github.com/Calypso27/medstock-ai
cd medstock-ai
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine :

```env
XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=medstock-ai
```

Lancer l'application :

```bash
streamlit run app.py
```

---

## Entraînement du modèle CNN

1. Télécharger le dataset depuis Kaggle :
   [CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)

2. Extraire dans `data/CT-KIDNEY-DATASET/`

3. Lancer l'entraînement et uploader sur HuggingFace Hub :

```bash
python train_cnn.py \
  --data_dir ./data/CT-KIDNEY-DATASET \
  --epochs 20 \
  --upload
```

4. Mettre à jour `HF_MODEL_REPO` dans `utils/config.py` avec ton username HuggingFace.

---

## Déploiement sur HuggingFace Spaces

### Secrets à configurer (Settings → Variables and secrets)

| Clé | Description |
|-----|-------------|
| `XAI_API_KEY` | Clé API xAI (Grok) |
| `LANGCHAIN_API_KEY` | Clé API LangSmith |
| `LANGCHAIN_TRACING_V2` | `true` |
| `LANGCHAIN_PROJECT` | `medstock-ai` |

### Déploiement

HuggingFace Spaces détecte automatiquement `sdk: streamlit` dans le README et lance `app.py`.
Le modèle CNN est chargé depuis HuggingFace Hub au premier démarrage (pas besoin de le commiter).

---

## Structure du projet

```
medstock-ai/
├── app.py                          # Point d'entrée Streamlit
├── train_cnn.py                    # Script d'entraînement CNN (hors app)
├── requirements.txt
├── .env                            # Secrets locaux (non commité)
├── .streamlit/
│   └── config.toml                 # Thème et config Streamlit
├── pages/
│   ├── classification_cnn.py       # Module 1 — interface complète
│   └── prediction_bourse.py        # Module 2 — interface complète
├── modules/
│   ├── ai_services/
│   │   ├── chatbot.py              # LLM Grok via API OpenAI-compatible
│   │   ├── translator.py           # Traduction → allemand
│   │   ├── summarizer.py           # Résumé IA (output distinct)
│   │   └── tts.py                  # gTTS → BytesIO → st.audio()
│   ├── cnn/
│   │   ├── predictor.py            # Inférence EfficientNet-B4
│   │   └── preprocessor.py         # Prétraitement images
│   └── stock/
│       ├── data_fetcher.py         # yfinance OHLCV
│       ├── lstm_model.py           # LSTM PyTorch
│       ├── prophet_model.py        # Facebook Prophet
│       ├── neuralprophet_model.py  # NeuralProphet
│       └── chart_builder.py        # Graphique Plotly
├── monitoring/
│   └── langsmith_tracker.py        # @traceable + dashboard traces
├── utils/
│   ├── config.py                   # Constantes globales
│   └── session.py                  # Helpers st.session_state
├── models/                         # Modèles entraînés (hors Git)
└── data/                           # Données brutes (hors Git)
```
