import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

from monitoring.langsmith_tracker import setup_langsmith
langsmith_ok = setup_langsmith()

st.set_page_config(
    page_title="MedStock AI — Groupe 2",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/classification_cnn.py",
            title="Classification CNN", icon=":material/biotech:"),
    st.Page("pages/prediction_bourse.py",
            title="Prédiction Boursière", icon=":material/candlestick_chart:"),
]

pg = st.navigation(pages)

with st.sidebar:
    st.markdown("### MedStock AI")
    st.caption("Groupe 2 · Master 2 Data Science")
    st.divider()
    st.caption("Statut des clés API")

    api_ok = bool(os.getenv("GROQ_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("XAI_API_KEY", ""))))
    ls_ok  = bool(os.getenv("LANGCHAIN_API_KEY", ""))

    if api_ok:
        st.success("Groq (LLaMA) ✓", icon=":material/check_circle:")
    else:
        st.error("GROQ_API_KEY manquante", icon=":material/error:")

    if ls_ok:
        st.success("LangSmith ✓", icon=":material/check_circle:")
    else:
        st.warning("LangSmith non configuré", icon=":material/warning:")

    if langsmith_ok:
        st.divider()
        st.caption("Monitoring LangSmith actif")
        st.markdown("[Ouvrir le dashboard →](https://smith.langchain.com)")

pg.run()
