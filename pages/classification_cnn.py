import streamlit as st
from PIL import Image

from utils.config import CNN_CLASS_INFO
from utils.session import init_cnn_session, clear_cnn_chat
from modules.cnn.predictor import predict_image
from modules.ai_services.chatbot import stream_grok
from modules.ai_services.translator import translate_to_german
from modules.ai_services.summarizer import generate_medical_summary
from modules.ai_services.tts import text_to_speech

init_cnn_session()


def _system(result):
    base = (
        "Tu es un assistant médical spécialisé en imagerie rénale. "
        "Tu réponds en français, de façon claire et bienveillante. "
        "Rappelle toujours que tes réponses sont informatives et ne remplacent "
        "pas une consultation médicale."
    )
    if result:
        cls  = result["predicted_class"]
        conf = round(result["confidence"] * 100, 1)
        base += (
            f"\n\nContexte : classe détectée = '{cls}' avec {conf}% de confiance. "
            f"Tiens compte de ce résultat dans toutes tes réponses."
        )
    return base


st.title(":material/biotech: Classification CNN — Maladies rénales")
st.caption("EfficientNet-B4 · 4 classes : Cyst · Normal · Stone · Tumor")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    with st.container(border=True):
        st.subheader("Image CT-scan")
        uploaded = st.file_uploader(
            "Glisser ou cliquer pour uploader",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)
            if st.button("Analyser avec le CNN", type="primary", use_container_width=True):
                with st.spinner("Inférence CNN en cours…"):
                    result = predict_image(image)
                st.session_state["cnn_result"] = result
                clear_cnn_chat()
                cls  = result["predicted_class"]
                conf = round(result["confidence"] * 100, 1)
                mode = " *(mode démo)*" if result["demo_mode"] else ""
                st.session_state["cnn_messages"] = [{
                    "role": "assistant",
                    "content": (
                        f"Analyse terminée{mode}. Classe : **{cls}** "
                        f"avec **{conf}%** de confiance. Posez vos questions."
                    ),
                }]
                st.rerun()
        else:
            st.info("Uploadez une image CT-scan rénale.", icon=":material/upload:")

    result = st.session_state.get("cnn_result")
    if result:
        with st.container(border=True):
            st.subheader("Résultats")
            if result["demo_mode"]:
                st.warning("Mode démo — probabilités aléatoires (modèle non chargé)")

            cls   = result["predicted_class"]
            conf  = round(result["confidence"] * 100, 1)
            info  = CNN_CLASS_INFO.get(cls, {})
            color = info.get("color", "#888")
            desc  = info.get("desc", cls)

            c1, c2, c3 = st.columns(3)
            c1.metric("Diagnostic", cls)
            c2.metric("Confiance", f"{conf}%")
            c3.metric("Inférence", f"{result['inference_ms']} ms")

            st.markdown(
                f"<div style='padding:8px;border-radius:6px;"
                f"background:{color}22;border-left:4px solid {color};margin:8px 0'>"
                f"<b>{cls}</b> — {desc}</div>",
                unsafe_allow_html=True,
            )
            st.divider()
            for c, p in sorted(result["all_probas"].items(), key=lambda x: x[1], reverse=True):
                col_c = CNN_CLASS_INFO.get(c, {}).get("color", "#888")
                pct   = round(p * 100, 1)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
                    f"<span style='width:50px;font-size:13px'>{c}</span>"
                    f"<div style='flex:1;background:rgba(128,128,128,0.15);"
                    f"border-radius:4px;height:8px'>"
                    f"<div style='width:{pct}%;background:{col_c};"
                    f"height:8px;border-radius:4px'></div></div>"
                    f"<span style='font-size:12px;width:38px;text-align:right'>{pct}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

with col_right:
    result = st.session_state.get("cnn_result")

    with st.container(border=True):
        st.subheader("Chatbot médical")
        if result:
            st.caption(f"Contexte : {result['predicted_class']} ({round(result['confidence']*100,1)}%)")
        else:
            st.caption("En attente d'une analyse CNN")

        for msg in st.session_state["cnn_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Posez votre question médicale…", disabled=(result is None)):
            st.session_state["cnn_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                response = st.write_stream(stream_grok(
                    messages=st.session_state["cnn_messages"],
                    system=_system(result),
                ))
            st.session_state["cnn_messages"].append({"role": "assistant", "content": response})
            st.session_state["cnn_translation"] = ""
            st.session_state["cnn_summary"]     = ""
            st.session_state["cnn_audio"]       = None
            st.rerun()

        if st.session_state["cnn_messages"]:
            if st.button("Effacer la conversation", key="clear_cnn"):
                clear_cnn_chat()
                st.rerun()

    messages = st.session_state["cnn_messages"]
    last_bot = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
        None,
    )

    if last_bot:
        with st.container(border=True):
            st.subheader("Traduction · Résumé · Audio")
            st.caption("Appliqués à la dernière réponse du chatbot")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Traduire en allemand", use_container_width=True,
                             icon=":material/translate:"):
                    with st.spinner("Traduction…"):
                        st.session_state["cnn_translation"] = translate_to_german(last_bot)
                        st.session_state["cnn_audio"]       = None
            with b2:
                if st.button("Générer le résumé", use_container_width=True,
                             icon=":material/summarize:"):
                    with st.spinner("Résumé…"):
                        st.session_state["cnn_summary"] = generate_medical_summary(last_bot)

            if st.session_state["cnn_translation"]:
                st.divider()
                st.markdown("**Traduction (Deutsch)**")
                st.info(st.session_state["cnn_translation"])

                if st.session_state["cnn_audio"] is None:
                    with st.spinner("Synthèse vocale…"):
                        st.session_state["cnn_audio"] = text_to_speech(
                            st.session_state["cnn_translation"], lang="de"
                        )
                if st.session_state["cnn_audio"]:
                    st.markdown("**Audio (Allemand)**")
                    st.audio(st.session_state["cnn_audio"], format="audio/mp3")

            if st.session_state["cnn_summary"]:
                st.divider()
                st.markdown("**Résumé IA**")
                st.success(st.session_state["cnn_summary"])
