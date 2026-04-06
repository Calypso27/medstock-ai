import streamlit as st

from utils.config import COMPANIES, HORIZON_MIN, HORIZON_MAX
from utils.session import init_stock_session, clear_stock_chat
from modules.stock.data_fetcher import get_close_series
from modules.stock.lstm_model import run_lstm
from modules.stock.prophet_model import run_prophet
from modules.stock.neuralprophet_model import run_neuralprophet
from modules.stock.chart_builder import build_forecast_chart
from modules.ai_services.chatbot import stream_grok
from modules.ai_services.summarizer import generate_financial_summary

MODELS = ["LSTM", "Prophet", "NeuralProphet"]

init_stock_session()


def _system(ticker, model_name, forecast):
    co   = COMPANIES[ticker]
    base = (
        f"Tu es un analyste financier quantitatif expert. "
        f"Tu analyses {co['name']} ({ticker}), secteur : {co['sector']}. "
        f"Tu réponds en français, de façon analytique. "
        f"Ne donne jamais de conseil d'investissement direct. "
        f"Mentionne toujours les risques."
    )
    if forecast is not None and not forecast.empty:
        first = round(float(forecast["yhat"].iloc[0]), 2)
        last  = round(float(forecast["yhat"].iloc[-1]), 2)
        base += (
            f"\n\nContexte : modèle {model_name}, "
            f"cours prédit à 1 mois : ${first}, "
            f"cours prédit en fin de période : ${last}."
        )
    return base


st.title(":material/candlestick_chart: Prédiction Boursière")
st.caption("NVIDIA · Oracle · IBM · Cisco — LSTM · Prophet · NeuralProphet")

col_params, col_main = st.columns([1, 2.5], gap="large")

with col_params:
    with st.container(border=True):
        st.subheader("Paramètres")

        ticker = st.selectbox(
            "Entreprise",
            options=list(COMPANIES.keys()),
            format_func=lambda k: f"{COMPANIES[k]['name']} ({k})",
            index=list(COMPANIES.keys()).index(st.session_state.get("stock_ticker", "NVDA")),
        )
        st.session_state["stock_ticker"] = ticker

        model_name = st.radio(
            "Modèle de prédiction",
            options=MODELS,
            index=MODELS.index(st.session_state.get("stock_model", "LSTM")),
        )
        st.session_state["stock_model"] = model_name

        horizon = st.slider(
            "Horizon (mois)",
            min_value=HORIZON_MIN,
            max_value=HORIZON_MAX,
            value=st.session_state.get("stock_horizon", 6),
        )
        st.session_state["stock_horizon"] = horizon

        run_btn = st.button("Lancer la prédiction", type="primary", use_container_width=True)

    with st.container(border=True):
        st.subheader(f"Métriques — {COMPANIES[ticker]['name']}")
        try:
            close   = get_close_series(ticker)
            current = round(float(close.iloc[-1]), 2)
            prev    = round(float(close.iloc[-2]), 2)
            chg_1d  = round((current - prev) / prev * 100, 2)
            chg_1m  = round(
                (current - float(close.iloc[-21])) / float(close.iloc[-21]) * 100, 2
            ) if len(close) > 21 else 0.0

            st.metric("Cours actuel", f"${current}", f"{chg_1d:+.2f}% (1j)")
            st.metric("Variation 30j", f"{chg_1m:+.2f}%")

            forecast = st.session_state.get("stock_forecast")
            if forecast is not None and not forecast.empty:
                target = round(float(forecast["yhat"].iloc[-1]), 2)
                upside = round((target - current) / current * 100, 2)
                st.metric(f"Cible {model_name} ({horizon}m)", f"${target}", f"{upside:+.2f}%")
        except Exception as e:
            st.warning(f"Données indisponibles : {e}")

if run_btn:
    clear_stock_chat()
    try:
        close = get_close_series(ticker)

        with st.spinner(f"Entraînement {model_name}…"):
            if model_name == "LSTM":
                forecast = run_lstm(close, horizon)
            elif model_name == "Prophet":
                forecast = run_prophet(close, horizon)
            else:
                forecast = run_neuralprophet(close, horizon)

        st.session_state["stock_forecast"] = forecast

        target = round(float(forecast["yhat"].iloc[-1]), 2)
        st.session_state["stock_messages"] = [{
            "role": "assistant",
            "content": (
                f"Prédiction **{model_name}** pour "
                f"**{COMPANIES[ticker]['name']} ({ticker})** "
                f"sur **{horizon} mois** terminée. "
                f"Cours cible estimé : **${target}**. "
                f"Posez vos questions d'analyse financière."
            ),
        }]

        summary_text = (
            f"Prédiction {model_name} pour {ticker} sur {horizon} mois. "
            f"Cours actuel : ${round(float(close.iloc[-1]), 2)}. "
            f"Cours cible : ${target}."
        )
        with st.spinner("Résumé IA…"):
            st.session_state["stock_summary"] = generate_financial_summary(
                summary_text, ticker, model_name
            )
        st.rerun()

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}", icon=":material/error:")

with col_main:
    forecast = st.session_state.get("stock_forecast")

    with st.container(border=True):
        st.subheader(f"Graphique — {COMPANIES[ticker]['name']} · {model_name}")
        if forecast is not None:
            try:
                close = get_close_series(ticker)
                fig   = build_forecast_chart(close, forecast, ticker, model_name)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur graphique : {e}")
        else:
            st.info("Lancez une prédiction pour afficher le graphique.",
                    icon=":material/show_chart:")

    if st.session_state["stock_summary"]:
        st.info(f"**Résumé IA :** {st.session_state['stock_summary']}")

    with st.container(border=True):
        st.subheader("Chatbot d'analyse financière")
        if forecast is not None:
            st.caption(f"Contexte : {COMPANIES[ticker]['name']} ({ticker}), {model_name}, {horizon} mois")
        else:
            st.caption("En attente d'une prédiction")

        for msg in st.session_state["stock_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Posez votre question financière…", disabled=(forecast is None)):
            st.session_state["stock_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                response = st.write_stream(stream_grok(
                    messages=st.session_state["stock_messages"],
                    system=_system(ticker, model_name, forecast),
                ))
            st.session_state["stock_messages"].append({"role": "assistant", "content": response})
            st.rerun()

        if st.session_state["stock_messages"]:
            if st.button("Effacer la conversation", key="clear_stock"):
                clear_stock_chat()
                st.rerun()
