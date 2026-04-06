import streamlit as st


def init_cnn_session():
    defaults = {
        "cnn_result":      None,
        "cnn_messages":    [],
        "cnn_translation": "",
        "cnn_summary":     "",
        "cnn_audio":       None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def init_stock_session():
    defaults = {
        "stock_ticker":   "NVDA",
        "stock_model":    "LSTM",
        "stock_horizon":  6,
        "stock_forecast": None,
        "stock_messages": [],
        "stock_summary":  "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_cnn_chat():
    st.session_state["cnn_messages"]    = []
    st.session_state["cnn_translation"] = ""
    st.session_state["cnn_summary"]     = ""
    st.session_state["cnn_audio"]       = None


def clear_stock_chat():
    st.session_state["stock_messages"] = []
    st.session_state["stock_summary"]  = ""
