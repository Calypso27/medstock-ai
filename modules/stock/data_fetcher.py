import pandas as pd
import yfinance as yf
import streamlit as st


@st.cache_data(ttl=3600, show_spinner="Téléchargement des données de marché…")
def fetch_ohlcv(ticker, period="2y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Aucune donnée pour {ticker}.")
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def get_close_series(ticker, period="2y"):
    df = fetch_ohlcv(ticker, period)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()
    close.index.freq = None  
    return close
