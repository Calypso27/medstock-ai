import warnings
import pandas as pd
import streamlit as st
from neuralprophet import NeuralProphet

warnings.filterwarnings("ignore")


@st.cache_data(show_spinner="Entraînement NeuralProphet en cours…")
def run_neuralprophet(close, horizon_months):
    horizon_days = horizon_months * 21

    df = pd.DataFrame({
        "ds": pd.to_datetime(close.index),
        "y":  close.values,
    })

    model = NeuralProphet(
        n_forecasts=1,
        n_lags=30,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        verbose=False,
    )
    model.fit(df, freq="B")

    last_date    = df["ds"].max()
    future_dates = list(pd.bdate_range(start=last_date, periods=horizon_days + 1)[1:])

    future   = model.make_future_dataframe(df, periods=horizon_days, n_historic_predictions=False)
    forecast = model.predict(future)

    future_fc = forecast[forecast["ds"] > last_date][["ds", "yhat1"]].copy()
    future_fc = future_fc.rename(columns={"yhat1": "yhat"})

    return future_fc.reset_index(drop=True)
