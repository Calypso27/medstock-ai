import pandas as pd
import streamlit as st
from prophet import Prophet


@st.cache_data(show_spinner="Entraînement Prophet en cours…")
def run_prophet(close, horizon_months):
    horizon_days = horizon_months * 21

    df = pd.DataFrame({"ds": close.index, "y": close.values})
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)

    # bdate_range pour éviter le bug pandas 2.x avec les Timestamp
    last_date    = df["ds"].max()
    future_dates = list(pd.bdate_range(start=last_date, periods=horizon_days + 1)[1:])
    future = pd.concat(
        [df[["ds"]], pd.DataFrame({"ds": future_dates})],
        ignore_index=True,
    )

    forecast  = model.predict(future)
    future_fc = forecast[forecast["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return future_fc.reset_index(drop=True)
