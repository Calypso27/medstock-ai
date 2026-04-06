import pandas as pd
import plotly.graph_objects as go

_COLORS = {
    "LSTM":          "#8B5CF6",
    "Prophet":       "#34D399",
    "NeuralProphet": "#F59E0B",
}


def build_forecast_chart(close, forecast, ticker, model_name):
    pred_color = _COLORS.get(model_name, "#8B5CF6")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        name="Historique", mode="lines",
        line=dict(color="#60A5FA", width=2),
        hovertemplate="%{x|%d/%m/%Y}<br>$%{y:.2f}<extra>Historique</extra>",
    ))

    # intervalle de confiance uniquement pour Prophet
    if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
        r, g, b = int(pred_color[1:3], 16), int(pred_color[3:5], 16), int(pred_color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle confiance",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        name=f"Prédiction {model_name}", mode="lines",
        line=dict(color=pred_color, width=2.5, dash="dot"),
        hovertemplate="%{x|%d/%m/%Y}<br>$%{y:.2f}<extra>" + model_name + "</extra>",
    ))

    # ligne verticale "aujourd'hui" — add_shape plus stable que add_vline avec des dates
    today_str = pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=today_str, x1=today_str, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="gray", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=today_str, y=1, yref="paper",
        text="Aujourd'hui", showarrow=False,
        xanchor="right", font=dict(size=11, color="gray"),
    )

    fig.update_layout(
        title=f"{ticker} — Prédiction {model_name}",
        xaxis_title="Date", yaxis_title="Cours (USD)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig
