import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


class _LSTMNet(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


@st.cache_data(show_spinner="Entraînement LSTM en cours…")
def run_lstm(close, horizon_months):
    SEQ_LEN = 60
    EPOCHS  = 30
    HORIZON = horizon_months * 21

    scaler = MinMaxScaler()
    vals   = close.values.reshape(-1, 1)
    scaled = scaler.fit_transform(vals).flatten()

    X, y = _make_sequences(scaled, SEQ_LEN)
    X_t  = torch.FloatTensor(X).unsqueeze(-1)
    y_t  = torch.FloatTensor(y).unsqueeze(-1)

    model     = _LSTMNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    window = scaled[-SEQ_LEN:].tolist()
    preds  = []
    with torch.no_grad():
        for _ in range(HORIZON):
            inp = torch.FloatTensor(window[-SEQ_LEN:]).unsqueeze(0).unsqueeze(-1)
            p   = model(inp).item()
            preds.append(p)
            window.append(p)

    preds_inv    = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    last_date    = pd.Timestamp(close.index[-1]).replace(tzinfo=None)
    future_dates = list(pd.bdate_range(start=last_date, periods=HORIZON + 1)[1:])

    return pd.DataFrame({"ds": future_dates, "yhat": preds_inv})
