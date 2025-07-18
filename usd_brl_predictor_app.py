
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from transformers import pipeline

st.set_page_config(page_title="USD to BRL Predictor", layout="wide")

@st.cache_data
def get_hourly_data():
    now = datetime.now()
    past = now - timedelta(days=5)
    data = yf.download("USDBRL=X", start=past.strftime("%Y-%m-%d"), interval="60m")
    if data.empty:
        raise ValueError("No data retrieved from yfinance.")
    return data

@st.cache_resource
def build_model_cached(data):
    return build_and_train(data)

def build_and_train(data, seq_len=60, epochs=5):
    if data.shape[0] < seq_len + 1:
        raise ValueError("Not enough data to train the model. Need at least 61 data points.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Training data is empty after sequence preparation.")

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    return model, scaler

def predict_next_value(model, scaler, recent_data):
    recent_scaled = scaler.transform(recent_data)
    X_input = recent_scaled[-60:].reshape((1, 60, 1))
    prediction = model.predict(X_input, verbose=0)
    return scaler.inverse_transform(prediction)[0][0]

def classify_news(text):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["positive for BRL", "negative for BRL", "neutral"]
    result = classifier(text, labels)
    return result["labels"][0], result["scores"][0]

# MAIN APP
st.title("💸 USD to BRL AI Price Predictor")

try:
    data = get_hourly_data()
    st.subheader("Historical USD/BRL Exchange Rate (Hourly)")
    st.line_chart(data['Close'])

    st.write("Latest value:", round(data['Close'].iloc[-1], 4))
    st.write("Training model...")

    model, scaler = build_model_cached(data[['Close']].values.reshape(-1, 1))

    prediction = predict_next_value(model, scaler, data[['Close']].values)
    st.success(f"Predicted next value: {round(prediction, 4)} BRL")

    with st.expander("📊 Raw Data"):
        st.dataframe(data.tail(100))

except Exception as e:
    st.error(f"App crashed: {e}")



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

st.subheader("📉 Forecast vs Actual for USD/BRL – Next 3 Months")

# Simulated prediction (replace with real model outputs if available)
future_days = pd.date_range(datetime.today(), periods=90, freq='D')
predicted = np.sin(np.linspace(0, 3 * np.pi, 90)) + 5.0 + np.random.normal(0, 0.1, 90)
actual = predicted + np.random.normal(0, 0.2, 90)

# Prediction bands (you could use model confidence intervals instead)
predicted_low = predicted - 0.15
predicted_high = predicted + 0.15

# Classification of actual vs predicted range
markers = []
for act, low, high in zip(actual, predicted_low, predicted_high):
    if act < low:
        markers.append("below")
    elif act > high:
        markers.append("above")
    else:
        markers.append("within")

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(future_days, predicted, label="Predicted", color="blue")
ax.fill_between(future_days, predicted_low, predicted_high, color="blue", alpha=0.2, label="Prediction Range")
ax.plot(future_days, actual, label="Actual", color="black", linestyle="--", linewidth=1)

for i in range(len(future_days)):
    if markers[i] == "below":
        ax.plot(future_days[i], actual[i], 'ro', label='Actual Below Range' if i == 0 else "", markersize=5)
    elif markers[i] == "above":
        ax.plot(future_days[i], actual[i], 'go', label='Actual Above Range' if i == 0 else "", markersize=5)

ax.set_title("Predicted vs Actual USD/BRL – 3-Month Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Exchange Rate")
ax.grid(True)
ax.legend()
fig.autofmt_xdate()

st.pyplot(fig)
