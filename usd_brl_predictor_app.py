
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
    st.subheader("Diego - Historical USD/BRL Exchange Rate (Hourly)")
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
