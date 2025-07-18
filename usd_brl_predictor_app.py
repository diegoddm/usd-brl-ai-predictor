
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils.preprocess import get_hourly_data, get_4hour_data, scrape_news
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import time

st.set_page_config(layout="wide")
st.title("Diego USD/BRL AI Predictor V2")

@st.cache_data(ttl=3600)
def load_data():
    return get_hourly_data(), get_4hour_data()

@st.cache_resource
def build_model(data, epochs=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(20, len(scaled)):
        X.append(scaled[i-20:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    return model, scaler

hourly_data, data_4h = load_data()

# Hourly chart
fig1, ax1 = plt.subplots()
price = hourly_data['Close']
ax1.plot(hourly_data.index, price, label="USD/BRL Hourly")
ax1.set_ylim(price.min() - 1, price.max() + 1)
ax1.set_title("USD/BRL - Hourly")
ax1.grid(True)
st.pyplot(fig1)

# 4-hour chart
fig2, ax2 = plt.subplots()
price_4h = data_4h['Close']
ax2.plot(data_4h.index, price_4h, label="USD/BRL 4H", color="green")
ax2.set_title("USD/BRL - 4 Hour Intervals")
ax2.grid(True)
st.pyplot(fig2)

# Forecasting
model, scaler = build_model(hourly_data['Close'].values.reshape(-1, 1))
last_sequence = scaler.transform(hourly_data['Close'].values[-20:].reshape(-1, 1))
preds = []
seq = last_sequence
for _ in range(90):
    next_pred = model.predict(seq.reshape(1, 20, 1), verbose=0)
    preds.append(next_pred[0][0])
    seq = np.append(seq[1:], next_pred, axis=0)

future_dates = pd.date_range(start=hourly_data.index[-1], periods=90, freq="D")
forecast = pd.DataFrame({'Date': future_dates, 'Forecast': scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()})
st.line_chart(forecast.set_index("Date"))

# News Impact
st.markdown("### News Impact Correlation (Top AI-relevant items)")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
news = scrape_news()
labels = ["economy", "politics", "trade", "inflation", "interest rate", "dollar up", "dollar down"]
for item in news:
    result = classifier(item['title'], labels)
    top_label = result['labels'][0]
    st.markdown(f"[{item['title']}]({item['url']}) — **{top_label}**")
