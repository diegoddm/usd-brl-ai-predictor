
import os
import streamlit as st
from dotenv import load_dotenv
from utils.preprocess import get_hourly_data
from utils.model import build_and_train, predict_next
from utils.news_scraper import fetch_and_classify_news
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

load_dotenv()
st.set_page_config(page_title="USD→BRL AI Predictor", layout="centered")
st.title("💸 USD→BRL Hourly Predictor with News Intelligence")

# Auto-refresh every hour
st_autorefresh(interval=60 * 60 * 1000)

# Load hourly exchange rate data
with st.spinner("📥 Loading hourly USD/BRL data..."):
    data = get_hourly_data()

st.subheader("📉 Hourly USD/BRL Exchange Rate")
st.line_chart(data)

# Cache model training
@st.cache_resource
def build_model_cached(data):
    return build_and_train(data)

# Train model and predict next price
with st.spinner("🤖 Training LSTM model and predicting next value..."):
    model, scaler = build_model_cached(data.values)
    prediction = predict_next(model, scaler, data.values)

current = data['Close'].iloc[-1]
delta = prediction - current

st.subheader("🔮 LSTM Model Prediction")
st.metric("Current Rate", f"{current:.4f}")
st.metric("Predicted Rate", f"{prediction:.4f}", delta=f"{delta:+.4f}")

# Scrape and score news sentiment
with st.spinner("🧠 Scraping and analyzing news..."):
    news = fetch_and_classify_news()

st.subheader("📰 News Impact Analysis")
if news:
    score = sum(n['score'] if 'up' in n['impact'].lower() else -n['score'] for n in news) / len(news)
    st.metric("Impact Score (Last Hour)", f"{score:+.2f}")
    for n in news[:5]:
        st.write(f"- **{n['impact']}** | {n['title']} ({n['published'][:16]})")
else:
    st.info("No recent news available.")

# Final recommendation logic
st.subheader("📌 Recommendation")
if prediction < current and score > 0:
    st.success("✅ Strong BUY signal – favorable forecast + positive news")
elif prediction < current:
    st.warning("⚠️ Buy signal, but news is mixed or negative")
elif score < 0:
    st.info("⏳ Wait – forecast rising and news negative")
else:
    st.info("🔍 Mixed signals – monitor closely")
