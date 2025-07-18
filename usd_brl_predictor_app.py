
import os
import streamlit as st
from dotenv import load_dotenv
# --- Inline replacements for broken imports ---

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sqlite3
import datetime
import feedparser
from transformers import pipeline
from googletrans import Translator

# Preprocess
def get_hourly_data(ticker="USDBRL=X", period="7d", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval)
    if 'Close' not in df.columns or df.empty:
        return pd.DataFrame(columns=["Close"])
    df = df[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    return df



# Model
def build_and_train(data, seq_len=60, epochs=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(len(X), seq_len, 1)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile("adam", "mean_squared_error")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    return model, scaler

def predict_next(model, scaler, data, seq_len=60):
    last = data[-seq_len:].reshape(1, seq_len, 1)
    pred = model.predict(last)
    return scaler.inverse_transform(pred)[0][0]

# News
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
translator = Translator()
labels = ["BRL will go up", "BRL will go down", "No significant effect"]

def fetch_and_classify_news():
    db_path = "brl_news_cache.sqlite"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY,
        title TEXT,
        summary TEXT,
        url TEXT,
        published TEXT,
        language TEXT,
        translated TEXT,
        impact TEXT,
        score REAL
    )''')
    conn.commit()

    now = datetime.datetime.utcnow()
    one_hour_ago = now - datetime.timedelta(hours=1)
    rss_feeds = [
        "https://g1.globo.com/rss/g1/economia/",
        "https://valor.globo.com/rss.xml",
        "https://feeds.folha.uol.com.br/mercado/rss091.xml",
        "http://feeds.reuters.com/reuters/businessNews",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml"
    ]
    results = []
    for url in rss_feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            published = datetime.datetime(*entry.published_parsed[:6])
            if published < one_hour_ago:
                continue
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            lang = "pt" if "globo.com" in url or "folha" in url else "en"
            text = f"{title}. {summary}"
            translated = translator.translate(text, src='pt', dest='en').text if lang == "pt" else text
            try:
                result = classifier(translated[:512], labels)
                top_label = result['labels'][0]
                score = result['scores'][0]
            except:
                top_label = "Unknown"
                score = 0
            results.append({
                "title": title,
                "summary": summary,
                "url": entry.get("link", ""),
                "published": published.isoformat(),
                "language": lang,
                "translated": translated,
                "impact": top_label,
                "score": score
            })
            c.execute('''INSERT INTO news (
                title, summary, url, published, language, translated, impact, score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (
                title, summary, entry.get("link", ""), published.isoformat(),
                lang, translated, top_label, score
            ))
            conn.commit()
    conn.close()
    return results
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
st.subheader("📈 USD to BRL - Last 7 Days (Hourly)")

if data.empty or 'Close' not in data.columns:
    st.error("⚠️ No data available for USD/BRL. Please try again later.")
else:
st.line_chart(data['Close'])
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
