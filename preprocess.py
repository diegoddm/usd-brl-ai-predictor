
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_hourly_data(symbol="USDBRL=X", period="7d", interval="60m"):
    return yf.download(tickers=symbol, period=period, interval=interval)

def get_4hour_data(symbol="USDBRL=X"):
    return yf.download(tickers=symbol, period="14d", interval="240m")

def scrape_news():
    return [
        {"title": "Fed rate decision impacts USD", "url": "https://example.com", "timestamp": datetime.now()},
        {"title": "Brazil announces new fiscal policy", "url": "https://example.com", "timestamp": datetime.now()},
    ]
