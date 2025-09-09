import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

class StockPredictor:
  def __init__(self, ticker):
    self.ticker = ticker.upper()
    self.data = self._fetch_data()
    self.model = None

  def _fetch_data(self):
    # Fetches last 5 years of daily data
    end = datetime.today()
    start = end - timedelta(days=5*365)
    df = yf.download(self.ticker, start=start, end=end)
    if df.empty:
      raise ValueError(f"No data found for ticker {self.ticker}")
    return df

  def get_info(self):
    stock = yf.Ticker(self.ticker)
    info = stock.info
    keys = ['longName', 'sector', 'industry', 'marketCap', 'previousClose', 'open', 'dayHigh', 'dayLow', 'volume']
    return {k: info.get(k, 'N/A') for k in keys}

  def train_model(self):
    df = self.data.copy()
    df = df.reset_index()
    df['Date_ordinal'] = pd.to_datetime(df['Date']).map(datetime.toordinal)
    X = df[['Date_ordinal']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    self.model = model
    self.last_date = df['Date'].max()
    self.last_date_ordinal = df['Date_ordinal'].max()

  def predict(self, period='weeks', amount=1):
    if self.model is None:
      self.train_model()
    days_map = {
      'weeks': 7,
      'months': 30,
      'years': 365,
      'decades': 3650
    }
    if period not in days_map:
      # raises error for unsupported time periods
      raise ValueError("Period must be one of: weeks, months, years, decades")
    days = days_map[period] * amount
    future_date_ordinal = self.last_date_ordinal + days
    pred = self.model.predict(np.array([[future_date_ordinal]]))
    return float(pred[0])

def main():
  # intro
  print("Welcome to Stock Predictor!")
  ticker = input("Enter stock ticker (Exampl: AMZN): ").strip().upper()
  predictor = StockPredictor(ticker)
  print("\nStock Information:")
  info = predictor.get_info()
  for k, v in info.items():
    print(f"{k}: {v}")
  print("\nPredictions:")
  for period in ['weeks', 'months', 'years', 'decades']:
    pred = predictor.predict(period=period, amount=1)
    print(f"Predicted price in 1 {period[:-1]}: ${pred:.2f}")

if __name__ == "__main__":
  main()