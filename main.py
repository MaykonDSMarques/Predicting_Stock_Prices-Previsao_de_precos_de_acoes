import pandas as pd
import prophet as Prophet
from prophet.plot import plot_plotly
from pandas_datareader import data as pdr
import yfinance as yfin
import matplotlib.pyplot as plt
yfin.pdr_override()

data = pdr.DataReader("TSLA", start="2020-01-01", end="2023-02-21")

data.to_csv("stock_data.csv")

data = pd.read_csv("stock_data.csv")

data = data[["Date", "Close"]]
data.columns = ["ds", "y"]
print(data)

prophet = Prophet.Prophet(daily_seasonality=True)
prophet.fit(data)

future_dates = prophet.make_future_dataframe(periods=365)
predictions = prophet.predict(future_dates)

grafico = plot_plotly(prophet, predictions)
grafico.show()
