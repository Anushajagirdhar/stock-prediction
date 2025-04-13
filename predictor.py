import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Download stock data
ticker = 'AAPL'  # You can change this to any stock symbol
df = yf.download(ticker, start='2015-01-01', end='2023-12-31')

# Use only 'Close' prices
df = df[['Close']]
df['Prediction'] = df['Close'].shift(-30)  # Predict 30 days into the future

# Create feature and target data
X = np.array(df.drop(['Prediction'], axis=1))[:-30]
y = np.array(df['Prediction'])[:-30]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the next 30 days
X_future = df.drop(['Prediction'], axis=1)[-30:]
forecast = model.predict(X_future)

# Visualize the prediction
plt.figure(figsize=(10, 5))
plt.plot(df['Close'], label='Actual Prices')
plt.plot(range(len(df)-30, len(df)), forecast, label='Predicted Prices', color='red')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
