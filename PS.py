# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import plotly.graph_objects as go

# Step 1: Data Collection using Yahoo Finance
ticker = 'AAPL'  # Example: Apple stock (change to other ticker symbols as needed)
data = yf.download(ticker, start="2015-01-01", end="2025-01-01")

# Step 2: Data Preprocessing
# We are interested in 'Close' price for stock prediction
data = data[['Close']]

# Handle missing values (forward fill)
data.fillna(method='ffill', inplace=True)

# Step 3: Feature Engineering - Add Lag features (Previous days' closing prices)
data['Prev_1'] = data['Close'].shift(1)
data['Prev_2'] = data['Close'].shift(2)
data['Prev_3'] = data['Close'].shift(3)
data.dropna(inplace=True)

# Step 4: Split Data into Train/Test (80/20 split, chronological)
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Normalize the data using MinMaxScaler for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train[['Close']])
test_scaled = scaler.transform(test[['Close']])

# Step 5: Model Building

## ARIMA Model (Traditional)
arima_model = ARIMA(train['Close'], order=(5, 1, 0))  # p=5, d=1, q=0 (can be optimized)
arima_fit = arima_model.fit()

# ARIMA Prediction
arima_pred = arima_fit.forecast(steps=len(test))
arima_pred = pd.Series(arima_pred, index=test.index)

## LSTM Model (Deep Learning)
# Prepare data for LSTM model (reshape into 3D array)
X_train = np.array([train_scaled[i-3:i, 0] for i in range(3, len(train_scaled))])
y_train = train_scaled[3:, 0]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test = np.array([test_scaled[i-3:i, 0] for i in range(3, len(test_scaled))])
y_test = test_scaled[3:, 0]
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train LSTM model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# LSTM Prediction
lstm_pred_scaled = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# Step 6: Model Evaluation
# Evaluate ARIMA model
arima_rmse = np.sqrt(mean_squared_error(test['Close'], arima_pred))
arima_mae = mean_absolute_error(test['Close'], arima_pred)
arima_r2 = r2_score(test['Close'], arima_pred)

# Evaluate LSTM model
lstm_rmse = np.sqrt(mean_squared_error(test['Close'][3:], lstm_pred))
lstm_mae = mean_absolute_error(test['Close'][3:], lstm_pred)
lstm_r2 = r2_score(test['Close'][3:], lstm_pred)

# Print Evaluation Metrics
print(f"ARIMA Model - RMSE: {arima_rmse}, MAE: {arima_mae}, R²: {arima_r2}")
print(f"LSTM Model - RMSE: {lstm_rmse}, MAE: {lstm_mae}, R²: {lstm_r2}")

# Step 7: Visualization
# Plot Actual vs Predicted for ARIMA
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Close'], color='blue', label='Actual Price')
plt.plot(test.index, arima_pred, color='red', label='ARIMA Predicted Price')
plt.title(f'{ticker} Stock Price Prediction (ARIMA)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot Actual vs Predicted for LSTM
plt.figure(figsize=(10, 6))
plt.plot(test.index[3:], test['Close'][3:], color='blue', label='Actual Price')
plt.plot(test.index[3:], lstm_pred, color='green', label='LSTM Predicted Price')
plt.title(f'{ticker} Stock Price Prediction (LSTM)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
