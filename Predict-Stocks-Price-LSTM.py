import yfinance as yf
import pandas as pd

ticker = 'VOO'
data = yf.download(ticker,start='2014-01-01', end='2024-12-31')

df = data[['Close']]
# print(df.head())

# Data Normalization (Scaling all the closed prices between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Sliding window
import numpy as np

X = []
y = []

# Set the window-size to 60 days
for i in range(60, len(scaled_data)):
    # Features, historical Data of 60 days, from i-60 to i
    X.append(scaled_data[i-60:i, 0])
    # Target, the price of the next days, from i to the end
    y.append(scaled_data[i,0])

# Convert list to array
X = np.array(X)
y = np.array(y)
# print (X.shape)
# print(y.shape)

# Reshape X to [Samples, Time Steps, Features]
X = np.reshape(X,(X.shape[0],X.shape[1],1))

#--LSTM Model--
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# If there is another LSTM layer following, return_sequence = True
model.add(LSTM(units=50, return_sequences = True, input_shape = (X.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = model.fit(X,y,epochs=20, batch_size=32)

# Prediction
predictions = model.predict(X)

# 2. Transform X to real price
# Through Inverse_transform
predictions = scaler.inverse_transform(predictions)

# Transform y to real value
# reshape y from array to table, -1 mean infer rows, 1 mean 1 column
y_actual = scaler.inverse_transform(y.reshape(-1, 1))

import matplotlib.pyplot as plt

# establish the figure sizze
plt.figure(figsize=(14, 6))

# Real Price (Blue)
# df.index[60:] : offset the index by 60 to align with the predictions
plt.plot(df.index[60:], y_actual, color='blue', label='Actual Price')

# Predicted Price (Red)
plt.plot(df.index[60:], predictions, color='red', label='Predicted Price')


plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.show()