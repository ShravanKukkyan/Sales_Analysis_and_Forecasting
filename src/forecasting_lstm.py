import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

def prepare_data(df):
    """Prepares data for LSTM model."""
    df = df.groupby("Date")["Sales"].sum()
    df = df.values.reshape(-1, 1)

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(10, len(df_scaled)):  # Using past 10 days to predict next
        X.append(df_scaled[i-10:i])
        y.append(df_scaled[i])

    return np.array(X), np.array(y), scaler

def train_lstm(X, y):
    """Trains an LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=32)

    model.save("../models/lstm_model.h5")
    print("✅ LSTM model trained and saved!")

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/sales_cleaned.csv", parse_dates=["Date"])
    X, y, scaler = prepare_data(df)
    train_lstm(X, y)
    
    with open("../models/lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
