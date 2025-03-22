import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

def train_arima(df):
    """Trains an ARIMA model for sales forecasting."""
    df = df.groupby("Date")["Sales"].sum()
    
    model = ARIMA(df, order=(5,1,0))  # (p,d,q) parameters
    model_fit = model.fit()

    # Save model
    with open("../models/arima_model.pkl", "wb") as f:
        pickle.dump(model_fit, f)
    
    print("✅ ARIMA model trained and saved!")

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/sales_cleaned.csv", parse_dates=["Date"])
    train_arima(df)
