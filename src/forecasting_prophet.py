import pandas as pd
from prophet import Prophet
import pickle

def train_prophet(df):
    """Trains a Facebook Prophet model."""
    df = df.groupby("Date")["Sales"].sum().reset_index()
    df.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df)

    # Save model
    with open("../models/prophet_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Prophet model trained and saved!")

if __name__ == "__main__":
    df = pd.read_csv("../data/processed/sales_cleaned.csv", parse_dates=["Date"])
    train_prophet(df)
