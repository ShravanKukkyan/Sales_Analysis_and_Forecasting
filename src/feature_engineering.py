import os
import pandas as pd

# Load cleaned data
def load_cleaned_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    filepath = os.path.join(script_dir, "../data/processed/sales_data_cleaned.csv")
    
    print(f"🔍 Loading data from: {os.path.abspath(filepath)}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Processed file not found: {os.path.abspath(filepath)}")
    
    df = pd.read_csv(filepath, parse_dates=["Date"])
    print("✅ Cleaned data loaded successfully!")
    
    return df

# Add new features
def add_features(df):
    print("⚙️ Adding new features...")

    # Extract time-based features
    df["Weekday"] = df["Date"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df["Quarter"] = df["Date"].dt.quarter
    df["YearMonth"] = df["Date"].dt.to_period("M")

    # Moving averages (Rolling window features)
    df["Sales_MA_7"] = df["Sales"].rolling(window=7, min_periods=1).mean()  # 7-day MA
    df["Sales_MA_30"] = df["Sales"].rolling(window=30, min_periods=1).mean()  # 30-day MA

    # Lag features (Previous sales values)
    df["Sales_Lag_1"] = df["Sales"].shift(1)
    df["Sales_Lag_7"] = df["Sales"].shift(7)

    print("✅ Feature engineering completed!")

    return df

# Save processed data
def save_data(df):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "../data/processed/sales_data_features.csv")

    df.to_csv(save_path, index=False)
    print(f"✅ Processed data saved at: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    df = load_cleaned_data()
    df = add_features(df)
    print(df.head())  # Preview the data
    save_data(df)
