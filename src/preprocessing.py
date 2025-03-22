import os
import pandas as pd

def load_data(filepath=None):
    """
    Loads sales dataset from CSV file.
    If no filepath is provided, it defaults to '../data/raw/sales_data.csv'.
    """
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
        filepath = os.path.join(script_dir, "../data/raw/sales_data.csv")

    abs_filepath = os.path.abspath(filepath)
    print(f"🔍 Checking file at: {abs_filepath}")

    if not os.path.exists(abs_filepath):
        raise FileNotFoundError(f"❌ File not found: {abs_filepath}")

    df = pd.read_csv(abs_filepath, parse_dates=["Date"])
    print("✅ Data loaded successfully!")
    return df

def preprocess_data(df):
    """
    Cleans and preprocesses the sales dataset.
    - Fills missing values
    - Adds new features if needed
    - Converts data types
    """
    print("⚙️ Preprocessing data...")

    # Handling missing values
    df.fillna(0, inplace=True)

    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract additional features (Year, Month, Day)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    print("✅ Preprocessing completed!")
    return df

if __name__ == "__main__":
    try:
        df = load_data()  # Load dataset
        print(df.head())  # Display first few rows

        df = preprocess_data(df)  # Preprocess dataset
        print(df.head())  # Display after preprocessing

        # Save processed data
        output_path = os.path.abspath("../sales_forecasting_project/data/processed/sales_data_cleaned.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
        df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved at: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
