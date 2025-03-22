import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Load feature-engineered data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
    filepath = os.path.join(script_dir, "../data/processed/sales_data_features.csv")
    
    print(f"🔍 Loading data from: {os.path.abspath(filepath)}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Processed file not found: {os.path.abspath(filepath)}")
    
    df = pd.read_csv(filepath, parse_dates=["Date"])
    print("✅ Data loaded successfully!")
    
    return df

# Train ARIMA model
def train_arima(df):
    print("📊 Training ARIMA model...")
    
    # Set Date as index
    df.set_index("Date", inplace=True)

    # Fit ARIMA model (p=5, d=1, q=0 is a simple baseline)
    model = ARIMA(df["Sales"], order=(5, 1, 0))  # p=5, d=1, q=0
    model_fit = model.fit()
    
    print("✅ ARIMA model trained successfully!")
    
    return model_fit

# Make future predictions
def forecast_sales(model_fit, df, steps=30):
    print(f"📅 Forecasting next {steps} days...")
    
    # Forecast next 30 days
    forecast = model_fit.forecast(steps=steps)
    
    # Create forecast DataFrame
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Sales": forecast})
    
    print("✅ Forecasting completed!")
    
    return forecast_df

# Plot actual vs forecasted sales
def plot_forecast(df, forecast_df):
    plt.figure(figsize=(10, 5))
    
    # Plot historical sales
    plt.plot(df.index, df["Sales"], label="Actual Sales", color="blue")
    
    # Plot forecasted sales
    plt.plot(forecast_df["Date"], forecast_df["Predicted_Sales"], label="Forecasted Sales", linestyle="dashed", color="red")
    
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Forecasting using ARIMA")
    plt.legend()
    
    # Save the plot
    save_path = os.path.join(os.path.dirname(__file__), "../reports/arima_forecast.png")
    plt.savefig(save_path)
    
    print(f"📊 Forecast plot saved at: {os.path.abspath(save_path)}")
    plt.show()

# Main execution
if __name__ == "__main__":
    df = load_data()
    model_fit = train_arima(df)
    forecast_df = forecast_sales(model_fit, df)
    
    print(forecast_df.head())  # Show first few predictions
    
    plot_forecast(df, forecast_df)
