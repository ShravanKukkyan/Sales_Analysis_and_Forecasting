import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Plot sales trends over time
def plot_sales_trends(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x="Date", y="Sales", data=df, marker="o", linestyle="-", color="b")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("📈 Sales Trend Over Time")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Plot sales by category
def plot_sales_by_category(df):
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Category", y="Sales", data=df, palette="viridis")
    plt.xlabel("Product Category")
    plt.ylabel("Total Sales")
    plt.title("🛒 Total Sales by Category")
    plt.xticks(rotation=30)
    plt.show()

# Monthly sales trend
def plot_monthly_sales(df):
    monthly_sales = df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    monthly_sales["Month_Year"] = monthly_sales["Year"].astype(str) + "-" + monthly_sales["Month"].astype(str)

    plt.figure(figsize=(12, 5))
    sns.lineplot(x="Month_Year", y="Sales", data=monthly_sales, marker="o", linestyle="-", color="g")
    plt.xlabel("Month-Year")
    plt.ylabel("Total Sales")
    plt.title("📊 Monthly Sales Trend")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    df = load_cleaned_data()

    print(df.head())  # Show first few rows

    plot_sales_trends(df)
    plot_sales_by_category(df)
    plot_monthly_sales(df)
