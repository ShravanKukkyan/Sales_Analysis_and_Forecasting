from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load models
with open("../models/arima_model.pkl", "rb") as f:
    arima_model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts future sales based on input date range."""
    data = request.get_json()
    steps = data.get("steps", 30)  # Default 30 days
    
    forecast = arima_model.forecast(steps=steps)
    
    return jsonify({"forecast": forecast.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
