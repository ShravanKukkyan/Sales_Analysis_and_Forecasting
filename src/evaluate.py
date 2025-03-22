import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(y_true, y_pred):
    """Evaluates the model performance using RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"🔹 RMSE: {rmse}")
    return rmse
