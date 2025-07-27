import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ============================
# Load model & test data
# ============================
def load_data(X_test_path, y_test_path):
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    return X_test, y_test

def load_trained_model(model_path):
    return load_model(model_path)

# ============================
# Convert predictions to weights
# ============================
def predictions_to_weights(preds):
    preds = np.maximum(preds, 0)
    row_sums = preds.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    weights = preds / row_sums
    return weights

# ============================
# Calculate portfolio returns
# ============================
def portfolio_returns(weights, actual_returns):
    daily_returns = np.sum(weights * actual_returns, axis=1)
    return daily_returns

# ============================
# Main function
# ============================
def run_portfolio_strategy(model_path, X_test, y_test):
    model = load_trained_model(model_path)
    preds = model.predict(X_test)
    weights = predictions_to_weights(preds)
    daily_ret = portfolio_returns(weights, y_test)
    return daily_ret, weights, preds

if __name__ == "__main__":
    # Load test data
    X_test = np.load("D:/AI_Portfolio_Project/results/X_test.npy")
    y_test = np.load("D:/AI_Portfolio_Project/results/y_test.npy")

    # Run LSTM model
    daily_ret_lstm, weights_lstm, preds_lstm = run_portfolio_strategy(
        "D:/AI_Portfolio_Project/results/lstm_model.keras", X_test, y_test
    )

    # Run Transformer model
    daily_ret_trans, weights_trans, preds_trans = run_portfolio_strategy(
        "D:/AI_Portfolio_Project/results/transformer_model.keras", X_test, y_test
    )

    # Save daily returns
    np.save("D:/AI_Portfolio_Project/results/daily_ret_lstm.npy", daily_ret_lstm)
    np.save("D:/AI_Portfolio_Project/results/daily_ret_trans.npy", daily_ret_trans)

    # ✅ Save weights for allocation plotting
    np.save("D:/AI_Portfolio_Project/results/weights_lstm.npy", weights_lstm)
    np.save("D:/AI_Portfolio_Project/results/weights_trans.npy", weights_trans)

    print("✅ Portfolio strategy run completed!")
    print("✅ Saved weights_lstm.npy and weights_trans.npy for allocation plots.")
