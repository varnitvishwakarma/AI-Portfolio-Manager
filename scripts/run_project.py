import numpy as np
from metrics import calculate_metrics
from portfolio_strategy import run_portfolio_strategy

# ============================
# Load test data
# ============================
X_test = np.load(r"D:\AI_Portfolio_Project\results\X_test.npy")
y_test = np.load(r"D:\AI_Portfolio_Project\results\y_test.npy")

# ============================
# Run LSTM portfolio
# ============================
lstm_returns, _, _ = run_portfolio_strategy(
    r"D:\AI_Portfolio_Project\results\lstm_model.keras", X_test, y_test
)
np.save(r"D:\AI_Portfolio_Project\results\daily_ret_lstm.npy", lstm_returns)

# ============================
# Run Transformer portfolio
# ============================
trans_returns, _, _ = run_portfolio_strategy(
    r"D:\AI_Portfolio_Project\results\transformer_model.keras", X_test, y_test
)
np.save(r"D:\AI_Portfolio_Project\results\daily_ret_trans.npy", trans_returns)

# ============================
# Evaluate metrics
# ============================
lstm_metrics = calculate_metrics(lstm_returns)
trans_metrics = calculate_metrics(trans_returns)

print("✅ LSTM Portfolio Metrics:")
print(f"Annualized Return: {lstm_metrics[0]:.4f}")
print(f"Annualized Volatility: {lstm_metrics[1]:.4f}")
print(f"Sharpe Ratio: {lstm_metrics[2]:.4f}")
print(f"Max Drawdown: {lstm_metrics[3]:.4f}")

print("✅ Transformer Portfolio Metrics:")
print(f"Annualized Return: {trans_metrics[0]:.4f}")
print(f"Annualized Volatility: {trans_metrics[1]:.4f}")
print(f"Sharpe Ratio: {trans_metrics[2]:.4f}")
print(f"Max Drawdown: {trans_metrics[3]:.4f}")
