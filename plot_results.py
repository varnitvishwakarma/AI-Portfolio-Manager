import numpy as np
import matplotlib.pyplot as plt
import os

# ===== Load daily returns =====
results_path = r"D:\AI_Portfolio_Project\results"
lstm_returns = np.load(os.path.join(results_path, "daily_ret_lstm.npy"))
trans_returns = np.load(os.path.join(results_path, "daily_ret_trans.npy"))

# ===== Compute cumulative returns =====
def cumulative_curve(daily_returns):
    return np.cumprod(1 + daily_returns)

lstm_curve = cumulative_curve(lstm_returns)
trans_curve = cumulative_curve(trans_returns)

# ===== Compute drawdowns =====
def drawdown_curve(curve):
    running_max = np.maximum.accumulate(curve)
    return (curve - running_max) / running_max

lstm_dd = drawdown_curve(lstm_curve)
trans_dd = drawdown_curve(trans_curve)

# ===== Plot equity curves =====
plt.figure(figsize=(10,6))
plt.plot(lstm_curve, label="LSTM Portfolio")
plt.plot(trans_curve, label="Transformer Portfolio")
plt.title("Equity Curves (Cumulative Returns)")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "equity_curves.png"))
plt.show()

# ===== Plot drawdowns =====
plt.figure(figsize=(10,6))
plt.plot(lstm_dd, label="LSTM Drawdown")
plt.plot(trans_dd, label="Transformer Drawdown")
plt.title("Drawdowns Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_path, "drawdowns.png"))
plt.show()
