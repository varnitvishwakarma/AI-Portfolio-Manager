import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from portfolio_strategy import run_portfolio_strategy  # make sure this is in same folder or adjust import

# ============================
# Core metrics calculation
# ============================
def calculate_metrics(daily_returns):
    ann_return = np.mean(daily_returns) * 252
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0
    cumulative = (1 + daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    return ann_return, ann_vol, sharpe, max_drawdown

# ============================
# Paths
# ============================
results_path = r"D:\AI_Portfolio_Project\results"
X_test = np.load(f"{results_path}\\X_test.npy")
y_test = np.load(f"{results_path}\\y_test.npy")

# ============================
# Run LSTM portfolio
# ============================
daily_ret_lstm, weights_lstm, _ = run_portfolio_strategy(
    f"{results_path}\\lstm_model.keras", X_test, y_test
)
np.save(f"{results_path}\\daily_ret_lstm.npy", daily_ret_lstm)
np.save(f"{results_path}\\weights_lstm.npy", weights_lstm)

# ============================
# Run Transformer portfolio
# ============================
daily_ret_trans, weights_trans, _ = run_portfolio_strategy(
    f"{results_path}\\transformer_model.keras", X_test, y_test
)
np.save(f"{results_path}\\daily_ret_trans.npy", daily_ret_trans)
np.save(f"{results_path}\\weights_trans.npy", weights_trans)

# ============================
# Performance Metrics Table
# ============================
lstm_metrics = calculate_metrics(daily_ret_lstm)
trans_metrics = calculate_metrics(daily_ret_trans)

metrics_df = pd.DataFrame([
    ["LSTM Strategy",
     f"{lstm_metrics[0]*100:.2f}%", f"{lstm_metrics[1]*100:.2f}%",
     f"{lstm_metrics[2]:.2f}", f"{lstm_metrics[3]*100:.2f}%"],
    ["Transformer Strategy",
     f"{trans_metrics[0]*100:.2f}%", f"{trans_metrics[1]*100:.2f}%",
     f"{trans_metrics[2]:.2f}", f"{trans_metrics[3]*100:.2f}%"]
], columns=["Strategy", "Annual Return (%)", "Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"])

print("\n📊 Performance Metrics Table:\n")
print(metrics_df)
metrics_df.to_csv(f"{results_path}\\performance_metrics.csv", index=False)
print(f"✅ Saved performance metrics to {results_path}\\performance_metrics.csv")

# ============================
# Prediction Performance Table
# ============================
lstm_model = load_model(f"{results_path}\\lstm_model.keras")
trans_model = load_model(f"{results_path}\\transformer_model.keras")

preds_lstm = lstm_model.predict(X_test)
preds_trans = trans_model.predict(X_test)

rmse_lstm = np.sqrt(mean_squared_error(y_test, preds_lstm))
rmse_trans = np.sqrt(mean_squared_error(y_test, preds_trans))

direction_actual = (y_test > 0).astype(int)
direction_lstm = (preds_lstm > 0).astype(int)
direction_trans = (preds_trans > 0).astype(int)

acc_lstm = (direction_actual == direction_lstm).mean()
acc_trans = (direction_actual == direction_trans).mean()

pred_df = pd.DataFrame([
    ["LSTM", f"{rmse_lstm*100:.2f}%", f"{acc_lstm*100:.2f}%"],
    ["Transformer", f"{rmse_trans*100:.2f}%", f"{acc_trans*100:.2f}%"]
], columns=["Model", "RMSE (% daily return)", "Directional Accuracy (%)"])

print("\n📊 Prediction Performance Table:\n")
print(pred_df)
pred_df.to_csv(f"{results_path}\\prediction_performance.csv", index=False)
print(f"✅ Saved prediction performance to {results_path}\\prediction_performance.csv")

print("🎯 All evaluation steps completed successfully!")
