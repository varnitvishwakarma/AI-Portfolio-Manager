import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import glob, os

# ===== Load processed CSVs and align by common dates =====
processed_path = r"D:\AI_Portfolio_Project\data\processed"
files = glob.glob(os.path.join(processed_path, "*.csv"))

data_dict = {}
for file in files:
    asset_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        raise ValueError(f"{asset_name} CSV has no Date column!")
    data_dict[asset_name] = df

common_index = None
for asset, df in data_dict.items():
    if common_index is None:
        common_index = df.index
    else:
        common_index = common_index.intersection(df.index)

print("✅ Common dates count:", len(common_index))
for asset in data_dict:
    data_dict[asset] = data_dict[asset].loc[common_index]

# ===== Build feature tensor and target tensor =====
features = ['LogReturn', 'MA_5', 'MA_21', 'Volatility_21', 'RSI']
asset_names = list(data_dict.keys())

X_all = []
for f in features:
    temp = []
    for asset in asset_names:
        temp.append(data_dict[asset][f].values)
    temp = np.array(temp).T
    X_all.append(temp)
X_all = np.stack(X_all, axis=2)
print("✅ Feature tensor shape:", X_all.shape)

y_all = []
for asset in asset_names:
    y_all.append(data_dict[asset]['LogReturn'].values)
y_all = np.array(y_all).T
print("✅ Target tensor shape:", y_all.shape)

# ===== Flatten features and create sliding windows =====
n_time, n_assets, n_features = X_all.shape
X_all_flat = X_all.reshape((n_time, n_assets * n_features))
print("Flattened X_all shape:", X_all_flat.shape)

lookback = 60  # ✅ unified lookback
X, y = [], []
for t in range(lookback, len(X_all_flat)):
    X.append(X_all_flat[t-lookback:t])
    y.append(y_all[t])
X = np.array(X)
y = np.array(y)
print("✅ Final X shape:", X.shape)
print("✅ Final y shape:", y.shape)

# ===== Train-test split =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ✅ Save for portfolio evaluation
np.save(r"D:\AI_Portfolio_Project\results\X_test.npy", X_test)
np.save(r"D:\AI_Portfolio_Project\results\y_test.npy", y_test)
print("✅ Saved X_test and y_test with lookback=60")

# ===== Build LSTM model =====
n_timesteps = X_train.shape[1]
n_features = X_train.shape[2]
n_outputs = y_train.shape[1]

model = Sequential()
model.add(LSTM(50, input_shape=(n_timesteps, n_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(n_outputs))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

# ===== Train LSTM model =====
history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.1
)

# ===== Evaluate and save =====
mse = model.evaluate(X_test, y_test)
print("Test MSE:", mse)

y_pred = model.predict(X_test)
print("Predicted shape:", y_pred.shape)

model.save("D:/AI_Portfolio_Project/results/lstm_model.keras")
print("✅ LSTM model saved in results/")
