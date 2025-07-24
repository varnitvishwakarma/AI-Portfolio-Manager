import pandas as pd
import numpy as np
import glob, os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# =======================
# Load processed data
# =======================
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

# =======================
# Prepare sequences
# =======================
n_time, n_assets, n_features = X_all.shape
X_all_flat = X_all.reshape((n_time, n_assets * n_features))
print("Flattened X_all shape:", X_all_flat.shape)

lookback = 60
X, y = [], []
for t in range(lookback, len(X_all_flat)):
    X.append(X_all_flat[t - lookback:t])
    y.append(y_all[t])
X = np.array(X)
y = np.array(y)
print("✅ Final X shape:", X.shape)
print("✅ Final y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# =======================
# Build Transformer model
# =======================
sequence_length = 60
n_features = X.shape[2]

inputs = layers.Input(shape=(sequence_length, n_features))
attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
x = layers.LayerNormalization(epsilon=1e-6)(attention + inputs)
x_ffn = layers.Dense(128, activation='relu')(x)
x_ffn = layers.Dense(n_features)(x_ffn)
x = layers.LayerNormalization(epsilon=1e-6)(x_ffn + x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(y.shape[1])(x)

model = models.Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

# =======================
# Train and evaluate
# =======================
history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.1
)

mse = model.evaluate(X_test, y_test)
print("Test MSE:", mse)

y_pred = model.predict(X_test)
print("Predicted shape:", y_pred.shape)

# =======================
# Save model
# =======================
model.save("D:/AI_Portfolio_Project/results/transformer_model.keras")
print("✅ Transformer model saved in results/")
