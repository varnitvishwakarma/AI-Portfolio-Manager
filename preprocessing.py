import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator

os.makedirs("data/processed", exist_ok=True)

raw_path = "data/raw/"
processed_path = "data/processed/"
files = ["GSPC.csv", "FTSE.csv", "N225.csv", "EEM.csv", "GC=F.csv", "TNX.csv"]

def add_features(df):
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_5'] = df['LogReturn'].rolling(window=5).mean()
    df['MA_21'] = df['LogReturn'].rolling(window=21).mean()
    df['Volatility_21'] = df['LogReturn'].rolling(window=21).std()
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()
    return df

for file in files:
    path = os.path.join(raw_path, file)
    df = pd.read_csv(path)
    df = df.rename(columns=str.title)
    df = add_features(df)
    df = df.dropna().reset_index(drop=True)
    feature_cols = ['LogReturn','MA_5','MA_21','Volatility_21','RSI']
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    out_path = os.path.join(processed_path, file)
    df.to_csv(out_path, index=False)
    print(f"✅ Processed and saved: {out_path}")

print("🎯 Preprocessing completed. All processed files are in data/processed/")
