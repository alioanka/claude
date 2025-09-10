# ml/offline_train.py
import os, joblib, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path("data")           # adjust if you store CSV elsewhere
MODEL_DIR = Path("storage/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["BTC/USDT","ETH/USDT","SOL/USDT"]  # or read from config

FEATURES = [
  "open","high","low","close","volume",
  "sma_20","sma_50","ema_12","ema_26",
  "rsi","macd","macd_signal","macd_hist",
  "bb_upper","bb_middle","bb_lower","bb_width",
  "volume_sma","volume_ratio",
  "price_change_1h","price_change_4h","price_change_24h",
  "volatility_1h","volatility_4h","volatility_24h"
]

def load_symbol_df(sym):
    # expect a features CSV you log during paper mode (one row per bar)
    # with 'label' column: 1 = up next period, 0 = down (or your exit rule)
    p = DATA_DIR / f"features_{sym.replace('/','_')}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    df = df.dropna()
    ok = [c for c in FEATURES if c in df.columns]
    if 'label' not in df or not ok: return None
    X, y = df[ok].values, df['label'].values
    return X, y

def train_one(sym):
    data = load_symbol_df(sym)
    if not data: 
        print(f"No data for {sym}")
        return
    X, y = data
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42,
        class_weight="balanced_subsample"
    )
    clf.fit(X, y)
    out = MODEL_DIR / f"{sym.replace('/','_')}_rf.joblib"
    joblib.dump({"model": clf, "features": FEATURES}, out)
    print(f"Saved {out}")

if __name__ == "__main__":
    for s in SYMBOLS:
        train_one(s)
