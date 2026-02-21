"""
합성 데이터 생성 -> 전처리 -> 6개 모델 학습 -> 아티팩트 저장
출력: scaler.pkl, feature_columns.json, isolation_forest.joblib,
      autoencoder.pt, vae.pt, xgboost.joblib, random_forest.joblib,
      logistic_regression.joblib
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.models.autoencoder import Autoencoder
from app.models.vae import VAE, vae_loss

# ── 설정 ──────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000
FRAUD_RATIO = 0.05
OUTLIER_RATIO = 0.02
EPOCHS = 50
LR = 0.001

CATEGORIES_COL = [
    "voice_match", "is_new_account_for_user", "is_nighttime",
    "is_new_device", "vpn", "payment_method", "intent",
    "authentication", "rooting",
]
NUMERICS_COL = [
    "TransactionAmt", "hour", "avg_amount_to_bank",
    "recent_transaction_gap",
]


# ── 합성 데이터 생성 ─────────────────────────────────
def generate_synthetic_data(n: int = N_SAMPLES) -> pd.DataFrame:
    np.random.seed(SEED)
    rows = []
    start = datetime(2025, 1, 1)
    for i in range(n):
        is_fraud = np.random.rand() < FRAUD_RATIO
        amount = (
            np.random.exponential(scale=300_000)
            if is_fraud
            else np.random.exponential(scale=30_000)
        )
        if np.random.rand() < OUTLIER_RATIO:
            amount *= 50

        dt = start + timedelta(seconds=int(np.random.uniform(0, 100 * 86400)))
        hour = dt.hour

        rows.append(
            {
                "TransactionDT": dt.isoformat(),
                "TransactionAmt": round(amount, 2),
                "hour": hour,
                "avg_amount_to_bank": round(np.random.normal(250_000, 50_000), 2),
                "recent_transaction_gap": round(np.random.exponential(3600), 2),
                "payment_method": np.random.choice(
                    ["P01", "P02", "P03", "P04", "P05"]
                ),
                "intent": np.random.choice(["T01", "T02", "T03", "T04", "T05"]),
                "authentication": np.random.choice(
                    ["A01", "A02", "A03", "A04", "A05", "A06"]
                ),
                "voice_match": np.random.choice(["Y", "N"]),
                "is_new_account_for_user": int(np.random.rand() < 0.15),
                "is_nighttime": 1 if 0 <= hour < 6 else 0,
                "is_new_device": int(np.random.rand() < 0.2),
                "vpn": int(np.random.rand() < 0.25),
                "rooting": int(np.random.rand() < 0.1),
                "is_fraud": int(is_fraud),
            }
        )
    return pd.DataFrame(rows)


# ── 전처리 ────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    feature_df = df[CATEGORIES_COL + NUMERICS_COL].copy()
    encoded = pd.get_dummies(feature_df, columns=CATEGORIES_COL, drop_first=True)
    encoded = encoded.astype(float)

    feature_columns = list(encoded.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(encoded)
    return X_scaled, scaler, feature_columns


# ── 학습 루프 ─────────────────────────────────────────
def train_all(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] 합성 데이터 생성...")
    df = generate_synthetic_data()
    y = df["is_fraud"].values

    print("[2/7] 전처리...")
    X, scaler, feature_columns = preprocess(df)
    input_dim = X.shape[1]
    print(f"       input_dim={input_dim}, features={len(feature_columns)}")

    # 저장: scaler, feature_columns
    joblib.dump(scaler, out / "scaler.pkl")
    with open(out / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, ensure_ascii=False)

    # ── Isolation Forest ──
    print("[3/7] Isolation Forest 학습...")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=SEED)
    iso.fit(X)
    joblib.dump(iso, out / "isolation_forest.joblib")

    # ── Autoencoder ──
    print("[4/7] Autoencoder 학습...")
    X_tensor = torch.FloatTensor(X)
    ae = Autoencoder(input_dim)
    optimizer_ae = optim.Adam(ae.parameters(), lr=LR)
    criterion = nn.MSELoss()
    ae.train()
    for epoch in range(EPOCHS):
        optimizer_ae.zero_grad()
        out_ae = ae(X_tensor)
        loss = criterion(out_ae, X_tensor)
        loss.backward()
        optimizer_ae.step()
        if epoch % 10 == 0:
            print(f"       Epoch {epoch} loss={loss.item():.4f}")
    torch.save({"input_dim": input_dim, "state_dict": ae.state_dict()}, out / "autoencoder.pt")

    # ── VAE ──
    print("[5/7] VAE 학습...")
    vae = VAE(input_dim)
    optimizer_vae = optim.Adam(vae.parameters(), lr=LR)
    vae.train()
    for epoch in range(EPOCHS):
        optimizer_vae.zero_grad()
        recon, mu, logvar = vae(X_tensor)
        loss = vae_loss(X_tensor, recon, mu, logvar)
        loss.backward()
        optimizer_vae.step()
        if epoch % 10 == 0:
            print(f"       Epoch {epoch} loss={loss.item():.4f}")
    torch.save({"input_dim": input_dim, "state_dict": vae.state_dict()}, out / "vae.pt")

    # ── XGBoost ──
    print("[6/7] XGBoost 학습...")
    xgb = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=(1 - FRAUD_RATIO) / FRAUD_RATIO,
        random_state=SEED,
        eval_metric="logloss",
    )
    xgb.fit(X, y)
    joblib.dump(xgb, out / "xgboost.joblib")

    # ── Random Forest + Logistic Regression ──
    print("[7/7] Random Forest & Logistic Regression 학습...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X, y)
    joblib.dump(rf, out / "random_forest.joblib")

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X, y)
    joblib.dump(lr, out / "logistic_regression.joblib")

    print(f"\n모든 아티팩트가 {out.resolve()} 에 저장됨")
    for p in sorted(out.iterdir()):
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="models/")
    args = parser.parse_args()
    train_all(args.output_dir)
