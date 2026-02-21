"""전처리 서비스: scaler/feature_columns 로드, 원핫인코딩 + 스케일링 변환."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import settings

CATEGORIES_COL = [
    "voice_match", "is_new_account_for_user", "is_nighttime",
    "is_new_device", "vpn", "payment_method", "intent",
    "authentication", "rooting",
]
NUMERICS_COL = [
    "TransactionAmt", "hour", "avg_amount_to_bank",
    "recent_transaction_gap",
]


class Preprocessor:
    def __init__(self):
        model_dir = Path(settings.MODEL_DIR)
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        with open(model_dir / "feature_columns.json") as f:
            self.feature_columns: list[str] = json.load(f)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        feature_df = df[CATEGORIES_COL + NUMERICS_COL].copy()
        encoded = pd.get_dummies(feature_df, columns=CATEGORIES_COL, drop_first=True)
        encoded = encoded.astype(float)
        # 학습 시 컬럼 순서에 맞춰 reindex (누락 컬럼은 0으로 채움)
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
        return self.scaler.transform(encoded)
