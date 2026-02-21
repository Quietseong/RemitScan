"""ML 파이프라인: 6개 모델 로드, 앙상블 추론 (0.5x비지도 + 0.5x지도)."""
from pathlib import Path

import joblib
import numpy as np
import torch

from app.config import settings
from app.models.autoencoder import Autoencoder
from app.models.vae import VAE


class MLPipeline:
    def __init__(self):
        model_dir = Path(settings.MODEL_DIR)

        # 비지도 모델
        self.iso = joblib.load(model_dir / "isolation_forest.joblib")

        ckpt_ae = torch.load(model_dir / "autoencoder.pt", map_location="cpu", weights_only=True)
        self.ae = Autoencoder(ckpt_ae["input_dim"])
        self.ae.load_state_dict(ckpt_ae["state_dict"])
        self.ae.eval()

        ckpt_vae = torch.load(model_dir / "vae.pt", map_location="cpu", weights_only=True)
        self.vae = VAE(ckpt_vae["input_dim"])
        self.vae.load_state_dict(ckpt_vae["state_dict"])
        self.vae.eval()

        # 지도 모델
        self.xgb = joblib.load(model_dir / "xgboost.joblib")
        self.rf = joblib.load(model_dir / "random_forest.joblib")
        self.lr = joblib.load(model_dir / "logistic_regression.joblib")

        # 앙상블 가중치
        self.w_unsup = settings.UNSUPERVISED_WEIGHT
        self.w_sup = settings.SUPERVISED_WEIGHT

        # 이상치 임계값 (초기값, 시뮬레이션 누적 시 갱신)
        self.threshold: float = 0.5

    def predict(self, X: np.ndarray) -> dict:
        """배치 추론. X: (n, features) 스케일링 완료 배열."""
        n = X.shape[0]

        # Isolation Forest: anomaly_score는 -score_samples 로 양수화
        if_raw = -self.iso.score_samples(X)
        if_score = _rank_pct(if_raw)

        # Autoencoder: MSE reconstruction error
        with torch.no_grad():
            t = torch.FloatTensor(X)
            ae_recon = self.ae(t)
            ae_raw = ((t - ae_recon) ** 2).mean(dim=1).numpy()
        ae_score = _rank_pct(ae_raw)

        # VAE: MSE reconstruction error
        with torch.no_grad():
            vae_recon, _, _ = self.vae(t)
            vae_raw = ((t - vae_recon) ** 2).mean(dim=1).numpy()
        vae_score = _rank_pct(vae_raw)

        # 비지도 앙상블 (0.33 IF + 0.33 AE + 0.34 VAE)
        unsup = 0.33 * if_score + 0.33 * ae_score + 0.34 * vae_score

        # 지도 모델: 이상 확률
        xgb_prob = self.xgb.predict_proba(X)[:, 1]
        rf_prob = self.rf.predict_proba(X)[:, 1]
        lr_prob = self.lr.predict_proba(X)[:, 1]
        sup = (xgb_prob + rf_prob + lr_prob) / 3.0

        # 최종 결합
        combined = self.w_unsup * unsup + self.w_sup * sup

        # 동적 임계값: 현재 배치 95th percentile
        if n >= 10:
            self.threshold = float(np.percentile(combined, settings.ANOMALY_THRESHOLD_PERCENTILE))

        is_anomaly = combined > self.threshold

        return {
            "if_score": if_score,
            "ae_score": ae_score,
            "vae_score": vae_score,
            "xgb_score": xgb_prob,
            "rf_score": rf_prob,
            "lr_score": lr_prob,
            "unsupervised_score": unsup,
            "supervised_score": sup,
            "combined_score": combined,
            "is_anomaly": is_anomaly,
        }


def _rank_pct(arr: np.ndarray) -> np.ndarray:
    """랭크 기반 백분위 정규화 [0,1]."""
    from scipy.stats import rankdata
    return rankdata(arr, method="average") / len(arr)
