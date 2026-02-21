from pydantic import BaseModel
from typing import Optional


class AnomalyScore(BaseModel):
    if_score: float
    ae_score: float
    vae_score: float
    xgb_score: float
    rf_score: float
    lr_score: float
    unsupervised_score: float
    supervised_score: float
    combined_score: float
    is_anomaly: bool


class AnomalyResult(BaseModel):
    transaction_id: str
    timestamp: str
    scores: AnomalyScore
    triggered_rules: list[str]
    transaction_amount: float
