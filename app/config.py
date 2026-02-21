from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    MODEL_DIR: str = str(Path(__file__).resolve().parent.parent / "models")
    SIMULATION_INTERVAL: float = 5.0
    BATCH_SIZE: int = 50
    ANOMALY_THRESHOLD_PERCENTILE: float = 95.0
    DRIFT_HISTORY_SIZE: int = 100
    UNSUPERVISED_WEIGHT: float = 0.5
    SUPERVISED_WEIGHT: float = 0.5

    model_config = {"env_prefix": "FDS_"}


settings = Settings()
