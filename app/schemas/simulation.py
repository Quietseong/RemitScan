from pydantic import BaseModel


class SimulationStatus(BaseModel):
    running: bool
    total_processed: int
    anomalies_detected: int
    interval_seconds: float
