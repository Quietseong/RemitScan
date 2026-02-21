"""드리프트 탐지: PSI/KS/JS/Wasserstein 실시간 계산, deque 히스토리."""
import asyncio
from collections import deque
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from app.config import settings
from app.schemas.drift import DriftMetrics


class DriftDetector:
    def __init__(self):
        self.reference: np.ndarray | None = None
        self.history: deque[DriftMetrics] = deque(maxlen=settings.DRIFT_HISTORY_SIZE)
        self._lock = asyncio.Lock()

    async def set_reference(self, scores: np.ndarray):
        async with self._lock:
            self.reference = scores.copy()

    async def compute(self, scores: np.ndarray) -> DriftMetrics | None:
        async with self._lock:
            if self.reference is None:
                self.reference = scores.copy()
                return None

            ts = datetime.now(timezone.utc).isoformat()
            psi = _calculate_psi(self.reference, scores)
            ks_stat, ks_pval = ks_2samp(self.reference, scores)
            js_div = _calculate_js(self.reference, scores)
            wd = float(wasserstein_distance(self.reference, scores))

            m = DriftMetrics(
                timestamp=ts,
                psi=float(psi),
                ks_stat=float(ks_stat),
                ks_pval=float(ks_pval),
                js_div=float(js_div),
                wasserstein=wd,
            )
            self.history.append(m)
            return m

    def get_latest(self) -> DriftMetrics | None:
        return self.history[-1] if self.history else None

    def get_history(self) -> list[DriftMetrics]:
        return list(self.history)


def _calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10, eps: float = 1e-6) -> float:
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
    if len(breakpoints) < 2:
        return 0.0
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    return float(np.sum((act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps))))


def _calculate_js(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    all_data = np.concatenate([expected, actual])
    bin_edges = np.linspace(all_data.min(), all_data.max(), bins + 1)
    exp_hist = np.histogram(expected, bins=bin_edges, density=True)[0] + 1e-10
    act_hist = np.histogram(actual, bins=bin_edges, density=True)[0] + 1e-10
    return float(jensenshannon(exp_hist, act_hist) ** 2)
