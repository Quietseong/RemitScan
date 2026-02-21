"""시뮬레이터: 더미 거래 생성 + asyncio.Task 백그라운드 루프."""
import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.config import settings
from app.schemas.anomaly import AnomalyResult, AnomalyScore
from app.services.drift_detector import DriftDetector
from app.services.ml_pipeline import MLPipeline
from app.services.preprocessing import Preprocessor
from app.services import rule_engine
from app import metrics as m


class Simulator:
    def __init__(
        self,
        preprocessor: Preprocessor,
        pipeline: MLPipeline,
        drift_detector: DriftDetector,
    ):
        self.preprocessor = preprocessor
        self.pipeline = pipeline
        self.drift_detector = drift_detector

        self._task: asyncio.Task | None = None
        self._running = False
        self.total_processed = 0
        self.anomalies_detected = 0
        self.results: deque[AnomalyResult] = deque(maxlen=500)
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._running

    async def start(self):
        if self._running:
            return
        self._running = True
        m.fds_simulation_running.set(1)
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        m.fds_simulation_running.set(0)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self):
        while self._running:
            try:
                df = _generate_batch(settings.BATCH_SIZE)
                X = self.preprocessor.transform(df)
                preds = self.pipeline.predict(X)

                # 규칙 엔진
                triggered_rules = rule_engine.evaluate(df)

                # Prometheus 메트릭 갱신
                m.fds_transactions_total.inc(len(df))
                m.fds_unsupervised_score.set(float(preds["unsupervised_score"].mean()))
                m.fds_supervised_score.set(float(preds["supervised_score"].mean()))
                m.fds_anomaly_threshold.set(self.pipeline.threshold)
                m.fds_if_score.set(float(preds["if_score"].mean()))
                m.fds_ae_score.set(float(preds["ae_score"].mean()))
                m.fds_vae_score.set(float(preds["vae_score"].mean()))
                m.fds_xgb_score.set(float(preds["xgb_score"].mean()))
                m.fds_rf_score.set(float(preds["rf_score"].mean()))
                m.fds_lr_score.set(float(preds["lr_score"].mean()))
                for score in preds["combined_score"]:
                    m.fds_combined_score.observe(float(score))

                # 드리프트 계산 + 메트릭
                drift_result = await self.drift_detector.compute(preds["combined_score"])
                if drift_result:
                    m.fds_drift_psi.set(drift_result.psi or 0)
                    m.fds_drift_ks_pval.set(drift_result.ks_pval or 0)
                    m.fds_drift_js.set(drift_result.js_div or 0)
                    m.fds_drift_wasserstein.set(drift_result.wasserstein or 0)

                # 결과 수집
                async with self._lock:
                    for i in range(len(df)):
                        is_anom = bool(preds["is_anomaly"][i])
                        result = AnomalyResult(
                            transaction_id=str(uuid.uuid4()),
                            timestamp=df.iloc[i]["TransactionDT"],
                            scores=AnomalyScore(
                                if_score=float(preds["if_score"][i]),
                                ae_score=float(preds["ae_score"][i]),
                                vae_score=float(preds["vae_score"][i]),
                                xgb_score=float(preds["xgb_score"][i]),
                                rf_score=float(preds["rf_score"][i]),
                                lr_score=float(preds["lr_score"][i]),
                                unsupervised_score=float(preds["unsupervised_score"][i]),
                                supervised_score=float(preds["supervised_score"][i]),
                                combined_score=float(preds["combined_score"][i]),
                                is_anomaly=is_anom,
                            ),
                            triggered_rules=triggered_rules[i],
                            transaction_amount=float(df.iloc[i]["TransactionAmt"]),
                        )
                        self.results.append(result)
                        self.total_processed += 1
                        if is_anom or triggered_rules[i]:
                            self.anomalies_detected += 1
                            m.fds_anomalies_total.inc()
                        for rule_name in triggered_rules[i]:
                            m.fds_rules_triggered_total.labels(rule_name=rule_name).inc()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Simulator] error: {e}")

            await asyncio.sleep(settings.SIMULATION_INTERVAL)


def _generate_batch(n: int) -> pd.DataFrame:
    """더미 거래 배치 생성."""
    rows = []
    now = datetime.now(timezone.utc)
    for _ in range(n):
        is_fraud = np.random.rand() < 0.05
        amount = (
            np.random.exponential(300_000) if is_fraud
            else np.random.exponential(30_000)
        )
        hour = now.hour

        rows.append({
            "TransactionDT": now.isoformat(),
            "user_id": f"U{np.random.randint(1000, 9999)}",
            "recipient": f"R{np.random.randint(1000, 9999)}",
            "device_id": f"D{np.random.randint(1000, 9999)}",
            "receiver_bank": f"B{np.random.randint(10, 99)}",
            "receiver_account": f"A{np.random.randint(100000, 999999)}",
            "ip_address": f"{np.random.randint(1,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}",
            "app_version": "2.1.0",
            "region": np.random.choice(["KR", "US", "JP", "VN", "PH"]),
            "TransactionAmt": round(amount, 2),
            "hour": hour,
            "avg_amount_to_bank": round(np.random.normal(250_000, 50_000), 2),
            "recent_transaction_gap": round(np.random.exponential(3600), 2),
            "payment_method": np.random.choice(["P01", "P02", "P03", "P04", "P05"]),
            "intent": np.random.choice(["T01", "T02", "T03", "T04", "T05"]),
            "authentication": np.random.choice(["A01", "A02", "A03", "A04", "A05", "A06"]),
            "voice_match": np.random.choice(["Y", "N"]),
            "is_new_account_for_user": int(np.random.rand() < 0.15),
            "is_nighttime": 1 if 0 <= hour < 6 else 0,
            "is_new_device": int(np.random.rand() < 0.2),
            "vpn": int(np.random.rand() < 0.25),
            "rooting": int(np.random.rand() < 0.1),
        })
    return pd.DataFrame(rows)
