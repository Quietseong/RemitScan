"""Prometheus 메트릭 정의: fds_ 접두사 17개."""
from prometheus_client import Counter, Gauge, Histogram

# ── 거래 처리 ──
fds_transactions_total = Counter(
    "fds_transactions_total", "총 처리 거래 수"
)
fds_anomalies_total = Counter(
    "fds_anomalies_total", "탐지된 이상 거래 수"
)
fds_rules_triggered_total = Counter(
    "fds_rules_triggered_total", "규칙 트리거 총 횟수", ["rule_name"]
)

# ── 앙상블 점수 ──
fds_combined_score = Histogram(
    "fds_combined_score", "최종 결합 점수 분포",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
fds_unsupervised_score = Gauge(
    "fds_unsupervised_score_avg", "비지도 점수 평균 (최근 배치)"
)
fds_supervised_score = Gauge(
    "fds_supervised_score_avg", "지도 점수 평균 (최근 배치)"
)
fds_anomaly_threshold = Gauge(
    "fds_anomaly_threshold", "현재 이상 탐지 임계값"
)

# ── 개별 모델 점수 ──
fds_if_score = Gauge("fds_if_score_avg", "Isolation Forest 평균 점수")
fds_ae_score = Gauge("fds_ae_score_avg", "Autoencoder 평균 점수")
fds_vae_score = Gauge("fds_vae_score_avg", "VAE 평균 점수")
fds_xgb_score = Gauge("fds_xgb_score_avg", "XGBoost 평균 점수")
fds_rf_score = Gauge("fds_rf_score_avg", "Random Forest 평균 점수")
fds_lr_score = Gauge("fds_lr_score_avg", "Logistic Regression 평균 점수")

# ── 드리프트 메트릭 ──
fds_drift_psi = Gauge("fds_drift_psi", "Population Stability Index")
fds_drift_ks_pval = Gauge("fds_drift_ks_pval", "KS test p-value")
fds_drift_js = Gauge("fds_drift_js_div", "Jensen-Shannon Divergence")
fds_drift_wasserstein = Gauge("fds_drift_wasserstein", "Wasserstein Distance")

# ── 시뮬레이션 ──
fds_simulation_running = Gauge(
    "fds_simulation_running", "시뮬레이션 실행 상태 (0/1)"
)
