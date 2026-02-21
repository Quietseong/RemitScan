"""Grafana 데이터소스 + FDS 대시보드 자동 프로비저닝."""
import requests
import sys

GRAFANA = "http://admin:admin@localhost:3001"


def get_datasource_uid():
    r = requests.get(f"{GRAFANA}/api/datasources/name/Prometheus")
    if r.status_code == 200:
        return r.json()["uid"]
    return None


def setup():
    ds_uid = get_datasource_uid()
    if not ds_uid:
        print("Prometheus datasource not found, skipping")
        sys.exit(1)

    print(f"Datasource UID: {ds_uid}")

    def t(expr, legend="", ref="A", instant=False):
        target = {"expr": expr, "legendFormat": legend, "refId": ref}
        if instant:
            target["instant"] = True
        return target

    dashboard = {
        "dashboard": {
            "id": None,
            "uid": "fds-main",
            "title": "RemitScan FDS Monitoring",
            "tags": ["fds", "anomaly", "drift"],
            "timezone": "browser",
            "refresh": "5s",
            "time": {"from": "now-30m", "to": "now"},
            "panels": [
                # ── Row 1: 핵심 KPI (Stat) ──
                {
                    "id": 1,
                    "type": "stat",
                    "title": "Total Transactions",
                    "gridPos": {"h": 5, "w": 6, "x": 0, "y": 0},
                    "targets": [t("fds_transactions_total")],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [{"color": "blue", "value": None}]
                            },
                        }
                    },
                },
                {
                    "id": 2,
                    "type": "stat",
                    "title": "Anomalies Detected",
                    "gridPos": {"h": 5, "w": 6, "x": 6, "y": 0},
                    "targets": [t("fds_anomalies_total")],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "orange", "value": 50},
                                    {"color": "red", "value": 200},
                                ]
                            },
                        }
                    },
                },
                {
                    "id": 3,
                    "type": "stat",
                    "title": "Simulation",
                    "gridPos": {"h": 5, "w": 6, "x": 12, "y": 0},
                    "targets": [t("fds_simulation_running")],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "mappings": [
                                {
                                    "type": "value",
                                    "options": {
                                        "0": {"text": "STOPPED", "color": "red"},
                                        "1": {"text": "RUNNING", "color": "green"},
                                    },
                                }
                            ],
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": None},
                                    {"color": "green", "value": 1},
                                ]
                            },
                        }
                    },
                },
                {
                    "id": 4,
                    "type": "gauge",
                    "title": "Anomaly Threshold",
                    "gridPos": {"h": 5, "w": 6, "x": 18, "y": 0},
                    "targets": [t("fds_anomaly_threshold")],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "min": 0,
                            "max": 1,
                            "color": {"mode": "thresholds"},
                            "thresholds": {
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 0.5},
                                    {"color": "red", "value": 0.8},
                                ]
                            },
                        }
                    },
                },
                # ── Row 2: 드리프트 메트릭 ──
                {
                    "id": 10,
                    "type": "timeseries",
                    "title": "Drift - PSI",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 5},
                    "targets": [t("fds_drift_psi", "PSI")],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"fixedColor": "orange", "mode": "fixed"},
                            "custom": {
                                "lineWidth": 2,
                                "fillOpacity": 20,
                                "pointSize": 5,
                                "showPoints": "auto",
                            },
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 0.1},
                                    {"color": "red", "value": 0.25},
                                ],
                            },
                        }
                    },
                },
                {
                    "id": 11,
                    "type": "timeseries",
                    "title": "Drift - KS p-value",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 5},
                    "targets": [t("fds_drift_ks_pval", "KS p-value")],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"fixedColor": "purple", "mode": "fixed"},
                            "custom": {"lineWidth": 2, "fillOpacity": 15},
                        }
                    },
                },
                {
                    "id": 12,
                    "type": "timeseries",
                    "title": "Drift - JS Divergence & Wasserstein",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 13},
                    "targets": [
                        t("fds_drift_js_div", "JS Divergence"),
                        t("fds_drift_wasserstein", "Wasserstein", "B"),
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"lineWidth": 2, "fillOpacity": 10},
                        }
                    },
                },
                # ── Row 2 right: 처리율 ──
                {
                    "id": 13,
                    "type": "timeseries",
                    "title": "Transaction & Anomaly Rate (/s)",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 13},
                    "targets": [
                        t("rate(fds_transactions_total[1m])", "Txns/s"),
                        t("rate(fds_anomalies_total[1m])", "Anomalies/s", "B"),
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"lineWidth": 2, "fillOpacity": 20},
                        }
                    },
                },
                # ── Row 3: 비지도 vs 지도 모델 ──
                {
                    "id": 20,
                    "type": "timeseries",
                    "title": "Unsupervised Scores (IF / AE / VAE)",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 21},
                    "targets": [
                        t("fds_if_score_avg", "Isolation Forest"),
                        t("fds_ae_score_avg", "Autoencoder", "B"),
                        t("fds_vae_score_avg", "VAE", "C"),
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"lineWidth": 2, "fillOpacity": 10},
                            "min": 0,
                            "max": 1,
                        }
                    },
                },
                {
                    "id": 21,
                    "type": "timeseries",
                    "title": "Supervised Scores (XGB / RF / LR)",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 21},
                    "targets": [
                        t("fds_xgb_score_avg", "XGBoost"),
                        t("fds_rf_score_avg", "Random Forest", "B"),
                        t("fds_lr_score_avg", "Logistic Regression", "C"),
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"lineWidth": 2, "fillOpacity": 10},
                            "min": 0,
                            "max": 1,
                        }
                    },
                },
                # ── Row 4: 앙상블 결합 & 규칙 트리거 ──
                {
                    "id": 30,
                    "type": "timeseries",
                    "title": "Ensemble: Unsupervised vs Supervised vs Threshold",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 29},
                    "targets": [
                        t("fds_unsupervised_score_avg", "Unsupervised"),
                        t("fds_supervised_score_avg", "Supervised", "B"),
                        t("fds_anomaly_threshold", "Threshold", "C"),
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "custom": {"lineWidth": 2, "fillOpacity": 10},
                            "min": 0,
                            "max": 1,
                        }
                    },
                },
                {
                    "id": 31,
                    "type": "barchart",
                    "title": "Rule Triggers (Total Count)",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 29},
                    "targets": [
                        t("fds_rules_triggered_total", "{{rule_name}}", instant=True)
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                    "fieldConfig": {
                        "defaults": {"color": {"mode": "palette-classic"}}
                    },
                    "options": {"orientation": "horizontal", "showValue": "always"},
                },
                # ── Row 5: Combined Score 분포 ──
                {
                    "id": 40,
                    "type": "heatmap",
                    "title": "Combined Score Distribution (Histogram)",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 37},
                    "targets": [
                        t(
                            "rate(fds_combined_score_bucket[1m])",
                            "le={{le}}",
                        )
                    ],
                    "datasource": {"type": "prometheus", "uid": ds_uid},
                },
            ],
        },
        "overwrite": True,
    }

    r = requests.post(f"{GRAFANA}/api/dashboards/db", json=dashboard)
    result = r.json()
    print(f"Dashboard: {r.status_code} {result.get('status', '')}")
    print(f"URL: http://localhost:3001{result.get('url', '')}")


if __name__ == "__main__":
    setup()
