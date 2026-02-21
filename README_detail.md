# RemitScan FDS 상세 가이드 (A to Z)

> 처음 접하는 사람도 환경 설정부터 Grafana 대시보드 구축까지 따라할 수 있도록 작성된 가이드입니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [환경 설정](#4-환경-설정)
5. [ML 모델 학습](#5-ml-모델-학습)
6. [로컬 실행 (Docker 없이)](#6-로컬-실행-docker-없이)
7. [Docker로 전체 서비스 실행](#7-docker로-전체-서비스-실행)
8. [API 사용법](#8-api-사용법)
9. [Prometheus 메트릭 확인](#9-prometheus-메트릭-확인)
10. [Grafana 대시보드 설정](#10-grafana-대시보드-설정)
11. [환경 변수 설정](#11-환경-변수-설정)
12. [ML 파이프라인 상세](#12-ml-파이프라인-상세)
13. [트러블슈팅](#13-트러블슈팅)

---

## 1. 프로젝트 개요

RemitScan은 송금 거래 패턴을 실시간으로 감시하여 이상 거래를 조기에 탐지하는 **이상거래탐지시스템(FDS)** 입니다.

**핵심 파이프라인:**

```
더미 거래 생성 → 전처리 → ML 앙상블 추론 → 규칙 기반 탐지 → 드리프트 계산 → Prometheus 메트릭 → Grafana 시각화
```

**사용 기술:**

| 구분 | 기술 |
|------|------|
| 백엔드 | FastAPI, Uvicorn, asyncio |
| ML (비지도) | Isolation Forest, Autoencoder, VAE |
| ML (지도) | XGBoost, Random Forest, Logistic Regression |
| 모니터링 | Prometheus, Grafana |
| 인프라 | Docker, Docker Compose |
| 전처리 | scikit-learn (StandardScaler), pandas (one-hot encoding) |

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (port 8001)                  │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │Simulator │──▶│Preprocessing │──▶│  ML Pipeline        │  │
│  │(더미 거래)│   │(scaler, OHE) │   │  (6개 모델 앙상블)  │  │
│  └──────────┘   └──────────────┘   └─────────┬──────────┘  │
│       │                                       │             │
│       │              ┌────────────────────────┤             │
│       │              ▼                        ▼             │
│       │     ┌────────────────┐    ┌───────────────────┐     │
│       │     │ Rule Engine    │    │ Drift Detector    │     │
│       │     │ (6개 규칙)     │    │ (PSI/KS/JS/W)     │     │
│       │     └────────────────┘    └───────────────────┘     │
│       │                                                     │
│       └──────────▶ Prometheus Metrics (17개 fds_* 메트릭)   │
│                          │                                  │
│   /health  /simulation  /anomaly  /drift  /metrics          │
└──────────────────────────┼──────────────────────────────────┘
                           │
                  ┌────────▼────────┐
                  │   Prometheus    │ (port 9091)
                  │  15초 스크래핑   │
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │    Grafana      │ (port 3001)
                  │  실시간 대시보드 │
                  └─────────────────┘
```

---

## 3. 프로젝트 구조

```
RemitScan/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 앱 (lifespan 패턴)
│   ├── config.py                # Pydantic Settings (환경변수)
│   ├── metrics.py               # Prometheus 메트릭 17개 정의
│   ├── models/
│   │   ├── autoencoder.py       # Autoencoder (input→16→8→16→input)
│   │   └── vae.py               # VAE (reparameterization trick)
│   ├── routes/
│   │   ├── health.py            # GET /health
│   │   ├── simulation.py        # POST start/stop, GET status
│   │   ├── anomaly.py           # GET results, latest
│   │   └── drift.py             # GET metrics, history
│   ├── schemas/
│   │   ├── transaction.py       # 거래 데이터 스키마
│   │   ├── anomaly.py           # 이상탐지 결과 스키마
│   │   ├── drift.py             # 드리프트 메트릭 스키마
│   │   └── simulation.py        # 시뮬레이션 상태 스키마
│   └── services/
│       ├── preprocessing.py     # scaler + one-hot encoding
│       ├── ml_pipeline.py       # 6개 모델 앙상블 추론
│       ├── rule_engine.py       # 규칙 기반 탐지 (6개 규칙)
│       ├── drift_detector.py    # PSI/KS/JS/Wasserstein 계산
│       └── simulator.py         # 더미 거래 생성 + 백그라운드 루프
├── scripts/
│   ├── train_models.py          # 합성 데이터 생성 → 모델 학습
│   └── setup_grafana.py         # Grafana 대시보드 자동 프로비저닝
├── models/                      # 학습된 모델 아티팩트 (.gitignore)
│   ├── scaler.pkl
│   ├── feature_columns.json
│   ├── isolation_forest.joblib
│   ├── autoencoder.pt
│   ├── vae.pt
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   └── logistic_regression.joblib
├── docker-compose.yml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── README.md
```

---

## 4. 환경 설정

### 4-1. 사전 요구사항

| 도구 | 버전 | 용도 |
|------|------|------|
| Python | 3.10 이상 | 앱 실행 및 모델 학습 |
| Docker Desktop | 최신 | 컨테이너 실행 |
| Git | 최신 | 소스 코드 관리 |

### 4-2. 저장소 클론

```bash
git clone https://github.com/Quietseong/RemitScan.git
cd RemitScan
```

### 4-3. Python 패키지 설치

```bash
pip install -r requirements.txt
```

`requirements.txt`에 포함된 주요 패키지:

```
numpy, pandas, scipy          # 데이터 처리
scikit-learn, torch, xgboost  # ML 모델
fastapi, uvicorn, pydantic    # 웹 프레임워크
pydantic-settings              # 환경변수 관리
prometheus-client              # 메트릭 수집
joblib                         # 모델 직렬화
```

---

## 5. ML 모델 학습

> Docker로 실행할 때도 모델 아티팩트가 필요합니다. **최초 1회** 학습을 실행하세요.

```bash
python scripts/train_models.py --output-dir models/
```

**실행 과정:**
1. 합성 거래 데이터 5,000건 생성 (정상 95% + 이상 5%)
2. 원핫 인코딩 + StandardScaler 전처리
3. 6개 모델 학습:
   - **비지도**: Isolation Forest, Autoencoder, VAE
   - **지도**: XGBoost, Random Forest, Logistic Regression
4. `models/` 디렉토리에 8개 아티팩트 저장

**예상 출력:**

```
[1/7] 합성 데이터 생성...
[2/7] 전처리...
       input_dim=23, features=23
[3/7] Isolation Forest 학습...
[4/7] Autoencoder 학습...
       Epoch 0 loss=1.0329
       ...
[5/7] VAE 학습...
       Epoch 0 loss=1.1686
       ...
[6/7] XGBoost 학습...
[7/7] Random Forest & Logistic Regression 학습...

모든 아티팩트가 models/ 에 저장됨
```

**생성되는 파일 확인:**

```bash
ls models/
# autoencoder.pt  feature_columns.json  isolation_forest.joblib
# logistic_regression.joblib  random_forest.joblib  scaler.pkl
# vae.pt  xgboost.joblib
```

---

## 6. 로컬 실행 (Docker 없이)

Docker 없이도 FastAPI 서버를 직접 실행할 수 있습니다.

```bash
# 모델 학습이 완료된 상태에서
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

```
INFO:     [FDS] services initialized
INFO:     Uvicorn running on http://0.0.0.0:8001
```

브라우저에서 http://localhost:8001/health 접속하여 `{"status":"ok"}` 확인.

---

## 7. Docker로 전체 서비스 실행

### 7-1. Docker Desktop 실행

- Windows: 시작 메뉴에서 **Docker Desktop** 을 실행합니다.
- 트레이 아이콘이 초록색으로 변할 때까지 기다립니다 ("Docker Desktop is running").

### 7-2. 서비스 빌드 및 실행

```bash
docker-compose up --build -d
```

- `--build`: 이미지를 새로 빌드합니다 (최초 실행 또는 코드 변경 시).
- `-d`: 백그라운드 실행 (터미널을 계속 사용할 수 있습니다).
- 최초 빌드 시 PyTorch(~900MB) 등 패키지 다운로드로 **5~10분** 소요됩니다.

### 7-3. 컨테이너 상태 확인

```bash
docker-compose ps
```

3개 서비스가 모두 **Up** 상태여야 합니다:

```
NAME                     STATUS    PORTS
remitscan-fastapi-1      Up        0.0.0.0:8001->8001/tcp
remitscan-prometheus-1   Up        0.0.0.0:9091->9090/tcp
remitscan-grafana-1      Up        0.0.0.0:3001->3000/tcp
```

### 7-4. 서비스 중지 및 재시작

```bash
# 중지
docker-compose down

# 로그 확인
docker-compose logs fastapi
docker-compose logs -f fastapi   # 실시간 로그
```

---

## 8. API 사용법

### 8-1. 헬스체크

```bash
curl http://localhost:8001/health
# {"status":"ok"}
```

### 8-2. 시뮬레이션 시작/중지

```bash
# 시뮬레이션 시작 (5초마다 50건 더미 거래 생성)
curl -X POST http://localhost:8001/simulation/start
# {"message":"simulation started"}

# 상태 확인
curl http://localhost:8001/simulation/status
# {"running":true,"total_processed":500,"anomalies_detected":73,"interval_seconds":5.0}

# 시뮬레이션 중지
curl -X POST http://localhost:8001/simulation/stop
# {"message":"simulation stopped"}
```

### 8-3. 이상탐지 결과 조회

```bash
# 최근 1건
curl http://localhost:8001/anomaly/latest

# 최근 N건 (기본 50건)
curl http://localhost:8001/anomaly/results?limit=10
```

**응답 예시:**

```json
{
  "transaction_id": "75355945-dfa5-49c1-a420-e2560121cb63",
  "timestamp": "2026-02-21T11:47:59+00:00",
  "scores": {
    "if_score": 0.92,
    "ae_score": 0.94,
    "vae_score": 0.96,
    "xgb_score": 0.0001,
    "rf_score": 0.03,
    "lr_score": 0.23,
    "unsupervised_score": 0.94,
    "supervised_score": 0.09,
    "combined_score": 0.51,
    "is_anomaly": false
  },
  "triggered_rules": ["루팅_탐지"],
  "transaction_amount": 23980.51
}
```

### 8-4. 드리프트 메트릭 조회

```bash
# 최신 드리프트
curl http://localhost:8001/drift/metrics
# {"timestamp":"...","psi":0.31,"ks_stat":0.12,"ks_pval":0.87,"js_div":0.04,"wasserstein":0.02}

# 전체 히스토리
curl http://localhost:8001/drift/history
```

### 8-5. Prometheus 메트릭

```bash
curl http://localhost:8001/metrics
# fds_transactions_total 6550.0
# fds_anomalies_total 914.0
# fds_drift_psi 0.31
# ...
```

### 8-6. API 전체 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/simulation/start` | 시뮬레이션 시작 |
| POST | `/simulation/stop` | 시뮬레이션 중지 |
| GET | `/simulation/status` | 시뮬레이션 상태 조회 |
| GET | `/anomaly/results?limit=N` | 이상탐지 결과 목록 |
| GET | `/anomaly/latest` | 최신 이상탐지 결과 |
| GET | `/drift/metrics` | 최신 드리프트 메트릭 |
| GET | `/drift/history` | 드리프트 히스토리 |
| GET | `/metrics` | Prometheus 메트릭 |

---

## 9. Prometheus 메트릭 확인

### 9-1. Prometheus 웹 UI 접속

브라우저에서 http://localhost:9091 접속.

### 9-2. 메트릭 쿼리

검색창에 `fds_` 를 입력하면 전체 메트릭 목록이 나타납니다.

**등록된 17개 메트릭:**

| 메트릭 | 타입 | 설명 |
|--------|------|------|
| `fds_transactions_total` | Counter | 총 처리 거래 수 |
| `fds_anomalies_total` | Counter | 탐지된 이상 거래 수 |
| `fds_rules_triggered_total` | Counter | 규칙별 트리거 횟수 |
| `fds_combined_score` | Histogram | 최종 결합 점수 분포 |
| `fds_unsupervised_score_avg` | Gauge | 비지도 점수 평균 |
| `fds_supervised_score_avg` | Gauge | 지도 점수 평균 |
| `fds_anomaly_threshold` | Gauge | 현재 이상 탐지 임계값 |
| `fds_if_score_avg` | Gauge | Isolation Forest 평균 |
| `fds_ae_score_avg` | Gauge | Autoencoder 평균 |
| `fds_vae_score_avg` | Gauge | VAE 평균 |
| `fds_xgb_score_avg` | Gauge | XGBoost 평균 |
| `fds_rf_score_avg` | Gauge | Random Forest 평균 |
| `fds_lr_score_avg` | Gauge | Logistic Regression 평균 |
| `fds_drift_psi` | Gauge | Population Stability Index |
| `fds_drift_ks_pval` | Gauge | KS test p-value |
| `fds_drift_js_div` | Gauge | Jensen-Shannon Divergence |
| `fds_drift_wasserstein` | Gauge | Wasserstein Distance |
| `fds_simulation_running` | Gauge | 시뮬레이션 실행 상태 |

### 9-3. Target 상태 확인

Prometheus UI → Status → Targets 에서 `fastapi:8001`이 **UP** 상태인지 확인합니다.

---

## 10. Grafana 대시보드 설정

### 방법 A: 자동 프로비저닝 (스크립트 실행)

Docker 서비스가 실행 중인 상태에서:

```bash
python scripts/setup_grafana.py
```

```
Datasource UID: efdwpjiihgge8c
Dashboard: 200 success
URL: http://localhost:3001/d/fds-main/remitscan-fds-monitoring
```

이후 http://localhost:3001/d/fds-main/remitscan-fds-monitoring 접속하면 바로 대시보드를 볼 수 있습니다.

---

### 방법 B: Grafana에서 직접 만들기 (수동)

처음부터 직접 구성하고 싶다면 아래 단계를 따릅니다.

#### Step 1: Grafana 로그인

1. 브라우저에서 http://localhost:3001 접속
2. 기본 계정: **admin** / **admin**
3. 비밀번호 변경 화면이 나오면 **Skip** 클릭

#### Step 2: Prometheus 데이터 소스 추가

1. 좌측 메뉴 → **Connections** → **Data sources**
2. **Add data source** 클릭
3. **Prometheus** 선택
4. Connection URL: `http://prometheus:9090`
5. 페이지 하단 **Save & test** 클릭
6. "Successfully queried the Prometheus API" 메시지 확인

#### Step 3: 대시보드 생성

1. 좌측 메뉴 → **Dashboards**
2. 우측 상단 **New** → **New dashboard**
3. **Add visualization** 클릭

#### Step 4: 패널 추가 - 핵심 KPI (Row 1)

##### 패널 1: Total Transactions (Stat)

1. Data source: **Prometheus** 선택
2. Metric 검색창에 `fds_transactions_total` 입력
3. **Run queries** 클릭
4. 우측 패널 설정:
   - Visualization: **Stat** 선택
   - Title: `Total Transactions`
5. **Apply** 클릭

##### 패널 2: Anomalies Detected (Stat)

1. 대시보드에서 **Add** → **Visualization**
2. Metric: `fds_anomalies_total`
3. Visualization: **Stat**
4. Title: `Anomalies Detected`
5. Thresholds 설정:
   - Base: Green
   - 50: Orange
   - 200: Red
6. **Apply**

##### 패널 3: Simulation Status (Stat)

1. Metric: `fds_simulation_running`
2. Visualization: **Stat**
3. Title: `Simulation`
4. Value mappings (우측 패널 하단):
   - 0 → "STOPPED" (Red)
   - 1 → "RUNNING" (Green)
5. **Apply**

##### 패널 4: Anomaly Threshold (Gauge)

1. Metric: `fds_anomaly_threshold`
2. Visualization: **Gauge**
3. Title: `Anomaly Threshold`
4. Min: 0, Max: 1
5. Thresholds: Green(0), Yellow(0.5), Red(0.8)
6. **Apply**

#### Step 5: 패널 추가 - 드리프트 메트릭 (Row 2)

##### 패널 5: Drift PSI (Time series)

1. Metric: `fds_drift_psi`
2. Visualization: **Time series**
3. Title: `Drift - PSI`
4. Legend: `PSI`
5. **Apply**

##### 패널 6: Drift KS p-value (Time series)

1. Metric: `fds_drift_ks_pval`
2. Title: `Drift - KS p-value`
3. Legend: `KS p-value`
4. **Apply**

##### 패널 7: JS Divergence & Wasserstein (Time series)

1. 첫 번째 쿼리: `fds_drift_js_div` (Legend: `JS Divergence`)
2. **+ Add query** 클릭
3. 두 번째 쿼리: `fds_drift_wasserstein` (Legend: `Wasserstein`)
4. Title: `Drift - JS Divergence & Wasserstein`
5. **Apply**

##### 패널 8: Transaction Rate (Time series)

1. 첫 번째 쿼리: `rate(fds_transactions_total[1m])` (Legend: `Txns/s`)
2. 두 번째 쿼리: `rate(fds_anomalies_total[1m])` (Legend: `Anomalies/s`)
3. Title: `Transaction & Anomaly Rate (/s)`
4. **Apply**

#### Step 6: 패널 추가 - 모델 점수 (Row 3)

##### 패널 9: Unsupervised Scores (Time series)

1. 쿼리 3개 추가:
   - `fds_if_score_avg` (Legend: `Isolation Forest`)
   - `fds_ae_score_avg` (Legend: `Autoencoder`)
   - `fds_vae_score_avg` (Legend: `VAE`)
2. Title: `Unsupervised Scores (IF / AE / VAE)`
3. Y축 Min: 0, Max: 1
4. **Apply**

##### 패널 10: Supervised Scores (Time series)

1. 쿼리 3개 추가:
   - `fds_xgb_score_avg` (Legend: `XGBoost`)
   - `fds_rf_score_avg` (Legend: `Random Forest`)
   - `fds_lr_score_avg` (Legend: `Logistic Regression`)
2. Title: `Supervised Scores (XGB / RF / LR)`
3. Y축 Min: 0, Max: 1
4. **Apply**

#### Step 7: 패널 추가 - 앙상블 & 규칙 (Row 4)

##### 패널 11: Ensemble Comparison (Time series)

1. 쿼리 3개:
   - `fds_unsupervised_score_avg` (Legend: `Unsupervised`)
   - `fds_supervised_score_avg` (Legend: `Supervised`)
   - `fds_anomaly_threshold` (Legend: `Threshold`)
2. Title: `Ensemble: Unsupervised vs Supervised vs Threshold`
3. **Apply**

##### 패널 12: Rule Triggers (Bar chart)

1. Metric: `fds_rules_triggered_total`
2. Options → Type: **Instant** 선택
3. Visualization: **Bar chart**
4. Legend: `{{rule_name}}`
5. Title: `Rule Triggers (Total Count)`
6. **Apply**

##### 패널 13: Combined Score Histogram (Heatmap)

1. Metric: `rate(fds_combined_score_bucket[1m])`
2. Visualization: **Heatmap**
3. Title: `Combined Score Distribution`
4. **Apply**

#### Step 8: 대시보드 저장 및 자동 새로고침

1. 대시보드 우측 상단 **Save** (Ctrl+S)
2. 이름: `RemitScan FDS Monitoring`
3. 우측 상단 시간 범위: **Last 30 minutes**
4. Refresh interval: **5s**

#### 완성된 대시보드

**상단 영역 (KPI + 드리프트):**

![Grafana Dashboard Top](/assets/grafana_dashboard_top.png)

**하단 영역 (모델 점수 + 규칙 + 히스토그램):**

![Grafana Dashboard Bottom](/assets/grafana_dashboard_bottom.png)

---

## 11. 환경 변수 설정

모든 설정은 `FDS_` 접두사 환경 변수로 오버라이드할 수 있습니다.

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `FDS_SIMULATION_INTERVAL` | 5.0 | 시뮬레이션 주기 (초) |
| `FDS_BATCH_SIZE` | 50 | 배치당 거래 건수 |
| `FDS_ANOMALY_THRESHOLD_PERCENTILE` | 95.0 | 이상 탐지 임계값 백분위 |
| `FDS_DRIFT_HISTORY_SIZE` | 100 | 드리프트 히스토리 보관 수 |
| `FDS_UNSUPERVISED_WEIGHT` | 0.5 | 비지도 앙상블 가중치 |
| `FDS_SUPERVISED_WEIGHT` | 0.5 | 지도 앙상블 가중치 |
| `FDS_MODEL_DIR` | `models/` | 모델 아티팩트 경로 |

`docker-compose.yml`의 `environment` 섹션에서 수정하거나, `.env` 파일을 사용할 수 있습니다.

---

## 12. ML 파이프라인 상세

### 12-1. 전처리

- **범주형 컬럼** (9개): `payment_method`, `intent`, `authentication`, `voice_match`, `is_new_account_for_user`, `is_nighttime`, `is_new_device`, `vpn`, `rooting`
- **수치형 컬럼** (4개): `TransactionAmt`, `hour`, `avg_amount_to_bank`, `recent_transaction_gap`
- 원핫 인코딩(`drop_first=True`) 후 StandardScaler 적용 → 최종 23개 피처

### 12-2. 앙상블 추론

```
비지도 점수 = 0.33 × IF + 0.33 × AE + 0.34 × VAE     (rank-percentile 정규화)
지도 점수   = mean(XGBoost, RF, LR)                    (확률값 평균)
최종 점수   = 0.5 × 비지도 + 0.5 × 지도
이상 판정   = 최종 점수 > 95th percentile
```

### 12-3. 규칙 기반 탐지

| 규칙명 | 조건 |
|--------|------|
| 야간_대량_송금 | 야간(0~6시) AND 금액 > 50만원 |
| VPN_대량_송금 | VPN 사용 AND 금액 > 30만원 |
| 약인증_대량_송금 | 인증 A01/A02 AND 금액 > 50만원 |
| 신규기기_대량_송금 | 새 기기 AND 금액 > 30만원 |
| 루팅_탐지 | 루팅 기기 사용 |
| 신규계정_야간_VPN | 신규계정 AND 야간 AND VPN |

### 12-4. 드리프트 탐지

| 지표 | 설명 | 기준 |
|------|------|------|
| PSI | Population Stability Index | > 0.25 → 심각한 드리프트 |
| KS p-value | Kolmogorov-Smirnov 검정 | < 0.05 → 분포 변화 |
| JS Divergence | Jensen-Shannon 발산 | 0=동일, 1=완전히 다름 |
| Wasserstein | Earth Mover's Distance | 값이 클수록 분포 차이 큼 |

---

## 13. 트러블슈팅

### Docker Desktop이 실행되지 않음

```
error: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

→ Docker Desktop을 먼저 실행하세요. 트레이 아이콘이 초록색이 될 때까지 기다립니다.

### 포트 충돌

```
Bind for 0.0.0.0:8001 failed: port is already allocated
```

→ 해당 포트를 사용 중인 프로세스를 종료하거나, `docker-compose.yml`에서 포트를 변경합니다.

### 모델 파일이 없음

```
FileNotFoundError: models/scaler.pkl
```

→ 모델 학습을 먼저 실행하세요:

```bash
python scripts/train_models.py --output-dir models/
```

### Grafana에서 "No data"

- 시뮬레이션이 실행 중인지 확인: `curl http://localhost:8001/simulation/status`
- 시뮬레이션이 꺼져 있으면 시작: `curl -X POST http://localhost:8001/simulation/start`
- Prometheus Target 상태 확인: http://localhost:9091/targets
- 시간 범위를 **Last 5 minutes** 등 짧게 조정

### Windows PowerShell에서 curl 에러

PowerShell의 `curl`은 `Invoke-WebRequest`의 별칭입니다. 대신 다음을 사용하세요:

```powershell
# PowerShell에서
Invoke-RestMethod http://localhost:8001/health

# 또는 curl.exe를 직접 호출
curl.exe http://localhost:8001/health
```
