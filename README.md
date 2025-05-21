[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Quietseong/RemitScan)

## 📌 Project Name: RemitScan

## 🧾 Overview
RemitScan은 송금 거래 패턴을 실시간으로 감시하여 이상 거래를 조기에 탐지하는 지능형 이상거래탐지(FDS, Fraud Detection System)입니다. 사용자의 거래 행태와 디바이스, 인증 방식, 송금 금액 등을 기반으로 데이터를 분석하고 금융 서비스가 리스크에 신속하게 대응할 수 있도록 지원합니다. 

## 🎯 Objective
- 규칙 기반 탐지와 머신러닝 기반 모델을 병행하여 송금 이상 거래를 식별
- 탐지된 이상 거래의 주요 특성을 분석하여 리스크 대응 전략 수립에 기여
- 거래 시간대, 인증 성공 여부, 새로운 디바이스 사용 등 다양한 이상 징후 패턴을 바탕으로 이상 거래의 확률을 예측

## 🚨 Definition of Fraudulent Transactions

## 🏗️ Architecture
![architecture](/assets/fds_diagram_example.png)

## 📊 Data Schema
https://docs.google.com/spreadsheets/d/13i9EEOPVrwfNdQjwzAHsoBYhe_UiJ6kofGHuEqlMCWU/edit?gid=0#gid=0

## 🛠️ Tech Stack
- **Python**: Data processing and ML modeling
- **Isolation Forest / Autoencoder / Variational Autoencoder**: Unsupervised modeling
- **XGBoost / LightGBM / Random Forest**: Supervised modeling
- **Jupyter Notebook**: EDA and feature engineering
- **Matplotlib / Seaborn**: Data visualization

## 👤 Team Role Assignment

| 이름     | 역할                   | 메일 |
|--------|----------------------|-|
| 김서령    | 총괄 / XGboost 알고리즘 적용 | ryeong2105@gmail.com |
| 조용성    | End-to-End 프로세스 구현 | j808esc@gmail.com |
| 임은서    |                      | |
| 김선민    | API 개발 및 디바이스 연동       | seonmin8284@gmail.com |

## 🔗 Project Planning (WBS)
https://gratis-catmint-235.notion.site/RemitScan-1dd38a80454880178f56c04edd60683d?pvs=4

# RemitScan 모니터링 프로젝트

이 프로젝트는 FastAPI 기반의 데이터 드리프트 API, Prometheus 기반의 메트릭 수집, Grafana 대시보드를 통한 실시간 모니터링 환경을 제공합니다.

---

## 1. 도커(Docker)로 전체 서비스 실행하기

### 1-1. 필수 조건
- Docker, Docker Compose 설치
- (윈도우 사용자는 Docker Desktop 권장)

### 1-2. 실행 명령어
```bash
git clone <이 저장소 주소>
cd <프로젝트 디렉토리>
docker-compose up --build -d
```
- 최초 실행 시 `--build` 옵션을 꼭 사용하세요.
- 모든 서비스가 자동으로 실행됩니다.

### 1-3. 주요 컨테이너 및 포트
| 서비스      | 설명                | 호스트 포트 |
|-------------|---------------------|-------------|
| FastAPI     | 드리프트 API        | 8001        |
| Prometheus  | 메트릭 수집         | 9091        |
| Grafana     | 대시보드            | 3001        |

---

## 2. FastAPI API 사용법

- 데이터 드리프트 결과 조회:
  - [http://localhost:8001/drift-metrics](http://localhost:8001/drift-metrics)
- Prometheus 메트릭 확인:
  - [http://localhost:8001/metrics](http://localhost:8001/metrics)

---

## 3. Prometheus 연동 및 확인

- Prometheus 웹 UI: [http://localhost:9091](http://localhost:9091)
- 메트릭 타겟이 정상적으로 UP 상태인지 확인하세요.
- 주요 메트릭:
  - `data_drift_psi`
  - `data_drift_ks_pval`
  - `data_drift_js_div`
  - `data_drift_wasserstein`

---

## 4. Grafana 대시보드에서 데이터 확인하기

### 4-1. Grafana 접속
- [http://localhost:3001](http://localhost:3001)
- 기본 계정: `admin` / `admin` (최초 로그인 시 비밀번호 변경 필요)

### 4-2. Prometheus 데이터 소스 추가
1. 왼쪽 메뉴 → Administration → Data sources
2. "Add data source" 클릭
3. Prometheus 선택
4. URL: `http://prometheus:9090` (컨테이너 환경)
5. Save & Test

### 4-3. 대시보드 생성 및 패널 추가
1. Create → Dashboard → Add new panel
2. Data source: Prometheus 선택
3. 쿼리 입력: 예) `data_drift_psi`
4. 원하는 시각화(그래프, 게이지 등) 선택
5. Apply로 저장

### 4-4. 여러 메트릭 패널 추가
- `data_drift_ks_pval`, `data_drift_js_div`, `data_drift_wasserstein` 등도 동일하게 추가

### 4-5. 자동 새로고침 및 시간 범위 설정
- 우측 상단에서 시간 범위 및 Auto-refresh(예: 10s) 설정

---

## 5. 기타

- 컨테이너 중지: `docker-compose down`
- 로그 확인: `docker-compose logs [서비스명]`
- 환경/포트 충돌 시 기존 컨테이너/프로세스 종료 후 재시작

---

## 6. 참고
- FastAPI, Prometheus, Grafana 공식 문서 참고
- 추가 문의는 이슈로 남겨주세요.
