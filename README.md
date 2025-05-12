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
