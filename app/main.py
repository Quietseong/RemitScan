# 파일명: main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.middleware.wsgi import WSGIMiddleware
from prometheus_client import Counter, Gauge, Histogram, generate_latest, make_wsgi_app
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import json
import os

app = FastAPI()

# Prometheus 메트릭 정의
PSI_GAUGE = Gauge('data_drift_psi', 'Population Stability Index')
KS_PVAL_GAUGE = Gauge('data_drift_ks_pval', 'Kolmogorov-Smirnov test p-value')
JS_DIV_GAUGE = Gauge('data_drift_js_div', 'Jensen-Shannon Divergence')
WASSERSTEIN_GAUGE = Gauge('data_drift_wasserstein', 'Wasserstein Distance')

@app.get("/drift-metrics")
async def get_drift_metrics():
    """
    drift_metrics_hourly.json 파일의 데이터 드리프트 결과를 반환하는 API

    Returns:
        JSONResponse: 시간별 데이터 드리프트 결과
    Raises:
        HTTPException: 파일이 없거나 읽기 오류 발생 시
    """
    # 상위 디렉토리의 data 폴더를 참조하도록 수정
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "drift_metrics_hourly.json")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"drift_metrics_hourly.json 파일이 없습니다. 경로: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 가장 최근 데이터로 Prometheus 메트릭 업데이트
        if data:
            latest = data[-1]
            PSI_GAUGE.set(latest['psi'])
            KS_PVAL_GAUGE.set(latest['ks_pval'])
            JS_DIV_GAUGE.set(latest['js_div'])
            WASSERSTEIN_GAUGE.set(latest['wasserstein'])
            
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 오류: {e}")

# WSGI 미들웨어로 /metrics 엔드포인트 등록
app.mount("/metrics", WSGIMiddleware(make_wsgi_app()))