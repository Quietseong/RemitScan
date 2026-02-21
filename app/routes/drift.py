from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.drift import DriftMetrics

router = APIRouter(prefix="/drift", tags=["drift"])


@router.get("/metrics", response_model=DriftMetrics | None)
async def get_drift_metrics(request: Request):
    dd = request.app.state.drift_detector
    latest = dd.get_latest()
    if latest is None:
        return JSONResponse(status_code=404, content={"detail": "reference 미설정 또는 데이터 부족"})
    return latest


@router.get("/history", response_model=list[DriftMetrics])
async def get_drift_history(request: Request):
    dd = request.app.state.drift_detector
    return dd.get_history()
