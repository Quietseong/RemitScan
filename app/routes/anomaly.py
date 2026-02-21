from fastapi import APIRouter, Request

from app.schemas.anomaly import AnomalyResult

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.get("/results", response_model=list[AnomalyResult])
async def get_results(request: Request, limit: int = 50):
    sim = request.app.state.simulator
    results = list(sim.results)
    return results[-limit:]


@router.get("/latest", response_model=AnomalyResult | None)
async def get_latest(request: Request):
    sim = request.app.state.simulator
    if not sim.results:
        return None
    return sim.results[-1]
