from fastapi import APIRouter, Request

from app.schemas.simulation import SimulationStatus

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/start")
async def start_simulation(request: Request):
    sim = request.app.state.simulator
    await sim.start()
    return {"message": "simulation started"}


@router.post("/stop")
async def stop_simulation(request: Request):
    sim = request.app.state.simulator
    await sim.stop()
    return {"message": "simulation stopped"}


@router.get("/status", response_model=SimulationStatus)
async def simulation_status(request: Request):
    sim = request.app.state.simulator
    return SimulationStatus(
        running=sim.running,
        total_processed=sim.total_processed,
        anomalies_detected=sim.anomalies_detected,
        interval_seconds=request.app.state.settings.SIMULATION_INTERVAL,
    )
