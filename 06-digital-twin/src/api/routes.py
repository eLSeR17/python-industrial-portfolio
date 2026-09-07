"""FastAPI routes for the Digital Twin Simulator."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.models.schemas import ScenarioComparison, SimulationConfig, SimulationResult
from src.services.scenario_runner import ScenarioRunner
from src.simulation.engine import SimulationEngine

router = APIRouter(prefix="/api/v1")

# In-memory result cache (a production system would use Redis)
_results_cache: dict[str, SimulationResult] = {}
_plant_topologies: dict[str, dict[str, Any]] = {
    "default": {
        "id": "default",
        "name": "Default 3-station line",
        "stations": ["m1", "m2", "m3"],
        "conveyors": ["c1", "c2"],
    }
}


class CompareRequest(BaseModel):
    """Request body for multi-scenario comparison."""

    scenarios: list[SimulationConfig]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/simulate", response_model=SimulationResult)
async def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a simulation and return the result.

    The result is cached so it can be retrieved later via
    ``GET /results/{sim_id}``.
    """
    engine = SimulationEngine(config)
    result = engine.run()
    _results_cache[config.id] = result
    return result


@router.get("/results/{sim_id}", response_model=SimulationResult)
async def get_results(sim_id: str) -> SimulationResult:
    """Retrieve cached simulation results by id."""
    result = _results_cache.get(sim_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found.")
    return result


@router.post("/compare", response_model=ScenarioComparison)
async def compare(request: CompareRequest) -> ScenarioComparison:
    """Compare multiple simulation scenarios and return recommendations."""
    if not request.scenarios:
        raise HTTPException(status_code=400, detail="No scenarios provided.")
    runner = ScenarioRunner()
    comparison = runner.compare_scenarios(request.scenarios, replications=3)
    return comparison


@router.get("/plant/{plant_id}/topology")
async def get_plant_topology(plant_id: str) -> dict[str, Any]:
    """Return the topology of a plant (stations, conveyors)."""
    topo = _plant_topologies.get(plant_id)
    if topo is None:
        raise HTTPException(status_code=404, detail=f"Plant '{plant_id}' not found.")
    return topo
