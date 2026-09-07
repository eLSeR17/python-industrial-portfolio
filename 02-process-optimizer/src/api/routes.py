"""FastAPI HTTP endpoints for the process optimizer.

Endpoints:
    POST /process/update     — ingest a batch of sensor readings
    GET  /optimize/{pid}     — trigger and return optimization results
    GET  /spc/{pid}          — run SPC analysis on a process
    GET  /dashboard-data     — full snapshot for the operator dashboard
    GET  /health             — liveness probe
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from src.models.process_state import holder
from src.models.schemas import (
    DashboardData,
    OptimizationMethod,
    OptimizationRequest,
    OptimizationResult,
    ProcessUpdateRequest,
    SPCResult,
)
from src.services.optimizer import optimizer
from src.services.stream_processor import stream_processor
from src.services.waste_analyzer import waste_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["process"])


# -----------------------------------------------------------------------
# Process Update
# -----------------------------------------------------------------------

@router.post("/process/update", status_code=200)
async def process_update(request: ProcessUpdateRequest) -> dict:
    """Ingest a batch of sensor readings for a process.

    The readings are validated, stored in the process state holder, and
    made available to the optimizer and SPC engine.

    Args:
        request: ProcessUpdateRequest with process_id, process_type, readings,
                 and optional setpoints.

    Returns:
        Confirmation with the number of variables updated.
    """
    state = await holder.update(request)
    return {
        "status": "ok",
        "process_id": request.process_id,
        "variables_updated": len(state.variables),
        "total_readings": len(request.readings),
    }


# -----------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------

@router.get("/optimize/{process_id}", response_model=OptimizationResult)
async def optimize_process(
    process_id: str,
    method: OptimizationMethod = Query(default=OptimizationMethod.NELDER_MEAD),
    max_iterations: int = Query(default=200, ge=1, le=10000),
) -> OptimizationResult:
    """Trigger an optimization run for the given process.

    The optimizer reads the current state and setpoints from the state
    holder, runs the selected gradient-free algorithm, and returns the
    recommended setpoints.

    Args:
        process_id: Process line to optimize.
        method: Optimization algorithm (nelder_mead, coordinate_descent, bayesian).
        max_iterations: Maximum iterations for the optimizer.

    Raises:
        HTTPException 404 if the process is unknown.
        HTTPException 400 if insufficient data for optimization.
    """
    state = await holder.get_state(process_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Process '{process_id}' not found")

    current_setpoints = {v.name: v.value for v in state.variables}
    if not current_setpoints:
        raise HTTPException(status_code=400, detail="No variables recorded for this process")

    variable_names = list(current_setpoints.keys())

    # Build variable limits from process variable bounds.
    variable_limits: dict[str, tuple[float, float]] = {}
    for var in state.variables:
        variable_limits[var.name] = (var.min_limit, var.max_limit)

    result = optimizer.optimize(
        process_id=process_id,
        current_setpoints=current_setpoints,
        variable_names=variable_names,
        variable_limits=variable_limits,
        method=method,
        max_iterations=max_iterations,
    )

    # Publish recommendation to Kafka for downstream consumers.
    await stream_processor.publish_recommendation(
        process_id=process_id,
        setpoints=result.recommended_setpoints,
        metadata={
            "method": result.method.value,
            "improvement_pct": result.predicted_improvement_pct,
            "objective_value": result.objective_value,
        },
    )

    return result


# -----------------------------------------------------------------------
# SPC Analysis
# -----------------------------------------------------------------------

@router.get("/spc/{process_id}", response_model=SPCResult)
async def spc_analysis(
    process_id: str,
    window: int = Query(default=100, ge=10, le=10000, description="Number of recent readings to analyze"),
    subgroup_size: int = Query(default=5, ge=2, le=10),
) -> SPCResult:
    """Run Statistical Process Control analysis on a process.

    Reads the historical data from the state holder and computes X-bar/R
    charts, CUSUM detection, and Western Electric rule violations.

    Args:
        process_id: Process to analyze.
        window: Number of most recent readings to include.
        subgroup_size: Subgroup size for control charts.

    Raises:
        HTTPException 404 if the process is unknown.
        HTTPException 400 if insufficient data.
    """
    state = await holder.get_state(process_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Process '{process_id}' not found")

    # Collect historical data for each variable.
    variable_data = {}
    for var in state.variables:
        history = await holder.get_history(process_id, var.name, last_n=window)
        if history:
            import numpy as np

            variable_data[var.name] = np.array(
                [h["value"] for h in history], dtype=np.float64
            )

    if not variable_data:
        raise HTTPException(status_code=400, detail="No historical data available")

    result = waste_analyzer.analyze(
        process_id=process_id,
        variable_data=variable_data,
        subgroup_size=subgroup_size,
    )
    return result


# -----------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------

@router.get("/dashboard-data", response_model=DashboardData)
async def dashboard_data() -> DashboardData:
    """Return a full snapshot of all processes for the operator dashboard.

    Includes current states, latest optimizations, SPC results, OEE,
    and active alarms.
    """
    states = await holder.get_all_states()

    latest_optimizations: dict[str, OptimizationResult] = {}
    spc_results: dict[str, SPCResult] = {}
    active_alarms = []

    for state in states:
        pid = state.process_id

        # Latest optimization.
        opt = optimizer.get_last_result(pid)
        if opt is not None:
            latest_optimizations[pid] = opt

        # Quick SPC check.
        variable_data = {}
        for var in state.variables:
            history = await holder.get_history(pid, var.name, last_n=100)
            if len(history) >= 10:
                import numpy as np

                variable_data[var.name] = np.array(
                    [h["value"] for h in history], dtype=np.float64
                )
        if variable_data:
            spc = waste_analyzer.analyze(pid, variable_data)
            spc_results[pid] = spc
            active_alarms.extend(spc.alarms)

    return DashboardData(
        processes=states,
        latest_optimizations=latest_optimizations,
        spc_results=spc_results,
        active_alarms=active_alarms,
    )


# -----------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------

@router.get("/health")
async def health() -> dict:
    """Liveness / readiness probe."""
    return {
        "status": "healthy",
        "processes_tracked": len(holder.process_ids),
        "stream_processor": stream_processor.status(),
    }
