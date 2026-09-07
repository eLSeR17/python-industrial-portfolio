"""FastAPI REST endpoints for the Energy Auditor."""

import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.models.schemas import (
    AnomalyListResponse,
    AuditReportRequest,
    AuditReportResponse,
    BenchmarkResponse,
    DemandAnalysisResponse,
    FacilityCreate,
    FacilityResponse,
    LoadProfileResponse,
    MeterReadingBatch,
    MeterReadingResponse,
    PowerFactorAnalysis,
    SavingsReportResponse,
)
from src.services.anomaly_detector import run_full_anomaly_detection
from src.services.benchmark_engine import benchmark_facilities, calculate_enpi
from src.services.consumption_analyzer import (
    analyze_demand,
    analyze_power_factor,
    build_load_profile,
    generate_tou_cost_breakdown,
)
from src.services.meter_reader import generate_synthetic_readings, resample_readings
from src.services.optimizer import generate_savings_report
from src.services.report_generator import generate_audit_report

router = APIRouter(prefix="/api/v1", tags=["energy-auditor"])

# ── In-memory store (production would use PostgreSQL via SQLAlchemy) ──────

_facilities: dict[str, dict] = {}
_readings: dict[str, pd.DataFrame] = {}


def _get_facility_or_404(facility_id: uuid.UUID) -> dict:
    fid = str(facility_id)
    if fid not in _facilities:
        raise HTTPException(status_code=404, detail=f"Facility {fid} not found")
    return _facilities[fid]


def _get_readings_df(facility_id: uuid.UUID) -> pd.DataFrame:
    fid = str(facility_id)
    return _readings.get(fid, pd.DataFrame())


# ── Facilities ─────────────────────────────────────────────────────────────

@router.post("/facilities", response_model=FacilityResponse, status_code=201)
def create_facility(data: FacilityCreate) -> FacilityResponse:
    """Register a new industrial facility for auditing."""
    fid = str(uuid.uuid4())
    _facilities[fid] = {
        "id": uuid.UUID(fid),
        "name": data.name,
        "code": data.code,
        "address": data.address,
        "facility_type": data.facility_type,
        "contract_demand_kva": data.contract_demand_kva,
        "tariff_profile": data.tariff_profile,
        "timezone": data.timezone,
    }
    return FacilityResponse(**_facilities[fid])


@router.get("/facilities/{facility_id}", response_model=FacilityResponse)
def get_facility(facility_id: uuid.UUID) -> FacilityResponse:
    """Get facility details."""
    f = _get_facility_or_404(facility_id)
    return FacilityResponse(**f)


# ── Meter Readings ─────────────────────────────────────────────────────────

@router.post("/readings/ingest", response_model=MeterReadingResponse)
def ingest_readings(batch: MeterReadingBatch) -> MeterReadingResponse:
    """Ingest a batch of smart meter readings.

    Accepts up to 10,000 readings per request. Duplicate timestamps for the
    same facility are skipped (last-write-wins within the batch).
    """
    if not batch.readings:
        raise HTTPException(status_code=400, detail="Empty readings batch")

    facility_id = batch.readings[0].facility_id
    records = []
    for r in batch.readings:
        if r.facility_id != facility_id:
            raise HTTPException(status_code=400, detail="All readings must target the same facility")
        records.append({
            "facility_id": r.facility_id,
            "meter_id": r.meter_id,
            "timestamp": r.timestamp,
            "active_energy_kwh": r.active_energy_kwh,
            "reactive_energy_kvarh": r.reactive_energy_kvarh,
            "apparent_energy_kvah": r.apparent_energy_kvah,
            "demand_kw": r.demand_kw,
            "demand_kva": r.demand_kva,
            "power_factor": r.power_factor,
            "voltage_avg": r.voltage_avg,
            "frequency_hz": r.frequency_hz,
            "current_a": r.current_a,
            "thd_voltage_pct": r.thd_voltage_pct,
            "temperature_c": r.temperature_c,
        })

    new_df = pd.DataFrame(records)
    fid = str(facility_id)

    if fid in _readings:
        existing = _readings[fid]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        duplicates = len(new_df) - (len(combined) - len(existing))
        _readings[fid] = combined
    else:
        _readings[fid] = new_df.sort_values("timestamp").reset_index(drop=True)
        duplicates = 0

    return MeterReadingResponse(
        ingested=len(batch.readings),
        facility_id=facility_id,
        duplicates_skipped=max(0, duplicates),
    )


@router.post("/readings/generate/{facility_id}", response_model=MeterReadingResponse)
def generate_readings(
    facility_id: uuid.UUID,
    days: int = Query(30, ge=1, le=365),
    meter_id: str = Query("MTR-001"),
) -> MeterReadingResponse:
    """Generate synthetic meter readings for testing/demo purposes.

    Creates realistic consumption data with daily, weekly, and seasonal patterns.
    """
    _get_facility_or_404(facility_id)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    df = generate_synthetic_readings(facility_id, meter_id, start, end)

    fid = str(facility_id)
    if fid in _readings:
        combined = pd.concat([_readings[fid], df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        _readings[fid] = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        _readings[fid] = df.sort_values("timestamp").reset_index(drop=True)

    return MeterReadingResponse(ingested=len(df), facility_id=facility_id)


# ── Analysis Endpoints ─────────────────────────────────────────────────────

@router.get("/dashboard/{facility_id}", response_model=LoadProfileResponse)
def get_dashboard(facility_id: uuid.UUID) -> LoadProfileResponse:
    """Get load profile and key metrics for the dashboard."""
    f = _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found for this facility")
    return build_load_profile(df, facility_id, f["tariff_profile"])


@router.get("/analysis/demand/{facility_id}", response_model=DemandAnalysisResponse)
def get_demand_analysis(facility_id: uuid.UUID) -> DemandAnalysisResponse:
    """Analyze demand patterns and contract utilization."""
    f = _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found")
    return analyze_demand(df, facility_id, f["contract_demand_kva"])


@router.get("/analysis/power-factor/{facility_id}", response_model=PowerFactorAnalysis)
def get_power_factor_analysis(facility_id: uuid.UUID) -> PowerFactorAnalysis:
    """Analyze power factor with capacitor bank sizing recommendations."""
    f = _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found")
    return analyze_power_factor(df, facility_id, f["tariff_profile"])


@router.get("/analysis/tou-cost/{facility_id}")
def get_tou_cost_breakdown(facility_id: uuid.UUID) -> dict:
    """Break down energy cost by time-of-use period."""
    f = _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found")
    return generate_tou_cost_breakdown(df, f["tariff_profile"])


@router.get("/analysis/anomalies/{facility_id}", response_model=AnomalyListResponse)
def get_anomalies(facility_id: uuid.UUID) -> AnomalyListResponse:
    """Run all anomaly detectors and return flagged events."""
    _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found")
    hourly = resample_readings(df, "1h")
    return run_full_anomaly_detection(hourly, facility_id)


# ── Audit & Reports ────────────────────────────────────────────────────────

@router.get("/audit/{facility_id}", response_model=AuditReportResponse)
def run_audit(
    facility_id: uuid.UUID,
    production_units: float | None = Query(None),
    floor_area_sqm: float | None = Query(None),
) -> AuditReportResponse:
    """Run a full energy audit and generate an HTML report."""
    f = _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found for audit")

    hourly = resample_readings(df, "1h")
    load_profile = build_load_profile(hourly, facility_id, f["tariff_profile"])
    tou_breakdown = generate_tou_cost_breakdown(hourly, f["tariff_profile"])
    demand = analyze_demand(hourly, facility_id, f["contract_demand_kva"])
    pf = analyze_power_factor(hourly, facility_id, f["tariff_profile"])
    anomalies_resp = run_full_anomaly_detection(hourly, facility_id)

    # Savings report for recommendations
    savings = generate_savings_report(hourly, facility_id, f["tariff_profile"])

    html, sections = generate_audit_report(
        facility_name=f["name"],
        facility_code=f["code"],
        readings_df=hourly,
        facility_id=facility_id,
        load_profile=load_profile.model_dump(),
        tou_breakdown=tou_breakdown,
        anomalies=anomalies_resp.anomalies,
        recommendations=savings.recommendations,
        demand_analysis=demand.model_dump(),
        pf_analysis=pf.model_dump(),
        contract_demand_kva=f["contract_demand_kva"],
        production_units=production_units,
        floor_area_sqm=floor_area_sqm,
    )

    return AuditReportResponse(
        facility_id=facility_id,
        report_html=html,
        generated_at=datetime.now(timezone.utc),
        sections=sections,
    )


@router.get("/benchmark", response_model=BenchmarkResponse)
def get_benchmark() -> BenchmarkResponse:
    """Benchmark all registered facilities against each other."""
    if not _facilities:
        return BenchmarkResponse(entries=[], rank_by_enpi=[])

    facility_data = []
    for fid_str, f in _facilities.items():
        fid = f["id"]
        df = _get_readings_df(fid)
        total_kwh = float(df["active_energy_kwh"].sum()) if not df.empty else 0.0
        avg_pf = float(df["power_factor"].mean()) if not df.empty and "power_factor" in df.columns else 0.92

        facility_data.append({
            "facility_id": fid,
            "facility_name": f["name"],
            "total_kwh": total_kwh,
            "avg_power_factor": avg_pf,
        })

    return benchmark_facilities(facility_data)


@router.get("/savings-report/{facility_id}", response_model=SavingsReportResponse)
def get_savings_report(
    facility_id: uuid.UUID,
    production_units: float | None = Query(None),
) -> SavingsReportResponse:
    """Generate a savings report with recommendations."""
    f = _get_facility_or_404(facility_id)
    df = _get_readings_df(facility_id)
    if df.empty:
        raise HTTPException(status_code=404, detail="No readings found")
    hourly = resample_readings(df, "1h")
    return generate_savings_report(hourly, facility_id, f["tariff_profile"], production_units)
