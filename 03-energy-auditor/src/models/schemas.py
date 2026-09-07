"""Pydantic schemas for API request / response validation."""

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────

class TariffPeriod(str, Enum):
    PEAK = "peak"
    SHOULDER = "shoulder"
    OFFPEAK = "offpeak"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Meter Reading Ingestion ───────────────────────────────────────────────

class MeterReadingCreate(BaseModel):
    """Single meter reading to ingest."""
    meter_id: str = Field(..., min_length=1, max_length=50)
    facility_id: uuid.UUID
    timestamp: datetime
    active_energy_kwh: float = Field(..., ge=0)
    reactive_energy_kvarh: float = Field(0.0, ge=0)
    apparent_energy_kvah: float = Field(0.0, ge=0)
    demand_kw: float = Field(0.0, ge=0)
    demand_kva: float = Field(0.0, ge=0)
    power_factor: float = Field(1.0, ge=0.0, le=1.0)
    voltage_avg: float | None = Field(None, gt=0)
    frequency_hz: float | None = Field(None, gt=0)
    current_a: float | None = Field(None, ge=0)
    thd_voltage_pct: float | None = Field(None, ge=0)
    temperature_c: float | None = None

    @field_validator("demand_kva")
    @classmethod
    def kva_must_be_ge_kw(cls, v: float, info) -> float:
        if "demand_kw" in info.data and v < info.data["demand_kw"]:
            raise ValueError("demand_kva must be >= demand_kw")
        return v


class MeterReadingBatch(BaseModel):
    """Batch of meter readings for bulk ingestion."""
    readings: list[MeterReadingCreate] = Field(..., min_length=1, max_length=10_000)


class MeterReadingResponse(BaseModel):
    """Response after ingesting readings."""
    ingested: int
    facility_id: uuid.UUID
    duplicates_skipped: int = 0


# ── Facility ───────────────────────────────────────────────────────────────

class FacilityCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., min_length=1, max_length=50)
    address: str | None = None
    facility_type: str = "manufacturing"
    contract_demand_kva: float = Field(..., gt=0)
    tariff_profile: str = "tou_general"
    timezone: str = "UTC"


class FacilityResponse(BaseModel):
    id: uuid.UUID
    name: str
    code: str
    facility_type: str
    contract_demand_kva: float
    tariff_profile: str

    model_config = {"from_attributes": True}


# ── Load Profile ───────────────────────────────────────────────────────────

class LoadProfileBucket(BaseModel):
    """One bucket of a load profile (e.g. one hour of the day)."""
    hour: int = Field(..., ge=0, le=23)
    avg_kw: float
    max_kw: float
    min_kw: float
    avg_pf: float
    readings_count: int
    tariff_period: TariffPeriod


class LoadProfileResponse(BaseModel):
    facility_id: uuid.UUID
    period_start: datetime
    period_end: datetime
    buckets: list[LoadProfileBucket]
    baseload_kw: float = Field(..., description="Minimum sustained load")
    peak_demand_kw: float
    peak_demand_timestamp: datetime | None
    load_factor: float = Field(..., description="Average / peak ratio (0-1)")


# ── Demand Analysis ────────────────────────────────────────────────────────

class DemandAnalysisResponse(BaseModel):
    facility_id: uuid.UUID
    period_start: datetime
    period_end: datetime
    avg_demand_kw: float
    peak_demand_kw: float
    peak_demand_timestamp: datetime | None
    contract_demand_kva: float
    demand_utilization_pct: float = Field(..., description="Peak / contract demand %")
    demand_exceeded_hours: int = Field(..., description="Hours above contract demand")
    demand_exceeded_pct: float
    recommended_contract_kva: float = Field(..., description="Optimal contract demand")


# ── Power Factor ───────────────────────────────────────────────────────────

class PowerFactorAnalysis(BaseModel):
    facility_id: uuid.UUID
    period_start: datetime
    period_end: datetime
    avg_power_factor: float
    min_power_factor: float
    avg_reactive_kvar: float
    penalty_hours: int = Field(..., description="Hours below PF threshold")
    estimated_penalty_usd: float
    capacitor_bank_kvar: float = Field(..., description="Recommended capacitor bank size")
    estimated_pf_after_correction: float
    annual_savings_usd: float


# ── Anomaly ────────────────────────────────────────────────────────────────

class AnomalyRecord(BaseModel):
    timestamp: datetime
    facility_id: uuid.UUID
    anomaly_type: str = Field(..., description="spike | baseline_shift | equipment_left_on | pf_drop")
    severity: AnomalySeverity
    measured_value: float
    expected_value: float
    deviation_pct: float
    description: str
    meter_id: str | None = None


class AnomalyListResponse(BaseModel):
    facility_id: uuid.UUID
    period_start: datetime
    period_end: datetime
    total_anomalies: int
    anomalies: list[AnomalyRecord]


# ── Benchmark ──────────────────────────────────────────────────────────────

class BenchmarkEntry(BaseModel):
    facility_id: uuid.UUID
    facility_name: str
    kwh_per_unit: float = Field(..., description="kWh per production unit")
    kwh_per_sqm: float = Field(..., description="kWh per square meter")
    demand_intensity_kw_per_unit: float
    power_factor_avg: float
    enpi_score: float = Field(..., description="Energy Performance Indicator (lower is better)")
    vs_baseline_pct: float = Field(..., description="% change vs baseline year")
    iso50001_compliant: bool


class BenchmarkResponse(BaseModel):
    entries: list[BenchmarkEntry]
    rank_by_enpi: list[uuid.UUID] = Field(..., description="Facility IDs ranked best to worst")


# ── Savings Report ─────────────────────────────────────────────────────────

class SavingsRecommendation(BaseModel):
    category: str
    title: str
    description: str
    estimated_savings_usd: float
    estimated_savings_kwh: float
    implementation_cost_usd: float
    payback_months: float | None
    priority: Priority


class SavingsReportResponse(BaseModel):
    facility_id: uuid.UUID
    report_date: datetime
    current_monthly_cost_usd: float
    potential_monthly_cost_usd: float
    monthly_savings_usd: float
    annual_savings_usd: float
    savings_pct: float
    recommendations: list[SavingsRecommendation]
    total_capex_usd: float
    weighted_payback_months: float | None


# ── Audit Report ───────────────────────────────────────────────────────────

class AuditReportRequest(BaseModel):
    facility_id: uuid.UUID
    period_start: datetime
    period_end: datetime
    include_recommendations: bool = True
    production_units: float | None = Field(None, description="Total production output for normalization")
    floor_area_sqm: float | None = Field(None, description="Facility floor area for per-sqm metrics")


class AuditReportResponse(BaseModel):
    facility_id: uuid.UUID
    report_html: str
    generated_at: datetime
    sections: list[str]
