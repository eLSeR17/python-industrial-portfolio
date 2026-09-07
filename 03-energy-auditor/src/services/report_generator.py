"""Energy audit report generation using Jinja2 + Plotly.

Generates professional HTML/PDF-ready reports with:
- Executive summary
- Load profile charts (Plotly)
- Demand analysis with graphs
- Power factor trends
- Anomaly timeline
- TOU cost breakdown pie chart
- Benchmarking radar chart
- Actionable recommendations with cost/benefit
"""

import uuid
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy Audit Report – {{ facility_name }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; color: #1a1a2e; background: #f8f9fa; line-height: 1.6; }
  .container { max-width: 1100px; margin: 0 auto; padding: 2rem; }
  .header { background: linear-gradient(135deg, #0f3460, #16213e); color: white; padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; }
  .header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  .header .subtitle { opacity: 0.85; font-size: 0.95rem; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
  .kpi { background: white; border-radius: 10px; padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }
  .kpi .value { font-size: 1.8rem; font-weight: 700; color: #0f3460; }
  .kpi .label { font-size: 0.8rem; color: #666; margin-top: 0.3rem; }
  .kpi .delta { font-size: 0.85rem; margin-top: 0.3rem; }
  .kpi .delta.positive { color: #27ae60; }
  .kpi .delta.negative { color: #e74c3c; }
  .section { background: white; border-radius: 10px; padding: 1.8rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
  .section h2 { font-size: 1.2rem; color: #0f3460; margin-bottom: 1rem; border-bottom: 2px solid #e8ecf1; padding-bottom: 0.5rem; }
  .chart-container { width: 100%; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
  th, td { padding: 0.7rem 1rem; text-align: left; border-bottom: 1px solid #eee; font-size: 0.9rem; }
  th { background: #f1f3f7; font-weight: 600; color: #333; }
  .recommendation { border-left: 4px solid #0f3460; padding: 1rem 1.2rem; margin-bottom: 1rem; background: #f8f9fc; border-radius: 0 8px 8px 0; }
  .recommendation .priority { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }
  .priority.critical { background: #fee2e2; color: #991b1b; }
  .priority.high { background: #fef3c7; color: #92400e; }
  .priority.medium { background: #dbeafe; color: #1e40af; }
  .priority.low { background: #e0e7ff; color: #3730a3; }
  .footer { text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem; padding: 1rem; }
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>Industrial Energy Audit Report</h1>
  <div class="subtitle">{{ facility_name }} ({{ facility_code }}) &bull; {{ period_start }} to {{ period_end }}</div>
  <div class="subtitle">Generated: {{ generated_at }}</div>
</div>

<div class="kpi-grid">
  <div class="kpi">
    <div class="value">{{ "{:,.0f}".format(total_kwh) }}</div>
    <div class="label">Total kWh</div>
  </div>
  <div class="kpi">
    <div class="value">${{ "{:,.0f}".format(current_monthly_cost) }}</div>
    <div class="label">Est. Monthly Cost</div>
  </div>
  <div class="kpi">
    <div class="value">{{ "{:,.0f}".format(peak_demand_kw) }}</div>
    <div class="label">Peak Demand (kW)</div>
  </div>
  <div class="kpi">
    <div class="value">{{ "{:.3f}".format(avg_pf) }}</div>
    <div class="label">Avg Power Factor</div>
  </div>
  <div class="kpi">
    <div class="value">${{ "{:,.0f}".format(annual_savings) }}</div>
    <div class="label">Annual Savings Potential</div>
    <div class="delta positive">{{ savings_pct }}% reduction</div>
  </div>
  <div class="kpi">
    <div class="value">{{ anomaly_count }}</div>
    <div class="label">Anomalies Detected</div>
  </div>
</div>

<div class="section">
  <h2>Load Profile</h2>
  <div class="chart-container">{{ load_profile_chart }}</div>
</div>

<div class="section">
  <h2>TOU Cost Breakdown</h2>
  <div class="chart-container">{{ tou_breakdown_chart }}</div>
</div>

<div class="section">
  <h2>Demand Analysis</h2>
  <p>Average demand: <strong>{{ "{:,.1f}".format(avg_demand_kw) }} kW</strong> &bull;
     Peak: <strong>{{ "{:,.1f}".format(peak_demand_kw) }} kW</strong> &bull;
     Contract: <strong>{{ "{:,.0f}".format(contract_demand_kva) }} kVA</strong></p>
  <p>Demand utilization: <strong>{{ demand_utilization_pct }}%</strong> &bull;
     Hours exceeding contract: <strong>{{ demand_exceeded_hours }}</strong></p>
</div>

<div class="section">
  <h2>Power Factor</h2>
  <p>Average PF: <strong>{{ avg_pf }}</strong> &bull;
     Minimum PF: <strong>{{ min_pf }}</strong> &bull;
     Penalty threshold: <strong>0.90</strong></p>
  {% if pf_penalty_hours > 0 %}
  <p style="color:#e74c3c;">⚠ {{ pf_penalty_hours }} hours below threshold — estimated penalty: ${{ pf_penalty_usd }}/period</p>
  {% else %}
  <p style="color:#27ae60;">✓ No PF penalties in this period</p>
  {% endif %}
</div>

{% if anomalies|length > 0 %}
<div class="section">
  <h2>Anomalies Detected ({{ anomaly_count }})</h2>
  <table>
    <tr><th>Timestamp</th><th>Type</th><th>Severity</th><th>Description</th></tr>
    {% for a in anomalies[:20] %}
    <tr>
      <td>{{ a.timestamp.strftime('%Y-%m-%d %H:%M') }}</td>
      <td>{{ a.anomaly_type }}</td>
      <td><span class="priority {{ a.severity.value }}">{{ a.severity.value }}</span></td>
      <td>{{ a.description }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
{% endif %}

<div class="section">
  <h2>Recommendations</h2>
  {% for r in recommendations %}
  <div class="recommendation">
    <span class="priority {{ r.priority.value }}">{{ r.priority.value }}</span>
    <strong>{{ r.title }}</strong>
    <p>{{ r.description }}</p>
    <p>Estimated savings: <strong>${{ "{:,.0f}".format(r.estimated_savings_usd) }}/month</strong>
       &bull; Investment: ${{ "{:,.0f}".format(r.implementation_cost_usd) }}
       {% if r.payback_months %}&bull; Payback: {{ r.payback_months }} months{% endif %}</p>
  </div>
  {% endfor %}
  {% if not recommendations %}
  <p>No recommendations — facility is operating efficiently.</p>
  {% endif %}
</div>

<div class="footer">
  Industrial Energy Auditor v1.0 &bull; Report generated {{ generated_at }} &bull; Confidential
</div>
</div>
</body>
</html>
"""


def _make_load_profile_chart(load_profile: dict) -> str:
    """Generate Plotly HTML for the load profile bar chart."""
    buckets = load_profile.get("buckets", [])
    if not buckets:
        return "<p>No load profile data available.</p>"

    hours = [b["hour"] for b in buckets]
    avg_kw = [b["avg_kw"] for b in buckets]
    max_kw = [b["max_kw"] for b in buckets]

    # Color by tariff period
    colors = []
    for b in buckets:
        period = b.get("tariff_period", "shoulder")
        if period == "peak":
            colors.append("#e74c3c")
        elif period == "offpeak":
            colors.append("#27ae60")
        else:
            colors.append("#3498db")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=avg_kw, name="Avg Demand (kW)",
                         marker_color=colors, opacity=0.85))
    fig.add_trace(go.Scatter(x=hours, y=max_kw, name="Max Demand (kW)",
                             mode="lines+markers", line=dict(color="#1a1a2e", dash="dash")))
    fig.update_layout(
        title="Hourly Load Profile",
        xaxis_title="Hour of Day",
        yaxis_title="Demand (kW)",
        xaxis=dict(dtick=1),
        template="plotly_white",
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _make_tou_breakdown_chart(tou_data: dict) -> str:
    """Generate Plotly pie chart for TOU cost breakdown."""
    labels = []
    values = []
    colors = {"peak": "#e74c3c", "shoulder": "#3498db", "offpeak": "#27ae60"}

    for period in ["peak", "shoulder", "offpeak"]:
        cost = tou_data.get(f"{period}_cost", 0)
        if cost > 0:
            labels.append(f"{period.title()} (${cost:.2f})")
            values.append(cost)

    if not values:
        return "<p>No TOU cost data available.</p>"

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker=dict(colors=[colors.get(l.split("(")[0].strip().lower(), "#999") for l in labels]),
        hole=0.4,
        textinfo="label+percent",
    )])
    fig.update_layout(
        title="Cost by TOU Period",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def generate_audit_report(
    facility_name: str,
    facility_code: str,
    readings_df: pd.DataFrame,
    facility_id: uuid.UUID,
    load_profile: dict | None = None,
    tou_breakdown: dict | None = None,
    anomalies: list | None = None,
    recommendations: list | None = None,
    demand_analysis: dict | None = None,
    pf_analysis: dict | None = None,
    contract_demand_kva: float = 500.0,
    production_units: float | None = None,
    floor_area_sqm: float | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Generate a complete HTML energy audit report.

    Parameters
    ----------
    facility_name, facility_code : Facility identifiers.
    readings_df : Meter readings for the audit period.
    facility_id : Facility UUID.
    load_profile : Output of build_load_profile().
    tou_breakdown : Output of generate_tou_cost_breakdown().
    anomalies : List of AnomalyRecord.
    recommendations : List of SavingsRecommendation.
    demand_analysis : Output of analyze_demand().
    pf_analysis : Output of analyze_power_factor().
    contract_demand_kva : Contracted demand level.
    production_units, floor_area_sqm : For normalization metrics.

    Returns
    -------
    Tuple of (html_report, section_names).
    """
    df = readings_df.copy() if not readings_df.empty else pd.DataFrame()
    now = datetime.now(timezone.utc)

    # Compute metrics
    total_kwh = float(df["active_energy_kwh"].sum()) if not df.empty and "active_energy_kwh" in df.columns else 0.0
    peak_demand_kw = float(df["demand_kw"].max()) if not df.empty else 0.0
    avg_demand_kw = float(df["demand_kw"].mean()) if not df.empty else 0.0
    avg_pf = float(df["power_factor"].mean()) if not df.empty and "power_factor" in df.columns else 0.92
    min_pf = float(df["power_factor"].min()) if not df.empty and "power_factor" in df.columns else 0.80

    # Cost
    from src.utils.tariff import get_tariff, calculate_monthly_bill
    tariff = get_tariff("tou_general")
    bill = calculate_monthly_bill(df, tariff, float(df["demand_kva"].max()) if "demand_kva" in df.columns else peak_demand_kw) if not df.empty else {"total": 0}

    # Anomalies
    anomaly_list = anomalies or []
    anomaly_count = len(anomaly_list)

    # Recommendations
    recs = recommendations or []
    annual_savings = sum(r.estimated_savings_usd for r in recs) * 12
    current_cost = bill["total"]
    savings_pct = round(annual_savings / (current_cost * 12) * 100 if current_cost > 0 else 0, 1)

    # Demand
    demand_util = demand_analysis.get("demand_utilization_pct", 0) if demand_analysis else 0
    demand_exceeded = demand_analysis.get("demand_exceeded_hours", 0) if demand_analysis else 0

    # PF
    pf_hours = pf_analysis.get("penalty_hours", 0) if pf_analysis else 0
    pf_penalty = pf_analysis.get("estimated_penalty_usd", 0) if pf_analysis else 0

    # Charts
    load_chart = _make_load_profile_chart(load_profile) if load_profile else "<p>No load profile data.</p>"
    tou_chart = _make_tou_breakdown_chart(tou_breakdown) if tou_breakdown else "<p>No TOU data.</p>"

    # Render
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        facility_name=facility_name,
        facility_code=facility_code,
        period_start=df["timestamp"].min().strftime("%Y-%m-%d") if not df.empty else "N/A",
        period_end=df["timestamp"].max().strftime("%Y-%m-%d") if not df.empty else "N/A",
        generated_at=now.strftime("%Y-%m-%d %H:%M UTC"),
        total_kwh=total_kwh,
        current_monthly_cost=round(current_cost, 2),
        peak_demand_kw=round(peak_demand_kw, 1),
        avg_pf=round(avg_pf, 3),
        avg_demand_kw=round(avg_demand_kw, 1),
        annual_savings=round(annual_savings, 2),
        savings_pct=savings_pct,
        anomaly_count=anomaly_count,
        contract_demand_kva=contract_demand_kva,
        demand_utilization_pct=round(demand_util, 1),
        demand_exceeded_hours=demand_exceeded,
        min_pf=round(min_pf, 3),
        pf_penalty_hours=pf_hours,
        pf_penalty_usd=round(pf_penalty, 2),
        load_profile_chart=load_chart,
        tou_breakdown_chart=tou_chart,
        anomalies=anomaly_list,
        recommendations=recs,
    )

    sections = [
        "Executive Summary", "Load Profile", "TOU Cost Breakdown",
        "Demand Analysis", "Power Factor", "Anomalies", "Recommendations",
    ]

    return html, sections
