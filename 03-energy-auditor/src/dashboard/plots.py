"""Plotly Dash interactive dashboard embedded in FastAPI.

Provides real-time energy visualization:
- Hourly load profile with TOU color coding
- Demand timeline with contract demand line
- Power factor trend chart
- Anomaly timeline
- TOU cost breakdown pie chart
- KPI cards
"""

import uuid
from datetime import datetime, timedelta, timezone

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, callback, dash_table, dcc, html

from src.services.anomaly_detector import run_full_anomaly_detection
from src.services.consumption_analyzer import (
    build_load_profile,
    generate_tou_cost_breakdown,
)
from src.services.meter_reader import resample_readings
from src.utils.tariff import get_tariff, classify_tou_period

# ── Dash app (runs as a sub-application) ──────────────────────────────────

dash_app = dash.Dash(
    __name__,
    server=False,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)

# Shared data store (populated by FastAPI endpoints)
_data_store: dict[str, pd.DataFrame] = {}
_facility_store: dict[str, dict] = {}


def update_data(facility_id: str, readings_df: pd.DataFrame, facility_info: dict) -> None:
    """Push new data into the dashboard's store."""
    _data_store[facility_id] = readings_df
    _facility_store[facility_id] = facility_info


# ── Layout ─────────────────────────────────────────────────────────────────

def _build_layout() -> html.Div:
    """Build the Dash layout with KPI cards and charts."""
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Energy Auditor Dashboard", className="text-primary mb-0"), width=8),
            dbc.Col(
                dcc.Dropdown(
                    id="facility-selector",
                    placeholder="Select facility...",
                    className="mt-2",
                ),
                width=4,
            ),
        ], className="my-3"),

        # KPI Row
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Total Consumption", className="text-muted"),
                    html.H3(id="kpi-total-kwh", children="--"),
                ])
            ], className="shadow-sm"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Peak Demand", className="text-muted"),
                    html.H3(id="kpi-peak-kw", children="--"),
                ])
            ], className="shadow-sm"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Avg Power Factor", className="text-muted"),
                    html.H3(id="kpi-avg-pf", children="--"),
                ])
            ], className="shadow-sm"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Anomalies", className="text-muted"),
                    html.H3(id="kpi-anomalies", children="--"),
                ])
            ], className="shadow-sm"), width=3),
        ], className="mb-4"),

        # Charts Row 1
        dbc.Row([
            dbc.Col(dcc.Graph(id="load-profile-chart", className="shadow-sm"), width=7),
            dbc.Col(dcc.Graph(id="tou-pie-chart", className="shadow-sm"), width=5),
        ], className="mb-4"),

        # Charts Row 2
        dbc.Row([
            dbc.Col(dcc.Graph(id="demand-timeline-chart", className="shadow-sm"), width=6),
            dbc.Col(dcc.Graph(id="pf-trend-chart", className="shadow-sm"), width=6),
        ], className="mb-4"),

        # Anomaly table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Detected Anomalies")),
                    dbc.CardBody(dash_table.DataTable(
                        id="anomaly-table",
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Type", "id": "type"},
                            {"name": "Severity", "id": "severity"},
                            {"name": "Measured", "id": "measured"},
                            {"name": "Expected", "id": "expected"},
                            {"name": "Description", "id": "description"},
                        ],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
                        style_header={"backgroundColor": "#f1f3f7", "fontWeight": "600"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{severity} = "critical"'},
                             "backgroundColor": "#fee2e2", "color": "#991b1b"},
                            {"if": {"filter_query": '{severity} = "high"'},
                             "backgroundColor": "#fef3c7", "color": "#92400e"},
                        ],
                        page_size=10,
                    )),
                ], className="shadow-sm"),
            ], width=12),
        ], className="mb-4"),

    ], fluid=True)


dash_app.layout = _build_layout


# ── Callbacks ──────────────────────────────────────────────────────────────

@callback(
    Output("facility-selector", "options"),
    Input("facility-selector", "value"),
)
def update_facility_dropdown(_):
    """Populate facility dropdown from data store."""
    return [{"label": info.get("name", fid), "value": fid}
            for fid, info in _facility_store.items()]


@callback(
    Output("kpi-total-kwh", "children"),
    Output("kpi-peak-kw", "children"),
    Output("kpi-avg-pf", "children"),
    Output("kpi-anomalies", "children"),
    Output("load-profile-chart", "figure"),
    Output("tou-pie-chart", "figure"),
    Output("demand-timeline-chart", "figure"),
    Output("pf-trend-chart", "figure"),
    Output("anomaly-table", "data"),
    Input("facility-selector", "value"),
)
def update_dashboard(facility_id: str):
    """Update all dashboard components when facility is selected."""
    import plotly.graph_objects as go

    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white", annotations=[{"text": "Select a facility", "showarrow": False}])

    if not facility_id or facility_id not in _data_store:
        return ("--", "--", "--", "--", empty_fig, empty_fig, empty_fig, empty_fig, [])

    df = _data_store[facility_id].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    info = _facility_store.get(facility_id, {})

    # KPIs
    total_kwh = f"{df['active_energy_kwh'].sum():,.0f} kWh"
    peak_kw = f"{df['demand_kw'].max():,.0f} kW"
    avg_pf = f"{df['power_factor'].mean():.3f}" if "power_factor" in df.columns else "--"

    # Anomalies
    hourly = resample_readings(df, "1h")
    anomalies_resp = run_full_anomaly_detection(hourly, uuid.UUID(facility_id))
    anomaly_count = f"{anomalies_resp.total_anomalies}"

    # Load profile chart
    tariff_name = info.get("tariff_profile", "tou_general")
    tariff = get_tariff(tariff_name)
    df_h = df.copy()
    df_h["hour"] = df_h["timestamp"].dt.hour
    hourly_profile = df_h.groupby("hour").agg(
        avg_kw=("demand_kw", "mean"),
        max_kw=("demand_kw", "max"),
    ).reset_index()

    colors = []
    for _, row in hourly_profile.iterrows():
        rep_ts = df_h[df_h["hour"] == row["hour"]]["timestamp"].iloc[0] if not df_h[df_h["hour"] == row["hour"]].empty else datetime.now(timezone.utc)
        period = classify_tou_period(rep_ts, tariff)
        colors.append({"peak": "#e74c3c", "shoulder": "#3498db", "offpeak": "#27ae60"}.get(period, "#999"))

    load_fig = go.Figure()
    load_fig.add_trace(go.Bar(
        x=hourly_profile["hour"], y=hourly_profile["avg_kw"],
        name="Avg kW", marker_color=colors, opacity=0.85,
    ))
    load_fig.add_trace(go.Scatter(
        x=hourly_profile["hour"], y=hourly_profile["max_kw"],
        name="Max kW", mode="lines+markers",
        line=dict(color="#1a1a2e", dash="dash"),
    ))
    load_fig.update_layout(
        title="Hourly Load Profile",
        xaxis_title="Hour", yaxis_title="Demand (kW)",
        template="plotly_white", xaxis=dict(dtick=1),
        margin=dict(l=50, r=20, t=40, b=40), height=350,
    )

    # TOU pie chart
    tou_data = generate_tou_cost_breakdown(df, tariff_name)
    pie_labels, pie_values, pie_colors = [], [], []
    color_map = {"peak": "#e74c3c", "shoulder": "#3498db", "offpeak": "#27ae60"}
    for period in ["peak", "shoulder", "offpeak"]:
        cost = tou_data.get(f"{period}_cost", 0)
        if cost > 0:
            pie_labels.append(f"{period.title()}")
            pie_values.append(cost)
            pie_colors.append(color_map[period])

    pie_fig = go.Figure(data=[go.Pie(
        labels=pie_labels, values=pie_values,
        marker=dict(colors=pie_colors), hole=0.4, textinfo="label+percent",
    )])
    pie_fig.update_layout(title="Cost by TOU Period", template="plotly_white",
                          height=350, margin=dict(l=20, r=20, t=40, b=20))

    # Demand timeline
    demand_fig = go.Figure()
    demand_fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["demand_kw"],
        name="Demand (kW)", mode="lines",
        line=dict(color="#3498db"),
    ))
    contract = info.get("contract_demand_kva", 500)
    demand_fig.add_hline(y=contract, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"Contract: {contract:.0f} kVA")
    demand_fig.update_layout(
        title="Demand Timeline", xaxis_title="Time", yaxis_title="Demand (kW)",
        template="plotly_white", height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    # PF trend
    pf_fig = go.Figure()
    if "power_factor" in df.columns:
        pf_fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["power_factor"],
            name="Power Factor", mode="lines",
            line=dict(color="#8e44ad"),
        ))
    pf_fig.add_hline(y=0.90, line_dash="dash", line_color="#e74c3c",
                     annotation_text="Penalty Threshold (0.90)")
    pf_fig.update_layout(
        title="Power Factor Trend", xaxis_title="Time", yaxis_title="PF",
        template="plotly_white", yaxis=dict(range=[0.6, 1.05]),
        height=350, margin=dict(l=50, r=20, t=40, b=40),
    )

    # Anomaly table data
    anomaly_rows = []
    for a in anomalies_resp.anomalies[:20]:
        anomaly_rows.append({
            "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M"),
            "type": a.anomaly_type,
            "severity": a.severity.value,
            "measured": f"{a.measured_value:.1f}",
            "expected": f"{a.expected_value:.1f}",
            "description": a.description[:80],
        })

    return (total_kwh, peak_kw, avg_pf, anomaly_count,
            load_fig, pie_fig, demand_fig, pf_fig, anomaly_rows)
