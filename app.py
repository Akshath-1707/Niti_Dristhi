"""
=============================================================================
NITI DRISHTI: Urban Education Infrastructure Decision Support System (DSS)
=============================================================================
Sovereign Municipal Intelligence Platform | Chhatrapati Sambhajinagar
Design System: Nordic Minimalist Slate & Deep Pine (Option B)
=============================================================================
"""

import os
import io
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. CORE DATA RESOLUTION & ENGINE INITIALIZATION
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Import analytics, RAG, and cognitive reasoning modules
from backend.analytics.readiness_index import EducationReadinessCalculator
from backend.analytics.forecasting import EducationForecaster
from backend.rag.pdf_exporter import generate_policy_pdf
from backend.analytics.geocoding import assign_school_coordinates, create_gis_infrastructure_map

# Cognitive Decision Support Modules (CDSS)
from backend.core.config import BENCHMARKS, DEFAULT_WEIGHTS, PRIORITY_THRESHOLDS
from backend.utils.data_quality import audit_school_record
from backend.reasoning.gaps import analyze_school_infrastructure_gaps
from backend.reasoning.decision_engine import formulate_phased_action_plan
from backend.reasoning.explanation import (
    generate_causal_why_factors,
    generate_citizen_explanation,
    generate_administrative_narrative
)

try:
    from backend.rag.service import generate_education_policy_brief
    RAG_READY = True
except Exception as e:
    print(f"[WARNING] RAG engine import note: {e}")
    RAG_READY = False

# Locate master processed dataset
def find_dataset(filenames):
    for f in filenames:
        p = BASE_DIR / f
        if p.exists():
            return p
    return None

school_file = find_dataset([
    "data/database_ready_schools_urc.csv",
    "data/processed/database_ready_schools_urc.csv",
    "data/model_school_features.csv"
])

if not school_file:
    raise FileNotFoundError("Could not find processed school dataset in data/ directory.")

df_raw = pd.read_csv(school_file)

# Standardize columns to guarantee 100% data consistency
if "total_enrolment" not in df_raw.columns and "total_students" in df_raw.columns:
    df_raw["total_enrolment"] = df_raw["total_students"]

if "is_operational" in df_raw.columns:
    df_raw["is_operational"] = (
        df_raw["is_operational"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "operational"])
    )
else:
    df_raw["is_operational"] = True

# Run Readiness Index Engine
calc = EducationReadinessCalculator()
df = calc.evaluate_schools(df_raw)
df = assign_school_coordinates(df)
district_summary = calc.get_district_summary(df)

# Load 2021-2037 Forecast Data (anchored to exact 278,945 actual baseline)
forecast_file = find_dataset(["data/timeseries/yearly_enrollment_forecast_2021_2037.csv"])
if forecast_file:
    df_forecast = pd.read_csv(forecast_file)
else:
    forecaster = EducationForecaster()
    df_forecast = forecaster.save_forecast()
    df_forecast = pd.read_csv(df_forecast)

# Dropdown options
block_options = [{"label": "All Areas / Blocks", "value": "ALL"}] + [
    {"label": str(b), "value": str(b)} for b in sorted(df["block_name"].dropna().unique())
] if "block_name" in df.columns else []

mgmt_options = [{"label": "All School Types", "value": "ALL"}] + [
    {"label": str(m), "value": str(m)} for m in sorted(df["school_management"].dropna().unique())
] if "school_management" in df.columns else []

category_options = [{"label": "All Grade Levels", "value": "ALL"}] + [
    {"label": str(c), "value": str(c)} for c in sorted(df["school_category"].dropna().unique())
] if "school_category" in df.columns else []

tier_options = [
    {"label": "All Status Tiers", "value": "ALL"},
    {"label": "Critical Attention Required (< 55)", "value": "CRITICAL_INTERVENTION"},
    {"label": "Developing Needs (55 - 74)", "value": "DEVELOPING_NEEDS"},
    {"label": "Optimal / Resilient (>= 75)", "value": "OPTIMAL_RESILIENT"},
]

school_select_options = [
    {"label": f"{r['school_name']} (UDISE: {r['udise_code']})", "value": str(r["udise_code"])}
    for _, r in df.sort_values("total_enrolment", ascending=False).iterrows()
]

# ---------------------------------------------------------------------------
# 2. DESIGN TOKENS: "NORDIC MINIMALIST SLATE & DEEP PINE" (OPTION B)
# ---------------------------------------------------------------------------
COLOR_PINE = "#0F766E"         # Deep Pine Teal
COLOR_CYAN = "#0284C7"         # Steel Cyan
COLOR_SAGE = "#16A34A"         # Sage Green
COLOR_OCHRE = "#B45309"        # Warm Ochre
COLOR_CRIMSON = "#B91C1C"      # Rust Crimson
COLOR_PITCH = "#020617"        # Pitch Slate
COLOR_MUTED = "#475569"        # Cool Grey
COLOR_BORDER = "#CBD5E1"       # Slate Border

PLOT_BG = "rgba(0,0,0,0)"

# ---------------------------------------------------------------------------
# 3. DASH APPLICATION LAYOUT
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="NITI DRISHTI | Decision Support System"
)

app.layout = html.Div(
    className="dashboard-container",
    children=[
        # Stores for smooth RAG streaming and PDF generation
        dcc.Store(id="rag-full-text-store", data=""),
        dcc.Store(id="rag-stream-cursor", data=0),
        dcc.Interval(id="rag-typewriter-interval", interval=25, n_intervals=0, disabled=True),
        dcc.Download(id="download-policy-pdf"),

        # ===================================================================
        # HEADER: Clean Institutional Identity
        # ===================================================================
        html.Div(
            className="platform-header",
            children=[
                html.Div([
                    html.H1(
                        "NITI DRISHTI",
                        style={"margin": "0", "fontSize": "26px", "fontWeight": "900", "letterSpacing": "1.5px", "color": COLOR_PITCH}
                    ),
                    html.P(
                        "Urban Infrastructure Analytics & Decision Support System • Chhatrapati Sambhajinagar",
                        style={"margin": "3px 0 0 0", "fontSize": "13px", "color": COLOR_PINE, "fontWeight": "600"}
                    )
                ]),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                    children=[
                        html.Span("STATUS: PILOT AUDIT ACTIVE", className="status-pill status-pill-green"),
                        html.Span("HORIZON: VISION 2037", className="status-pill status-pill-cyan")
                    ]
                )
            ]
        ),

        # ===================================================================
        # COGNITIVE DECISION SUPPORT: MULTI-PERSONA PORTAL NAVIGATION
        # ===================================================================
        html.Div(
            className="portal-nav-bar",
            children=[
                html.Span("ACTIVE PORTAL VIEW:", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_MUTED, "letterSpacing": "0.5px", "marginLeft": "4px"}),
                html.Button("🏛️ Public Citizen View", id="portal-tab-citizen", n_clicks=0, className="portal-tab-btn portal-tab-btn-active"),
                html.Button("🏫 School Administrator Portal", id="portal-tab-school", n_clicks=0, className="portal-tab-btn"),
                html.Button("📋 Government Administrator Portal", id="portal-tab-govt", n_clicks=0, className="portal-tab-btn"),
                dcc.Store(id="active-portal-store", data="citizen"),
            ]
        ),

        # ===================================================================
        # TOP COMMAND FILTER RIBBON (Moved to Top — Fixes Overshadowing)
        # ===================================================================
        html.Div(
            className="command-ribbon",
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "10px"},
                    children=[
                        html.Span("Select Area & Category Filters (Controls Dashboard & KPIs)", className="ribbon-title"),
                        html.Span("Real-Time Filter Active", style={"fontSize": "11px", "color": COLOR_MUTED, "fontWeight": "600"})
                    ]
                ),
                html.Div(
                    className="ribbon-grid",
                    children=[
                        html.Div([
                            html.Label("Select Area / Block", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-block",
                                options=block_options,
                                value="ALL",
                                clearable=False
                            )
                        ]),
                        html.Div([
                            html.Label("School Type", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-mgmt",
                                options=mgmt_options,
                                value="ALL",
                                clearable=False
                            )
                        ]),
                        html.Div([
                            html.Label("Grade Level", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-category",
                                options=category_options,
                                value="ALL",
                                clearable=False
                            )
                        ]),
                        html.Div([
                            html.Label("Performance Status", className="filter-label"),
                            dcc.Dropdown(
                                id="filter-tier",
                                options=tier_options,
                                value="ALL",
                                clearable=False
                            )
                        ]),
                    ]
                )
            ]
        ),

        # ===================================================================
        # GOVERNMENT ADMIN "WHAT-IF" SIMULATION SANDBOX (Dynamic Weight Recalibration)
        # ===================================================================
        html.Div(
            id="section-govt-sim-sandbox",
            className="sandbox-container",
            style={"display": "none"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.Span("GOVERNMENT DECISION SUPPORT: WHAT-IF SIMULATION SANDBOX", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_CYAN, "letterSpacing": "0.5px"}),
                            html.H4("Dynamic Multi-Criteria Weight Recalibration & Capital Infrastructure Simulation", style={"margin": "2px 0 0 0", "fontSize": "15px", "color": COLOR_PITCH}),
                        ]),
                        html.Span("COGNITIVE WHAT-IF ENGINE", className="status-pill status-pill-cyan")
                    ]
                ),
                html.P(
                    "Simulate capital classroom infusion and adjust statutory pillar weights on the fly. Watch the District Readiness Score and the 2037 Classroom Deficit recalculate live across all charts.",
                    style={"fontSize": "12.5px", "color": COLOR_MUTED, "marginBottom": "14px"}
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "16px"},
                    children=[
                        html.Div([
                            html.Label("Capital Classroom Infusion", style={"fontSize": "11.5px", "fontWeight": "700", "color": COLOR_PITCH}),
                            dcc.Slider(id="sim-classroom-slider", min=0, max=1000, step=50, value=0, marks={0: "+0", 500: "+500", 1000: "+1k"}, tooltip={"placement": "bottom", "always_visible": False})
                        ]),
                        html.Div([
                            html.Label("Equity Weight (w_E)", style={"fontSize": "11.5px", "fontWeight": "700", "color": COLOR_PITCH}),
                            dcc.Slider(id="sim-w-equity", min=0.10, max=0.40, step=0.05, value=0.25, marks={0.1: "0.1", 0.25: "0.25", 0.4: "0.4"}, tooltip={"placement": "bottom", "always_visible": False})
                        ]),
                        html.Div([
                            html.Label("Promotion Weight (w_P)", style={"fontSize": "11.5px", "fontWeight": "700", "color": COLOR_PITCH}),
                            dcc.Slider(id="sim-w-promo", min=0.10, max=0.40, step=0.05, value=0.30, marks={0.1: "0.1", 0.3: "0.3", 0.4: "0.4"}, tooltip={"placement": "bottom", "always_visible": False})
                        ]),
                        html.Div([
                            html.Label("Retention Weight (w_R)", style={"fontSize": "11.5px", "fontWeight": "700", "color": COLOR_PITCH}),
                            dcc.Slider(id="sim-w-ret", min=0.10, max=0.40, step=0.05, value=0.25, marks={0.1: "0.1", 0.25: "0.25", 0.4: "0.4"}, tooltip={"placement": "bottom", "always_visible": False})
                        ]),
                        html.Div([
                            html.Label("Capacity Weight (w_C)", style={"fontSize": "11.5px", "fontWeight": "700", "color": COLOR_PITCH}),
                            dcc.Slider(id="sim-w-cap", min=0.10, max=0.40, step=0.05, value=0.20, marks={0.1: "0.1", 0.2: "0.2", 0.4: "0.4"}, tooltip={"placement": "bottom", "always_visible": False})
                        ]),
                    ]
                )
            ]
        ),

        # ===================================================================
        # SCHOOL ADMINISTRATOR PORTAL: DATA ENTRY & DIAGNOSTIC AUDIT
        # ===================================================================
        html.Div(
            id="section-school-admin-view",
            className="executive-card",
            style={"display": "none", "border": f"1px solid {COLOR_PINE}", "marginBottom": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "14px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.Span("INSTITUTIONAL DATA ENTRY & VERIFICATION PORTAL", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "letterSpacing": "0.5px"}),
                            html.H3("School Administrator Facility Audit & Live Diagnostic Intake", style={"margin": "2px 0 0 0", "fontSize": "17px", "fontWeight": "800", "color": COLOR_PITCH}),
                            html.P("Select a school from the district registry to inspect and update, or enter new capacity data to run an automated Data Quality & Completeness Audit.", style={"margin": "4px 0 0 0", "fontSize": "12.5px", "color": COLOR_MUTED})
                        ]),
                        html.Span("PRINCIPAL PORTAL ACTIVE", className="status-pill status-pill-green")
                    ]
                ),

                # School Pre-fill Selector
                html.Div(
                    style={"marginBottom": "16px"},
                    children=[
                        html.Label("Select Registered School to Pre-fill & Audit:", className="filter-label"),
                        dcc.Dropdown(
                            id="school-admin-select-dropdown",
                            options=school_select_options[:150],
                            placeholder="Choose a school from the district registry to pre-fill its official metrics...",
                            clearable=True
                        )
                    ]
                ),

                # School Form Grid
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px", "marginBottom": "16px"},
                    children=[
                        html.Div([
                            html.Label("School Name", className="filter-label"),
                            dcc.Input(id="input-sch-name", type="text", className="search-input-field", placeholder="e.g. Z.P. High School")
                        ]),
                        html.Div([
                            html.Label("UDISE Code (11 Digits)", className="filter-label"),
                            dcc.Input(id="input-sch-udise", type="number", className="search-input-field", placeholder="e.g. 27191001234")
                        ]),
                        html.Div([
                            html.Label("Total Active Enrolment", className="filter-label"),
                            dcc.Input(id="input-sch-enrol", type="number", className="search-input-field", placeholder="e.g. 350")
                        ]),
                        html.Div([
                            html.Label("Boys Enrolment", className="filter-label"),
                            dcc.Input(id="input-sch-boys", type="number", className="search-input-field", placeholder="e.g. 180")
                        ]),
                        html.Div([
                            html.Label("Girls Enrolment", className="filter-label"),
                            dcc.Input(id="input-sch-girls", type="number", className="search-input-field", placeholder="e.g. 170")
                        ]),
                        html.Div([
                            html.Label("Functional Classrooms", className="filter-label"),
                            dcc.Input(id="input-sch-rooms", type="number", className="search-input-field", placeholder="e.g. 8")
                        ]),
                        html.Div([
                            html.Label("Total Teaching Staff", className="filter-label"),
                            dcc.Input(id="input-sch-teachers", type="number", className="search-input-field", placeholder="e.g. 10")
                        ]),
                        html.Div([
                            html.Label("Promotion Pass Rate (%)", className="filter-label"),
                            dcc.Input(id="input-sch-promo", type="number", className="search-input-field", placeholder="e.g. 85.5")
                        ]),
                        html.Div([
                            html.Label("Pending Remedial Students", className="filter-label"),
                            dcc.Input(id="input-sch-pend", type="number", className="search-input-field", placeholder="e.g. 25")
                        ]),
                    ]
                ),

                html.Div(
                    style={"display": "flex", "gap": "12px", "marginBottom": "16px"},
                    children=[
                        html.Button("Run Data Quality & Diagnostic Audit", id="btn-sch-audit", n_clicks=0, className="btn-primary"),
                    ]
                ),

                # Output of Audit & Phased Plan
                html.Div(id="school-audit-output-container")
            ]
        ),

        # ===================================================================
        # SECTION 1: DISTRICT SCORE & 4 DYNAMIC KPI CARDS (Responsive to Filter)
        # ===================================================================
        html.Div(
            className="kpi-row",
            children=[
                # District Score Gauge Card
                html.Div(
                    className="executive-card",
                    style={"textAlign": "center", "padding": "18px 20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "2px"},
                            children=[
                                html.Span("DISTRICT SCHOOL SCORE", className="kpi-title"),
                                html.Span("0–100 SCALE", style={"fontSize": "10px", "color": COLOR_PINE, "fontWeight": "700"})
                            ]
                        ),
                        dcc.Graph(
                            id="readiness-radial-gauge",
                            config={"displayModeBar": False},
                            style={"height": "165px", "margin": "-10px 0 -15px 0"}
                        ),
                        html.Div(
                            id="gauge-status-pill",
                            className="status-pill status-pill-green",
                            style={"marginBottom": "10px"}
                        ),
                        html.Div(
                            id="gauge-explanation-text",
                            style={"fontSize": "11.5px", "color": COLOR_MUTED, "lineHeight": "1.45", "textAlign": "left", "backgroundColor": "#F8FAFC", "padding": "10px 12px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}
                        )
                    ]
                ),

                # 4 Dynamically Reactive KPI Cards
                html.Div(
                    className="kpi-cards-grid",
                    children=[
                        html.Div(
                            className="executive-card",
                            children=[
                                html.P("OPEN SCHOOLS", className="kpi-title"),
                                html.H2(id="kpi-open-schools", className="kpi-value"),
                                html.Span(id="kpi-open-schools-sub", className="kpi-badge", style={"color": COLOR_SAGE}),
                                html.P("Mapped operational schools actively holding classes in the selected area.", className="kpi-desc")
                            ]
                        ),
                        html.Div(
                            className="executive-card",
                            children=[
                                html.P("TOTAL STUDENTS", className="kpi-title"),
                                html.H2(id="kpi-total-students", className="kpi-value"),
                                html.Span(id="kpi-total-students-sub", className="kpi-badge", style={"color": COLOR_CYAN}),
                                html.P("Total enrolled students across all educational cohorts.", className="kpi-desc")
                            ]
                        ),
                        html.Div(
                            className="executive-card",
                            children=[
                                html.P("STUDENTS HELD BACK", className="kpi-title"),
                                html.H2(id="kpi-remedial-students", className="kpi-value", style={"color": COLOR_CRIMSON}),
                                html.Span("Need Remedial Help", className="kpi-badge", style={"color": COLOR_CRIMSON}),
                                html.P("Students pending promotion who require fast-track bridge learning.", className="kpi-desc")
                            ]
                        ),
                        html.Div(
                            className="executive-card",
                            children=[
                                html.P("NEW CLASSROOMS NEEDED BY 2037", className="kpi-title"),
                                html.H2(id="kpi-classroom-deficit", className="kpi-value", style={"color": COLOR_OCHRE}),
                                html.Span("RTE 30:1 Standard", className="kpi-badge", style={"color": COLOR_OCHRE}),
                                html.P("Projected classrooms required to meet statutory student ratio.", className="kpi-desc")
                            ]
                        ),
                    ]
                )
            ]
        ),

        # ===================================================================
        # SECTION 2: ANALYTICAL CHARTS (2x2 Grid with Plain-English Titles)
        # ===================================================================
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                # Chart 1: Student Drop-out by Grade
                html.Div(
                    className="executive-card",
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "8px"},
                            children=[
                                html.Div([
                                    html.H3("Student Drop-out by Grade (Class 1 to 12)", style={"margin": "0", "fontSize": "16px", "fontWeight": "800", "color": COLOR_PITCH}),
                                    html.P("Tracking student volumes across educational milestones", style={"margin": "2px 0 0 0", "fontSize": "12px", "color": COLOR_MUTED})
                                ]),
                                html.Span("38.4% DROP-OFF AT CLASS 9", className="status-pill status-pill-red")
                            ]
                        ),
                        html.Div(
                            "Key Finding: Notice the sharp drop from Upper Primary (76,819) to Secondary (47,322). Over 38.4% of students drop out between Class 8 and Class 9 because the area lacks secondary schools.",
                            className="plain-banner plain-banner-red"
                        ),
                        dcc.Graph(id="cohort-funnel-chart", config={"displayModeBar": False}, style={"height": "300px"})
                    ]
                ),

                # Chart 2: Student Enrollment Forecast
                html.Div(
                    className="executive-card",
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "8px"},
                            children=[
                                html.Div([
                                    html.H3("Student Enrollment Forecast (2021 to 2037)", style={"margin": "0", "fontSize": "16px", "fontWeight": "800", "color": COLOR_PITCH}),
                                    html.P("Predictive demographic trajectory with 85% statistical confidence bounds", style={"margin": "2px 0 0 0", "fontSize": "12px", "color": COLOR_MUTED})
                                ]),
                                html.Span("PROJECTION TO 2037", className="status-pill status-pill-cyan")
                            ]
                        ),
                        html.Div(
                            "Key Finding: Solid cyan is historical enrollment (2021–2025). Dashed line projects growth to 350,573 students by 2037, creating an infrastructure deficit of 2,388 new classrooms under RTE 30:1 norms.",
                            className="plain-banner plain-banner-cyan"
                        ),
                        dcc.Graph(id="forecast-envelope-chart", config={"displayModeBar": False}, style={"height": "300px"})
                    ]
                ),
            ]
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
            children=[
                # Chart 3: Schools Needing Urgent Attention
                html.Div(
                    className="executive-card",
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "8px"},
                            children=[
                                html.Div([
                                    html.H3("Schools Needing Urgent Attention", style={"margin": "0", "fontSize": "16px", "fontWeight": "800", "color": COLOR_PITCH}),
                                    html.P("Promotion Pass Rate vs. School Size (Bubble size = Pending Remedial Students)", style={"margin": "2px 0 0 0", "fontSize": "12px", "color": COLOR_MUTED})
                                ]),
                                html.Span("INTERVENTION MAP", className="status-pill status-pill-cyan")
                            ]
                        ),
                        html.Div(
                            "Key Finding: Bubbles in the bottom-right represent High-Risk Schools (Large student population with low pass rates). These schools must receive priority remedial teaching funds.",
                            className="plain-banner plain-banner-ochre"
                        ),
                        dcc.Graph(id="vulnerability-scatter-chart", config={"displayModeBar": False}, style={"height": "300px"})
                    ]
                ),

                # Chart 4: Education Quality Breakdown
                html.Div(
                    className="executive-card",
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "8px"},
                            children=[
                                html.Div([
                                    html.H3("Education Quality Breakdown", style={"margin": "0", "fontSize": "16px", "fontWeight": "800", "color": COLOR_PITCH}),
                                    html.P("Statutory compliance scores across 4 core educational pillars", style={"margin": "2px 0 0 0", "fontSize": "12px", "color": COLOR_MUTED})
                                ]),
                                html.Span("PILLAR AUDIT", className="status-pill status-pill-green")
                            ]
                        ),
                        html.Div(
                            "Key Finding: Capacity (91.7) and Retention (86.1) are resilient; Gender Equity (70.4) requires municipal investment in dedicated female sanitation and safety.",
                            className="plain-banner plain-banner-teal"
                        ),
                        dcc.Graph(id="pillar-radar-chart", config={"displayModeBar": False}, style={"height": "300px"})
                    ]
                ),
            ]
        ),

        # ===================================================================
        # SECTION 2.5: INTERACTIVE INFRASTRUCTURE GIS MAP (Chhatrapati Sambhajinagar)
        # ===================================================================
        html.Div(
            className="executive-card",
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.Span("GEOSPATIAL DECISION INTELLIGENCE • WARD MAPPING", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "letterSpacing": "0.5px"}),
                            html.H3("Interactive Infrastructure GIS Map • Chhatrapati Sambhajinagar", style={"margin": "2px 0 0 0", "fontSize": "17px", "fontWeight": "800", "color": COLOR_PITCH}),
                            html.P("Geospatial distribution of 954 institutions across URC-1 and URC-2. Marker color reflects Priority Tier; marker radius scales with student enrollment.", style={"margin": "3px 0 0 0", "fontSize": "12.5px", "color": COLOR_MUTED})
                        ]),
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Span(id="map-school-count-badge", className="status-pill status-pill-cyan"),
                                html.Span("CLICK PIN TO INSPECT DOSSIER", className="status-pill status-pill-green")
                            ]
                        )
                    ]
                ),
                dcc.Graph(
                    id="gis-infrastructure-map",
                    config={"displayModeBar": True, "scrollZoom": True},
                    style={"borderRadius": "8px", "overflow": "hidden", "border": f"1px solid {COLOR_BORDER}"}
                )
            ]
        ),

        # ===================================================================
        # SECTION 3: INTERACTIVE SEARCH & SCHOOL REGISTER DIRECTORY
        # ===================================================================
        html.Div(
            className="executive-card",
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "16px", "flexWrap": "wrap", "gap": "14px"},
                    children=[
                        html.Div([
                            html.H3("School Directory & Facility Diagnostic Inspector", style={"margin": "0", "fontSize": "17px", "fontWeight": "800", "color": COLOR_PITCH}),
                            html.P("Search any school by name or UDISE code to inspect its diagnostic profile", style={"margin": "2px 0 0 0", "fontSize": "12px", "color": COLOR_MUTED})
                        ]),
                        html.Div(id="table-row-count", style={"fontSize": "12px", "color": COLOR_PINE, "fontWeight": "700"})
                    ]
                ),

                # Search Bar Row
                html.Div(
                    style={"display": "flex", "gap": "12px", "marginBottom": "16px"},
                    children=[
                        dcc.Input(
                            id="school-search-input",
                            type="text",
                            className="search-input-field",
                            placeholder="Search by school name, UDISE code, or locality to inspect facility profile..."
                        ),
                        html.Button(
                            "Clear Search",
                            id="clear-search-btn",
                            n_clicks=0,
                            className="btn-outline"
                        )
                    ]
                ),

                # Dynamic Facility Profile Inspector Card
                html.Div(id="facility-inspector-container", style={"marginBottom": "16px"}),

                # Paginated Clean Data Table
                dash_table.DataTable(
                    id="school-data-table",
                    columns=[
                        {"name": "UDISE Code", "id": "udise_code"},
                        {"name": "School Name", "id": "school_name"},
                        {"name": "Area / Block", "id": "block_name"},
                        {"name": "School Type", "id": "school_management"},
                        {"name": "Students 2025", "id": "total_enrolment"},
                        {"name": "Pass Rate (%)", "id": "promotion_rate"},
                        {"name": "Held Back", "id": "pending_students"},
                        {"name": "School Score", "id": "readiness_score"},
                        {"name": "Status Tier", "id": "priority_tier"}
                    ],
                    page_size=8,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "backgroundColor": "#FFFFFF",
                        "color": COLOR_PITCH,
                        "border": f"1px solid {COLOR_BORDER}",
                        "padding": "10px 14px",
                        "fontSize": "12px",
                        "textAlign": "left"
                    },
                    style_header={
                        "backgroundColor": "#F8FAFC",
                        "color": COLOR_PINE,
                        "fontWeight": "800",
                        "border": f"1px solid {COLOR_BORDER}",
                        "fontSize": "12px",
                        "letterSpacing": "0.5px"
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": '{priority_tier} = "CRITICAL_INTERVENTION"'},
                            "color": COLOR_CRIMSON,
                            "fontWeight": "700"
                        },
                        {
                            "if": {"filter_query": '{priority_tier} = "OPTIMAL_RESILIENT"'},
                            "color": COLOR_SAGE
                        },
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#F8FAFC"
                        }
                    ]
                )
            ]
        ),

        # ===================================================================
        # SECTION 4: AI POLICY ADVISOR (SMOOTH FLICKER-FREE STREAMING)
        # ===================================================================
        html.Div(
            className="executive-card",
            style={"border": f"1px solid {COLOR_PINE}", "backgroundColor": "#F8FAFC"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "16px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.H3("AI Policy Advisor (Ask Questions & Formulate Directives)", style={"margin": "0", "fontSize": "17px", "fontWeight": "800", "color": COLOR_PITCH}),
                            html.P(
                                "Cross-references live district data with statutory documents (NEP 2020, RTE Act, Samagra Shiksha) using FAISS and local Qwen 2.5 on RTX 5050 GPU.",
                                style={"margin": "4px 0 0 0", "fontSize": "13px", "color": COLOR_MUTED}
                            )
                        ]),
                        html.Span("LOCAL QWEN 2.5 (RTX 5050)", className="status-pill status-pill-cyan")
                    ]
                ),

                # Quick Preset Chips
                html.Div(
                    style={"display": "flex", "gap": "10px", "marginBottom": "12px", "flexWrap": "wrap"},
                    children=[
                        html.Button("Explain Current District Data", id="preset-btn-1", n_clicks=0, className="btn-outline"),
                        html.Button("Explain 2021–2037 Forecast", id="preset-btn-2", n_clicks=0, className="btn-outline"),
                        html.Button("Analyze 38.4% Secondary Drop-off", id="preset-btn-3", n_clicks=0, className="btn-outline"),
                        html.Button("27k Remedial Directive", id="preset-btn-4", n_clicks=0, className="btn-outline"),
                    ]
                ),

                # Prompt Text Area
                dcc.Textarea(
                    id="rag-prompt-input",
                    placeholder="Ask an open-ended question or request a policy directive (e.g. 'Explain current data', 'Explain demographic forecast', 'Synthesize remedial policy')...",
                    value="Explain the Student Enrollment Forecast (2021 to 2037), the predictive demographic trajectory with 85% statistical confidence bounds, and the classroom deficit under RTE 30:1 norms.",
                    style={
                        "width": "100%",
                        "height": "75px",
                        "backgroundColor": "#FFFFFF",
                        "color": COLOR_PITCH,
                        "border": f"1px solid {COLOR_BORDER}",
                        "borderRadius": "8px",
                        "padding": "12px 14px",
                        "fontSize": "13px",
                        "boxSizing": "border-box",
                        "resize": "none",
                        "outline": "none"
                    }
                ),

                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginTop": "14px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Button("Ask AI Advisor / Synthesize Directive", id="rag-submit-btn", n_clicks=0, className="btn-primary"),
                        html.Button("Download Official Policy Brief (PDF)", id="rag-download-pdf-btn", n_clicks=0, disabled=True, className="btn-pdf")
                    ]
                ),

                # Dynamic Output Area (Flicker-Free Smooth Streaming with Active Agentic Thinking)
                dcc.Loading(
                    id="rag-loading-wrapper",
                    custom_spinner=html.Div(
                        className="thinking-container",
                        style={"marginTop": "18px", "padding": "18px 22px"},
                        children=[
                            html.Div(
                                className="thinking-header",
                                children=[
                                    html.Span(className="pulse-dot"),
                                    html.Span("THINKING & EXECUTING REASONING PIPELINE...", style={"fontWeight": "800", "fontSize": "13px", "letterSpacing": "0.5px"})
                                ]
                            ),
                            html.Div(
                                style={"marginTop": "12px", "display": "flex", "flexDirection": "column", "gap": "6px"},
                                children=[
                                    html.Div(
                                        className="thinking-step-row",
                                        children=[
                                            html.Span(">", style={"color": COLOR_PINE, "fontWeight": "800", "fontFamily": "monospace"}),
                                            html.Span("Inspecting district master database: 278,945 students across 954 schools...", style={"color": COLOR_PITCH, "fontWeight": "600"})
                                        ]
                                    ),
                                    html.Div(
                                        className="thinking-step-row",
                                        children=[
                                            html.Span(">", style={"color": COLOR_PINE, "fontWeight": "800", "fontFamily": "monospace"}),
                                            html.Span("Scanning 2021–2037 demographic forecasts (85% confidence bounds & RTE 30:1 PTR)...", style={"color": COLOR_MUTED})
                                        ]
                                    ),
                                    html.Div(
                                        className="thinking-step-row",
                                        children=[
                                            html.Span(">", style={"color": COLOR_PINE, "fontWeight": "800", "fontFamily": "monospace"}),
                                            html.Span("Querying FAISS vector index for statutory mandates (NEP 2020, RTE Act, Samagra Shiksha)...", style={"color": COLOR_MUTED})
                                        ]
                                    ),
                                    html.Div(
                                        className="thinking-step-row",
                                        children=[
                                            html.Span(">", style={"color": COLOR_PINE, "fontWeight": "800", "fontFamily": "monospace"}),
                                            html.Span("Synthesizing tailored answer via local Qwen 2.5 on RTX 5050 GPU...", style={"color": COLOR_PINE, "fontWeight": "700"})
                                        ]
                                    ),
                                ]
                            )
                        ]
                    ),
                    children=html.Div(
                        id="rag-output-container",
                        style={"marginTop": "18px"},
                        children=[
                            html.Div(
                                "Ask any question about the district data, demographic forecast, or request a statutory policy directive.",
                                style={"padding": "20px", "backgroundColor": "#FFFFFF", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}", "color": COLOR_MUTED, "fontSize": "13px", "textAlign": "center"}
                            )
                        ]
                    )
                )
            ]
        ),

        # ===================================================================
        # FOOTER
        # ===================================================================
        html.Div(
            style={"textAlign": "center", "padding": "24px 0 10px 0", "color": COLOR_MUTED, "fontSize": "12px", "borderTop": f"1px solid {COLOR_BORDER}", "marginTop": "24px"},
            children=[
                html.P("NITI DRISHTI • National Infrastructure Readiness Decision Support System", style={"margin": "0", "fontWeight": "700", "color": COLOR_PITCH}),
                html.P("B.Tech Final Year Project | Focus District: Chhatrapati Sambhajinagar | Data Window: 2021–2025 | Horizon: 2037", style={"margin": "4px 0 0 0"})
            ]
        )
    ]
)


# ---------------------------------------------------------------------------
# 4. DASHBOARD INTERACTIVE CALLBACKS (FULLY REACTIVE TO TOP COMMAND RIBBON)
# ---------------------------------------------------------------------------

@app.callback(
    # Top Cockpit & Gauges
    Output("readiness-radial-gauge", "figure"),
    Output("gauge-status-pill", "children"),
    Output("gauge-explanation-text", "children"),
    Output("kpi-open-schools", "children"),
    Output("kpi-open-schools-sub", "children"),
    Output("kpi-total-students", "children"),
    Output("kpi-total-students-sub", "children"),
    Output("kpi-remedial-students", "children"),
    Output("kpi-classroom-deficit", "children"),
    # 4 Charts
    Output("cohort-funnel-chart", "figure"),
    Output("forecast-envelope-chart", "figure"),
    Output("vulnerability-scatter-chart", "figure"),
    Output("pillar-radar-chart", "figure"),
    # GIS Infrastructure Map
    Output("gis-infrastructure-map", "figure"),
    Output("map-school-count-badge", "children"),
    # Table & Search
    Output("school-data-table", "data"),
    Output("table-row-count", "children"),
    Output("facility-inspector-container", "children"),
    # Inputs from Top Command Ribbon & Search
    Input("filter-block", "value"),
    Input("filter-mgmt", "value"),
    Input("filter-category", "value"),
    Input("filter-tier", "value"),
    Input("school-search-input", "value"),
    Input("sim-classroom-slider", "value"),
    Input("sim-w-equity", "value"),
    Input("sim-w-promo", "value"),
    Input("sim-w-ret", "value"),
    Input("sim-w-cap", "value")
)
def update_entire_dashboard(selected_block, selected_mgmt, selected_cat, selected_tier, search_query, sim_classrooms, w_eq, w_pr, w_ret, w_cap):
    filtered = df.copy()

    if selected_block and selected_block != "ALL" and "block_name" in filtered.columns:
        filtered = filtered[filtered["block_name"] == selected_block]

    if selected_mgmt and selected_mgmt != "ALL" and "school_management" in filtered.columns:
        filtered = filtered[filtered["school_management"] == selected_mgmt]

    if selected_cat and selected_cat != "ALL" and "school_category" in filtered.columns:
        filtered = filtered[filtered["school_category"] == selected_cat]

    if selected_tier and selected_tier != "ALL" and "priority_tier" in filtered.columns:
        filtered = filtered[filtered["priority_tier"] == selected_tier]

    # Search filter & Cognitive Explainable Decision Dossier Card
    inspector_card = None
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        search_mask = (
            filtered["school_name"].astype(str).str.lower().str.contains(q)
            | filtered["udise_code"].astype(str).str.lower().str.contains(q)
            | filtered["block_name"].astype(str).str.lower().str.contains(q)
        )
        filtered = filtered[search_mask]

        if not filtered.empty:
            top_school = filtered.iloc[0]
            s_name = top_school.get("school_name", "N/A")
            s_code = top_school.get("udise_code", "N/A")
            s_block = top_school.get("block_name", "N/A")
            s_enrol = int(top_school.get("total_enrolment", 0))
            s_score = float(top_school.get("readiness_score", 0.0))

            # Run Cognitive Decision & Explainability Engine (XAI)
            school_dict = top_school.to_dict()
            dossier = formulate_phased_action_plan(school_dict)
            gaps = dossier["gaps"]
            why_factors = generate_causal_why_factors(dossier)
            citizen_exp = generate_citizen_explanation(dossier)

            badge_color = dossier["priority_color"]
            priority_label = dossier["overall_priority"]
            dominant = dossier["dominant_bottleneck"]

            why_rows = [
                html.Div(
                    className="why-factor-row",
                    children=[
                        html.Span("•", style={"color": badge_color, "fontWeight": "bold", "fontSize": "13px"}),
                        html.Span(f, style={"color": COLOR_PITCH, "fontSize": "12px"})
                    ]
                )
                for f in why_factors
            ]

            inspector_card = html.Div(
                className="dossier-card",
                style={"borderLeft": f"5px solid {badge_color}"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "flexWrap": "wrap", "gap": "10px"},
                        children=[
                            html.Div([
                                html.Span("EXPLAINABLE COGNITIVE DECISION DOSSIER", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "letterSpacing": "0.5px"}),
                                html.H4(f"{s_name}", style={"margin": "4px 0 2px 0", "fontSize": "17px", "fontWeight": "800", "color": COLOR_PITCH}),
                                html.P(f"UDISE Code: {s_code} • Area: {s_block} • Dominant Constraint: {dominant}", style={"margin": "0", "fontSize": "12px", "color": COLOR_MUTED})
                            ]),
                            html.Div(
                                style={"textAlign": "right"},
                                children=[
                                    html.Span(f"Score: {s_score}/100", style={"fontSize": "16px", "fontWeight": "900", "color": badge_color, "display": "block"}),
                                    html.Span(f"PRIORITY: {priority_label}", className=f"status-pill {'status-pill-red' if priority_label == 'CRITICAL' else ('status-pill-amber' if priority_label == 'HIGH' else 'status-pill-green')}")
                                ]
                            )
                        ]
                    ),
                    # Physical Gaps Grid
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginTop": "12px", "borderTop": f"1px solid {COLOR_BORDER}", "paddingTop": "12px"},
                        children=[
                            html.Div([html.Span("Total Students", style={"fontSize": "11px", "color": COLOR_MUTED}), html.H5(f"{s_enrol:,}", style={"margin": "2px 0 0 0", "fontSize": "14px", "color": COLOR_PITCH})]),
                            html.Div([html.Span("Teacher Gap (ΔT)", style={"fontSize": "11px", "color": COLOR_MUTED}), html.H5(f"+{gaps['teacher_gap']} Teachers" if gaps['teacher_gap'] > 0 else "Compliant", style={"margin": "2px 0 0 0", "fontSize": "14px", "color": COLOR_CRIMSON if gaps['teacher_gap'] > 0 else COLOR_SAGE})]),
                            html.Div([html.Span("Classroom Gap (ΔC)", style={"fontSize": "11px", "color": COLOR_MUTED}), html.H5(f"+{gaps['classroom_gap']} Rooms" if gaps['classroom_gap'] > 0 else "Compliant", style={"margin": "2px 0 0 0", "fontSize": "14px", "color": COLOR_CRIMSON if gaps['classroom_gap'] > 0 else COLOR_SAGE})]),
                            html.Div([html.Span("Capacity Status", style={"fontSize": "11px", "color": COLOR_MUTED}), html.H5("SURGE OVERFLOW" if gaps['is_capacity_saturated'] else "Capacity Safe", style={"margin": "2px 0 0 0", "fontSize": "13px", "color": COLOR_CRIMSON if gaps['is_capacity_saturated'] else COLOR_PINE})])
                        ]
                    ),
                    # Plain-English Citizen Box
                    html.Div(
                        style={"backgroundColor": "#F8FAFC", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "12px 14px", "marginTop": "12px"},
                        children=[
                            html.Span("Plain-Language Explanation for Citizens & Parents:", style={"fontSize": "11.5px", "fontWeight": "800", "color": COLOR_PINE, "display": "block", "marginBottom": "4px"}),
                            html.P(citizen_exp, style={"margin": "0", "fontSize": "12.5px", "color": COLOR_PITCH, "lineHeight": "1.5"})
                        ]
                    ),
                    # Causal "WHY" factors
                    html.Div(
                        style={"marginTop": "12px"},
                        children=[
                            html.Span("Causal 'WHY' Factor Trail (Primary Drivers):", style={"fontSize": "11.5px", "fontWeight": "800", "color": COLOR_MUTED, "display": "block", "marginBottom": "6px"}),
                            html.Div(why_rows)
                        ]
                    ),
                    # Phased Action Recommendation
                    html.Div(
                        style={"marginTop": "10px", "paddingTop": "8px", "borderTop": f"1px dashed {COLOR_BORDER}", "fontSize": "12px"},
                        children=[
                            html.Span("First Recommended Action: ", style={"fontWeight": "800", "color": COLOR_PINE}),
                            html.Span(dossier['phased_plan']['phase_1_immediate'][0], style={"color": COLOR_PITCH})
                        ]
                    )
                ]
            )

    # 1. DYNAMIC CALCULATIONS FOR FILTERED SUBSET (WITH WHAT-IF RECALIBRATION)
    active_subset = filtered[filtered["is_operational"] == True] if "is_operational" in filtered.columns else filtered
    total_schools = len(filtered)
    open_schools = len(active_subset)
    total_students = int(filtered["total_enrolment"].sum())
    total_boys = int(filtered["total_boys"].sum()) if "total_boys" in filtered.columns else int(total_students * 0.52)
    total_girls = int(filtered["total_girls"].sum()) if "total_girls" in filtered.columns else int(total_students * 0.48)
    remedial_students = int(filtered["pending_students"].sum()) if "pending_students" in filtered.columns else 0

    gpi = round(total_girls / max(total_boys, 1), 2)

    # Multi-Criteria Weight Recalibration
    sim_classrooms = int(sim_classrooms or 0)
    w_eq = float(w_eq or 0.25)
    w_pr = float(w_pr or 0.30)
    w_ret = float(w_ret or 0.25)
    w_cap = float(w_cap or 0.20)
    tot_w = max(0.001, w_eq + w_pr + w_ret + w_cap)
    w_eq, w_pr, w_ret, w_cap = w_eq / tot_w, w_pr / tot_w, w_ret / tot_w, w_cap / tot_w

    if not active_subset.empty and "score_equity" in active_subset.columns:
        composite_series = (
            active_subset["score_equity"] * w_eq +
            active_subset["score_promotion"] * w_pr +
            active_subset["score_retention"] * w_ret +
            active_subset["score_capacity"] * w_cap
        )
        avg_score = round(float(composite_series.mean()), 1)
    else:
        avg_score = round(float(active_subset["readiness_score"].mean()), 1) if not active_subset.empty else 0.0

    # 2037 Deficit for filtered subset with live capital infusion reduction
    baseline_classrooms = int(total_students / 30.0)
    projected_2037_students = int(total_students * (350573 / 278945)) if total_students > 0 else 0
    projected_classrooms = int(projected_2037_students / 30.0)
    deficit_classrooms = max(0, projected_classrooms - (baseline_classrooms + sim_classrooms))

    # 2. RADIAL GAUGE FIGURE
    gauge_color = COLOR_SAGE if avg_score >= 75 else (COLOR_OCHRE if avg_score >= 55 else COLOR_CRIMSON)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score,
        number={"font": {"size": 32, "color": COLOR_PITCH, "family": "Inter"}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOR_MUTED, "nticks": 5},
            "bar": {"color": gauge_color, "thickness": 0.28},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 55], "color": "#FEE2E2"},
                {"range": [55, 75], "color": "#FEF3C7"},
                {"range": [75, 100], "color": "#DCFCE7"}
            ],
            "threshold": {
                "line": {"color": COLOR_PINE, "width": 3},
                "thickness": 0.8,
                "value": avg_score
            }
        }
    ))
    gauge_fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=165
    )

    tier_label = "Optimal • Resilient" if avg_score >= 75 else ("Developing • Needs Attention" if avg_score >= 55 else "Critical • Urgent Action Required")
    gauge_status = f"Score: {avg_score}/100 • {tier_label}"
    gauge_explanation = f"Score is {avg_score}/100. Operational stability is {((open_schools / max(total_schools, 1)) * 100):.1f}%. Equity and remedial programs will raise overall capacity."

    # 3. KPI OUTPUTS
    kpi_open = f"{open_schools:,} / {total_schools:,}"
    kpi_open_sub = f"{((open_schools / max(total_schools, 1)) * 100):.1f}% Active"
    kpi_total = f"{total_students:,}"
    kpi_total_sub = f"GPI: {gpi} • {total_girls:,} Girls"
    kpi_remedial = f"{remedial_students:,}"
    kpi_deficit = f"+{deficit_classrooms:,} Classrooms"

    # 4. COHORT PROGRESSION FUNNEL (Exact Real Data)
    pri = int(filtered["enrolment_primary"].sum()) if "enrolment_primary" in filtered.columns else 121184
    upri = int(filtered["enrolment_upper_primary"].sum()) if "enrolment_upper_primary" in filtered.columns else 76819
    sec = int(filtered["enrolment_secondary"].sum()) if "enrolment_secondary" in filtered.columns else 47322
    hsec = int(filtered["enrolment_higher_secondary"].sum()) if "enrolment_higher_secondary" in filtered.columns else 30285

    funnel_fig = go.Figure(go.Funnel(
        y=["Primary (1-5)", "Upper Primary (6-8)", "Secondary (9-10)", "Higher Sec (11-12)"],
        x=[pri, upri, sec, hsec],
        textinfo="value+percent previous",
        textfont={"color": COLOR_PITCH, "size": 12, "family": "Inter"},
        marker={
            "color": [COLOR_CYAN, COLOR_PINE, COLOR_OCHRE, COLOR_CRIMSON],
            "line": {"width": 1, "color": "#FFFFFF"}
        },
        connector={"line": {"color": COLOR_BORDER, "width": 1.5}}
    ))
    funnel_fig.update_layout(
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": COLOR_MUTED, "family": "Inter"},
        height=300
    )

    # 5. FORECAST ENVELOPE FIGURE (Exact Real 2021-2037 Data)
    forecast_fig = go.Figure()

    hist_subset = df_forecast[df_forecast["year"] <= 2025]
    forecast_fig.add_trace(go.Scatter(
        x=hist_subset["year"],
        y=hist_subset["total_students"],
        mode="lines+markers",
        name="Historical Actuals (2021–2025)",
        line=dict(color=COLOR_CYAN, width=3),
        marker=dict(size=7, color=COLOR_CYAN)
    ))

    proj_subset = df_forecast[df_forecast["year"] >= 2025]
    forecast_fig.add_trace(go.Scatter(
        x=proj_subset["year"],
        y=proj_subset["total_students"],
        mode="lines+markers",
        name="Projected Trend (2026–2037)",
        line=dict(color=COLOR_PINE, width=3, dash="dash"),
        marker=dict(size=6, color=COLOR_PINE)
    ))

    forecast_fig.add_trace(go.Scatter(
        x=list(proj_subset["year"]) + list(proj_subset["year"])[::-1],
        y=list(proj_subset["yhat_upper"]) + list(proj_subset["yhat_lower"])[::-1],
        fill="toself",
        fillcolor="rgba(15, 118, 110, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="85% Uncertainty Envelope"
    ))

    forecast_fig.update_layout(
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": COLOR_MUTED, "family": "Inter"},
        xaxis=dict(gridcolor="#E2E8F0", dtick=2),
        yaxis=dict(gridcolor="#E2E8F0", title="Student Volume"),
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        height=300
    )

    # 6. VULNERABILITY SCATTER MATRIX
    scatter_df = filtered.sample(min(len(filtered), 250), random_state=42) if len(filtered) > 250 else filtered
    scatter_fig = px.scatter(
        scatter_df,
        x="total_enrolment",
        y="promotion_rate",
        size="pending_students",
        color="priority_deficit_score",
        color_continuous_scale=[COLOR_SAGE, COLOR_OCHRE, COLOR_CRIMSON],
        hover_name="school_name" if "school_name" in scatter_df.columns else "udise_code",
        labels={"total_enrolment": "School Enrolment", "promotion_rate": "Pass Rate (%)", "priority_deficit_score": "Deficit Score"}
    )
    scatter_fig.add_hline(y=80.0, line_dash="dot", line_color=COLOR_OCHRE, annotation_text="80% Standard Norm", annotation_position="bottom right")
    scatter_fig.update_layout(
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": COLOR_MUTED, "family": "Inter"},
        xaxis=dict(gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0"),
        coloraxis_colorbar=dict(title="Deficit", len=0.8),
        height=300
    )

    # 7. SUB-PILLAR RADAR
    p_equity = round(float(active_subset["score_equity"].mean()), 1) if not active_subset.empty else 70.4
    p_promo = round(float(active_subset["score_promotion"].mean()), 1) if not active_subset.empty else 83.6
    p_ret = round(float(active_subset["score_retention"].mean()), 1) if not active_subset.empty else 86.1
    p_cap = round(float(active_subset["score_capacity"].mean()), 1) if not active_subset.empty else 91.7

    radar_fig = go.Figure(data=go.Scatterpolar(
        r=[p_equity, p_promo, p_ret, p_cap, p_equity],
        theta=["1. Gender Equity", "2. Pass Rate", "3. Cohort Retention", "4. Capacity/PTR", "1. Gender Equity"],
        fill="toself",
        fillcolor="rgba(15, 118, 110, 0.15)",
        line=dict(color=COLOR_PINE, width=2.5),
        marker=dict(size=6, color=COLOR_PINE)
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E2E8F0", linecolor="#CBD5E1"),
            angularaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1")
        ),
        margin=dict(l=35, r=35, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        font={"color": COLOR_MUTED, "family": "Inter"},
        height=300
    )

    # 8. GIS INFRASTRUCTURE MAP
    gis_map_fig = create_gis_infrastructure_map(filtered)
    map_badge = f"{len(filtered):,} Institutions Geospatially Plotted"

    # 9. TABLE DATA
    display_cols = [
        "udise_code", "school_name", "block_name", "school_management",
        "total_enrolment", "promotion_rate", "pending_students", "readiness_score", "priority_tier"
    ]
    table_records = filtered[[c for c in display_cols if c in filtered.columns]].sort_values("readiness_score").to_dict("records")
    row_count_str = f"Displaying {len(table_records):,} Institutions Matching Filter"

    return (
        gauge_fig, gauge_status, gauge_explanation,
        kpi_open, kpi_open_sub,
        kpi_total, kpi_total_sub,
        kpi_remedial, kpi_deficit,
        funnel_fig, forecast_fig, scatter_fig, radar_fig,
        gis_map_fig, map_badge,
        table_records, row_count_str, inspector_card
    )


@app.callback(
    Output("school-search-input", "value"),
    Input("clear-search-btn", "n_clicks"),
    prevent_initial_call=True
)
def handle_clear_search(n_clicks):
    return ""


@app.callback(
    Output("school-search-input", "value", allow_duplicate=True),
    Input("gis-infrastructure-map", "clickData"),
    prevent_initial_call=True
)
def handle_map_pin_click(click_data):
    if not click_data or "points" not in click_data or not click_data["points"]:
        return no_update
    point = click_data["points"][0]
    if "customdata" in point and point["customdata"]:
        udise = str(point["customdata"][0])
        return udise
    elif "hovertext" in point and point["hovertext"]:
        return str(point["hovertext"])
    return no_update


# ---------------------------------------------------------------------------
# 5. PRESETS & SMOOTH FLICKER-FREE RAG STREAMING CALLBACKS
# ---------------------------------------------------------------------------

@app.callback(
    Output("rag-prompt-input", "value"),
    Input("preset-btn-1", "n_clicks"),
    Input("preset-btn-2", "n_clicks"),
    Input("preset-btn-3", "n_clicks"),
    Input("preset-btn-4", "n_clicks"),
    prevent_initial_call=True
)
def handle_preset_prompts(btn1, btn2, btn3, btn4):
    from dash import callback_context
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "preset-btn-1":
        return "Explain the current district education data, total enrollment, gender parity, grade distribution, and operational metrics for Chhatrapati Sambhajinagar."
    elif triggered_id == "preset-btn-2":
        return "Explain the Student Enrollment Forecast (2021 to 2037), the predictive demographic trajectory with 85% statistical confidence bounds, and the classroom deficit under RTE 30:1 norms."
    elif triggered_id == "preset-btn-3":
        return "Analyze the severe student drop-off between Upper Primary (76k) and Secondary (47k) stages. Formulate statutory directives under NEP 2020 for secondary school expansion."
    elif triggered_id == "preset-btn-4":
        return "Propose an urgent administrative directive for the 27,235 non-promoted students, citing Samagra Shiksha remedial bridge funding and fast-track learning recovery."
    return no_update


@app.callback(
    Output("rag-full-text-store", "data"),
    Output("rag-stream-cursor", "data"),
    Output("rag-typewriter-interval", "disabled"),
    Output("rag-output-container", "children"),
    Output("rag-download-pdf-btn", "disabled"),
    Input("rag-submit-btn", "n_clicks"),
    State("rag-prompt-input", "value"),
    prevent_initial_call=True
)
def initiate_rag_generation(n_clicks, prompt_text):
    if not prompt_text or not prompt_text.strip():
        return "", 0, True, html.Div("Please provide an executive policy prompt.", style={"color": COLOR_CRIMSON, "padding": "16px"}), True

    try:
        res = generate_education_policy_brief(user_prompt=prompt_text)
    except Exception as e:
        return "", 0, True, html.Div(f"Error during RAG synthesis: {str(e)}", style={"color": COLOR_CRIMSON, "padding": "16px"}), True

    full_policy_text = res.get("policy_report", "")

    # Clean Thinking Accordion (No casual emojis)
    thinking_items = [
        html.Div(
            style={"display": "flex", "alignItems": "flex-start", "gap": "10px", "marginBottom": "6px"},
            children=[
                html.Span("•", style={"color": COLOR_PINE, "fontSize": "14px", "fontWeight": "bold"}),
                html.Span(step, style={"color": COLOR_MUTED, "fontSize": "12.5px", "lineHeight": "1.5"})
            ]
        )
        for step in res.get("thinking_steps", [])
    ]

    source_badges = [
        html.Span(
            s,
            style={
                "backgroundColor": "#F1F5F9",
                "color": COLOR_PINE,
                "padding": "3px 8px",
                "borderRadius": "4px",
                "fontSize": "11px",
                "marginRight": "6px",
                "border": f"1px solid {COLOR_BORDER}",
                "fontWeight": "600"
            }
        )
        for s in res.get("sources", [])
    ]

    initial_view = html.Div([
        # Reasoning Process Card
        html.Details(
            style={
                "backgroundColor": "#F8FAFC",
                "border": f"1px solid {COLOR_BORDER}",
                "borderRadius": "8px",
                "padding": "12px 16px",
                "marginBottom": "14px"
            },
            open=True,
            children=[
                html.Summary(
                    style={"cursor": "pointer", "fontWeight": "800", "fontSize": "12px", "color": COLOR_PINE, "letterSpacing": "0.5px", "outline": "none"},
                    children=f"THOUGHT PROCESS & EVIDENCE TRACE ({res.get('engine', 'RAG Engine')})"
                ),
                html.Div(style={"marginTop": "10px"}, children=thinking_items),
                html.Div(
                    style={"marginTop": "10px", "paddingTop": "8px", "borderTop": f"1px solid {COLOR_BORDER}"},
                    children=[
                        html.Span("Statutory Sources Cited: ", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_MUTED}),
                        html.Span(source_badges)
                    ]
                )
            ]
        ),

        # Smooth Plain-Text Streaming Box (Prevents Markdown DOM parsing flicker!)
        html.Div(
            id="rag-streaming-text-target",
            className="streaming-text-box",
            children="Initiating generation stream... ▌"
        )
    ])

    return full_policy_text, 0, False, initial_view, False


@app.callback(
    Output("rag-streaming-text-target", "children"),
    Output("rag-stream-cursor", "data", allow_duplicate=True),
    Output("rag-typewriter-interval", "disabled", allow_duplicate=True),
    Input("rag-typewriter-interval", "n_intervals"),
    State("rag-full-text-store", "data"),
    State("rag-stream-cursor", "data"),
    prevent_initial_call=True
)
def stream_text_smooth(n, full_text, current_cursor):
    if not full_text:
        return no_update, 0, True

    # Advance chunk by 7 characters per tick (25ms) for smooth satisfying typewriter animation
    chunk_size = 7
    next_cursor = current_cursor + chunk_size

    if next_cursor >= len(full_text):
        # Finalize and render formatted markdown document cleanly
        return dcc.Markdown(full_text), len(full_text), True

    partial_text = full_text[:next_cursor]
    # Render with blinking cursor span without re-parsing markdown
    return html.Span([
        html.Span(partial_text),
        html.Span(" ▌", className="cursor-blink")
    ]), next_cursor, False


@app.callback(
    Output("download-policy-pdf", "data"),
    Input("rag-download-pdf-btn", "n_clicks"),
    State("rag-full-text-store", "data"),
    prevent_initial_call=True
)
def handle_pdf_download(n_clicks, policy_markdown):
    if not policy_markdown or not policy_markdown.strip():
        return no_update

    pdf_buffer = generate_policy_pdf(policy_markdown)
    return dcc.send_bytes(
        pdf_buffer.getvalue(),
        filename="NitiDrishti_Official_Policy_Directive.pdf"
    )


# ---------------------------------------------------------------------------
# 6. COGNITIVE MULTI-PERSONA PORTAL & SCHOOL AUDIT CALLBACKS
# ---------------------------------------------------------------------------

@app.callback(
    Output("active-portal-store", "data"),
    Output("portal-tab-citizen", "className"),
    Output("portal-tab-school", "className"),
    Output("portal-tab-govt", "className"),
    Output("section-govt-sim-sandbox", "style"),
    Output("section-school-admin-view", "style"),
    Input("portal-tab-citizen", "n_clicks"),
    Input("portal-tab-school", "n_clicks"),
    Input("portal-tab-govt", "n_clicks"),
    prevent_initial_call=True
)
def switch_portal_view(btn_cit, btn_sch, btn_gov):
    from dash import callback_context
    if not callback_context.triggered:
        return no_update
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    style_hidden = {"display": "none"}
    style_visible = {"display": "block"}

    if triggered == "portal-tab-school":
        return "school", "portal-tab-btn", "portal-tab-btn portal-tab-btn-active", "portal-tab-btn", style_hidden, style_visible
    elif triggered == "portal-tab-govt":
        return "govt", "portal-tab-btn", "portal-tab-btn", "portal-tab-btn portal-tab-btn-active", style_visible, style_hidden
    else:
        return "citizen", "portal-tab-btn portal-tab-btn-active", "portal-tab-btn", "portal-tab-btn", style_hidden, style_hidden


@app.callback(
    Output("input-sch-name", "value"),
    Output("input-sch-udise", "value"),
    Output("input-sch-enrol", "value"),
    Output("input-sch-boys", "value"),
    Output("input-sch-girls", "value"),
    Output("input-sch-rooms", "value"),
    Output("input-sch-teachers", "value"),
    Output("input-sch-promo", "value"),
    Output("input-sch-pend", "value"),
    Input("school-admin-select-dropdown", "value"),
    prevent_initial_call=True
)
def prefill_school_admin_form(selected_udise):
    if not selected_udise:
        return "", None, None, None, None, None, None, None, None

    matched = df[df["udise_code"].astype(str) == str(selected_udise)]
    if matched.empty:
        return "", None, None, None, None, None, None, None, None

    row = matched.iloc[0]
    enrol = int(row.get("total_enrolment", 0))
    boys = int(row.get("total_boys", round(enrol * 0.52)))
    girls = int(row.get("total_girls", round(enrol * 0.48)))
    rooms = int(row.get("functional_classrooms", max(1, round(enrol / 38.0))))
    teachers = int(row.get("total_teachers", max(1, round(enrol / 34.0))))
    promo = float(row.get("promotion_rate", 85.0))
    pend = int(row.get("pending_students", 0))

    return (
        str(row.get("school_name", "")),
        int(row.get("udise_code", 0)),
        enrol,
        boys,
        girls,
        rooms,
        teachers,
        promo,
        pend
    )


@app.callback(
    Output("school-audit-output-container", "children"),
    Input("btn-sch-audit", "n_clicks"),
    State("input-sch-name", "value"),
    State("input-sch-udise", "value"),
    State("input-sch-enrol", "value"),
    State("input-sch-boys", "value"),
    State("input-sch-girls", "value"),
    State("input-sch-rooms", "value"),
    State("input-sch-teachers", "value"),
    State("input-sch-promo", "value"),
    State("input-sch-pend", "value"),
    prevent_initial_call=True
)
def handle_school_admin_audit(n_clicks, name, udise, enrol, boys, girls, rooms, teachers, promo, pend):
    if not name and not enrol:
        return html.Div("Please enter school name and enrollment to run audit.", style={"color": COLOR_CRIMSON, "padding": "12px", "fontSize": "13px"})

    record = {
        "school_name": name or "Audited School",
        "udise_code": udise or 27191000000,
        "total_enrolment": float(enrol or 0),
        "total_boys": float(boys or 0),
        "total_girls": float(girls or 0),
        "functional_classrooms": float(rooms) if rooms is not None else None,
        "total_teachers": float(teachers) if teachers is not None else None,
        "promotion_rate": float(promo) if promo is not None else 85.0,
        "pending_students": float(pend or 0)
    }

    # 1. Run Data Quality Auditor
    audit_res = audit_school_record(record)
    conf_status = audit_res["confidence_status"]
    conf_color = COLOR_SAGE if conf_status == "Sufficient" else (COLOR_OCHRE if conf_status == "Moderate Confidence" else COLOR_CRIMSON)

    # 2. Run Phased Action Planner & Gaps
    dossier = formulate_phased_action_plan(record)
    gaps = dossier["gaps"]
    priority_tier = dossier["overall_priority"]
    priority_color = dossier["priority_color"]
    dominant = dossier["dominant_bottleneck"]

    why_factors = generate_causal_why_factors(dossier)
    admin_narrative = generate_administrative_narrative(dossier)

    return html.Div(
        style={"marginTop": "18px", "padding": "18px 20px", "backgroundColor": "#F8FAFC", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "10px"},
        children=[
            # Header
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px", "flexWrap": "wrap", "gap": "10px"},
                children=[
                    html.Div([
                        html.Span("OFFICIAL INSTITUTIONAL DIAGNOSTIC REPORT", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "letterSpacing": "0.5px"}),
                        html.H4(f"{record['school_name']}", style={"margin": "2px 0 0 0", "fontSize": "17px", "color": COLOR_PITCH}),
                        html.P(f"UDISE: {record['udise_code']} • Dominant Bottleneck: {dominant}", style={"margin": "2px 0 0 0", "fontSize": "12px", "color": COLOR_MUTED})
                    ]),
                    html.Div(
                        style={"display": "flex", "gap": "8px"},
                        children=[
                            html.Span(f"DATA AUDIT: {conf_status.upper()}", style={"backgroundColor": "#FFFFFF", "color": conf_color, "border": f"1px solid {conf_color}", "padding": "4px 10px", "borderRadius": "6px", "fontSize": "11px", "fontWeight": "800"}),
                            html.Span(f"PRIORITY: {priority_tier}", style={"backgroundColor": priority_color, "color": "#FFFFFF", "padding": "4px 12px", "borderRadius": "6px", "fontSize": "11px", "fontWeight": "800"})
                        ]
                    )
                ]
            ),

            # Data Quality Alert Box
            html.Div(
                style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "12px 14px", "marginBottom": "14px"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "6px"},
                        children=[
                            html.Span(f"Data Completeness Score: {audit_res['completeness_score']}%", style={"fontSize": "12px", "fontWeight": "700", "color": COLOR_PITCH}),
                            html.Span(audit_res["confidence_message"], style={"fontSize": "11.5px", "color": conf_color, "fontWeight": "600"})
                        ]
                    ),
                    html.Div(
                        [html.P(f"⚠️ Validation Warning: {v}", style={"margin": "2px 0", "fontSize": "11.5px", "color": COLOR_CRIMSON}) for v in audit_res["domain_violations"]]
                        if audit_res["domain_violations"] else
                        html.Span("✓ Zero domain validity contradictions detected in submission.", style={"fontSize": "11.5px", "color": COLOR_SAGE, "fontWeight": "600"})
                    )
                ]
            ),

            # Physical Gaps Breakdown Grid
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "14px"},
                children=[
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 12px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Teacher Gap (ΔT)", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4(f"+{gaps['teacher_gap']} Teachers" if gaps['teacher_gap'] > 0 else "✓ Compliant", style={"margin": "3px 0 0 0", "color": COLOR_CRIMSON if gaps['teacher_gap'] > 0 else COLOR_SAGE})
                    ]),
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 12px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Classroom Gap (ΔC)", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4(f"+{gaps['classroom_gap']} Classrooms" if gaps['classroom_gap'] > 0 else "✓ Compliant", style={"margin": "3px 0 0 0", "color": COLOR_CRIMSON if gaps['classroom_gap'] > 0 else COLOR_SAGE})
                    ]),
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 12px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Pupil-Teacher Ratio", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4(f"{gaps['actual_ptr']}:1", style={"margin": "3px 0 0 0", "color": COLOR_CRIMSON if gaps['actual_ptr'] > 30 else COLOR_PITCH})
                    ]),
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "10px 12px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Capacity Status", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4("OVERFLOW ALERT" if gaps['is_capacity_saturated'] else "Capacity Safe", style={"margin": "3px 0 0 0", "color": COLOR_CRIMSON if gaps['is_capacity_saturated'] else COLOR_PINE})
                    ]),
                ]
            ),

            # Phased Action Plan Roadmap
            html.Div(
                style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "14px 16px"},
                children=[
                    html.Span("TRI-PHASED STATUTORY ACTION ROADMAP:", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "letterSpacing": "0.5px", "display": "block", "marginBottom": "8px"}),
                    html.Div([
                        html.Span("Phase 1 (Immediate 0–90 Days): ", style={"fontWeight": "800", "color": COLOR_PITCH, "fontSize": "12px"}),
                        html.Span(dossier["phased_plan"]["phase_1_immediate"][0], style={"fontSize": "12px", "color": COLOR_MUTED})
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span("Phase 2 (Medium-Term Civil Works 6–24 Months): ", style={"fontWeight": "800", "color": COLOR_PITCH, "fontSize": "12px"}),
                        html.Span(dossier["phased_plan"]["phase_2_medium"][0], style={"fontSize": "12px", "color": COLOR_MUTED})
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span("Phase 3 (Strategic Horizon 2037): ", style={"fontWeight": "800", "color": COLOR_PITCH, "fontSize": "12px"}),
                        html.Span(dossier["phased_plan"]["phase_3_strategic"][0], style={"fontSize": "12px", "color": COLOR_MUTED})
                    ])
                ]
            )
        ]
    )


# ---------------------------------------------------------------------------
# 7. RUN DASH SERVER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NITI DRISHTI: Launching Nordic Minimalist Decision Support Platform...")
    print("=" * 70)

    app.run(
        debug=False,
        port=8050
    )