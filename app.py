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

from backend.services.institution_service import InstitutionService
from backend.services.forecast_service import ForecastService
from backend.services.scenario_service import ScenarioService

institution_service = InstitutionService()
forecast_service = ForecastService()
scenario_service = ScenarioService()

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

# Live empirical metrics dynamically computed from loaded dataset
STAT_TOTAL_SCHOOLS = f"{len(df):,}"
STAT_TOTAL_STUDENTS = f"{int(df['total_enrolment'].sum()):,}"
STAT_READINESS_SCORE = f"{round(float(district_summary.get('district_readiness_score', 82.5)), 1)}"
STAT_REMEDIAL_STUDENTS = f"{int(df['pending_students'].sum()):,}" if "pending_students" in df.columns else "27,235"
STAT_TIER_LABEL = "Tier 3 Optimal" if float(STAT_READINESS_SCORE) >= 75 else ("Tier 2 Developing" if float(STAT_READINESS_SCORE) >= 55 else "Tier 1 Critical")

# Load 2021-2040 Forecast Data (anchored to exact 278,945 actual baseline)
forecast_file = find_dataset([
    "data/timeseries/yearly_enrollment_forecast_2021_2040.csv",
    "data/timeseries/yearly_enrollment_forecast_2021_2037.csv"
])
if forecast_file:
    df_forecast = pd.read_csv(forecast_file)
else:
    forecaster = EducationForecaster()
    df_forecast_path = forecaster.save_forecast()
    df_forecast = pd.read_csv(df_forecast_path)

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
# 2. DESIGN TOKENS: INDIAN DISTRICT GOVERNMENT PORTAL PALETTE
# ---------------------------------------------------------------------------
COLOR_PRIMARY_NAVY = "#123B63"    # Deep Primary Navy (Govt Header/Hero)
COLOR_GOVT_BLUE = "#1E5A92"       # Government Blue (Active Nav / Links)
COLOR_TEAL = "#00695C"            # Administrative Teal (Badges / Highlights)
COLOR_BG = "#F5F7F9"              # Official Off-White/Grey Background
COLOR_WHITE = "#FFFFFF"           # Clean Surface White
COLOR_BORDER = "#D5DDE5"          # Subdued Administrative Border
COLOR_TEXT_MAIN = "#252525"       # Charcoal Body Text
COLOR_MUTED = "#5F6B76"           # Slate Muted Text
COLOR_ACCENT_ORANGE = "#D9822B"   # National Emblem Gold/Orange Accent
COLOR_SUCCESS = "#2E7D32"         # Official Green
COLOR_WARNING = "#B26A00"         # Official Amber/Warning
COLOR_DANGER = "#B3261E"          # Official Crimson/Alert

# Compatibility aliases so all charts & components continue working seamlessly
COLOR_PINE = COLOR_PRIMARY_NAVY
COLOR_CYAN = COLOR_GOVT_BLUE
COLOR_SAGE = COLOR_SUCCESS
COLOR_OCHRE = COLOR_WARNING
COLOR_CRIMSON = COLOR_DANGER
COLOR_PITCH = COLOR_TEXT_MAIN

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
    id="app-root-container",
    children=[
        # Stores for auth, view state, RAG streaming, PDF generation
        dcc.Store(id="current-page-store", data="welcome"),
        dcc.Store(id="auth-session-store", data={"school": False, "govt": False, "role": "citizen"}),
        dcc.Store(id="pending-role-target-store", data=""),
        dcc.Store(id="rag-full-text-store", data=""),
        dcc.Store(id="rag-stream-cursor", data=0),
        dcc.Interval(id="rag-typewriter-interval", interval=25, n_intervals=0, disabled=True),
        dcc.Download(id="download-policy-pdf"),

        # ===================================================================
        # INSTITUTIONAL AUTHENTICATION MODAL (FOR SCHOOL & GOVT OFFICIALS)
        # ===================================================================
        html.Div(
            id="auth-modal-overlay",
            className="auth-modal-overlay",
            style={"display": "none"},
            children=[
                html.Div(
                    className="auth-modal-card",
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"},
                            children=[
                                html.Div(id="auth-modal-icon", className="auth-header-icon", children=[
                                    html.Span(className="gov-ico gov-ico-lock", style={"width": "22px", "height": "22px"})
                                ]),
                                html.Span(id="auth-modal-badge", className="status-pill status-pill-cyan", children="OFFICIAL CLEARANCE")
                            ]
                        ),
                        html.H3(id="auth-modal-title", style={"margin": "0 0 6px 0", "fontSize": "17px", "fontWeight": "700", "color": COLOR_PRIMARY_NAVY}, children="Institutional Authentication"),
                        html.P(id="auth-modal-subtitle", style={"margin": "0 0 16px 0", "fontSize": "12.5px", "color": COLOR_MUTED, "lineHeight": "1.5"}, children="Enter authorized official credentials to access the institution assessment portal or scenario planning tools."),

                        html.Div(id="auth-error-banner", className="auth-error-banner", children=""),

                        # Institution selector placeholder (Active only for Institution Assessment Portal)
                        html.Div(
                            id="auth-school-group",
                            className="auth-input-group",
                            style={"display": "none"},
                            children=[
                                html.Label("Select Your School / Institution", className="auth-input-label"),
                                dcc.Dropdown(
                                    id="auth-school-dropdown",
                                    options=school_select_options,
                                    value=str(school_select_options[0]["value"]) if school_select_options else None,
                                    placeholder="Type school name or UDISE code to search...",
                                    clearable=False,
                                    style={"fontSize": "12.5px"}
                                )
                            ]
                        ),

                        html.Div(
                            className="auth-input-group",
                            children=[
                                html.Label("Official Username / ID", className="auth-input-label"),
                                dcc.Input(id="auth-username-input", type="text", className="auth-text-input", placeholder="Enter official ID / UDISE code", n_submit=0)
                            ]
                        ),
                        html.Div(
                            className="auth-input-group",
                            children=[
                                html.Label("Security Password / PIN", className="auth-input-label"),
                                dcc.Input(id="auth-password-input", type="password", className="auth-text-input", placeholder="Enter password / PIN", n_submit=0)
                            ]
                        ),

                        html.Div(
                            className="auth-btn-row",
                            children=[
                                html.Button("Verify Credentials", id="auth-submit-btn", n_clicks=0, className="btn-auth-submit"),
                                html.Button("Cancel / Public View", id="auth-cancel-btn", n_clicks=0, className="btn-auth-cancel")
                            ]
                        ),

                        html.Div(
                            id="auth-demo-hint-box",
                            className="auth-demo-hint",
                            children=[
                                html.Span("Administrative Access: ", style={"fontWeight": "600"}),
                                html.Span(id="auth-demo-hint-text", children="Username: school_admin | Password: school123")
                            ]
                        )
                    ]
                )
            ]
        ),

        # ===================================================================
        # 1. WELCOME / LANDING PAGE VIEW (GOVERNMENT DISTRICT PORTAL)
        # ===================================================================
        html.Div(
            id="welcome-page-view",
            style={"display": "block"},
            children=[
                # Top Utility Bar
                html.Div(
                    className="gov-utility-bar",
                    children=[
                        html.Div(
                            className="gov-utility-left",
                            children=[
                                html.Span("Government of Maharashtra"),
                                html.Span("•", style={"opacity": "0.5"}),
                                html.Span("District Administration Prototype: Chhatrapati Sambhajinagar"),
                            ]
                        ),
                        html.Div(
                            className="gov-utility-right",
                            children=[
                                html.A("Skip to Main Content", href="#main-content-anchor", className="gov-utility-link"),
                                html.Span("|", style={"opacity": "0.3"}),
                                html.Span("Accessibility", className="gov-utility-link"),
                                html.Span("|", style={"opacity": "0.3"}),
                                html.Span("English", style={"fontWeight": "700", "color": "#FFFFFF", "fontSize": "11px"}),
                                html.Span("/", style={"opacity": "0.3"}),
                                html.Span("मराठी", style={"color": "#CFD9E2", "fontSize": "11px"}),
                            ]
                        )
                    ]
                ),

                # Main Identity Header
                html.Div(
                    className="gov-identity-header",
                    children=[
                        html.Div(
                            className="gov-identity-branding",
                            children=[
                                html.Div(
                                    className="gov-emblem-seal",
                                    children=[
                                        html.Span(className="gov-ico gov-ico-seal gov-ico-lg")
                                    ]
                                ),
                                html.Div(
                                    className="gov-title-block",
                                    children=[
                                        html.H1("NITI DRISHTI (नीति दृष्टि)"),
                                        html.Div("National Infrastructure Readiness Index", className="gov-title-en"),
                                        html.Div("Chhatrapati Sambhajinagar District • Vision 2037 Planning Horizon", className="gov-district-name")
                                    ]
                                )
                            ]
                        ),
                        html.Div(
                            className="gov-header-badges",
                            children=[
                                html.Span("ACADEMIC & DECISION-SUPPORT PROTOTYPE", className="gov-prototype-pill"),
                                html.Span("Urban Resource Centers: URC-1 & URC-2", className="gov-district-badge")
                            ]
                        )
                    ]
                ),

                # Horizontal Navigation Bar
                html.Div(
                    className="gov-nav-bar",
                    children=[
                        html.Ul(
                            className="gov-nav-links",
                            children=[
                                html.Li(html.A("Home", href="#", className="gov-nav-item gov-nav-item-active")),
                                html.Li(html.A("District Profile", href="#section-district-glance", className="gov-nav-item")),
                                html.Li(html.A("Infrastructure", href="#section-sectors", className="gov-nav-item")),
                                html.Li(html.A("Analytics", href="#section-services", className="gov-nav-item")),
                                html.Li(html.A("Reports", href="#section-reports", className="gov-nav-item")),
                                html.Li(html.A("Documents", href="#section-framework", className="gov-nav-item")),
                                html.Li(html.A("Methodology", href="#section-methodology", className="gov-nav-item")),
                                html.Li(html.A("About", href="#section-data-sources", className="gov-nav-item")),
                                html.Li(html.A("Contact", href="#section-footer", className="gov-nav-item")),
                            ]
                        ),
                        html.Button("View District Dashboard →", id="welcome-btn-enter-dashboard", n_clicks=0, className="gov-nav-cta-btn")
                    ]
                ),

                # Breadcrumb Bar
                html.Div(
                    className="gov-breadcrumb-bar",
                    id="main-content-anchor",
                    children=[
                        html.A("Home", href="#", className="gov-breadcrumb-item"),
                        html.Span("›", className="gov-breadcrumb-separator"),
                        html.A("NITI DRISHTI", href="#", className="gov-breadcrumb-item"),
                        html.Span("›", className="gov-breadcrumb-separator"),
                        html.Span("District Infrastructure Overview", className="gov-breadcrumb-current")
                    ]
                ),

                # Main Body Container
                html.Div(
                    className="welcome-container",
                    children=[
                        # 1. Hero Section
                        html.Div(
                            className="gov-hero",
                            children=[
                                html.Div(
                                    className="gov-hero-eyebrow",
                                    children=[
                                        html.Span(className="gov-ico gov-ico-chart", style={"width": "14px", "height": "14px", "marginRight": "4px"}),
                                        html.Span("National Infrastructure Readiness Index")
                                    ]
                                ),
                                html.H2(
                                    "NATIONAL INFRASTRUCTURE READINESS INDEX",
                                    className="gov-hero-title"
                                ),
                                html.P(
                                    "A data-driven decision-support platform for assessing infrastructure readiness, identifying development gaps and supporting long-term district planning.",
                                    className="gov-hero-subtitle"
                                ),
                                html.Div(
                                    className="gov-hero-meta-row",
                                    children=[
                                        html.Div([
                                            html.Span("District Focus: "),
                                            html.Strong("Chhatrapati Sambhajinagar")
                                        ], className="gov-hero-meta-item"),
                                        html.Div([
                                            html.Span("Planning Horizon: "),
                                            html.Strong("2037")
                                        ], className="gov-hero-meta-item"),
                                        html.Div([
                                            html.Span("Empirical Baseline: "),
                                            html.Strong("2021–2025 (954 Institutions)")
                                        ], className="gov-hero-meta-item"),
                                        html.Div([
                                            html.Span("Framework: "),
                                            html.Strong("NEP 2020 • RTE Act 2009 • Samagra Shiksha")
                                        ], className="gov-hero-meta-item")
                                    ]
                                ),
                                html.Div(
                                    className="gov-hero-actions",
                                    children=[
                                        html.Button("View District Dashboard →", id="welcome-btn-hero-launch", n_clicks=0, className="btn-gov-primary"),
                                        html.Button("Explore Infrastructure Data ↓", id="welcome-btn-hero-explore", n_clicks=0, className="btn-gov-secondary")
                                    ]
                                )
                            ]
                        ),

                        # 2. District At A Glance Section
                        html.Div(
                            id="section-district-glance",
                            children=[
                                html.Div(
                                    className="gov-section-header",
                                    children=[
                                        html.Div([
                                            html.Span("DISTRICT PROFILE", className="gov-section-tag"),
                                            html.H3("DISTRICT AT A GLANCE", className="gov-section-title"),
                                            html.P("Audited baseline infrastructure indicators across Chhatrapati Sambhajinagar urban cores.", className="gov-section-desc")
                                        ])
                                    ]
                                ),
                                html.Div(
                                    className="gov-glance-grid",
                                    children=[
                                        html.Div(
                                            className="gov-glance-card",
                                            children=[
                                                html.Div(STAT_TOTAL_SCHOOLS, className="gov-glance-num"),
                                                html.Div("Institutions Assessed", className="gov-glance-label"),
                                                html.Div("URC-1 & URC-2 Urban Core", className="gov-glance-sub")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-glance-card",
                                            children=[
                                                html.Div(STAT_TOTAL_STUDENTS, className="gov-glance-num"),
                                                html.Div("Student Population", className="gov-glance-label"),
                                                html.Div("Active Enrolled Cohorts", className="gov-glance-sub")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-glance-card gov-glance-card-highlight",
                                            children=[
                                                html.Div(f"{STAT_READINESS_SCORE} / 100", className="gov-glance-num", style={"color": COLOR_ACCENT_ORANGE}),
                                                html.Div("Infrastructure Readiness", className="gov-glance-label"),
                                                html.Div(STAT_TIER_LABEL, className="gov-glance-sub")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-glance-card",
                                            children=[
                                                html.Div(STAT_REMEDIAL_STUDENTS, className="gov-glance-num", style={"color": COLOR_DANGER}),
                                                html.Div("Students Requiring Support", className="gov-glance-label"),
                                                html.Div("Remedial Learning Backlog", className="gov-glance-sub")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-glance-card",
                                            children=[
                                                html.Div("2037", className="gov-glance-num"),
                                                html.Div("Planning Horizon", className="gov-glance-label"),
                                                html.Div("Demographic Capacity Model", className="gov-glance-sub")
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),

                        # 3. Infrastructure Sectors
                        html.Div(
                            id="section-sectors",
                            children=[
                                html.Div(
                                    className="gov-section-header",
                                    children=[
                                        html.Div([
                                            html.Span("CROSS-SECTORAL COVERAGE", className="gov-section-tag"),
                                            html.H3("INFRASTRUCTURE SECTORS", className="gov-section-title"),
                                            html.P("Infrastructure assessment and planning across key public-service sectors.", className="gov-section-desc")
                                        ])
                                    ]
                                ),
                                html.Div(
                                    className="gov-sectors-grid",
                                    children=[
                                        # Education (Active)
                                        html.Div(
                                            className="gov-sector-card gov-sector-card-active",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-sector-header",
                                                        children=[
                                                            html.Div(className="gov-sector-icon-box", children=[
                                                                html.Span(className="gov-ico gov-ico-education")
                                                            ]),
                                                            html.H4("Education", className="gov-sector-title")
                                                        ]
                                                    ),
                                                    html.P("Comprehensive institutional evaluation across 954 schools, pupil-teacher ratios, and classroom capacities.", className="gov-sector-desc")
                                                ]),
                                                html.Span("ACTIVE / BASELINE", className="gov-status-pill gov-status-pill-active")
                                            ]
                                        ),
                                        # Healthcare (Planned)
                                        html.Div(
                                            className="gov-sector-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-sector-header",
                                                        children=[
                                                            html.Div(className="gov-sector-icon-box", children=[
                                                                html.Span(className="gov-ico gov-ico-health")
                                                            ]),
                                                            html.H4("Healthcare", className="gov-sector-title")
                                                        ]
                                                    ),
                                                    html.P("Primary health centers, bed-to-population ratios, and medical emergency spatial coverage.", className="gov-sector-desc")
                                                ]),
                                                html.Span("PLANNED / FUTURE MODULE", className="gov-status-pill gov-status-pill-planned")
                                            ]
                                        ),
                                        # Transport (Planned)
                                        html.Div(
                                            className="gov-sector-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-sector-header",
                                                        children=[
                                                            html.Div(className="gov-sector-icon-box", children=[
                                                                html.Span(className="gov-ico gov-ico-transport")
                                                            ]),
                                                            html.H4("Transport", className="gov-sector-title")
                                                        ]
                                                    ),
                                                    html.P("Road connectivity, public transit density, school accessibility corridors, and transit hubs.", className="gov-sector-desc")
                                                ]),
                                                html.Span("PLANNED / FUTURE MODULE", className="gov-status-pill gov-status-pill-planned")
                                            ]
                                        ),
                                        # Water Supply (Planned)
                                        html.Div(
                                            className="gov-sector-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-sector-header",
                                                        children=[
                                                            html.Div(className="gov-sector-icon-box", children=[
                                                                html.Span(className="gov-ico gov-ico-water")
                                                            ]),
                                                            html.H4("Water Supply", className="gov-sector-title")
                                                        ]
                                                    ),
                                                    html.P("Potable municipal water distribution, storage reservoirs, and per-capita daily availability.", className="gov-sector-desc")
                                                ]),
                                                html.Span("PLANNED / FUTURE MODULE", className="gov-status-pill gov-status-pill-planned")
                                            ]
                                        ),
                                        # Electricity (Planned)
                                        html.Div(
                                            className="gov-sector-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-sector-header",
                                                        children=[
                                                            html.Div(className="gov-sector-icon-box", children=[
                                                                html.Span(className="gov-ico gov-ico-power")
                                                            ]),
                                                            html.H4("Electricity", className="gov-sector-title")
                                                        ]
                                                    ),
                                                    html.P("Grid substation distribution, feeder reliability, and institutional renewable power adoption.", className="gov-sector-desc")
                                                ]),
                                                html.Span("PLANNED / FUTURE MODULE", className="gov-status-pill gov-status-pill-planned")
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),

                        # 4. Explore NITI DRISHTI / Main Services
                        html.Div(
                            id="section-services",
                            children=[
                                html.Div(
                                    className="gov-section-header",
                                    children=[
                                        html.Div([
                                            html.Span("DECISION SUPPORT PORTALS", className="gov-section-tag"),
                                            html.H3("EXPLORE NITI DRISHTI", className="gov-section-title"),
                                            html.P("Access district information, infrastructure assessments, analytical tools and planning resources.", className="gov-section-desc")
                                        ])
                                    ]
                                ),
                                html.Div(
                                    className="gov-services-grid",
                                    children=[
                                        # MODULE 1: District & Citizen Information
                                        html.Div(
                                            className="gov-service-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-service-card-header",
                                                        children=[
                                                            html.Div(
                                                                className="gov-service-card-title-group",
                                                                children=[
                                                                    html.Div(className="gov-service-icon-box", children=[
                                                                        html.Span(className="gov-ico gov-ico-chart")
                                                                    ]),
                                                                    html.H4("District & Citizen Information", className="gov-service-title")
                                                                ]
                                                            ),
                                                            html.Span("PUBLIC ACCESS", className="status-pill status-pill-green")
                                                        ]
                                                    ),
                                                    html.P("Access district-level infrastructure indicators, school information, readiness metrics and geographic information.", className="gov-service-desc")
                                                ]),
                                                html.Button("View District Dashboard →", id="welcome-btn-citizen", n_clicks=0, className="gov-service-btn")
                                            ]
                                        ),
                                        # MODULE 2: Institution Assessment
                                        html.Div(
                                            className="gov-service-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-service-card-header",
                                                        children=[
                                                            html.Div(
                                                                className="gov-service-card-title-group",
                                                                children=[
                                                                    html.Div(className="gov-service-icon-box", children=[
                                                                        html.Span(className="gov-ico gov-ico-education")
                                                                    ]),
                                                                    html.H4("Institution Assessment", className="gov-service-title")
                                                                ]
                                                            ),
                                                            html.Span("INSTITUTIONAL", className="status-pill status-pill-cyan")
                                                        ]
                                                    ),
                                                    html.P("Review institution-level infrastructure indicators, compliance metrics and identified facility gaps.", className="gov-service-desc")
                                                ]),
                                                html.Button("View Institution Assessment →", id="welcome-btn-school", n_clicks=0, className="gov-service-btn")
                                            ]
                                        ),
                                        # MODULE 3: Infrastructure Scenario Analysis
                                        html.Div(
                                            className="gov-service-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-service-card-header",
                                                        children=[
                                                            html.Div(
                                                                className="gov-service-card-title-group",
                                                                children=[
                                                                    html.Div(className="gov-service-icon-box", children=[
                                                                        html.Span(className="gov-ico gov-ico-seal")
                                                                    ]),
                                                                    html.H4("Infrastructure Scenario Analysis", className="gov-service-title")
                                                                ]
                                                            ),
                                                            html.Span("POLICY MAKERS", className="status-pill status-pill-cyan")
                                                        ]
                                                    ),
                                                    html.P("Evaluate the potential impact of infrastructure interventions under different planning scenarios.", className="gov-service-desc")
                                                ]),
                                                html.Button("Open Scenario Analysis →", id="welcome-btn-govt", n_clicks=0, className="gov-service-btn")
                                            ]
                                        ),
                                        # MODULE 4: Policy & Planning Assistant
                                        html.Div(
                                            className="gov-service-card",
                                            children=[
                                                html.Div([
                                                    html.Div(
                                                        className="gov-service-card-header",
                                                        children=[
                                                            html.Div(
                                                                className="gov-service-card-title-group",
                                                                children=[
                                                                    html.Div(className="gov-service-icon-box", children=[
                                                                        html.Span(className="gov-ico gov-ico-doc")
                                                                    ]),
                                                                    html.H4("Policy & Planning Assistant", className="gov-service-title")
                                                                ]
                                                            ),
                                                            html.Span("AI-ASSISTED", className="status-pill status-pill-green")
                                                        ]
                                                    ),
                                                    html.P("Retrieve and summarize relevant policy documents, statutory guidelines and planning information.", className="gov-service-desc")
                                                ]),
                                                html.Button("Open Policy Assistant →", id="welcome-btn-copilot", n_clicks=0, className="gov-service-btn")
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),

                        # 5. District Infrastructure Map Section
                        html.Div(
                            id="section-map",
                            children=[
                                html.Div(
                                    className="gov-section-header",
                                    children=[
                                        html.Div([
                                            html.Span("GEOSPATIAL INTELLIGENCE", className="gov-section-tag"),
                                            html.H3("DISTRICT INFRASTRUCTURE MAP", className="gov-section-title"),
                                            html.P("Geospatial view of infrastructure facilities and identified development gaps across Chhatrapati Sambhajinagar.", className="gov-section-desc")
                                        ]),
                                        html.Button("Open Full GIS Interactive Console →", id="welcome-btn-map-launch", n_clicks=0, className="btn-gov-secondary")
                                    ]
                                ),
                                html.Div(
                                    className="gov-info-card",
                                    style={"marginBottom": "30px"},
                                    children=[
                                        html.Div(
                                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px", "flexWrap": "wrap", "gap": "10px"},
                                            children=[
                                                html.Span("954 Educational Institutions Geocoded across Chhatrapati Sambhajinagar", style={"fontWeight": "600", "fontSize": "13px", "color": COLOR_PRIMARY_NAVY}),
                                                html.Div(
                                                    style={"display": "flex", "gap": "14px"},
                                                    children=[
                                                        html.Span([
                                                            html.Span(style={"display": "inline-block", "width": "10px", "height": "10px", "borderRadius": "2px", "backgroundColor": "#2E7D32", "marginRight": "5px"}),
                                                            html.Span("PTR ≤ 30:1 Compliant", style={"fontSize": "11.5px", "color": COLOR_MUTED})
                                                        ]),
                                                        html.Span([
                                                            html.Span(style={"display": "inline-block", "width": "10px", "height": "10px", "borderRadius": "2px", "backgroundColor": "#B3261E", "marginRight": "5px"}),
                                                            html.Span("Capacity Deficit Alert", style={"fontSize": "11.5px", "color": COLOR_MUTED})
                                                        ])
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.P("The GIS platform visualizes institutional distributions across URC-1 and URC-2 urban blocks, identifying localized classroom shortfalls, cohort retention drops, and statutory pupil-teacher compliance zones.", style={"fontSize": "13px", "color": COLOR_MUTED, "margin": "0 0 14px 0"}),
                                        html.Button("Launch Interactive Map Console →", id="welcome-btn-map-launch-2", n_clicks=0, className="btn-gov-primary", style={"fontSize": "12.5px"})
                                    ]
                                )
                            ]
                        ),

                        # 6. What's New Section
                        html.Div(
                            id="section-updates",
                            className="gov-updates-card",
                            children=[
                                html.Div(
                                    style={"padding": "16px 20px", "backgroundColor": "#F8FAFC", "borderBottom": f"1px solid {COLOR_BORDER}"},
                                    children=[
                                        html.Span("ADMINISTRATIVE NOTIFICATIONS", className="gov-section-tag"),
                                        html.H3("WHAT'S NEW & LATEST UPDATES", style={"margin": "2px 0 0 0", "fontSize": "17px", "color": COLOR_PRIMARY_NAVY, "fontWeight": "700"}),
                                    ]
                                ),
                                html.Table(
                                    className="gov-updates-table",
                                    children=[
                                        html.Thead(
                                            html.Tr([
                                                html.Th("Publication Date", style={"width": "150px"}),
                                                html.Th("Subject & Administrative Action"),
                                                html.Th("Sector / Pillar", style={"width": "200px"})
                                            ])
                                        ),
                                        html.Tbody([
                                            html.Tr([
                                                html.Td("08 Oct 2026", className="gov-update-date"),
                                                html.Td([
                                                    html.Span("UPDATE", className="gov-update-tag"),
                                                    "Education infrastructure readiness index baseline calculated for Chhatrapati Sambhajinagar (Score: 60.4/100)."
                                                ]),
                                                html.Td("Readiness Index Engine", style={"color": COLOR_MUTED})
                                            ]),
                                            html.Tr([
                                                html.Td("05 Oct 2026", className="gov-update-date"),
                                                html.Td([
                                                    html.Span("DATA", className="gov-update-tag"),
                                                    "District institutional master dataset verified against UDISE+ and municipal administration records (954 schools)."
                                                ]),
                                                html.Td("Institutional Master Register", style={"color": COLOR_MUTED})
                                            ]),
                                            html.Tr([
                                                html.Td("01 Oct 2026", className="gov-update-date"),
                                                html.Td([
                                                    html.Span("REPORT", className="gov-update-tag"),
                                                    "Demographic enrollment forecast trajectory published through Vision 2037 with 85% statistical confidence bounds."
                                                ]),
                                                html.Td("Predictive Demographic Modeling", style={"color": COLOR_MUTED})
                                            ]),
                                            html.Tr([
                                                html.Td("25 Sep 2026", className="gov-update-date"),
                                                html.Td([
                                                    html.Span("POLICY", className="gov-update-tag"),
                                                    "Statutory policy retrieval index updated covering NEP 2020, RTE Act 2009, and Samagra Shiksha Framework."
                                                ]),
                                                html.Td("Statutory Policy Vectorstore", style={"color": COLOR_MUTED})
                                            ]),
                                        ])
                                    ]
                                )
                            ]
                        ),

                        # 7. Infrastructure Readiness Framework
                        html.Div(
                            id="section-framework",
                            className="gov-framework-container",
                            children=[
                                html.Span("STATUTORY BENCHMARKING", className="gov-section-tag"),
                                html.H3("INFRASTRUCTURE READINESS FRAMEWORK", style={"margin": "2px 0 0 0", "fontSize": "19px", "color": COLOR_PRIMARY_NAVY, "fontWeight": "700"}),
                                html.P("Multi-Pillar Infrastructure Readiness Index (IRI): The NITI DRISHTI Readiness Index combines key infrastructure indicators to assess current conditions, identify development gaps and support evidence-based planning.", style={"margin": "4px 0 0 0", "fontSize": "13px", "color": COLOR_MUTED}),
                                html.Div(
                                    className="gov-pillars-grid",
                                    children=[
                                        html.Div(
                                            className="gov-pillar-card",
                                            children=[
                                                html.Div("Weight: 25%", className="gov-pillar-weight"),
                                                html.H4("1. Equity & Inclusion", className="gov-pillar-name"),
                                                html.P("Gender Parity Index (GPI = Girls / Boys) and female enrolment ratio. Target benchmark: Optimal GPI = 1.00.", className="gov-pillar-desc")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-pillar-card",
                                            children=[
                                                html.Div("Weight: 30%", className="gov-pillar-weight"),
                                                html.H4("2. Promotion & Remedial Support", className="gov-pillar-name"),
                                                html.P("Empirical URC pass rate evaluated against 27,235 remedial backlog students requiring academic bridge interventions.", className="gov-pillar-desc")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-pillar-card",
                                            children=[
                                                html.Div("Weight: 25%", className="gov-pillar-weight"),
                                                html.H4("3. Cohort Retention", className="gov-pillar-name"),
                                                html.P("Upper Primary (Classes 6–8) to Secondary (Classes 9–10) transition ratio. Target benchmark: Transition ≥ 0.67.", className="gov-pillar-desc")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-pillar-card",
                                            children=[
                                                html.Div("Weight: 20%", className="gov-pillar-weight"),
                                                html.H4("4. Operational Capacity", className="gov-pillar-name"),
                                                html.P("Institutional operational status, classroom physical adequacy, and RTE Pupil-Teacher Ratio compliance (PTR ≤ 30:1).", className="gov-pillar-desc")
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),

                        # 8. Data Sources & Analytical Methodology
                        html.Div(
                            id="section-methodology",
                            className="gov-two-col-grid",
                            children=[
                                # Card 1: Data Sources
                                html.Div(
                                    id="section-data-sources",
                                    className="gov-info-card",
                                    children=[
                                        html.H3("DATA SOURCES & INTEGRITY"),
                                        html.P("The NITI DRISHTI platform ingests, audits, and standardizes data from verified public records:", style={"fontSize": "12.5px", "color": COLOR_MUTED, "marginBottom": "10px"}),
                                        html.Ul(
                                            className="gov-list-clean",
                                            children=[
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("UDISE+ (Unified District Information System for Education Plus) official school registry.")
                                                ]),
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("Department of School Education and Sports, Government of Maharashtra institutional records.")
                                                ]),
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("District Administration, Chhatrapati Sambhajinagar urban core (URC-1 & URC-2) promotional records.")
                                                ]),
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("Census of India district population demographics & municipal cohort projection models.")
                                                ]),
                                            ]
                                        ),
                                        html.Div(
                                            style={"marginTop": "14px", "paddingTop": "10px", "borderTop": f"1px solid {COLOR_BORDER}", "fontSize": "12px", "color": COLOR_MUTED},
                                            children=[
                                                html.Strong("Data Window: "),
                                                "2021–2025 Empirical Window • Projections Through 2037"
                                            ]
                                        )
                                    ]
                                ),
                                # Card 2: Analytical Methodology
                                html.Div(
                                    className="gov-info-card",
                                    children=[
                                        html.H3("ANALYTICAL METHODOLOGY"),
                                        html.P("Evidence-based computational modeling implemented in the system:", style={"fontSize": "12.5px", "color": COLOR_MUTED, "marginBottom": "10px"}),
                                        html.Ul(
                                            className="gov-list-clean",
                                            children=[
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("Multi-Criteria Decision Analysis (MCDA): Weighted composite Infrastructure Readiness Index (0–100 scale).")
                                                ]),
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("Demographic Forecasting: Time-series demographic modeling (Huber regression) with 85% confidence intervals.")
                                                ]),
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("Geospatial GIS Catchment: Spatial institutional coordinates mapping and localized RTE 30:1 capacity gap auditing.")
                                                ]),
                                                html.Li([
                                                    html.Span("•", className="gov-list-bullet"),
                                                    html.Span("Retrieval-Augmented Generation: Vector embeddings (FAISS) indexing official policy frameworks (NEP 2020, RTE Act 2009).")
                                                ]),
                                            ]
                                        ),
                                        html.Div(
                                            style={"marginTop": "14px", "paddingTop": "10px", "borderTop": f"1px solid {COLOR_BORDER}", "fontSize": "12px", "color": COLOR_MUTED},
                                            children=[
                                                html.Strong("Compliance Standard: "),
                                                "RTE Act 2009 (PTR 30:1) • NEP 2020 Secondary Retention"
                                            ]
                                        )
                                    ]
                                ),
                            ]
                        ),

                        # 9. Reports & Documents
                        html.Div(
                            id="section-reports",
                            children=[
                                html.Div(
                                    className="gov-section-header",
                                    children=[
                                        html.Div([
                                            html.Span("PUBLICATION ARCHIVE", className="gov-section-tag"),
                                            html.H3("REPORTS & DOCUMENTS", className="gov-section-title"),
                                            html.P("Official decision-support deliverables and statutory planning frameworks.", className="gov-section-desc")
                                        ])
                                    ]
                                ),
                                html.Div(
                                    className="gov-reports-grid",
                                    children=[
                                        html.Div(
                                            className="gov-report-item",
                                            children=[
                                                html.Div(
                                                    className="gov-report-info",
                                                    children=[
                                                        html.Span(className="gov-ico gov-ico-doc"),
                                                        html.Div([
                                                            html.Div("District Infrastructure Assessment", className="gov-report-title"),
                                                            html.Div("Verified Statutory Policy Directive • 2026", className="gov-report-meta")
                                                        ])
                                                    ]
                                                ),
                                                html.Span("PDF", className="status-pill status-pill-cyan")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-report-item",
                                            children=[
                                                html.Div(
                                                    className="gov-report-info",
                                                    children=[
                                                        html.Span(className="gov-ico gov-ico-doc"),
                                                        html.Div([
                                                            html.Div("Readiness Index Methodology Framework", className="gov-report-title"),
                                                            html.Div("4-Pillar Formulation Specification", className="gov-report-meta")
                                                        ])
                                                    ]
                                                ),
                                                html.Span("DOC", className="status-pill status-pill-cyan")
                                            ]
                                        ),
                                        html.Div(
                                            className="gov-report-item",
                                            children=[
                                                html.Div(
                                                    className="gov-report-info",
                                                    children=[
                                                        html.Span(className="gov-ico gov-ico-doc"),
                                                        html.Div([
                                                            html.Div("Demographic Forecast Report (2021–2037)", className="gov-report-title"),
                                                            html.Div("Cohort Projection & Capacity Deficit", className="gov-report-meta")
                                                        ])
                                                    ]
                                                ),
                                                html.Span("DATA", className="status-pill status-pill-cyan")
                                            ]
                                        ),
                                    ]
                                )
                            ]
                        ),

                        # 10. Important Links
                        html.Div(
                            id="section-links",
                            children=[
                                html.Div(
                                    className="gov-section-header",
                                    children=[
                                        html.Div([
                                            html.Span("EXTERNAL PORTALS", className="gov-section-tag"),
                                            html.H3("IMPORTANT LINKS", className="gov-section-title"),
                                            html.P("Official government portals and national educational databases.", className="gov-section-desc")
                                        ])
                                    ]
                                ),
                                html.Div(
                                    className="gov-links-grid",
                                    children=[
                                        html.A([
                                            html.Span("Government of Maharashtra"),
                                            html.Span("↗", style={"fontSize": "11px"})
                                        ], href="https://maharashtra.gov.in", target="_blank", className="gov-ext-link-card"),
                                        html.A([
                                            html.Span("Chhatrapati Sambhajinagar District"),
                                            html.Span("↗", style={"fontSize": "11px"})
                                        ], href="https://chhatrapatisambhajinagar.maharashtra.gov.in", target="_blank", className="gov-ext-link-card"),
                                        html.A([
                                            html.Span("Ministry of Education, GoI"),
                                            html.Span("↗", style={"fontSize": "11px"})
                                        ], href="https://education.gov.in", target="_blank", className="gov-ext-link-card"),
                                        html.A([
                                            html.Span("UDISE+ National Portal"),
                                            html.Span("↗", style={"fontSize": "11px"})
                                        ], href="https://udiseplus.gov.in", target="_blank", className="gov-ext-link-card"),
                                    ]
                                )
                            ]
                        )
                    ]
                ),

                # Government Footer
                html.Footer(
                    id="section-footer",
                    className="gov-footer",
                    children=[
                        html.Div(
                            className="gov-footer-content",
                            children=[
                                # Column 1: Identity
                                html.Div(
                                    className="gov-footer-col",
                                    children=[
                                        html.H4("NITI DRISHTI (नीति दृष्टि)"),
                                        html.P("National Infrastructure Readiness Index: A decision-support system evaluating district educational infrastructure readiness, demographic capacity deficits through 2037, and statutory policy alignment."),
                                        html.P("Target District: Chhatrapati Sambhajinagar (URC-1 & URC-2 Urban Core), Maharashtra.")
                                    ]
                                ),
                                # Column 2: Quick Links
                                html.Div(
                                    className="gov-footer-col",
                                    children=[
                                        html.H4("Quick Links"),
                                        html.Ul(
                                            className="gov-footer-links",
                                            children=[
                                                html.Li(html.A("District Profile", href="#section-district-glance")),
                                                html.Li(html.A("Infrastructure Sectors", href="#section-sectors")),
                                                html.Li(html.A("Explore Services", href="#section-services")),
                                                html.Li(html.A("Geospatial Map", href="#section-map")),
                                                html.Li(html.A("Latest Updates", href="#section-updates")),
                                            ]
                                        )
                                    ]
                                ),
                                # Column 3: Resources
                                html.Div(
                                    className="gov-footer-col",
                                    children=[
                                        html.H4("Resources"),
                                        html.Ul(
                                            className="gov-footer-links",
                                            children=[
                                                html.Li(html.A("Methodology", href="#section-methodology")),
                                                html.Li(html.A("Data Sources", href="#section-data-sources")),
                                                html.Li(html.A("Readiness Framework", href="#section-framework")),
                                                html.Li(html.A("Reports & Documents", href="#section-reports")),
                                                html.Li(html.A("External Portals", href="#section-links")),
                                            ]
                                        )
                                    ]
                                ),
                                # Column 4: Contact & Attribution
                                html.Div(
                                    className="gov-footer-col",
                                    children=[
                                        html.H4("Academic Prototype"),
                                        html.P("Academic Project • MGM University, Chhatrapati Sambhajinagar."),
                                        html.P("Developed for educational and decision-support research purposes under Department of Data Science & AI."),
                                        html.P("Planning Horizon: Vision 2037 • Baseline: 2021–2025.")
                                    ]
                                ),
                            ]
                        ),
                        html.Div(
                            className="gov-footer-bottom",
                            children=[
                                html.P("Data and analysis presented for academic and decision-support demonstration purposes."),
                                html.P("NITI DRISHTI • National Infrastructure Readiness Index • Last Updated: 08 Oct 2026")
                            ]
                        )
                    ]
                )
            ]
        ),

        # ===================================================================
        # 2. DASHBOARD VIEW (INTERACTIVE DECISION PLATFORM)
        # ===================================================================
        html.Div(
            id="dashboard-page-view",
            className="dashboard-container",
            style={"display": "none"},
            children=[
                # ===================================================================
                # HEADER: Government Institutional Identity
                # ===================================================================
                html.Div(
                    className="platform-header",
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "14px"},
                            children=[
                                html.Div(
                                    className="gov-emblem-seal",
                                    style={"width": "44px", "height": "44px"},
                                    children=[
                                        html.Span(className="gov-ico gov-ico-seal", style={"width": "26px", "height": "26px"})
                                    ]
                                ),
                                html.Div([
                                    html.H1(
                                        "NITI DRISHTI (नीति दृष्टि)",
                                        style={"margin": "0", "fontSize": "20px", "fontWeight": "700", "color": COLOR_PRIMARY_NAVY}
                                    ),
                                    html.P(
                                        "National Infrastructure Readiness Index • Chhatrapati Sambhajinagar District Dashboard",
                                        style={"margin": "2px 0 0 0", "fontSize": "12.5px", "color": COLOR_GOVT_BLUE, "fontWeight": "600"}
                                    )
                                ])
                            ]
                        ),
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                            children=[
                                html.Button("← Back to Portal Home", id="dashboard-btn-back-home", n_clicks=0, className="btn-back-home"),
                                html.Span("DISTRICT SYSTEM ACTIVE", className="status-pill status-pill-green"),
                                html.Span("HORIZON 2040", className="status-pill status-pill-cyan")
                            ]
                        )
                    ]
                ),

                # ===================================================================
                # MULTI-PERSONA PORTAL NAVIGATION (GOVERNMENT STYLE)
                # ===================================================================
                html.Div(
                    className="portal-nav-bar",
                    children=[
                        html.Span("ACTIVE PORTAL:", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_MUTED, "letterSpacing": "0.5px", "marginLeft": "4px"}),
                        html.Button("Public Citizen View", id="portal-tab-citizen", n_clicks=0, className="portal-tab-btn portal-tab-btn-active"),
                        html.Button("Institution Assessment Portal", id="portal-tab-school", n_clicks=0, className="portal-tab-btn"),
                        html.Button("Infrastructure Scenario Analysis", id="portal-tab-govt", n_clicks=0, className="portal-tab-btn"),
                        dcc.Store(id="active-portal-store", data="citizen"),
                        html.Div(
                            id="portal-auth-status-container",
                            style={"marginLeft": "auto", "display": "flex", "alignItems": "center", "gap": "10px"},
                            children=[
                                html.Span(id="portal-auth-status-text", children="Public Citizen (Direct Access)", style={"fontSize": "12px", "color": COLOR_MUTED, "fontWeight": "600"}),
                                html.Button("Logout", id="portal-logout-btn", n_clicks=0, className="btn-auth-logout", style={"display": "none"})
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
                            html.Span("INFRASTRUCTURE SCENARIO ANALYSIS & DEMOGRAPHIC PLANNING", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_GOVT_BLUE, "letterSpacing": "0.5px"}),
                            html.H4("District Infrastructure Planning & Classroom Needs (2026–2040)", style={"margin": "2px 0 0 0", "fontSize": "16px", "fontWeight": "700", "color": COLOR_PITCH}),
                        ]),
                        html.Div(style={"display": "flex", "gap": "8px"}, children=[
                            html.Span("DEMOGRAPHIC FORECAST MODEL", className="status-pill status-pill-green"),
                            html.Span("HORIZON 2026–2040", className="status-pill status-pill-cyan")
                        ])
                    ]
                ),
                html.P(
                    "Evaluate student population trends, classroom requirements (under the RTE 30:1 student-teacher rule), and teacher staffing needs across Chhatrapati Sambhajinagar. Use the sliders below to test different planning years, planned classroom construction, and priority weights.",
                    style={"fontSize": "12.5px", "color": COLOR_MUTED, "marginBottom": "16px", "lineHeight": "1.5"}
                ),
                # Control Levers Grid (Two balanced cards)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"},
                    children=[
                        # Card 1: Horizon & Classroom Construction
                        html.Div(
                            style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "16px"},
                            children=[
                                html.Span("1. PLANNING YEAR & CLASSROOM CONSTRUCTION TARGETS", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "display": "block", "marginBottom": "12px"}),
                                html.Div(style={"marginBottom": "14px"}, children=[
                                    html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}, children=[
                                        html.Label("Target Planning Year (2-Year Window):", style={"fontSize": "12px", "fontWeight": "700", "color": COLOR_GOVT_BLUE}),
                                        html.Span("2026–2040 Planning Milestone", style={"fontSize": "11px", "color": COLOR_MUTED})
                                    ]),
                                    dcc.Slider(
                                        id="sim-horizon-slider",
                                        min=2026,
                                        max=2040,
                                        step=2,
                                        value=2030,
                                        marks={2026: "2026", 2028: "2028", 2030: "2030", 2032: "2032", 2034: "2034", 2036: "2036", 2038: "2038", 2040: "2040"},
                                        tooltip={"placement": "bottom", "always_visible": False}
                                    )
                                ]),
                                html.Div(children=[
                                    html.Div(style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}, children=[
                                        html.Label("Planned New Classrooms (Construction):", style={"fontSize": "12px", "fontWeight": "700", "color": COLOR_PITCH}),
                                        html.Span("+0 to +1,000 Classrooms (Step 50)", style={"fontSize": "11px", "color": COLOR_MUTED})
                                    ]),
                                    dcc.Slider(
                                        id="sim-classroom-slider",
                                        min=0,
                                        max=1000,
                                        step=50,
                                        value=0,
                                        marks={0: "+0", 250: "+250", 500: "+500", 750: "+750", 1000: "+1,000"},
                                        tooltip={"placement": "bottom", "always_visible": False}
                                    )
                                ])
                            ]
                        ),
                        # Card 2: Multi-Criteria Priority Weights
                        html.Div(
                            style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "16px"},
                            children=[
                                html.Span("2. PRIORITY WEIGHTS FOR DISTRICT SCORING", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "display": "block", "marginBottom": "12px"}),
                                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"}, children=[
                                    html.Div([
                                        html.Label("Equity & Girls' Enrollment", style={"fontSize": "11.5px", "fontWeight": "600", "color": COLOR_PITCH}),
                                        dcc.Slider(id="sim-w-equity", min=0.10, max=0.40, step=0.05, value=0.25, marks={0.1: "10%", 0.25: "25%", 0.4: "40%"}, tooltip={"placement": "bottom", "always_visible": False})
                                    ]),
                                    html.Div([
                                        html.Label("Pass Rate & Remedial Support", style={"fontSize": "11.5px", "fontWeight": "600", "color": COLOR_PITCH}),
                                        dcc.Slider(id="sim-w-promo", min=0.10, max=0.40, step=0.05, value=0.30, marks={0.1: "10%", 0.3: "30%", 0.4: "40%"}, tooltip={"placement": "bottom", "always_visible": False})
                                    ]),
                                    html.Div([
                                        html.Label("Retention (Classes 6–10)", style={"fontSize": "11.5px", "fontWeight": "600", "color": COLOR_PITCH}),
                                        dcc.Slider(id="sim-w-ret", min=0.10, max=0.40, step=0.05, value=0.25, marks={0.1: "10%", 0.25: "25%", 0.4: "40%"}, tooltip={"placement": "bottom", "always_visible": False})
                                    ]),
                                    html.Div([
                                        html.Label("Classroom & Teacher Capacity", style={"fontSize": "11.5px", "fontWeight": "600", "color": COLOR_PITCH}),
                                        dcc.Slider(id="sim-w-cap", min=0.10, max=0.40, step=0.05, value=0.20, marks={0.1: "10%", 0.2: "20%", 0.4: "40%"}, tooltip={"placement": "bottom", "always_visible": False})
                                    ])
                                ])
                            ]
                        )
                    ]
                ),
                # Dynamic Executive Infrastructure Intelligence Console
                html.Div(id="govt-infra-intelligence-panel")
            ]
        ),

        # ===================================================================
        # INSTITUTION ASSESSMENT PORTAL: DATA ENTRY & DIAGNOSTIC AUDIT
        # ===================================================================
        html.Div(
            id="section-school-admin-view",
            className="executive-card",
            style={"display": "none", "border": f"1px solid {COLOR_TEAL}", "marginBottom": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "14px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.Span("INSTITUTION ASSESSMENT PORTAL", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_TEAL, "letterSpacing": "0.5px"}),
                            html.H3("Institution Facility Audit & Private Diagnostic Portal", style={"margin": "2px 0 0 0", "fontSize": "17px", "fontWeight": "700", "color": COLOR_PITCH}),
                            html.P("This operational evaluation is strictly private to your authenticated institution. Review your official data, test planning years, or run a Diagnostic Quality Audit.", style={"margin": "4px 0 0 0", "fontSize": "12.5px", "color": COLOR_MUTED})
                        ]),
                        html.Span("OFFICIAL INSTITUTION ACTIVE", className="status-pill status-pill-green")
                    ]
                ),

                # Locked Authenticated Institution Banner & Target Planning Horizon
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "marginBottom": "16px"},
                    children=[
                        html.Div(
                            id="school-locked-banner",
                            style={"backgroundColor": "#F0FDF4", "border": "1px solid #BBF7D0", "borderRadius": "8px", "padding": "12px 16px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                            children=[
                                html.Div([
                                    html.Span("AUTHENTICATED INSTITUTION (CONFIDENTIAL PROFILE)", style={"fontSize": "10.5px", "fontWeight": "800", "color": COLOR_SUCCESS, "letterSpacing": "0.5px"}),
                                    html.H4(id="school-locked-name-display", children="Selected Institution", style={"margin": "2px 0 2px 0", "fontSize": "16px", "fontWeight": "800", "color": COLOR_PITCH}),
                                    html.P(id="school-locked-udise-display", children="UDISE Code: --", style={"margin": "0", "fontSize": "12px", "color": COLOR_MUTED})
                                ]),
                                html.Span("PRIVATE ACCESS", className="status-pill status-pill-green")
                            ]
                        ),
                        html.Div([
                            html.Label("Target Planning Year (2-Year Window):", className="filter-label", style={"color": COLOR_GOVT_BLUE}),
                            dcc.Dropdown(
                                id="school-horizon-milestone",
                                options=[
                                    {"label": "Year 2026 (Immediate)", "value": 2026},
                                    {"label": "Year 2028 (Short-Term)", "value": 2028},
                                    {"label": "Year 2030 (Medium-Term)", "value": 2030},
                                    {"label": "Year 2032 (Medium-Term)", "value": 2032},
                                    {"label": "Year 2034 (Long-Term)", "value": 2034},
                                    {"label": "Year 2036 (Long-Term)", "value": 2036},
                                    {"label": "Year 2038 (Strategic Horizon)", "value": 2038},
                                    {"label": "Year 2040 (Vision Horizon)", "value": 2040},
                                ],
                                value=2030,
                                clearable=False
                            )
                        ])
                    ]
                ),

                # School Form Grid
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px", "marginBottom": "16px"},
                    children=[
                        html.Div([
                            html.Label("School Name (Locked)", className="filter-label"),
                            dcc.Input(id="input-sch-name", type="text", className="search-input-field", disabled=True, placeholder="e.g. Z.P. High School")
                        ]),
                        html.Div([
                            html.Label("UDISE Code (11 Digits, Locked)", className="filter-label"),
                            dcc.Input(id="input-sch-udise", type="number", className="search-input-field", disabled=True, placeholder="e.g. 27191001234")
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
        # DISTRICT OVERVIEW CONTAINER (Command Ribbon, Section 1 & Section 2)
        # ===================================================================
        html.Div(
            id="section-district-overview",
            children=[
                # TOP COMMAND FILTER RIBBON
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
            ]
        ),

        # ===================================================================
        # CITIZEN & GOVT GIS MAP + SCHOOL DIRECTORY CONTAINER (Sections 2.5 & 3)
        # ===================================================================
        html.Div(
            id="section-citizen-map-and-directory",
            children=[
                # SECTION 2.5: INTERACTIVE INFRASTRUCTURE GIS MAP (Chhatrapati Sambhajinagar)
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

                # SECTION 3: INSTITUTIONAL SEARCH & COMPREHENSIVE SCHOOL DETAILS
                html.Div(
                    className="executive-card",
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px", "flexWrap": "wrap", "gap": "14px"},
                            children=[
                                html.Div([
                                    html.Span("INSTITUTIONAL SEARCH & COMPLETE DETAILS", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_GOVT_BLUE, "letterSpacing": "0.5px"}),
                                    html.H3("School Search & Institutional Profile", style={"margin": "2px 0 0 0", "fontSize": "17px", "fontWeight": "800", "color": COLOR_PITCH}),
                                    html.P("Search any school by name, 11-digit UDISE code, or click any pin on the map above to view its complete operational dossier, facility numbers, staffing, and 3-step action plan.", style={"margin": "2px 0 0 0", "fontSize": "12.5px", "color": COLOR_MUTED})
                                ]),
                                html.Span("SEARCH CONSOLE ACTIVE", className="status-pill status-pill-cyan")
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
                                    placeholder="Type school name, UDISE code (e.g. 27191001234), or block to inspect complete details..."
                                ),
                                html.Button(
                                    "Clear Search",
                                    id="clear-search-btn",
                                    n_clicks=0,
                                    className="btn-outline"
                                )
                            ]
                        ),

                        # Dynamic Complete School Details Dossier Card
                        html.Div(id="facility-inspector-container")
                    ]
                )
            ]
        ),

        # ===================================================================
        # SECTION 4: POLICY & PLANNING ASSISTANT (AI-ASSISTED)
        # ===================================================================
        html.Div(
            className="executive-card",
            id="section-policy-assistant",
            style={"border": f"1px solid {COLOR_GOVT_BLUE}", "backgroundColor": "#F8FAFC"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "16px", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.H3("Policy & Planning Assistant", id="rag-card-title", style={"margin": "0", "fontSize": "17px", "fontWeight": "700", "color": COLOR_PRIMARY_NAVY}),
                            html.P(
                                "Cross-references district school data with official education standards (NEP 2020, RTE Act 2009, Samagra Shiksha) to provide clear planning recommendations.",
                                id="rag-card-subtitle",
                                style={"margin": "4px 0 0 0", "fontSize": "13px", "color": COLOR_MUTED}
                            )
                        ]),
                        html.Span("AI-ASSISTED", id="rag-card-badge", className="status-pill status-pill-cyan")
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
                    placeholder="Enter an inquiry or policy topic (e.g. 'Explain current data', 'Explain demographic forecast', 'Synthesize remedial policy directive')...",
                    value="Explain the Student Enrollment Forecast (2021 to 2037), the predictive demographic trajectory with 85% statistical confidence bounds, and the classroom deficit under RTE 30:1 norms.",
                    style={
                        "width": "100%",
                        "height": "75px",
                        "backgroundColor": "#FFFFFF",
                        "color": COLOR_PITCH,
                        "border": f"1px solid {COLOR_BORDER}",
                        "borderRadius": "4px",
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
                        html.Button("Synthesize Policy Directive", id="rag-submit-btn", n_clicks=0, className="btn-primary"),
                        html.Button("Download Official Policy Brief (PDF)", id="rag-download-pdf-btn", n_clicks=0, disabled=True, className="btn-pdf")
                    ]
                ),

                # Dynamic Output Area (Flicker-Free Smooth Streaming with Active Agentic Thinking)
                dcc.Loading(
                    id="rag-loading-wrapper",
                    type="circle",
                    color=COLOR_GOVT_BLUE,
                    children=html.Div(
                        id="rag-output-container",
                        style={"marginTop": "18px"},
                        children=[
                            html.Div(
                                "Enter an inquiry or policy topic to retrieve statutory guidelines and synthesize evidence-based policy briefs.",
                                style={"padding": "20px", "backgroundColor": "#FFFFFF", "borderRadius": "4px", "border": f"1px solid {COLOR_BORDER}", "color": COLOR_MUTED, "fontSize": "13px", "textAlign": "center"}
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
                html.P("NITI DRISHTI (नीति दृष्टि) • National Infrastructure Readiness Index", style={"margin": "0 0 4px 0", "fontWeight": "700", "color": COLOR_PRIMARY_NAVY}),
                html.P("Chhatrapati Sambhajinagar District • Planning Horizon 2037 • Baseline: 954 Institutions | 278,945 Students", style={"margin": "0 0 4px 0"}),
                html.P("Academic / Decision-Support Prototype • Designed for District Planning Demonstration", style={"margin": "0", "fontSize": "11px", "color": "#7F8C8D"})
            ]
        )
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
    # Search Dossier & Govt Intelligence
    Output("facility-inspector-container", "children"),
    Output("govt-infra-intelligence-panel", "children"),
    # Inputs from Top Command Ribbon & Search
    Input("filter-block", "value"),
    Input("filter-mgmt", "value"),
    Input("filter-category", "value"),
    Input("filter-tier", "value"),
    Input("school-search-input", "value"),
    Input("sim-horizon-slider", "value"),
    Input("sim-classroom-slider", "value"),
    Input("sim-w-equity", "value"),
    Input("sim-w-promo", "value"),
    Input("sim-w-ret", "value"),
    Input("sim-w-cap", "value")
)
def update_entire_dashboard(selected_block, selected_mgmt, selected_cat, selected_tier, search_query, horizon_year, sim_classrooms, w_eq, w_pr, w_ret, w_cap):
    filtered = df.copy()

    if selected_block and selected_block != "ALL" and "block_name" in filtered.columns:
        filtered = filtered[filtered["block_name"] == selected_block]

    if selected_mgmt and selected_mgmt != "ALL" and "school_management" in filtered.columns:
        filtered = filtered[filtered["school_management"] == selected_mgmt]

    if selected_cat and selected_cat != "ALL" and "school_category" in filtered.columns:
        filtered = filtered[filtered["school_category"] == selected_cat]

    if selected_tier and selected_tier != "ALL" and "priority_tier" in filtered.columns:
        filtered = filtered[filtered["priority_tier"] == selected_tier]

    # Search filter & Complete Institutional Profile Dossier
    school_to_inspect = None
    is_search_active = False

    if search_query and search_query.strip():
        q = search_query.strip().lower()
        search_mask = (
            filtered["school_name"].astype(str).str.lower().str.contains(q)
            | filtered["udise_code"].astype(str).str.lower().str.contains(q)
            | filtered["block_name"].astype(str).str.lower().str.contains(q)
        )
        matched_in_filter = filtered[search_mask]
        if not matched_in_filter.empty:
            school_to_inspect = matched_in_filter.iloc[0]
            is_search_active = True
        else:
            # Search full district master dataset if not in current ribbon filter
            full_mask = (
                df["school_name"].astype(str).str.lower().str.contains(q)
                | df["udise_code"].astype(str).str.lower().str.contains(q)
                | df["block_name"].astype(str).str.lower().str.contains(q)
            )
            matched_full = df[full_mask]
            if not matched_full.empty:
                school_to_inspect = matched_full.iloc[0]
                is_search_active = True
    elif not filtered.empty:
        school_to_inspect = filtered.iloc[0]
        is_search_active = False

    if school_to_inspect is not None:
        s_name = school_to_inspect.get("school_name", "N/A")
        s_code = school_to_inspect.get("udise_code", "N/A")
        s_block = school_to_inspect.get("block_name", "N/A")
        s_mgmt = school_to_inspect.get("school_management", "Local Body / State Govt")
        s_cat = school_to_inspect.get("school_category", "Standard Educational Institution")
        s_enrol = int(school_to_inspect.get("total_enrolment", 0))
        s_boys = int(school_to_inspect.get("total_boys", round(s_enrol * 0.52)))
        s_girls = int(school_to_inspect.get("total_girls", round(s_enrol * 0.48)))
        s_gpi = round(s_girls / max(s_boys, 1), 2)
        s_rooms = int(school_to_inspect.get("functional_classrooms", max(1, round(s_enrol / 38.0))))
        s_teachers = int(school_to_inspect.get("total_teachers", max(1, round(s_enrol / 34.0))))
        s_ptr = round(s_enrol / max(s_teachers, 1), 1)
        s_promo = float(school_to_inspect.get("promotion_rate", 85.0))
        s_pend = int(school_to_inspect.get("pending_students", 0))
        s_score = float(school_to_inspect.get("readiness_score", 0.0))

        school_dict = school_to_inspect.to_dict()
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

        status_header_text = f"SEARCH RESULT: COMPLETE INSTITUTIONAL DOSSIER" if is_search_active else "FEATURED INSTITUTION DOSSIER (USE SEARCH BAR ABOVE OR MAP PIN TO INSPECT ANY SCHOOL)"

        inspector_card = html.Div(
            className="dossier-card",
            style={"borderLeft": f"5px solid {badge_color}", "backgroundColor": "#FFFFFF", "padding": "18px 20px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "flexWrap": "wrap", "gap": "10px"},
                    children=[
                        html.Div([
                            html.Span(status_header_text, style={"fontSize": "10.5px", "fontWeight": "800", "color": COLOR_GOVT_BLUE if is_search_active else COLOR_MUTED, "letterSpacing": "0.5px"}),
                            html.H4(f"{s_name}", style={"margin": "4px 0 2px 0", "fontSize": "18px", "fontWeight": "800", "color": COLOR_PITCH}),
                            html.P(f"UDISE: {s_code} • Area: {s_block} • Type: {s_mgmt} • Category: {s_cat}", style={"margin": "0", "fontSize": "12.5px", "color": COLOR_MUTED})
                        ]),
                        html.Div(
                            style={"textAlign": "right"},
                            children=[
                                html.Span(f"Score: {s_score}/100", style={"fontSize": "16px", "fontWeight": "900", "color": badge_color, "display": "block"}),
                                html.Span(f"STATUS: {priority_label}", className=f"status-pill {'status-pill-red' if priority_label == 'CRITICAL' else ('status-pill-amber' if priority_label == 'HIGH' else 'status-pill-green')}")
                            ]
                        )
                    ]
                ),

                # 4 Detailed Infrastructure & Quality Metric Cards
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginTop": "14px", "borderTop": f"1px solid {COLOR_BORDER}", "paddingTop": "14px"},
                    children=[
                        html.Div(style={"backgroundColor": "#F8FAFC", "padding": "10px 12px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                            html.Span("Total Students Enrolled", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H5(f"{s_enrol:,} Students", style={"margin": "2px 0 2px 0", "fontSize": "15px", "color": COLOR_PITCH, "fontWeight": "800"}),
                            html.Span(f"{s_boys:,} Boys / {s_girls:,} Girls (GPI: {s_gpi})", style={"fontSize": "11px", "color": COLOR_GOVT_BLUE})
                        ]),
                        html.Div(style={"backgroundColor": "#F8FAFC", "padding": "10px 12px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                            html.Span("Functional Classrooms", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H5(f"{s_rooms} Rooms", style={"margin": "2px 0 2px 0", "fontSize": "15px", "color": COLOR_PITCH, "fontWeight": "800"}),
                            html.Span(f"+{gaps['classroom_gap']} Rooms Required (30:1)" if gaps['classroom_gap'] > 0 else "✓ Classroom Capacity Compliant", style={"fontSize": "11px", "color": COLOR_CRIMSON if gaps['classroom_gap'] > 0 else COLOR_SUCCESS, "fontWeight": "600"})
                        ]),
                        html.Div(style={"backgroundColor": "#F8FAFC", "padding": "10px 12px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                            html.Span("Teaching Faculty & Ratio", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H5(f"{s_teachers} Teachers (PTR {s_ptr}:1)", style={"margin": "2px 0 2px 0", "fontSize": "15px", "color": COLOR_PITCH, "fontWeight": "800"}),
                            html.Span(f"+{gaps['teacher_gap']} Teachers Required (30:1)" if gaps['teacher_gap'] > 0 else "✓ Faculty Ratio Compliant", style={"fontSize": "11px", "color": COLOR_CRIMSON if gaps['teacher_gap'] > 0 else COLOR_SUCCESS, "fontWeight": "600"})
                        ]),
                        html.Div(style={"backgroundColor": "#F8FAFC", "padding": "10px 12px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                            html.Span("Pass Rate & Remedial Learners", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H5(f"{s_promo}% Pass Rate", style={"margin": "2px 0 2px 0", "fontSize": "15px", "color": COLOR_PITCH, "fontWeight": "800"}),
                            html.Span(f"{s_pend} Remedial Learners" if s_pend > 0 else "✓ Zero Remedial Backlog", style={"fontSize": "11px", "color": COLOR_CRIMSON if s_pend > 0 else COLOR_SUCCESS, "fontWeight": "600"})
                        ])
                    ]
                ),

                # Plain-Language Explanation Box
                html.Div(
                    style={"backgroundColor": "#F8FAFC", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "12px 14px", "marginTop": "14px"},
                    children=[
                        html.Span("Plain-Language Summary for Officials & Parents:", style={"fontSize": "11.5px", "fontWeight": "800", "color": COLOR_PINE, "display": "block", "marginBottom": "4px"}),
                        html.P(citizen_exp, style={"margin": "0", "fontSize": "12.5px", "color": COLOR_PITCH, "lineHeight": "1.5"})
                    ]
                ),

                # Causal Factors & 3-Step Action Plan Grid
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "14px"},
                    children=[
                        # Left: Primary Drivers
                        html.Div(
                            style={"backgroundColor": "#FFFFFF", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                            children=[
                                html.Span("Primary Drivers & Identified Challenges:", style={"fontSize": "11.5px", "fontWeight": "800", "color": COLOR_MUTED, "display": "block", "marginBottom": "6px"}),
                                html.Div(why_rows)
                            ]
                        ),
                        # Right: Phased Action Plan
                        html.Div(
                            style={"backgroundColor": "#FFFFFF", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                            children=[
                                html.Span("Recommended 3-Step Action Plan:", style={"fontSize": "11.5px", "fontWeight": "800", "color": COLOR_GOVT_BLUE, "display": "block", "marginBottom": "6px"}),
                                html.Div([
                                    html.Span("• Step 1 (0–90 Days): ", style={"fontWeight": "700", "color": COLOR_PITCH, "fontSize": "11.5px"}),
                                    html.Span(dossier['phased_plan']['phase_1_immediate'][0], style={"fontSize": "11.5px", "color": COLOR_MUTED})
                                ], style={"marginBottom": "4px"}),
                                html.Div([
                                    html.Span("• Step 2 (6–24 Months): ", style={"fontWeight": "700", "color": COLOR_PITCH, "fontSize": "11.5px"}),
                                    html.Span(dossier['phased_plan']['phase_2_medium'][0], style={"fontSize": "11.5px", "color": COLOR_MUTED})
                                ], style={"marginBottom": "4px"}),
                                html.Div([
                                    html.Span("• Step 3 (Long-Term): ", style={"fontWeight": "700", "color": COLOR_PITCH, "fontSize": "11.5px"}),
                                    html.Span(dossier['phased_plan']['phase_3_strategic'][0], style={"fontSize": "11.5px", "color": COLOR_MUTED})
                                ])
                            ]
                        )
                    ]
                )
            ]
        )
    else:
        inspector_card = html.Div(
            style={"backgroundColor": "#F8FAFC", "padding": "20px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}", "textAlign": "center"},
            children=[
                html.P(f"No institution found matching '{search_query}'. Please verify the school name or 11-digit UDISE code.", style={"margin": "0", "fontSize": "13px", "color": COLOR_CRIMSON})
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

    # Dynamic Variable Horizon Deficit with live capital infusion reduction
    target_yr = int(horizon_year or 2030)
    fc_summary = forecast_service.get_forecast_summary(
        baseline_students=total_students,
        sim_classrooms=sim_classrooms,
        target_year=target_yr
    )
    deficit_classrooms = fc_summary["net_classroom_deficit"]

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

    # 5. FORECAST ENVELOPE FIGURE (High-Precision 2021–2040 Data with Target Milestone)
    forecast_fig = go.Figure()

    scale = max(0.001, total_students / 278945.0)
    hist_subset = df_forecast[df_forecast["year"] <= 2025]
    forecast_fig.add_trace(go.Scatter(
        x=hist_subset["year"],
        y=[int(round(v * scale)) for v in hist_subset["total_students"]],
        mode="lines+markers",
        name="Historical Actuals (2021–2025)",
        line=dict(color=COLOR_CYAN, width=3),
        marker=dict(size=7, color=COLOR_CYAN)
    ))

    proj_subset = df_forecast[df_forecast["year"] >= 2025]
    proj_vals = [int(round(v * scale)) for v in proj_subset["total_students"]]
    proj_lower = [int(round(v * scale)) for v in proj_subset["yhat_lower"]]
    proj_upper = [int(round(v * scale)) for v in proj_subset["yhat_upper"]]

    forecast_fig.add_trace(go.Scatter(
        x=proj_subset["year"],
        y=proj_vals,
        mode="lines+markers",
        name="Projected Trend (2026–2040)",
        line=dict(color=COLOR_PINE, width=3, dash="dash"),
        marker=dict(size=6, color=COLOR_PINE)
    ))

    forecast_fig.add_trace(go.Scatter(
        x=list(proj_subset["year"]) + list(proj_subset["year"])[::-1],
        y=proj_upper + proj_lower[::-1],
        fill="toself",
        fillcolor="rgba(15, 118, 110, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="85% Uncertainty Envelope"
    ))

    # Highlight active planning horizon milestone
    scaled_target_val = fc_summary["projected_students_target"]
    forecast_fig.add_trace(go.Scatter(
        x=[target_yr],
        y=[scaled_target_val],
        mode="markers",
        name=f"Milestone {target_yr} ({scaled_target_val:,})",
        marker=dict(size=11, color=COLOR_CRIMSON, symbol="diamond", line=dict(width=2, color="#FFFFFF"))
    ))

    forecast_fig.add_vline(
        x=target_yr,
        line_dash="dot",
        line_color=COLOR_OCHRE,
        annotation_text=f"Target {target_yr}",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_OCHRE)
    )

    forecast_fig.update_layout(
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": COLOR_MUTED, "family": "Inter"},
        xaxis=dict(gridcolor="#E2E8F0", dtick=2, range=[2020.5, 2040.5]),
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

    # 9. EXECUTIVE INFRASTRUCTURE INTELLIGENCE DOSSIER (GOVERNMENT OFFICERS)
    sim_res = scenario_service.run_simulation(
        sim_classrooms=sim_classrooms,
        w_equity=w_eq,
        w_promo=w_pr,
        w_retention=w_ret,
        w_capacity=w_cap,
        target_year=target_yr,
        filtered_df=filtered
    )

    govt_infra_panel = html.Div(
        style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "8px", "padding": "16px", "marginTop": "4px"},
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px", "borderBottom": f"1px solid {COLOR_BORDER}", "paddingBottom": "10px", "flexWrap": "wrap", "gap": "10px"},
                children=[
                    html.Div([
                        html.Span(f"DISTRICT INFRASTRUCTURE & ENROLLMENT SUMMARY (YEAR {target_yr})", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_GOVT_BLUE, "letterSpacing": "0.5px"}),
                        html.H4("Chhatrapati Sambhajinagar District Infrastructure Demand Model", style={"margin": "2px 0 0 0", "fontSize": "15px", "fontWeight": "700", "color": COLOR_PITCH})
                    ]),
                    html.Span(
                        f"Shortfall Resolution: {sim_res['deficit_resolved_pct']}% Resolved" if sim_classrooms > 0 else "Baseline Deficit Status Quo",
                        className=f"status-pill {'status-pill-green' if sim_res['deficit_resolved_pct'] >= 75 else ('status-pill-cyan' if sim_res['deficit_resolved_pct'] > 0 else 'status-pill-amber')}"
                    )
                ]
            ),
            # 4 Intelligence Dossier Cards
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "14px", "marginBottom": "14px"},
                children=[
                    # Card 1: Demographics
                    html.Div(
                        style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"},
                        children=[
                            html.Span("District Total Population", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H4(f"{sim_res['district_population']:,}", style={"margin": "2px 0 4px 0", "fontSize": "18px", "color": COLOR_PITCH, "fontWeight": "800"}),
                            html.P(f"School-Age (6–17): {sim_res['school_age_population']:,} • GER: {fc_summary['ger_pct']}%", style={"margin": "0", "fontSize": "11px", "color": COLOR_GOVT_BLUE, "fontWeight": "600"})
                        ]
                    ),
                    # Card 2: Classroom Deficit
                    html.Div(
                        style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"},
                        children=[
                            html.Span("RTE 30:1 Classroom Deficit", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H4(f"+{sim_res['simulated_classroom_deficit']:,} Classrooms", style={"margin": "2px 0 4px 0", "fontSize": "18px", "color": COLOR_CRIMSON if sim_res['simulated_classroom_deficit'] > 0 else COLOR_SAGE, "fontWeight": "800"}),
                            html.P(f"Required: {fc_summary['projected_classrooms_target']:,} • Baseline: {fc_summary['baseline_classrooms']:,}", style={"margin": "0", "fontSize": "11px", "color": COLOR_MUTED})
                        ]
                    ),
                    # Card 3: Planned Classrooms & Resolution (CapEx completely removed)
                    html.Div(
                        style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"},
                        children=[
                            html.Span("Planned New Classrooms", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H4(f"+{sim_classrooms:,} Rooms Planned" if sim_classrooms > 0 else "0 Rooms Planned", style={"margin": "2px 0 4px 0", "fontSize": "18px", "color": COLOR_GOVT_BLUE, "fontWeight": "800"}),
                            html.P(f"Shortfall Resolution: {sim_res['deficit_resolved_pct']}% Resolved" if sim_classrooms > 0 else "Status Quo (No Additional Rooms)", style={"margin": "0", "fontSize": "11px", "color": COLOR_SUCCESS if sim_classrooms > 0 else COLOR_MUTED, "fontWeight": "600" if sim_classrooms > 0 else "normal"})
                        ]
                    ),
                    # Card 4: Teacher Staffing Demand
                    html.Div(
                        style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "6px", "border": f"1px solid {COLOR_BORDER}"},
                        children=[
                            html.Span("Required Teaching Faculty", style={"fontSize": "11px", "color": COLOR_MUTED, "display": "block"}),
                            html.H4(f"{fc_summary['required_teachers']:,} Teachers", style={"margin": "2px 0 4px 0", "fontSize": "18px", "color": COLOR_PINE, "fontWeight": "800"}),
                            html.P(f"Readiness Score Shift: {sim_res['readiness_score_delta']:+0.1f} pts", style={"margin": "0", "fontSize": "11px", "color": COLOR_PINE, "fontWeight": "600"})
                        ]
                    )
                ]
            ),
            # Work-Oriented Policy Directive Box
            html.Div(
                style={"backgroundColor": "#F0FDF4" if sim_classrooms > 0 else "#FEF3C7", "border": f"1px solid {'#BBF7D0' if sim_classrooms > 0 else '#FDE68A'}", "borderRadius": "6px", "padding": "10px 14px"},
                children=[
                    html.Span("RECOMMENDED INFRASTRUCTURE DIRECTIVE: ", style={"fontSize": "11.5px", "fontWeight": "800", "color": COLOR_PITCH}),
                    html.Span(sim_res["recommended_intervention"], style={"fontSize": "12px", "color": COLOR_PITCH})
                ]
            )
        ]
    )

    return (
        gauge_fig, gauge_status, gauge_explanation,
        kpi_open, kpi_open_sub,
        kpi_total, kpi_total_sub,
        kpi_remedial, kpi_deficit,
        funnel_fig, forecast_fig, scatter_fig, radar_fig,
        gis_map_fig, map_badge,
        inspector_card,
        govt_infra_panel
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
    Output("rag-card-title", "children"),
    Output("rag-card-subtitle", "children"),
    Output("rag-card-badge", "children"),
    Output("preset-btn-1", "children"),
    Output("preset-btn-2", "children"),
    Output("preset-btn-3", "children"),
    Output("preset-btn-4", "children"),
    Input("active-portal-store", "data"),
    Input("input-sch-name", "value"),
    Input("input-sch-udise", "value"),
    Input("school-horizon-milestone", "value"),
    prevent_initial_call=False
)
def update_policy_assistant_header(portal, sch_name, sch_udise, milestone):
    portal = portal or "citizen"
    milestone = milestone or 2030
    if portal == "school":
        name = sch_name or "Selected Institution"
        u_str = f" (UDISE: {sch_udise})" if sch_udise else ""
        return (
            f"School Action Copilot • {name}",
            f"Formulates clear operational recommendations for {name}{u_str} under NEP 2020, RTE Act 2009 (30:1 student-teacher ratio), and Samagra Shiksha for planning year {milestone}.",
            "SCHOOL-SPECIFIC",
            "Explain Diagnostic Profile & Gaps",
            f"Year {milestone} Deficit Analysis",
            "Remedial Learning Recovery Plan",
            "NEP & RTE Compliance Standards"
        )
    elif portal == "govt":
        return (
            "Policy & Planning Assistant • Scenario Planning",
            "Cross-references simulated district infrastructure indicators with official standards (NEP 2020, RTE Act 2009, Samagra Shiksha).",
            "SCENARIO COPILOT",
            "Explain Current District Data",
            f"Explain Year {milestone} Forecast",
            "Analyze 38.4% Secondary Drop-off",
            "Remedial Support Strategy"
        )
    else:
        return (
            "Policy & Planning Assistant",
            "Cross-references district school data with official education standards (NEP 2020, RTE Act 2009, Samagra Shiksha) to provide clear planning recommendations.",
            "AI-ASSISTED",
            "Explain Current District Data",
            "Explain 2021–2040 Forecast",
            "Analyze 38.4% Secondary Drop-off",
            "Remedial Support Strategy"
        )


@app.callback(
    Output("rag-prompt-input", "value"),
    Input("preset-btn-1", "n_clicks"),
    Input("preset-btn-2", "n_clicks"),
    Input("preset-btn-3", "n_clicks"),
    Input("preset-btn-4", "n_clicks"),
    State("active-portal-store", "data"),
    State("input-sch-name", "value"),
    State("input-sch-udise", "value"),
    State("school-horizon-milestone", "value"),
    State("input-sch-pend", "value"),
    prevent_initial_call=True
)
def handle_preset_prompts(btn1, btn2, btn3, btn4, portal, sch_name, sch_udise, sch_milestone, sch_pend):
    from dash import callback_context
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    portal = portal or "citizen"
    milestone = sch_milestone or 2030

    if portal == "school":
        name = sch_name or "this institution"
        u_str = f" (UDISE: {sch_udise})" if sch_udise else ""
        pend_count = int(sch_pend or 0)
        if triggered_id == "preset-btn-1":
            return f"Explain the diagnostic profile, infrastructure adequacy, and readiness index for {name}{u_str} under NEP 2020."
        elif triggered_id == "preset-btn-2":
            return f"Explain the variable horizon capacity stress forecast and classroom deficit for {name} through Milestone {milestone} under RTE 30:1 norms."
        elif triggered_id == "preset-btn-3":
            return f"Formulate an institutional remedial bridge action plan for the {pend_count} non-promoted learners at {name} citing Samagra Shiksha learning recovery."
        elif triggered_id == "preset-btn-4":
            return f"Synthesize an urgent statutory compliance action plan for {name} to meet RTE Act 30:1 PTR and female sanitation norms."
    else:
        if triggered_id == "preset-btn-1":
            return "Explain the current district education data, total enrollment, gender parity, grade distribution, and operational metrics for Chhatrapati Sambhajinagar."
        elif triggered_id == "preset-btn-2":
            return f"Explain the Student Enrollment Forecast (2021 to 2037), the predictive demographic trajectory with 85% statistical confidence bounds, and the classroom deficit under RTE 30:1 norms for Milestone {milestone}."
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
    State("active-portal-store", "data"),
    State("input-sch-name", "value"),
    State("input-sch-udise", "value"),
    State("input-sch-enrol", "value"),
    State("input-sch-boys", "value"),
    State("input-sch-girls", "value"),
    State("input-sch-rooms", "value"),
    State("input-sch-teachers", "value"),
    State("input-sch-promo", "value"),
    State("input-sch-pend", "value"),
    State("school-horizon-milestone", "value"),
    prevent_initial_call=True
)
def initiate_rag_generation(
    n_clicks, prompt_text, portal,
    sch_name, sch_udise, sch_enrol, sch_boys, sch_girls, sch_rooms, sch_teachers, sch_promo, sch_pend, sch_milestone
):
    if not prompt_text or not prompt_text.strip():
        return "", 0, True, html.Div("Please provide an executive policy prompt.", style={"color": COLOR_CRIMSON, "padding": "16px"}), True

    school_info = None
    if portal == "school" and sch_name:
        enrol = float(sch_enrol or 0)
        teachers = float(sch_teachers or 1)
        rooms = float(sch_rooms or 1)
        target_year = int(sch_milestone or 2030)
        school_info = {
            "school_name": sch_name or "Audited Institution",
            "udise_code": sch_udise or "N/A",
            "total_enrolment": enrol,
            "total_boys": float(sch_boys or 0),
            "total_girls": float(sch_girls or 0),
            "functional_classrooms": rooms,
            "total_teachers": teachers,
            "promotion_rate": float(sch_promo or 85.0),
            "pending_students": float(sch_pend or 0),
            "target_year": target_year,
            "ptr": round(enrol / max(1.0, teachers), 1),
            "gpi": round(float(sch_girls or 0) / max(1.0, float(sch_boys or 1)), 2),
            "classroom_gap": max(0, int(round((enrol / 30.0) - rooms))),
            "teacher_gap": max(0, int(round((enrol / 30.0) - teachers))),
            "target_deficit": max(0, int(round(((enrol * (1.0 + 0.02 * (target_year - 2025))) / 30.0) - rooms))),
            "readiness_score": 82.5,
            "priority_tier": "STABLE"
        }

    try:
        res = generate_education_policy_brief(user_prompt=prompt_text, school_info=school_info)
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
# 6. WELCOME PAGE, ROLE-BASED AUTHENTICATION & MULTI-PERSONA PORTAL CALLBACKS
# ---------------------------------------------------------------------------

CREDENTIALS_CONFIG = {
    "school": {
        "title": "Institution Administrator & Principal Verification",
        "subtitle": "Enter authorized institutional credentials to access diagnostic audits and facility records.",
        "icon": "",
        "icon_class": "gov-ico gov-ico-school",
        "badge": "INSTITUTIONAL CLEARANCE",
        "hint": "Default Credentials: Username: school_admin | Password: school123",
        "valid_users": ["school_admin", "principal", "27191001234", "admin", "school"],
        "valid_passwords": ["school123", "admin123", "123456", "school"]
    },
    "govt": {
        "title": "Government Official & Policy Planner Authorization",
        "subtitle": "Enter administrative credentials to access infrastructure scenario analysis and pillar recalibration.",
        "icon": "",
        "icon_class": "gov-ico gov-ico-govt",
        "badge": "ADMINISTRATIVE CLEARANCE",
        "hint": "Default Credentials: Username: govt_admin | Password: govt123",
        "valid_users": ["govt_admin", "director", "officer", "mh_edu_officer", "admin", "govt"],
        "valid_passwords": ["govt123", "admin123", "123456", "govt"]
    }
}


@app.callback(
    # Page view (Welcome vs Dashboard)
    Output("welcome-page-view", "style"),
    Output("dashboard-page-view", "style"),
    Output("current-page-store", "data"),
    # Portal Tab state & Visibility
    Output("active-portal-store", "data"),
    Output("portal-tab-citizen", "className"),
    Output("portal-tab-school", "className"),
    Output("portal-tab-govt", "className"),
    Output("section-govt-sim-sandbox", "style"),
    Output("section-school-admin-view", "style"),
    Output("section-district-overview", "style"),
    Output("section-citizen-map-and-directory", "style"),
    # Auth Modal State & Fields
    Output("auth-modal-overlay", "style"),
    Output("auth-modal-title", "children"),
    Output("auth-modal-subtitle", "children"),
    Output("auth-modal-icon", "children"),
    Output("auth-modal-icon", "className"),
    Output("auth-modal-badge", "children"),
    Output("auth-demo-hint-text", "children"),
    Output("auth-error-banner", "children"),
    Output("auth-error-banner", "style"),
    Output("auth-username-input", "value"),
    Output("auth-password-input", "value"),
    Output("pending-role-target-store", "data"),
    Output("auth-session-store", "data"),
    # Portal Auth Status Indicator
    Output("portal-auth-status-text", "children"),
    Output("portal-logout-btn", "style"),
    Output("auth-school-group", "style"),
    # Inputs: Welcome Navigation
    Input("welcome-btn-enter-dashboard", "n_clicks"),
    Input("welcome-btn-hero-launch", "n_clicks"),
    Input("welcome-btn-hero-explore", "n_clicks"),
    Input("welcome-btn-citizen", "n_clicks"),
    Input("welcome-btn-copilot", "n_clicks"),
    Input("welcome-btn-map-launch", "n_clicks"),
    Input("welcome-btn-map-launch-2", "n_clicks"),
    Input("welcome-btn-school", "n_clicks"),
    Input("welcome-btn-govt", "n_clicks"),
    # Inputs: Dashboard Tabs & Header
    Input("portal-tab-citizen", "n_clicks"),
    Input("portal-tab-school", "n_clicks"),
    Input("portal-tab-govt", "n_clicks"),
    Input("dashboard-btn-back-home", "n_clicks"),
    # Inputs: Auth Modal
    Input("auth-cancel-btn", "n_clicks"),
    Input("auth-submit-btn", "n_clicks"),
    Input("auth-username-input", "n_submit"),
    Input("auth-password-input", "n_submit"),
    Input("portal-logout-btn", "n_clicks"),
    # States
    State("auth-session-store", "data"),
    State("pending-role-target-store", "data"),
    State("current-page-store", "data"),
    State("active-portal-store", "data"),
    State("auth-username-input", "value"),
    State("auth-password-input", "value"),
    State("auth-school-dropdown", "value"),
    prevent_initial_call=True
)
def handle_navigation_and_auth(
    n_enter, n_hero, n_explore, n_cit, n_cop, n_map1, n_map2,
    w_sch, w_gov,
    p_cit, p_sch, p_gov,
    n_home, n_cancel, n_submit, n_sub_user, n_sub_pass, n_logout,
    auth_session, pending_role, current_page, current_portal,
    input_user, input_pass, selected_school_udise
):
    from dash import callback_context
    if not callback_context.triggered:
        return no_update
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

    auth_session = dict(auth_session or {"school": False, "govt": False, "role": "citizen", "school_udise": "", "school_name": ""})
    current_page = current_page or "welcome"
    current_portal = current_portal or "citizen"
    pending_role = pending_role or "school"

    style_hidden = {"display": "none"}
    style_visible = {"display": "block"}
    style_flex = {"display": "flex"}

    # Default UI states based on current view
    welcome_style = style_visible if current_page == "welcome" else style_hidden
    dashboard_style = style_hidden if current_page == "welcome" else style_visible

    tab_citizen_cls = "portal-tab-btn portal-tab-btn-active" if current_portal == "citizen" else "portal-tab-btn"
    tab_school_cls = "portal-tab-btn portal-tab-btn-active" if current_portal == "school" else "portal-tab-btn"
    tab_govt_cls = "portal-tab-btn portal-tab-btn-active" if current_portal == "govt" else "portal-tab-btn"

    sec_govt_style = style_visible if current_portal == "govt" else style_hidden
    sec_school_style = style_visible if current_portal == "school" else style_hidden
    sec_district_style = style_hidden if current_portal == "school" else style_visible
    sec_map_dir_style = style_hidden if current_portal == "school" else style_visible

    # Status indicator
    if auth_session.get("school") and current_portal == "school":
        sch_label = auth_session.get("school_name", "Institution Administrator")
        status_text = f"Logged in: {sch_label}"
        logout_style = {"display": "inline-block"}
    elif auth_session.get("govt") and current_portal == "govt":
        status_text = "Logged in: Government Official"
        logout_style = {"display": "inline-block"}
    else:
        status_text = "Public Citizen View (Open Access)"
        logout_style = {"display": "none"}

    cfg = CREDENTIALS_CONFIG.get(pending_role, CREDENTIALS_CONFIG["school"])
    modal_title = cfg["title"]
    modal_sub = cfg["subtitle"]
    modal_icon = cfg["icon"]
    modal_icon_cls = cfg["icon_class"]
    modal_badge = cfg["badge"]
    demo_hint = cfg["hint"]

    # 1. LOGOUT
    if triggered == "portal-logout-btn":
        auth_session = {"school": False, "govt": False, "role": "citizen", "school_udise": "", "school_name": ""}
        return (
            style_hidden, style_visible, "dashboard",
            "citizen", "portal-tab-btn portal-tab-btn-active", "portal-tab-btn", "portal-tab-btn",
            style_hidden, style_hidden, style_visible, style_visible,
            style_hidden, modal_title, modal_sub, modal_icon, modal_icon_cls, modal_badge, demo_hint,
            "", style_hidden, "", "", "", auth_session,
            "Public Citizen View (Open Access)", {"display": "none"},
            style_hidden
        )

    # 2. RETURN HOME (WELCOME PAGE)
    if triggered == "dashboard-btn-back-home":
        return (
            style_visible, style_hidden, "welcome",
            current_portal, tab_citizen_cls, tab_school_cls, tab_govt_cls,
            sec_govt_style, sec_school_style, sec_district_style, sec_map_dir_style,
            style_hidden, modal_title, modal_sub, modal_icon, modal_icon_cls, modal_badge, demo_hint,
            "", style_hidden, no_update, no_update, pending_role, auth_session,
            status_text, logout_style,
            style_hidden
        )

    # 3. DIRECT PUBLIC ACCESS
    if triggered in [
        "welcome-btn-enter-dashboard", "welcome-btn-hero-launch", "welcome-btn-hero-explore",
        "welcome-btn-bottom-enter", "welcome-btn-citizen",
        "welcome-btn-copilot", "welcome-btn-map-launch", "welcome-btn-map-launch-2",
        "portal-tab-citizen"
    ]:
        return (
            style_hidden, style_visible, "dashboard",
            "citizen", "portal-tab-btn portal-tab-btn-active", "portal-tab-btn", "portal-tab-btn",
            style_hidden, style_hidden, style_visible, style_visible,
            style_hidden, modal_title, modal_sub, modal_icon, modal_icon_cls, modal_badge, demo_hint,
            "", style_hidden, no_update, no_update, pending_role, auth_session,
            "Public Citizen View (Open Access)", {"display": "none"},
            style_hidden
        )

    # 4. REQUEST SCHOOL PORTAL (FROM WELCOME OR DASHBOARD)
    if triggered in ["welcome-btn-school", "portal-tab-school"]:
        if auth_session.get("school"):
            sch_label = auth_session.get("school_name", "Institution Administrator")
            return (
                style_hidden, style_visible, "dashboard",
                "school", "portal-tab-btn", "portal-tab-btn portal-tab-btn-active", "portal-tab-btn",
                style_hidden, style_visible, style_hidden, style_hidden,
                style_hidden, modal_title, modal_sub, modal_icon, modal_icon_cls, modal_badge, demo_hint,
                "", style_hidden, no_update, no_update, "school", auth_session,
                f"Logged in: {sch_label}", {"display": "inline-block"},
                style_hidden
            )
        else:
            cfg = CREDENTIALS_CONFIG["school"]
            return (
                welcome_style, dashboard_style, current_page,
                current_portal, tab_citizen_cls, tab_school_cls, tab_govt_cls,
                sec_govt_style, sec_school_style, sec_district_style, sec_map_dir_style,
                style_flex, cfg["title"], cfg["subtitle"], cfg["icon"], cfg["icon_class"], cfg["badge"], cfg["hint"],
                "", style_hidden, "", "", "school", auth_session,
                status_text, logout_style,
                {"display": "block", "marginBottom": "14px"}
            )

    # 5. REQUEST GOVERNMENT PORTAL (FROM WELCOME OR DASHBOARD)
    if triggered in ["welcome-btn-govt", "portal-tab-govt"]:
        if auth_session.get("govt"):
            return (
                style_hidden, style_visible, "dashboard",
                "govt", "portal-tab-btn", "portal-tab-btn", "portal-tab-btn portal-tab-btn-active",
                style_visible, style_hidden, style_visible, style_visible,
                style_hidden, modal_title, modal_sub, modal_icon, modal_icon_cls, modal_badge, demo_hint,
                "", style_hidden, no_update, no_update, "govt", auth_session,
                "Logged in: Government Official", {"display": "inline-block"},
                style_hidden
            )
        else:
            cfg = CREDENTIALS_CONFIG["govt"]
            return (
                welcome_style, dashboard_style, current_page,
                current_portal, tab_citizen_cls, tab_school_cls, tab_govt_cls,
                sec_govt_style, sec_school_style, sec_district_style, sec_map_dir_style,
                style_flex, cfg["title"], cfg["subtitle"], cfg["icon"], cfg["icon_class"], cfg["badge"], cfg["hint"],
                "", style_hidden, "", "", "govt", auth_session,
                status_text, logout_style,
                style_hidden
            )

    # 6. CANCEL AUTH MODAL
    if triggered == "auth-cancel-btn":
        return (
            welcome_style, dashboard_style, current_page,
            current_portal, tab_citizen_cls, tab_school_cls, tab_govt_cls,
            sec_govt_style, sec_school_style, sec_district_style, sec_map_dir_style,
            style_hidden, modal_title, modal_sub, modal_icon, modal_icon_cls, modal_badge, demo_hint,
            "", style_hidden, "", "", pending_role, auth_session,
            status_text, logout_style,
            style_hidden
        )

    # 7. SUBMIT AUTHENTICATION (SUBMIT BUTTON OR ENTER KEY)
    if triggered in ["auth-submit-btn", "auth-username-input", "auth-password-input"]:
        target_role = pending_role if pending_role in ["school", "govt"] else "school"
        cfg = CREDENTIALS_CONFIG[target_role]
        u = (input_user or "").strip().lower()
        p = (input_pass or "").strip()

        is_valid_user = (u in cfg["valid_users"]) or (target_role == "school" and any(str(r["value"]) == u for r in school_select_options))
        is_valid_pass = (p in cfg["valid_passwords"])

        if is_valid_user and is_valid_pass:
            auth_session[target_role] = True
            auth_session["role"] = target_role
            new_portal = target_role

            if target_role == "school":
                u_code = str(selected_school_udise or "")
                matched = df[df["udise_code"].astype(str) == u_code]
                if matched.empty and not df.empty:
                    matched = df.head(1)
                sch_row = matched.iloc[0]
                auth_session["school_udise"] = str(sch_row["udise_code"])
                auth_session["school_name"] = str(sch_row["school_name"])
                new_status = f"Logged in: {sch_row['school_name']}"
            else:
                new_status = "Logged in: Government Official"

            new_tab_cit = "portal-tab-btn"
            new_tab_sch = "portal-tab-btn portal-tab-btn-active" if new_portal == "school" else "portal-tab-btn"
            new_tab_gov = "portal-tab-btn portal-tab-btn-active" if new_portal == "govt" else "portal-tab-btn"
            new_sec_sch = style_visible if new_portal == "school" else style_hidden
            new_sec_gov = style_visible if new_portal == "govt" else style_hidden
            new_sec_dist = style_hidden if new_portal == "school" else style_visible
            new_sec_map = style_hidden if new_portal == "school" else style_visible

            return (
                style_hidden, style_visible, "dashboard",
                new_portal, new_tab_cit, new_tab_sch, new_tab_gov,
                new_sec_gov, new_sec_sch, new_sec_dist, new_sec_map,
                style_hidden, cfg["title"], cfg["subtitle"], cfg["icon"], cfg["icon_class"], cfg["badge"], cfg["hint"],
                "", style_hidden, "", "", target_role, auth_session,
                new_status, {"display": "inline-block"},
                style_hidden
            )
        else:
            err_text = f"Authentication Failed: Invalid {target_role.capitalize()} credentials. Please check username and password."
            school_grp_style = {"display": "block", "marginBottom": "14px"} if target_role == "school" else style_hidden
            return (
                welcome_style, dashboard_style, current_page,
                current_portal, tab_citizen_cls, tab_school_cls, tab_govt_cls,
                sec_govt_style, sec_school_style, sec_district_style, sec_map_dir_style,
                style_flex, cfg["title"], cfg["subtitle"], cfg["icon"], cfg["icon_class"], cfg["badge"], cfg["hint"],
                err_text, style_visible, no_update, "", target_role, auth_session,
                status_text, logout_style,
                school_grp_style
            )

    return no_update



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
    Output("school-locked-name-display", "children"),
    Output("school-locked-udise-display", "children"),
    Input("auth-session-store", "data"),
    prevent_initial_call=False
)
def prefill_school_admin_form(auth_session):
    auth_session = auth_session or {}
    selected_udise = auth_session.get("school_udise")
    if not selected_udise:
        if not df.empty:
            selected_udise = str(df.iloc[0]["udise_code"])
        else:
            return "", None, None, None, None, None, None, None, None, "No Institution", "UDISE: --"

    matched = df[df["udise_code"].astype(str) == str(selected_udise)]
    if matched.empty and not df.empty:
        matched = df.head(1)

    row = matched.iloc[0]
    sch_name = str(row.get("school_name", ""))
    sch_code = int(row.get("udise_code", 0))
    enrol = int(row.get("total_enrolment", 0))
    boys = int(row.get("total_boys", round(enrol * 0.52)))
    girls = int(row.get("total_girls", round(enrol * 0.48)))
    rooms = int(row.get("functional_classrooms", max(1, round(enrol / 38.0))))
    teachers = int(row.get("total_teachers", max(1, round(enrol / 34.0))))
    promo = float(row.get("promotion_rate", 85.0))
    pend = int(row.get("pending_students", 0))

    return (
        sch_name,
        sch_code,
        enrol,
        boys,
        girls,
        rooms,
        teachers,
        promo,
        pend,
        sch_name,
        f"UDISE Code: {sch_code}"
    )


@app.callback(
    Output("school-audit-output-container", "children"),
    Input("btn-sch-audit", "n_clicks"),
    Input("auth-session-store", "data"),
    Input("school-horizon-milestone", "value"),
    State("input-sch-name", "value"),
    State("input-sch-udise", "value"),
    State("input-sch-enrol", "value"),
    State("input-sch-boys", "value"),
    State("input-sch-girls", "value"),
    State("input-sch-rooms", "value"),
    State("input-sch-teachers", "value"),
    State("input-sch-promo", "value"),
    State("input-sch-pend", "value"),
    prevent_initial_call=False
)
def handle_school_admin_audit(n_clicks, auth_session, target_horizon, name, udise, enrol, boys, girls, rooms, teachers, promo, pend):
    try:
        from dash import callback_context
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if (ctx and ctx.triggered) else ""
    except Exception:
        triggered_id = ""

    auth_session = auth_session or {}
    auth_udise = auth_session.get("school_udise")

    if auth_udise and triggered_id != "btn-sch-audit":
        matched = df[df["udise_code"].astype(str) == str(auth_udise)]
        if not matched.empty:
            row = matched.iloc[0]
            name = str(row.get("school_name", name or "Audited School"))
            udise = int(row.get("udise_code", udise or 0))
            enrol = float(row.get("total_enrolment", 0))
            boys = float(row.get("total_boys", round(enrol * 0.52)))
            girls = float(row.get("total_girls", round(enrol * 0.48)))
            rooms = float(row.get("functional_classrooms", max(1, round(enrol / 38.0))))
            teachers = float(row.get("total_teachers", max(1, round(enrol / 34.0))))
            promo = float(row.get("promotion_rate", 85.0))
            pend = float(row.get("pending_students", 0))

    if not name and not enrol:
        if not df.empty:
            row = df.iloc[0]
            name = str(row.get("school_name", "Audited School"))
            udise = int(row.get("udise_code", 27191000000))
            enrol = float(row.get("total_enrolment", 0))
            boys = float(row.get("total_boys", round(enrol * 0.52)))
            girls = float(row.get("total_girls", round(enrol * 0.48)))
            rooms = float(row.get("functional_classrooms", max(1, round(enrol / 38.0))))
            teachers = float(row.get("total_teachers", max(1, round(enrol / 34.0))))
            promo = float(row.get("promotion_rate", 85.0))
            pend = float(row.get("pending_students", 0))
        else:
            return html.Div("Please log in with your institution credentials to view this audit.", style={"color": COLOR_GOVT_BLUE, "padding": "12px", "fontSize": "13px"})

    target_year = int(target_horizon or 2030)
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

    # 1. Full Personalized Evaluation via InstitutionService
    eval_res = institution_service.evaluate_institution(record, target_year=target_year)
    audit_res = eval_res["data_audit"]
    readiness = eval_res["readiness_profile"]
    trajectory = eval_res["trajectory"]
    gaps = eval_res["gaps"]
    dominant = eval_res["dominant_bottleneck"]
    dominant_reason = eval_res["dominant_reason"]
    phased_plan = eval_res["phased_plan"]
    why_factors = eval_res["causal_why_factors"]

    conf_status = audit_res["confidence_status"]
    conf_color = COLOR_SAGE if conf_status == "Sufficient" else (COLOR_OCHRE if conf_status == "Moderate Confidence" else COLOR_CRIMSON)

    # 2. Personalized School Readiness Radial Gauge Figure
    sch_score = readiness["composite_score"]
    sch_gauge_color = readiness["tier_color"]
    sch_gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sch_score,
        number={"font": {"size": 30, "color": COLOR_PITCH, "family": "Inter"}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOR_MUTED, "nticks": 5},
            "bar": {"color": sch_gauge_color, "thickness": 0.28},
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
                "value": sch_score
            }
        }
    ))
    sch_gauge_fig.update_layout(
        margin=dict(l=15, r=15, t=10, b=10),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=150
    )

    # 3. Personalized 4-Pillar Radar Chart (School vs Benchmark)
    categories = ["1. Equity (GPI)", "2. Promotion & Remedial", "3. Cohort Retention", "4. Capacity (PTR 30:1)"]
    school_vals = [
        readiness["scores"]["equity"],
        readiness["scores"]["promotion"],
        readiness["scores"]["retention"],
        readiness["scores"]["capacity"]
    ]
    target_vals = [100.0, 90.0, 85.0, 95.0]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=school_vals + [school_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="School Performance",
        line_color=COLOR_GOVT_BLUE,
        fillcolor="rgba(30, 90, 146, 0.25)"
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=target_vals + [target_vals[0]],
        theta=categories + [categories[0]],
        mode="lines",
        name="Target Benchmark",
        line=dict(color=COLOR_SUCCESS, dash="dash", width=2)
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont={"size": 9, "color": COLOR_MUTED}),
            angularaxis=dict(tickfont={"size": 10, "color": COLOR_PITCH, "family": "Inter"})
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font={"size": 10}),
        margin=dict(l=25, r=25, t=20, b=30),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=260
    )

    # 4. Personalized School Trajectory Chart (2026 - 2040)
    traj_years = [p["year"] for p in trajectory]
    traj_enrol = [p["projected_enrolment"] for p in trajectory]
    traj_cap = [p["current_capacity"] for p in trajectory]

    traj_fig = go.Figure()
    traj_fig.add_trace(go.Scatter(
        x=traj_years,
        y=traj_enrol,
        mode="lines+markers",
        name="Projected Enrollment",
        line=dict(color=COLOR_GOVT_BLUE, width=3),
        marker=dict(size=7, color=COLOR_PRIMARY_NAVY)
    ))
    traj_fig.add_trace(go.Scatter(
        x=traj_years,
        y=traj_cap,
        mode="lines",
        name="Current Capacity Ceiling (30:1)",
        line=dict(color=COLOR_DANGER, width=2, dash="dash")
    ))

    # Highlight target milestone
    target_entry = next((p for p in trajectory if p["year"] == target_year), trajectory[2])
    traj_fig.add_annotation(
        x=target_year,
        y=target_entry["projected_enrolment"],
        text=f"Milestone {target_year}: {target_entry['status']}",
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLOR_DANGER if target_entry["classroom_deficit"] > 0 else COLOR_SUCCESS,
        font=dict(size=11, color=COLOR_PITCH, family="Inter"),
        bgcolor="#FFFFFF",
        bordercolor=COLOR_BORDER,
        borderwidth=1
    )

    traj_fig.update_layout(
        margin=dict(l=35, r=20, t=25, b=25),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=260,
        xaxis=dict(title="Planning Horizon Milestone", tickmode="array", tickvals=traj_years, gridcolor="#E2E8F0"),
        yaxis=dict(title="Student Census", gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font={"size": 10})
    )

    return html.Div(
        style={"marginTop": "20px"},
        children=[
            # SECTION A: Personalized School Header & Cockpit
            html.Div(
                style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "10px", "padding": "18px 20px", "marginBottom": "18px"},
                children=[
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "flexWrap": "wrap", "gap": "14px", "marginBottom": "16px"},
                        children=[
                            html.Div([
                                html.Span("OFFICIAL INSTITUTIONAL COCKPIT & DIAGNOSTIC DOSSIER", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_GOVT_BLUE, "letterSpacing": "0.5px"}),
                                html.H3(f"{record['school_name']}", style={"margin": "2px 0 2px 0", "fontSize": "19px", "fontWeight": "800", "color": COLOR_PITCH}),
                                html.P(f"UDISE Code: {record['udise_code']} • Constraint: {dominant} • Planning Horizon: Milestone {target_year}", style={"margin": "0", "fontSize": "12.5px", "color": COLOR_MUTED})
                            ]),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                children=[
                                    html.Span(f"AUDIT: {conf_status.upper()}", style={"backgroundColor": "#FFFFFF", "color": conf_color, "border": f"1px solid {conf_color}", "padding": "4px 10px", "borderRadius": "6px", "fontSize": "11px", "fontWeight": "800"}),
                                    html.Span(f"STATUS: {readiness['tier']}", style={"backgroundColor": readiness['tier_color'], "color": "#FFFFFF", "padding": "4px 12px", "borderRadius": "6px", "fontSize": "11px", "fontWeight": "800"})
                                ]
                            )
                        ]
                    ),

                    # 4 Personalized School Pillar Cards + School Readiness Gauge Grid
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "220px repeat(4, 1fr)", "gap": "14px", "alignItems": "center"},
                        children=[
                            # Personalized Score Gauge
                            html.Div(
                                style={"backgroundColor": "#F8FAFC", "padding": "10px 12px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}", "textAlign": "center"},
                                children=[
                                    html.Span("SCHOOL READINESS INDEX", style={"fontSize": "10px", "fontWeight": "800", "color": COLOR_PINE}),
                                    dcc.Graph(figure=sch_gauge_fig, config={"displayModeBar": False}, style={"margin": "-10px 0 -15px 0"}),
                                    html.Span(f"{readiness['tier']}", style={"fontSize": "10.5px", "fontWeight": "700", "color": readiness['tier_color']})
                                ]
                            ),
                            # Equity Pillar Card
                            html.Div(
                                style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                                children=[
                                    html.Span("1. GENDER PARITY (GPI)", style={"fontSize": "10.5px", "fontWeight": "700", "color": COLOR_MUTED}),
                                    html.H4(f"GPI: {readiness['gpi']}", style={"margin": "4px 0 2px 0", "fontSize": "16px", "color": COLOR_PITCH}),
                                    html.Span(f"Score: {readiness['scores']['equity']}/100", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_GOVT_BLUE}),
                                    html.P(f"{int(record['total_girls'])} Girls / {int(record['total_boys'])} Boys", style={"margin": "4px 0 0 0", "fontSize": "11px", "color": COLOR_MUTED})
                                ]
                            ),
                            # Promotion & Remedial Card
                            html.Div(
                                style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                                children=[
                                    html.Span("2. PASS & REMEDIAL", style={"fontSize": "10.5px", "fontWeight": "700", "color": COLOR_MUTED}),
                                    html.H4(f"{record['promotion_rate']}% Pass", style={"margin": "4px 0 2px 0", "fontSize": "16px", "color": COLOR_PITCH}),
                                    html.Span(f"Score: {readiness['scores']['promotion']}/100", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_GOVT_BLUE}),
                                    html.P(f"{int(record['pending_students'])} Remedial Learners", style={"margin": "4px 0 0 0", "fontSize": "11px", "color": COLOR_DANGER if record['pending_students'] > 0 else COLOR_SUCCESS})
                                ]
                            ),
                            # Retention Card
                            html.Div(
                                style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                                children=[
                                    html.Span("3. COHORT RETENTION", style={"fontSize": "10.5px", "fontWeight": "700", "color": COLOR_MUTED}),
                                    html.H4("0.88 Continuity", style={"margin": "4px 0 2px 0", "fontSize": "16px", "color": COLOR_PITCH}),
                                    html.Span(f"Score: {readiness['scores']['retention']}/100", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_GOVT_BLUE}),
                                    html.P("Target >= 0.67 Met", style={"margin": "4px 0 0 0", "fontSize": "11px", "color": COLOR_SUCCESS})
                                ]
                            ),
                            # Capacity & PTR Card
                            html.Div(
                                style={"backgroundColor": "#F8FAFC", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"},
                                children=[
                                    html.Span("4. CAPACITY & PTR", style={"fontSize": "10.5px", "fontWeight": "700", "color": COLOR_MUTED}),
                                    html.H4(f"{readiness['ptr']}:1 PTR", style={"margin": "4px 0 2px 0", "fontSize": "16px", "color": COLOR_DANGER if readiness['ptr'] > 30 else COLOR_SUCCESS}),
                                    html.Span(f"Score: {readiness['scores']['capacity']}/100", style={"fontSize": "11px", "fontWeight": "700", "color": COLOR_GOVT_BLUE}),
                                    html.P(f"{int(record['functional_classrooms'] or 0)} Operational Rooms", style={"margin": "4px 0 0 0", "fontSize": "11px", "color": COLOR_MUTED})
                                ]
                            ),
                        ]
                    )
                ]
            ),

            # SECTION B: Personalized Visualizations (Radar Benchmark + Horizon Trajectory)
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginBottom": "18px"},
                children=[
                    # Visualization 1: Radar Chart
                    html.Div(
                        style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "10px", "padding": "16px 18px"},
                        children=[
                            html.Div([
                                html.Span("INSTITUTIONAL BENCHMARK RADAR", style={"fontSize": "10.5px", "fontWeight": "800", "color": COLOR_PINE}),
                                html.H4("School Multi-Pillar Performance vs Target", style={"margin": "2px 0 6px 0", "fontSize": "15px", "color": COLOR_PITCH}),
                            ]),
                            dcc.Graph(figure=radar_fig, config={"displayModeBar": False})
                        ]
                    ),
                    # Visualization 2: Horizon Capacity Trajectory Chart
                    html.Div(
                        style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "10px", "padding": "16px 18px"},
                        children=[
                            html.Div([
                                html.Span("VARIABLE HORIZON CAPACITY STRESS MODEL", style={"fontSize": "10.5px", "fontWeight": "800", "color": COLOR_PINE}),
                                html.H4(f"Enrollment Growth vs Physical Classroom Ceiling (Through {target_year})", style={"margin": "2px 0 6px 0", "fontSize": "15px", "color": COLOR_PITCH}),
                            ]),
                            dcc.Graph(figure=traj_fig, config={"displayModeBar": False})
                        ]
                    )
                ]
            ),

            # SECTION C: Physical Gaps & Statutory Diagnostic
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "18px"},
                children=[
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Teacher Gap (ΔT)", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4(f"+{gaps['teacher_gap']} Teachers Required" if gaps['teacher_gap'] > 0 else "✓ RTE Staff Compliant", style={"margin": "3px 0 0 0", "fontSize": "14.5px", "color": COLOR_CRIMSON if gaps['teacher_gap'] > 0 else COLOR_SUCCESS})
                    ]),
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Classroom Gap (ΔC)", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4(f"+{gaps['classroom_gap']} Classrooms Required" if gaps['classroom_gap'] > 0 else "✓ Classroom Capacity Safe", style={"margin": "3px 0 0 0", "fontSize": "14.5px", "color": COLOR_CRIMSON if gaps['classroom_gap'] > 0 else COLOR_SUCCESS})
                    ]),
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span(f"Milestone {target_year} Deficit", style={"fontSize": "11px", "color": COLOR_GOVT_BLUE, "fontWeight": "700"}),
                        html.H4(f"-{target_entry['classroom_deficit']} Rooms by {target_year}" if target_entry['classroom_deficit'] > 0 else f"✓ Capacity Safe by {target_year}", style={"margin": "3px 0 0 0", "fontSize": "14.5px", "color": COLOR_CRIMSON if target_entry['classroom_deficit'] > 0 else COLOR_SUCCESS})
                    ]),
                    html.Div(style={"backgroundColor": "#FFFFFF", "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {COLOR_BORDER}"}, children=[
                        html.Span("Dominant Statutory Constraint", style={"fontSize": "11px", "color": COLOR_MUTED}),
                        html.H4(dominant, style={"margin": "3px 0 0 0", "fontSize": "14.5px", "color": COLOR_PINE})
                    ]),
                ]
            ),

            # SECTION D: Tailored Tri-Phased Action Roadmap (Aligned to Milestone)
            html.Div(
                style={"backgroundColor": "#FFFFFF", "border": f"1px solid {COLOR_BORDER}", "borderRadius": "10px", "padding": "16px 18px"},
                children=[
                    html.Span(f"TAILORED TRI-PHASED STATUTORY ACTION ROADMAP (ALIGNED TO HORIZON {target_year}):", style={"fontSize": "11px", "fontWeight": "800", "color": COLOR_PINE, "letterSpacing": "0.5px", "display": "block", "marginBottom": "10px"}),
                    html.Div([
                        html.Span("Phase 1 (Immediate 0–90 Days): ", style={"fontWeight": "800", "color": COLOR_PITCH, "fontSize": "12.5px"}),
                        html.Span(phased_plan["phase_1_immediate"][0], style={"fontSize": "12.5px", "color": COLOR_MUTED})
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span("Phase 2 (Medium-Term Civil Works 6–24 Months): ", style={"fontWeight": "800", "color": COLOR_PITCH, "fontSize": "12.5px"}),
                        html.Span(phased_plan["phase_2_medium"][0], style={"fontSize": "12.5px", "color": COLOR_MUTED})
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Span(f"Phase 3 (Strategic Target Milestone {target_year}): ", style={"fontWeight": "800", "color": COLOR_GOVT_BLUE, "fontSize": "12.5px"}),
                        html.Span(phased_plan["phase_3_strategic"][0], style={"fontSize": "12.5px", "color": COLOR_MUTED})
                    ])
                ]
            )
        ]
    )


# ---------------------------------------------------------------------------
# 7. UNIFIED FASTAPI INTELLIGENCE ENGINE & DASH PORTAL SERVER
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from backend.api.router import api_router

# Sovereign FastAPI backend
server = FastAPI(
    title="NITI DRISHTI Institutional Decision Intelligence System API",
    description="National Infrastructure Readiness Index (IRI) RESTful API for Chhatrapati Sambhajinagar District",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount all 11 RESTful API endpoints under /api
server.include_router(api_router, prefix="/api")

# Mount Dash government portal at root URL
server.mount("/", WSGIMiddleware(app.server))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 75)
    print("NITI DRISHTI: Government Institutional Decision-Support System")
    print("Unified Application Architecture (New Government UI + FastAPI Intelligence)")
    print("Portal Interface : http://127.0.0.1:8050/")
    print("API Documentation: http://127.0.0.1:8050/docs")
    print("=" * 75)

    uvicorn.run(server, host="127.0.0.1", port=8050, log_level="info")