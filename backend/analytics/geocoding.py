"""
=============================================================================
NITI DRISHTI: Geospatial Analytics & GIS Mapping Engine
=============================================================================
Anchors 954 educational institutions to authentic municipal ward centroids
across Chhatrapati Sambhajinagar (URC-1 & URC-2) and generates interactive GIS maps.
=============================================================================
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Authentic Ward Centroids in Chhatrapati Sambhajinagar
CLUSTER_COORDS = {
    # URC-1 (Historic Core, West & North)
    "CLU3605027": (19.8775, 75.3280),  # Shahganj / Kranti Chowk
    "CLU3605028": (19.8720, 75.3620),  # Chikalthana Industrial / Jalna Road West
    "CLU3605031": (19.9200, 75.3350),  # Harsul / North Jalgaon Road
    "CLU3605032": (19.8920, 75.3180),  # Begumpura / Khadkeshwar
    "CLU3605034": (19.8650, 75.3150),  # Padampura / Railway Station Parisar
    # URC-2 (Modern East, CIDCO & Garkheda)
    "CLU3605041": (19.8850, 75.3680),  # CIDCO Sectors N-1 to N-4
    "CLU3605042": (19.9120, 75.3580),  # HUDCO / TV Centre
    "CLU3605045": (19.8520, 75.3450),  # Paithan Road / South Wards
    "CLU3605046": (19.8780, 75.3850),  # Mukundwadi / Al Hilal Colony
    "CLU3605047": (19.8680, 75.3520),  # Garkheda Parisar / Seven Hills
}

COLOR_CRIMSON = "#B91C1C"  # Rust Crimson (< 55)
COLOR_OCHRE = "#B45309"    # Warm Ochre (55 - 74)
COLOR_SAGE = "#16A34A"     # Sage Green (>= 75)
COLOR_PINE = "#0F766E"     # Deep Pine Teal
COLOR_BORDER = "#CBD5E1"   # Slate Border


def assign_school_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns realistic municipal geographic coordinates (lat/lon) to each school
    based on administrative cluster centroid and deterministic UDISE jitter.
    """
    df_geo = df.copy()

    def get_coords(row):
        clu = str(row.get("cluster_code", "")).strip()
        base_lat, base_lon = CLUSTER_COORDS.get(clu, (19.8762, 75.3433))

        # Deterministic seed from 11-digit UDISE code guarantees spatial consistency
        udise_str = str(row.get("udise_code", "0"))
        try:
            seed = int(udise_str[-6:])
        except ValueError:
            seed = 42

        rng = np.random.RandomState(seed)
        lat_offset = rng.uniform(-0.0075, 0.0075)
        lon_offset = rng.uniform(-0.0075, 0.0075)
        return pd.Series([round(base_lat + lat_offset, 6), round(base_lon + lon_offset, 6)])

    if "latitude" not in df_geo.columns or "longitude" not in df_geo.columns:
        df_geo[["latitude", "longitude"]] = df_geo.apply(get_coords, axis=1)

    return df_geo


def create_gis_infrastructure_map(df_filtered: pd.DataFrame) -> go.Figure:
    """
    Builds an interactive geospatial map plotting all filtered schools
    across Chhatrapati Sambhajinagar with priority colors and sizing.
    """
    if df_filtered.empty or "latitude" not in df_filtered.columns:
        # Fallback empty figure centered on Chhatrapati Sambhajinagar
        fig = go.Figure()
        fig.update_layout(
            title="No schools matching filter.",
            paper_bgcolor="#FFFFFF",
            height=420
        )
        return fig

    # Clamp size between 6 and 22 for clean visual balance
    df_plot = df_filtered.copy()
    enrol = df_plot.get("total_enrolment", df_plot.get("total_students", 50)).fillna(50)
    df_plot["marker_size"] = np.clip(np.sqrt(enrol) * 0.45, 6, 22)

    # Clean priority labels
    tier_labels = {
        "CRITICAL_INTERVENTION": "Critical Priority (< 55)",
        "DEVELOPING_NEEDS": "Developing Needs (55–74)",
        "OPTIMAL_RESILIENT": "Optimal / Resilient (≥ 75)"
    }
    df_plot["tier_display"] = df_plot["priority_tier"].map(tier_labels).fillna("Developing Needs")

    color_map = {
        "Critical Priority (< 55)": COLOR_CRIMSON,
        "Developing Needs (55–74)": COLOR_OCHRE,
        "Optimal / Resilient (≥ 75)": COLOR_SAGE,
    }

    # Center coordinates of Chhatrapati Sambhajinagar urban core
    center_lat = 19.8780
    center_lon = 75.3520

    hover_data = {
        "latitude": False,
        "longitude": False,
        "marker_size": False,
        "tier_display": False,
        "school_name": True,
        "udise_code": True,
        "total_enrolment": ":,",
        "readiness_score": ":.1f",
        "promotion_rate": ":.1f%",
        "pending_students": True,
    }

    try:
        # Plotly 6.x modern scatter_map
        fig = px.scatter_map(
            df_plot,
            lat="latitude",
            lon="longitude",
            color="tier_display",
            color_discrete_map=color_map,
            size="marker_size",
            hover_name="school_name",
            hover_data=hover_data,
            custom_data=["udise_code", "school_name", "readiness_score"],
            zoom=11.2,
            center={"lat": center_lat, "lon": center_lon},
            map_style="carto-positron"
        )
    except AttributeError:
        # Backward compatibility for px.scatter_mapbox
        fig = px.scatter_mapbox(
            df_plot,
            lat="latitude",
            lon="longitude",
            color="tier_display",
            color_discrete_map=color_map,
            size="marker_size",
            hover_name="school_name",
            hover_data=hover_data,
            custom_data=["udise_code", "school_name", "readiness_score"],
            zoom=11.2,
            center={"lat": center_lat, "lon": center_lon},
            mapbox_style="carto-positron"
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=440,
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(size=11, family="Inter", color="#020617")
        )
    )

    return fig
