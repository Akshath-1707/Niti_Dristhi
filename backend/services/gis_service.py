"""
=============================================================================
NITI DRISHTI: Geospatial GIS Service
=============================================================================
Manages coordinates, spatial distribution across URC-1 and URC-2 wards in
Chhatrapati Sambhajinagar, and Plotly map figures.
=============================================================================
"""

import pandas as pd
import plotly.graph_objects as go
from backend.analytics.geocoding import create_gis_infrastructure_map
from backend.services.readiness_service import readiness_service


class GISService:
    @staticmethod
    def get_spatial_records(filtered_df: pd.DataFrame = None) -> list[dict]:
        df = filtered_df if filtered_df is not None else readiness_service.get_master_df()
        cols = [
            "udise_code", "school_name", "block_name", "school_management",
            "latitude", "longitude", "total_enrolment", "readiness_score", "priority_tier"
        ]
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols].to_dict(orient="records")

    @staticmethod
    def get_gis_figure(filtered_df: pd.DataFrame = None) -> go.Figure:
        df = filtered_df if filtered_df is not None else readiness_service.get_master_df()
        return create_gis_infrastructure_map(df)


# Global singleton instance
gis_service = GISService()
