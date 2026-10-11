"""
=============================================================================
NITI DRISHTI: Readiness & District Intelligence Service
=============================================================================
Manages master dataset resolution, evaluation caching, and multi-criteria
district readiness calculations across the 4 statutory pillars.
=============================================================================
"""

import os
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np

from backend.analytics.readiness_index import EducationReadinessCalculator
from backend.analytics.geocoding import assign_school_coordinates
from backend.core.config import DEFAULT_WEIGHTS, BENCHMARKS, PRIORITY_THRESHOLDS

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def find_dataset(filenames: list[str]) -> Path | None:
    """Safely locates datasets trying primary, processed, and legacy fallback paths."""
    for f in filenames:
        p = BASE_DIR / f
        if p.exists():
            return p
    return None


class ReadinessService:
    _instance = None
    _df_master = None
    _district_summary = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ReadinessService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        school_file = find_dataset([
            "data/database_ready_schools_urc.csv",
            "data/processed/database_ready_schools_urc.csv",
            "data/model_school_features.csv",
            "data/database_ready_schools.csv",
            "data/clean_school_master.csv"
        ])

        if not school_file:
            raise FileNotFoundError("Master school dataset not found in data/ directory.")

        df_raw = pd.read_csv(school_file)

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

        # Run Readiness Index Engine & Geocoding
        calculator = EducationReadinessCalculator()
        df_eval = calculator.evaluate_schools(df_raw)
        df_geo = assign_school_coordinates(df_eval)

        self._df_master = df_geo
        self._district_summary = calculator.get_district_summary(self._df_master)

    def get_master_df(self) -> pd.DataFrame:
        return self._df_master.copy()

    def get_district_summary(self, filtered_df: pd.DataFrame = None) -> dict:
        if filtered_df is None:
            return self._district_summary.copy()
        calculator = EducationReadinessCalculator()
        return calculator.get_district_summary(filtered_df)

    def filter_schools(
        self,
        block: str = "ALL",
        management: str = "ALL",
        category: str = "ALL",
        tier: str = "ALL",
        search_query: str = None
    ) -> pd.DataFrame:
        df = self._df_master.copy()

        if block and block != "ALL" and "block_name" in df.columns:
            df = df[df["block_name"] == block]

        if management and management != "ALL" and "school_management" in df.columns:
            df = df[df["school_management"] == management]

        if category and category != "ALL" and "school_category" in df.columns:
            df = df[df["school_category"] == category]

        if tier and tier != "ALL" and "priority_tier" in df.columns:
            df = df[df["priority_tier"] == tier]

        if search_query and str(search_query).strip():
            q = str(search_query).strip().lower()
            mask = (
                df["school_name"].astype(str).str.lower().str.contains(q)
                | df["udise_code"].astype(str).str.lower().str.contains(q)
                | df["block_name"].astype(str).str.lower().str.contains(q)
            )
            df = df[mask]

        return df

    def get_filter_options(self) -> dict:
        df = self._df_master
        blocks = [{"label": "All Areas / Blocks", "value": "ALL"}] + [
            {"label": str(b), "value": str(b)} for b in sorted(df["block_name"].dropna().unique())
        ] if "block_name" in df.columns else []

        managements = [{"label": "All School Types", "value": "ALL"}] + [
            {"label": str(m), "value": str(m)} for m in sorted(df["school_management"].dropna().unique())
        ] if "school_management" in df.columns else []

        categories = [{"label": "All Grade Levels", "value": "ALL"}] + [
            {"label": str(c), "value": str(c)} for c in sorted(df["school_category"].dropna().unique())
        ] if "school_category" in df.columns else []

        tiers = [
            {"label": "All Status Tiers", "value": "ALL"},
            {"label": "Critical Attention Required (< 55)", "value": "CRITICAL_INTERVENTION"},
            {"label": "Developing Needs (55 - 74)", "value": "DEVELOPING_NEEDS"},
            {"label": "Optimal / Resilient (>= 75)", "value": "OPTIMAL_RESILIENT"},
        ]

        school_select = [
            {"label": f"{r['school_name']} (UDISE: {r['udise_code']})", "value": str(r["udise_code"])}
            for _, r in df.sort_values("total_enrolment", ascending=False).iterrows()
        ]

        return {
            "blocks": blocks,
            "managements": managements,
            "categories": categories,
            "tiers": tiers,
            "school_select": school_select
        }


# Global singleton instance
readiness_service = ReadinessService()
