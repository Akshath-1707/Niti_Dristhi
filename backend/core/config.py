"""
=============================================================================
NITI DRISHTI: Centralized Regulatory & Policy Configuration
=============================================================================
Manages statutory benchmarks (RTE Act 2009, NEP 2020, UDISE+) and multi-criteria
weight configurations in a single inspectable module.
=============================================================================
"""

# Regulatory Statutory Benchmarks
BENCHMARKS = {
    "ptr_target_elementary": 30.0,    # RTE Act Section 25 target (30 students / teacher)
    "ptr_critical_threshold": 45.0,   # Emergency overcrowding ceiling
    "pcr_target": 40.0,               # UDISE+ target (40 students / functional classroom)
    "pcr_critical_threshold": 50.0,   # Severe classroom deficit threshold
    "boys_per_toilet_target": 50.0,   # RTE sanitary standard for boys
    "girls_per_toilet_target": 40.0,  # RTE sanitary standard for girls
}

# Default Multi-Criteria Composite Weights (Sum = 1.0)
DEFAULT_WEIGHTS = {
    "equity": 0.25,       # Gender Parity Index & female inclusion
    "promotion": 0.30,    # Pass rate & remedial backlog remediation
    "retention": 0.25,    # Upper Primary to Secondary cohort progression
    "capacity": 0.20,     # Physical uptime & student scale balance
}

# 4-Tier Cognitive Priority Classification Thresholds
PRIORITY_THRESHOLDS = {
    "critical_max_score": 40.0,   # Score <= 40.0: Critical Emergency
    "high_max_score": 60.0,       # Score <= 60.0: High Priority
    "moderate_max_score": 75.0,   # Score <= 75.0: Moderate Needs
    # Score > 75.0: Low Priority / Model Resilient
}

# Standard School Category Labels
CATEGORY_LABELS = {
    1: "Primary (Classes 1–5)",
    2: "Primary with Upper Primary (Classes 1–8)",
    3: "Higher Secondary with Primary (Classes 1–12)",
    4: "Upper Primary Only (Classes 6–8)",
    5: "Higher Secondary with Upper Primary (Classes 6–12)",
    6: "Secondary with Primary (Classes 1–10)",
    7: "Secondary Only (Classes 9–10)",
    8: "Secondary with Upper Primary (Classes 6–10)",
    10: "Higher Secondary Only (Classes 11–12)",
    11: "Pre-Primary Only",
}
