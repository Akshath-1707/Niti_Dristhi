"""
=============================================================================
NITI DRISHTI: Declarative Statutory Policy Rule Engine
=============================================================================
Defines 8 deterministic, inspectable policy rules derived directly from
the RTE Act 2009, NEP 2020, and Samagra Shiksha statutory guidelines.
=============================================================================
"""

STATUTORY_RULES = [
    {
        "id": "RULE-PTR-EMERGENCY",
        "name": "Emergency Teacher Deprivation Alert",
        "condition": lambda g: g["actual_ptr"] >= 45.0 or g["teacher_gap"] >= 4,
        "priority": "CRITICAL",
        "statute": "Right to Education (RTE) Act 2009, Section 25 & Schedule",
        "intervention": "Immediate administrative redeployment of surplus teachers from neighboring cluster schools within 30 days."
    },
    {
        "id": "RULE-CLASSROOM-CRITICAL",
        "name": "Severe Classroom Overcrowding Alert",
        "condition": lambda g: g["actual_pcr"] >= 50.0 or g["classroom_gap"] >= 4,
        "priority": "CRITICAL",
        "statute": "RTE Act 2009, Section 19 & UDISE+ Norms",
        "intervention": "Sanction priority civil works budget under Samagra Shiksha for new classroom construction and implement temporary dual-shift scheduling."
    },
    {
        "id": "RULE-FUTURE-SURGE",
        "name": "Imminent Demography Capacity Overflow",
        "condition": lambda g: g["future_capacity_gap"] >= 30,
        "priority": "HIGH",
        "statute": "National Education Policy (NEP) 2020, Chapter 7 (School Complexes)",
        "intervention": "Integrate institution into urban school complex cluster for shared infrastructure and master capital expansion."
    },
    {
        "id": "RULE-REMEDIAL-BACKLOG",
        "name": "Learning Recovery & Retention Crisis",
        "condition": lambda g: g["backlog_ratio"] >= 25.0 or g["pending_students"] >= 50,
        "priority": "HIGH",
        "statute": "Samagra Shiksha Framework 2024 (Remedial & Bridge Courses)",
        "intervention": "Establish 90-day fast-track remedial learning recovery camps funded by Samagra Shiksha learning enhancement grants."
    },
    {
        "id": "RULE-SANITATION-DEFICIT",
        "name": "Girl-Child Sanitation Inadequacy",
        "condition": lambda g: g["girls_toilet_gap"] >= 2,
        "priority": "HIGH",
        "statute": "RTE Act 2009 Schedule & Swachh Vidyalaya Directive",
        "intervention": "Fast-track construction of dedicated, barrier-free sanitation units for female students with continuous water supply."
    },
    {
        "id": "RULE-CLASSROOM-MODERATE",
        "name": "Moderate Classroom Deficit",
        "condition": lambda g: 1 <= g["classroom_gap"] < 4,
        "priority": "HIGH",
        "statute": "RTE Act 2009, Section 19",
        "intervention": "Include additional classroom construction in the upcoming District Annual Work Plan & Budget (AWP&B)."
    },
    {
        "id": "RULE-TEACHER-MODERATE",
        "name": "Moderate Faculty Deficit",
        "condition": lambda g: 1 <= g["teacher_gap"] < 4,
        "priority": "MODERATE",
        "statute": "RTE Act 2009, Section 25",
        "intervention": "Request contractual resource teacher allocation pending formal municipal teacher recruitment cycle."
    },
    {
        "id": "RULE-MODEL-COMPLIANT",
        "name": "Statutory Norm Compliance",
        "condition": lambda g: g["teacher_gap"] == 0 and g["classroom_gap"] == 0 and g["backlog_ratio"] < 15.0,
        "priority": "LOW",
        "statute": "PM SHRI / Model School Quality Norms",
        "intervention": "Maintain standards; benchmark as a demonstration school for best pedagogical and administrative practices."
    }
]


def evaluate_statutory_rules(gap_analysis: dict) -> list[dict]:
    """
    Evaluates all statutory rules against computed school gaps.
    Returns list of matched rule objects.
    """
    matched = []
    for rule in STATUTORY_RULES:
        try:
            if rule["condition"](gap_analysis):
                matched.append({
                    "id": rule["id"],
                    "name": rule["name"],
                    "priority": rule["priority"],
                    "statute": rule["statute"],
                    "intervention": rule["intervention"]
                })
        except Exception:
            continue
    return matched
