"""
=============================================================================
NITI DRISHTI: Phased Action Planner & Decision Engine
=============================================================================
Synthesizes physical gaps and triggered statutory rules into a structured,
tri-phased municipal intervention roadmap.
=============================================================================
"""

from backend.reasoning.gaps import analyze_school_infrastructure_gaps
from backend.reasoning.rules import evaluate_statutory_rules


def formulate_phased_action_plan(school_row: dict) -> dict:
    """
    Formulates a comprehensive 3-phase action plan for a school:
      - Phase 1 (Immediate 0–90 Days): Emergency administrative adjustments
      - Phase 2 (Medium-Term 6–24 Months): Civil works & teacher staffing
      - Phase 3 (Strategic Horizon 2037): Cluster integration & smart digital facilities
    """
    gaps = analyze_school_infrastructure_gaps(school_row)
    matched_rules = evaluate_statutory_rules(gaps)

    # Determine Priority Tier
    priorities = [r["priority"] for r in matched_rules]
    if "CRITICAL" in priorities:
        overall_priority = "CRITICAL"
        priority_color = "#DC2626"  # Crimson
    elif "HIGH" in priorities:
        overall_priority = "HIGH"
        priority_color = "#EA580C"  # Amber Orange
    elif "MODERATE" in priorities:
        overall_priority = "MODERATE"
        priority_color = "#D97706"  # Warm Yellow
    else:
        overall_priority = "LOW"
        priority_color = "#16A34A"  # Green

    # Construct Phase 1: Immediate Actions (0–90 Days)
    phase_1_actions = []
    if gaps["teacher_gap"] >= 3:
        phase_1_actions.append(f"Deploy {gaps['teacher_gap']} guest or cluster-shared teachers immediately to relieve PTR of {gaps['actual_ptr']}:1.")
    elif gaps["teacher_gap"] > 0:
        phase_1_actions.append(f"Temporarily reassign 1–2 resource teachers from lower-density neighboring schools.")

    if gaps["classroom_gap"] >= 3:
        phase_1_actions.append(f"Adopt temporary dual-shift school timing (Morning: Primary / Afternoon: Upper Primary) to mitigate {gaps['classroom_gap']} classroom shortage.")

    if gaps["pending_students"] > 20:
        phase_1_actions.append(f"Launch 60-day accelerated remedial bridge classes for {gaps['pending_students']} pending non-promoted students.")

    if not phase_1_actions:
        phase_1_actions.append("Conduct routine quarterly administrative audit and maintain operational standards.")

    # Construct Phase 2: Medium-Term Civil Works (6–24 Months)
    phase_2_actions = []
    if gaps["classroom_gap"] > 0:
        phase_2_actions.append(f"Sanction capital civil works budget to construct {gaps['classroom_gap']} permanent standard classrooms (approx. ₹15–20 Lakhs under Samagra Shiksha).")

    if gaps["girls_toilet_gap"] > 0:
        phase_2_actions.append(f"Construct {gaps['girls_toilet_gap']} dedicated, barrier-free sanitation units for female students with running water.")

    if gaps["teacher_gap"] > 0:
        phase_2_actions.append(f"Submit formal roster requisition to Maharashtra State Education Dept for {gaps['teacher_gap']} permanent cadre teachers.")

    if not phase_2_actions:
        phase_2_actions.append("Perform preventive maintenance on existing civil and electrical infrastructure.")

    # Construct Phase 3: Strategic Modernization (Vision 2037)
    phase_3_actions = []
    if gaps["is_capacity_saturated"]:
        phase_3_actions.append(f"Expand master site plan by 2037 to accommodate projected demographic surge (+{gaps['future_capacity_gap']} students over built limit).")

    phase_3_actions.append("Equip school with 10-seat smart ICT computer lab and high-speed broadband under NEP 2020 digital inclusion guidelines.")
    phase_3_actions.append("Incorporate institution into PM SHRI school complex cluster for shared athletic and science laboratory facilities.")

    return {
        "school_name": school_row.get("school_name", "Unknown School"),
        "udise_code": school_row.get("udise_code", 0),
        "overall_priority": overall_priority,
        "priority_color": priority_color,
        "dominant_bottleneck": gaps["dominant_bottleneck"],
        "dominant_reason": gaps["dominant_reason"],
        "gaps": gaps,
        "matched_rules": matched_rules,
        "phased_plan": {
            "phase_1_immediate": phase_1_actions,
            "phase_2_medium": phase_2_actions,
            "phase_3_strategic": phase_3_actions
        }
    }
