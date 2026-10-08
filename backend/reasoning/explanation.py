"""
=============================================================================
NITI DRISHTI: Cognitive Explainability (XAI) Engine
=============================================================================
Translates quantitative gap metrics and rule activations into human-centric,
causal decision explanations tailored for both Citizens and Administrators.
=============================================================================
"""


def generate_causal_why_factors(decision_dossier: dict) -> list[str]:
    """
    Extracts a numbered list of concrete causal factors driving the school's priority status.
    """
    factors = []
    gaps = decision_dossier.get("gaps", {})
    rules = decision_dossier.get("matched_rules", [])

    if gaps.get("actual_ptr", 0) > 30.0:
        factors.append(f"Pupil-Teacher Ratio is {gaps['actual_ptr']}:1 (exceeds statutory RTE benchmark of 30:1 by {round(gaps['actual_ptr'] - 30.0, 1)} students/teacher).")

    if gaps.get("classroom_gap", 0) > 0:
        factors.append(f"Classroom deficit of {gaps['classroom_gap']} rooms leads to an overcrowded Pupil-Classroom Ratio of {gaps['actual_pcr']}:1 (norm is 40:1).")

    if gaps.get("is_capacity_saturated"):
        factors.append(f"Demographic surge trajectory: 3-year projected enrollment ({gaps['projected_3yr_enrol']}) overflows physical seat capacity ({gaps['built_capacity']}) by {gaps['future_capacity_gap']} students.")

    if gaps.get("pending_students", 0) > 15:
        factors.append(f"Academic promotion vulnerability: {gaps['pending_students']} students ({gaps['backlog_ratio']}% of school body) are currently held back or pending remedial intervention.")

    if gaps.get("girls_toilet_gap", 0) > 0:
        factors.append(f"Sanitation accessibility deficit: shortage of {gaps['girls_toilet_gap']} dedicated female washroom units violating RTE gender parity guidelines.")

    if not factors:
        factors.append("Institutional operations comply with RTE Act and UDISE+ statutory benchmarks across teacher, classroom, and sanitation metrics.")

    return factors


def generate_citizen_explanation(decision_dossier: dict) -> str:
    """
    Generates a simple, jargon-free explanation for parents and normal citizens.
    """
    school_name = decision_dossier.get("school_name", "This school")
    priority = decision_dossier.get("overall_priority", "MODERATE")
    gaps = decision_dossier.get("gaps", {})

    if priority == "CRITICAL":
        explanation = (
            f"{school_name} is marked as a Top Municipal Priority because it currently faces severe overcrowding. "
            f"There are too many students for the available rooms ({gaps.get('classroom_gap', 0)} extra classrooms are needed) "
            f"and teachers are managing over {int(gaps.get('actual_ptr', 30))} children per class. "
            f"Immediate municipal support has been recommended to add teaching staff and temporary rooms."
        )
    elif priority == "HIGH":
        explanation = (
            f"{school_name} is performing reasonably well, but requires upcoming municipal investment. "
            f"Specifically, it needs {gaps.get('classroom_gap', 1)} more classrooms and support for {gaps.get('pending_students', 0)} "
            f"students who need extra tutoring to pass their upcoming exams."
        )
    elif priority == "MODERATE":
        explanation = (
            f"{school_name} has steady everyday operations. Minor improvements are needed in specialized facilities, "
            f"such as extra washrooms or digital equipment, but the core classroom capacity is functioning normally."
        )
    else:
        explanation = (
            f"{school_name} is a high-performing model school in the district. It has enough classrooms, "
            f"balanced teacher-student ratios, and meets all standard government health and safety guidelines."
        )

    return explanation


def generate_administrative_narrative(decision_dossier: dict) -> str:
    """
    Generates an executive briefing narrative for District Collectors and Education Officers.
    """
    school_name = decision_dossier.get("school_name", "School")
    udise = decision_dossier.get("udise_code", "N/A")
    priority = decision_dossier.get("overall_priority", "MODERATE")
    gaps = decision_dossier.get("gaps", {})
    dominant = decision_dossier.get("dominant_bottleneck", "Capacity Balance")
    first_step = decision_dossier.get("phased_plan", {}).get("phase_1_immediate", ["Review quarterly metrics"])[0]

    narrative = (
        f"AUDIT BRIEF: {school_name} (UDISE: {udise}) is categorized under {priority} PRIORITY tier, "
        f"anchored by primary operational constraint: '{dominant}'. "
        f"Current student census is {gaps.get('total_students', 0)} against a built capacity of {gaps.get('built_capacity', 0)} seats. "
        f"To restore compliance with Section 19/25 of the RTE Act 2009, the competent municipal authority should execute: "
        f"'{first_step}' as immediate Phase 1 intervention."
    )
    return narrative
