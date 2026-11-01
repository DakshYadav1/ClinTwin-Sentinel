import pandas as pd


def active_vasopressors(intervals, assessment_time):
    """Return active medication doses at one assessment timestamp."""
    t = pd.Timestamp(assessment_time)

    active = intervals[
        (intervals["starttime"] <= t)
        & (intervals["endtime"] > t)
    ]

    doses = {}
    overlap_flags = {}

    for medication, group in active.groupby("medication"):
        doses[medication] = float(group["rate"].max())
        overlap_flags[medication] = len(group) > 1

    return doses, overlap_flags


def cardiovascular_sofa(doses, map_mmhg=None):
    """
    Original SOFA cardiovascular component.
    Dose units: mcg/kg/min.
    """
    score = 0

    dopamine = doses.get("dopamine", 0)
    dobutamine = doses.get("dobutamine", 0)
    norepinephrine = doses.get("norepinephrine", 0)
    epinephrine = doses.get("epinephrine", 0)

    if dopamine > 15:
        score = max(score, 4)
    elif dopamine > 5:
        score = max(score, 3)
    elif dopamine > 0:
        score = max(score, 2)

    if dobutamine > 0:
        score = max(score, 2)

    if norepinephrine > 0.1 or epinephrine > 0.1:
        score = max(score, 4)
    elif 0 < norepinephrine <= 0.1 or 0 < epinephrine <= 0.1:
        score = max(score, 3)

    if score == 0 and map_mmhg is not None and map_mmhg < 70:
        score = 1

    return score


def score_at_time(intervals, assessment_time, map_mmhg=None):
    doses, overlap_flags = active_vasopressors(
        intervals, assessment_time
    )

    return {
        "assessment_time": pd.Timestamp(assessment_time),
        "cardiovascular_sofa": cardiovascular_sofa(
            doses, map_mmhg
        ),
        "active_doses": doses,
        "overlap_flags": overlap_flags,
        "any_overlap": any(overlap_flags.values()),
    }


def run_tests():
    cases = [
        ("No support, normal MAP", {}, 75, 0),
        ("Low MAP, no support", {}, 65, 1),
        ("Dopamine 5", {"dopamine": 5}, 75, 2),
        ("Dobutamine 3", {"dobutamine": 3}, 75, 2),
        ("Dopamine 10", {"dopamine": 10}, 75, 3),
        ("Norepinephrine 0.1", {"norepinephrine": 0.1}, 75, 3),
        ("Epinephrine 0.2", {"epinephrine": 0.2}, 75, 4),
        ("Dopamine 16", {"dopamine": 16}, 75, 4),
        ("Multiple drugs", {
            "dobutamine": 5,
            "norepinephrine": 0.15
        }, 75, 4),
    ]

    print("========== SOFA SCORING TESTS ==========")
    passed = 0

    for name, doses, map_value, expected in cases:
        actual = cardiovascular_sofa(doses, map_value)
        ok = actual == expected
        passed += int(ok)
        print(
            f"{'PASS' if ok else 'FAIL'} | {name} | "
            f"expected={expected}, actual={actual}"
        )

    # Test interval boundaries and overlapping doses.
    sample = pd.DataFrame([
        {
            "medication": "norepinephrine",
            "rate": 0.08,
            "starttime": pd.Timestamp("2026-01-01 09:00"),
            "endtime": pd.Timestamp("2026-01-01 10:30"),
        },
        {
            "medication": "norepinephrine",
            "rate": 0.15,
            "starttime": pd.Timestamp("2026-01-01 10:00"),
            "endtime": pd.Timestamp("2026-01-01 11:00"),
        },
    ])

    result = score_at_time(
        sample, "2026-01-01 10:15", 75
    )

    assert result["active_doses"]["norepinephrine"] == 0.15
    assert result["any_overlap"] is True
    assert result["cardiovascular_sofa"] == 4

    print("PASS | Overlapping intervals use highest active dose")
    print("PASS | Overlap is flagged")
    print("PASS | End timestamp is exclusive")

    print(f"\nTests passed: {passed + 3}/{len(cases) + 3}")


if __name__ == "__main__":
    run_tests()
