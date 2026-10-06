"""Amendment 1 to docs/NEW_WEEKS_RULES.md: the Table 1 column check.

PDF extraction adds a trailing blank column to otherwise valid tables. Only
trailing blanks may be removed; everything else that departs from the 27-column
schema makes the week missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")  # build_corrected_cases needs it; CI installs only numpy

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis" / "_build"))

from build_corrected_cases import WER_COLUMNS  # noqa: E402
from parse_new_weeks import table_columns  # noqa: E402

COLS = list(WER_COLUMNS)
# As printed in the 2024-2026 reports: two spellings differ from the column names.
PRINTED = [{"NuwaraEliya": "Nuwara Eliya", "Moneragala": "Monaragala"}.get(c, c) for c in COLS]
B = [str(10 + k) for k in range(27)]
A = [str(k) for k in range(27)]


def _call(header_cells, b_cells, a_cells):
    return table_columns(
        ["RDHS", "", *header_cells],
        ["Dengue Fever", "B", *b_cells],
        ["", "A", *a_cells],
        COLS,
    )


def test_27_columns_plus_one_trailing_blank_is_accepted():
    out = _call([*PRINTED, ""], [*B, ""], [*A, None])
    assert out == (B, A)


def test_exactly_27_columns_is_accepted():
    assert _call(PRINTED, B, A) == (B, A)


def test_a_nonblank_trailing_cell_is_rejected():
    assert isinstance(_call([*PRINTED, ""], [*B, ""], [*A, "7"]), str)
    assert isinstance(_call([*PRINTED, "Extra"], [*B, ""], [*A, ""]), str)


def test_an_internal_blank_column_is_rejected():
    header = [*PRINTED[:5], "", *PRINTED[5:]]
    assert isinstance(_call(header, [*B[:5], "", *B[5:]], [*A[:5], "", *A[5:]]), str)


def test_a_wrong_or_misordered_district_name_is_rejected():
    swapped = [PRINTED[1], PRINTED[0], *PRINTED[2:]]
    assert isinstance(_call(swapped, B, A), str)
    wrong = [*PRINTED[:3], "Kandi", *PRINTED[4:]]
    assert isinstance(_call(wrong, B, A), str)


def test_only_the_listed_spellings_are_normalised():
    other = ["Nuwara-Eliya" if c == "Nuwara Eliya" else c for c in PRINTED]
    assert isinstance(_call(other, B, A), str)


def test_a_short_data_row_is_rejected():
    assert isinstance(_call(PRINTED, B[:-1], A), str)


# ---- amendment 2: cross-week rules --------------------------------------------------------------
from parse_new_weeks import weekly_vector  # noqa: E402


def _vec(seed: int, scale: int = 40) -> list[int]:
    """A valid 27-cell weekly vector: 25 districts, Kalmunai, national = their sum."""
    rng = __import__("numpy").random.default_rng(seed)
    d = [int(x) for x in rng.integers(1, scale, 26)]
    return [*d, sum(d)]


def _add(*vs):
    return [sum(c) for c in zip(*vs, strict=True)]


def _broken(v):
    return [*v[:26], v[26] // 10]  # national total misprinted: fails R1


def _report(week, a, b):
    return {"A": a, "B": b, "week_label": week}


def test_rollover_printed_week_1_uses_row_b_alone():
    w1 = _vec(1)
    vec, status, _ = weekly_vector(_report(1, _broken(w1), w1), _report(52, _vec(2), _vec(3, 9000)))
    assert (vec, status) == (w1, "corrected")


def test_report_number_1_carrying_week_52_uses_the_difference_not_the_year_total():
    b51, w52 = _vec(4, 9000), _vec(5)
    cur = _report(52, _broken(w52), _add(b51, w52))
    vec, status, _ = weekly_vector(cur, _report(51, _vec(6), b51))
    assert (vec, status) == (w52, "corrected")


def test_no_difference_across_a_gap_in_printed_weeks():
    w = _vec(7)
    vec, status, _ = weekly_vector(
        _report(10, _broken(w), _vec(8, 9000)), _report(8, _vec(9), _vec(10))
    )
    assert (vec, status) == (None, "missing")


def test_no_difference_without_a_parsed_predecessor():
    vec, status, _ = weekly_vector(_report(10, _broken(_vec(11)), _vec(12, 9000)), None)
    assert (vec, status) == (None, "missing")


def test_repeated_row_a_is_replaced_by_a_valid_difference():
    prev_a, b9, w10 = _vec(13), _vec(14, 9000), _vec(15)
    vec, status, log = weekly_vector(_report(10, prev_a, _add(b9, w10)), _report(9, prev_a, b9))
    assert (vec, status) == (w10, "corrected")
    assert any("repeats" in e["reason"] for e in log if e["action"] == "qc")


def test_a_fully_reprinted_table_is_missing():
    a, b = _vec(16), _vec(17, 9000)
    vec, status, _ = weekly_vector(_report(10, a, b), _report(9, a, b))
    assert (vec, status) == (None, "missing")


def test_a_negative_difference_is_never_used_and_is_flagged():
    b9 = _vec(18, 9000)
    b10 = [*b9[:3], b9[3] - 5, *b9[4:26], b9[26] - 5]  # one district's cumulative drops
    vec, status, log = weekly_vector(_report(10, _broken(_vec(19)), b10), _report(9, _vec(20), b9))
    assert (vec, status) == (None, "missing")
    assert any("decreases" in e["reason"] for e in log)


def test_week_1_row_b_differing_from_row_a_is_flagged_but_row_a_is_kept():
    a = _vec(21)
    vec, status, log = weekly_vector(_report(1, a, _vec(22)), None)
    assert (vec, status) == (a, "observed")
    assert any("week 1" in e["reason"] for e in log if e["action"] == "qc")


def test_large_but_consistent_weeks_are_kept():
    a = _vec(23, 2000)
    vec, status, _ = weekly_vector(
        _report(30, a, _vec(24, 90000)), _report(29, _vec(25, 2000), _vec(26))
    )
    assert (vec, status) == (a, "observed")


def test_a_difference_that_reproduces_the_repeated_row_is_not_evidence():
    # Amendment 2b: row B advanced by the repeated row A itself (B10 = B9 + A9).
    a9, b9 = _vec(27), _vec(28, 9000)
    vec, status, log = weekly_vector(_report(10, a9, _add(b9, a9)), _report(9, a9, b9))
    assert (vec, status) == (None, "missing")
    assert any("2b" in e["reason"] for e in log)
