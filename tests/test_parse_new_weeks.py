"""Amendment 1 to docs/NEW_WEEKS_RULES.md: the Table 1 column check.

PDF extraction adds a trailing blank column to otherwise valid tables. Only
trailing blanks may be removed; everything else that departs from the 27-column
schema makes the week missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
