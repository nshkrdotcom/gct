from __future__ import annotations

import pytest

from gct.models.behavior import parse_numeric_answer


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("83.25", 83.25),
        ("work\nFINAL=-1.2e1", -12.0),
        ("work 2 then\nFINAL = 3.5 C", 3.5),
    ],
)
def test_numeric_parser(text: str, value: float) -> None:
    parsed = parse_numeric_answer(text)
    assert parsed.status == "parsed"
    assert parsed.value == value


def test_numeric_parser_records_failure() -> None:
    parsed = parse_numeric_answer("underdetermined")
    assert parsed.status == "missing_final_marker"
    assert parsed.value is None


def test_numeric_parser_does_not_mistake_copied_context_for_answer() -> None:
    parsed = parse_numeric_answer("Given:\n- Concentration M: 0.75")
    assert parsed.status == "missing_final_marker"
    assert parsed.value is None


def test_numeric_parser_accepts_prefilled_answer_before_trailing_text() -> None:
    parsed = parse_numeric_answer("FINAL=104.2\nExplanation omitted.")
    assert parsed.status == "parsed"
    assert parsed.value == 104.2
