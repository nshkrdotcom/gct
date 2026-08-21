from typing import Any

from gct.reporting.report import _hypothesis_line, _interpretation_level, _root_report_text


def _hypotheses() -> dict[str, dict[str, Any]]:
    return {
        "H1": {"status": "not_supported", "ci_95": [-0.1, 0.1]},
        "H2": {"status": "not_supported"},
        "H3": {"status": "not_supported"},
        "H4": {"status": "not_supported"},
        "H5": {"status": "not_supported"},
        "H6": {"status": "control_pass"},
        "H7": {"status": "not_supported"},
        "H8": {"status": "not_supported"},
    }


def test_level_one_allows_a_reliable_opposite_direction_h1_effect() -> None:
    hypotheses = _hypotheses()
    hypotheses["H1"]["ci_95"] = [1.4, 1.6]
    assert _interpretation_level(hypotheses) == 1


def test_interpretation_ladder_is_cumulative() -> None:
    hypotheses = _hypotheses()
    assert _interpretation_level(hypotheses) == 0
    hypotheses["H1"] = {"status": "supported", "ci_95": [-1.0, -0.5]}
    hypotheses["H2"]["status"] = "supported"
    assert _interpretation_level(hypotheses) == 2
    hypotheses["H3"]["status"] = "supported"
    hypotheses["H4"]["status"] = "supported"
    assert _interpretation_level(hypotheses) == 4
    for name in ("H5", "H7", "H8"):
        hypotheses[name]["status"] = "supported"
    assert _interpretation_level(hypotheses) == 5


def test_root_report_links_resolve_into_the_run_directory() -> None:
    text = "[plot](figures/test.png) `metrics/a.parquet` `statistics/b.json`"
    assert _root_report_text(text, "gct-v0.1-example") == (
        "[plot](runs/gct-v0.1-example/figures/test.png) "
        "`runs/gct-v0.1-example/metrics/a.parquet` "
        "`runs/gct-v0.1-example/statistics/b.json`"
    )


def test_h7_report_is_explicit_when_behavior_is_unparseable() -> None:
    line = _hypothesis_line(
        "H7",
        {
            "title": "Informative base lift",
            "status": "not_supported",
            "structural_gain": {"irrelevant_q": {"estimate": 0.1}},
            "behavioral_gain": {"status": "inconclusive_due_to_parse_failures"},
        },
    )
    assert "inconclusive_due_to_parse_failures" in line
