from typing import Any

from gct.reporting.report import _interpretation_level


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
