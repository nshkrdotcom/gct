from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from gct.config import ExperimentConfig
from gct.data.prompts import (
    COMMON_TASK_SUFFIX,
    LEXICAL_ALIAS_INVERSE,
    LEXICAL_ALIAS_MAP,
    PromptRenderer,
)
from gct.worlds.toythermo import State, ToyThermo


class SuffixTokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Sequence[int]:
        assert tokenize and add_generation_prompt and kwargs["enable_thinking"] is False
        body = [ord(char) % 251 for char in conversation[-1]["content"]]
        return [*body, 9001, 9002, 9003, 9004]


class MappingSuffixTokenizer(SuffixTokenizer):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any:
        values = super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        return {"input_ids": values}


def test_all_renderers_share_common_text_suffix(ci_config: ExperimentConfig) -> None:
    renderer = PromptRenderer(ToyThermo(ci_config.world))
    state = State("Aquila", 1.1, 0.5, 0.8)
    prompts = [
        renderer.render(state, "inferable_unnamed_coordinate", renamed=False, variant=variant).text
        for variant in (
            "prose",
            "bullets",
            "json_like",
            "active_passive",
            "clause_order",
            "lexical_alias",
            "paraphrase_0",
        )
    ]
    assert all(prompt.endswith(COMMON_TASK_SUFFIX) for prompt in prompts)
    assert renderer.assert_common_anchor(SuffixTokenizer(), prompts, 4) == (9001, 9002, 9003, 9004)
    assert renderer.assert_common_anchor(MappingSuffixTokenizer(), prompts, 4) == (
        9001,
        9002,
        9003,
        9004,
    )


def test_unobservable_is_exactly_pressure_blind(ci_config: ExperimentConfig) -> None:
    renderer = PromptRenderer(ToyThermo(ci_config.world))
    low = State("Boreal", 0.5, 0.75, 1.0)
    high = State("Boreal", 2.5, 0.75, 1.0)
    first = renderer.render(low, "unobservable_coordinate", renamed=False, variant="prose")
    second = renderer.render(high, "unobservable_coordinate", renamed=False, variant="prose")
    assert first.text == second.text
    assert first.prompt_hash == second.prompt_hash
    assert "pressure" not in first.observable
    assert "sensor_reading" not in first.observable
    assert "Pressure P" not in first.text
    assert "sealed coordinate U" in first.text


def test_renamed_unobservable_uses_opaque_isomorphic_hidden_symbol(
    ci_config: ExperimentConfig,
) -> None:
    renderer = PromptRenderer(ToyThermo(ci_config.world))
    state = State("Aquila", 1.1, 0.5, 0.8)
    rendered = renderer.render(state, "unobservable_coordinate", renamed=True, variant="prose")
    assert "sealed coordinate V" in rendered.text
    assert "Control X" not in rendered.text
    assert "ln(V)" in rendered.text


def test_alias_world_changes_labels_not_oracle(ci_config: ExperimentConfig) -> None:
    world = ToyThermo(ci_config.world)
    renderer = PromptRenderer(world)
    state = State("Aquila", 1.1, 0.5, 0.8)
    primary = renderer.render(state, "explicit_coordinate", renamed=False, variant="prose")
    alias = renderer.render(state, "explicit_coordinate", renamed=True, variant="prose")
    assert "Aquila" in primary.text and "Water" in alias.text
    assert "T(Aquila,P,M)" in primary.text and "Z(Water,X,Y)" in alias.text
    assert "Pressure P" in primary.text and "Control X" in alias.text
    assert "Calibration law: R" in primary.text and "Calibration law: G" in alias.text
    assert world.oracle(state) == world.oracle(state)


def test_required_nuisance_renderers_are_semantically_reversible(
    ci_config: ExperimentConfig,
) -> None:
    renderer = PromptRenderer(ToyThermo(ci_config.world))
    state = State("Boreal", 1.25, 0.75, 0.4)
    rendered = {
        variant: renderer.render(state, "explicit_coordinate", renamed=False, variant=variant)
        for variant in ("prose", "active_passive", "clause_order", "lexical_alias")
    }
    assert all(item.observable == rendered["prose"].observable for item in rendered.values())
    assert "was recorded as" in rendered["active_passive"].text
    assert rendered["clause_order"].text.index("Pressure P") < rendered["clause_order"].text.index(
        "Fluid:"
    )
    assert "Substance code: Boreal" in rendered["lexical_alias"].text
    assert all(
        LEXICAL_ALIAS_INVERSE[alias] == source for source, alias in LEXICAL_ALIAS_MAP.items()
    )
