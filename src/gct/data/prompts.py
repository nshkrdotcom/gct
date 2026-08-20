"""Leakage-resistant deterministic prompt rendering."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from gct.worlds.toythermo import State, ToyThermo

SYSTEM_MESSAGE = (
    "You are evaluating an axiomatic synthetic world. Use only the supplied rules; "
    "outside scientific knowledge is irrelevant."
)
COMMON_TASK_SUFFIX = (
    "\n\nTask:\nCompute the oracle-defined synthetic temperature. "
    "You may calculate internally, but end with a new line exactly in the form "
    "FINAL=<one decimal number>."
)
LEXICAL_ALIAS_MAP = {
    "Fluid": "Substance code",
    "Concentration M": "Mixture coordinate M",
    "Pressure P": "Load coordinate P",
    "Calibration reading R": "Gauge value R",
    "Nuisance Q": "Auxiliary value Q",
    "Entity": "Object code",
    "Composition Y": "Mixture index Y",
    "Control X": "Load index X",
    "Proxy G": "Gauge alias G",
    "Nuisance W": "Auxiliary value W",
}
LEXICAL_ALIAS_INVERSE = {value: key for key, value in LEXICAL_ALIAS_MAP.items()}
if len(LEXICAL_ALIAS_INVERSE) != len(LEXICAL_ALIAS_MAP):  # pragma: no cover - import invariant
    raise RuntimeError("lexical nuisance alias map must be one-to-one")


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    observable: dict[str, Any]
    prompt_hash: str


def _number(value: float) -> str:
    return format(value, ".8f")


class PromptRenderer:
    def __init__(self, world: ToyThermo) -> None:
        self.world = world

    def _law_lines(self, renamed: bool, hidden_symbol: str = "P") -> list[str]:
        lines = []
        temperature_symbol = self.world.config.renamed_fields["temperature"] if renamed else "T"
        concentration_symbol = self.world.config.renamed_fields["concentration"] if renamed else "M"
        for fluid, c in sorted(self.world.config.fluids.items()):
            label = self.world.label(fluid, renamed)
            lines.append(
                f"{temperature_symbol}({label},{hidden_symbol},{concentration_symbol}) = "
                f"{c.a:.6g} + {c.b:.6g} ln({hidden_symbol}) "
                f"+ {c.k:.6g} {concentration_symbol} + {c.q:.6g} "
                f"{concentration_symbol} ln({hidden_symbol})"
            )
        return lines

    def render(
        self,
        state: State,
        coordinate_condition: str,
        *,
        renamed: bool,
        variant: str,
        persona: str = "neutral",
        irrelevant_fact: bool = False,
    ) -> RenderedPrompt:
        label = self.world.label(state.fluid, renamed)
        observable: dict[str, Any] = {
            "fluid": label,
            "concentration": state.concentration,
            "coordinate_condition": coordinate_condition,
            "world_variant": "renamed" if renamed else "primary",
        }
        preface = "Synthetic-world rules override real-world chemistry."
        pressure_symbol = self.world.config.renamed_fields["pressure"] if renamed else "P"
        sensor_symbol = self.world.config.renamed_fields["sensor"] if renamed else "R"
        renamed_fields = self.world.config.renamed_fields
        renamed_legend = (
            f"Renamed field legend: {renamed_fields['temperature']} is the requested scalar output; "
            f"{renamed_fields['pressure']} is the control coordinate; "
            f"{renamed_fields['concentration']} is the composition coordinate; "
            f"{renamed_fields['sensor']} is the calibration proxy; "
            f"{renamed_fields['nuisance']} is nuisance-only."
            if renamed
            else None
        )
        if coordinate_condition == "unobservable_coordinate":
            manual = [
                preface,
                *([renamed_legend] if renamed_legend is not None else []),
                f"The oracle also depends on a sealed coordinate {pressure_symbol} whose value "
                "and proxies are withheld.",
                *self._law_lines(renamed, hidden_symbol=pressure_symbol),
            ]
        else:
            manual = [
                preface,
                *([renamed_legend] if renamed_legend is not None else []),
                *self._law_lines(renamed, hidden_symbol=pressure_symbol),
            ]
            sensor = self.world.config.calibration_sensor
            manual.append(
                f"Calibration law: {sensor_symbol} = {sensor.r0:.6g} + {sensor.r1:.6g} "
                f"ln({pressure_symbol})."
            )
            if coordinate_condition == "explicit_coordinate":
                observable["pressure"] = state.pressure
            elif coordinate_condition in {"inferable_unnamed_coordinate", "irrelevant_coordinate"}:
                observable["sensor_reading"] = self.world.sensor(state.pressure)
            else:
                raise ValueError(f"unknown coordinate condition: {coordinate_condition}")
            if coordinate_condition == "irrelevant_coordinate":
                observable["irrelevant_q"] = state.irrelevant_q
                nuisance_symbol = self.world.config.renamed_fields["nuisance"] if renamed else "Q"
                temperature_symbol = (
                    self.world.config.renamed_fields["temperature"] if renamed else "T"
                )
                manual.append(
                    f"{nuisance_symbol} is an independently sampled nuisance and has no role in "
                    f"{temperature_symbol} or {sensor_symbol}."
                )

        if renamed:
            manual.append(
                "Familiar labels are aliases only; the supplied synthetic laws take precedence."
            )

        context = self._render_context(observable, variant, renamed=renamed)
        persona_line = {
            "neutral": "",
            "engineer": "Framing only: an engineer recorded this synthetic calibration.\n",
            "student": "Framing only: a student is checking this synthetic exercise.\n",
        }.get(persona)
        if persona_line is None:
            raise ValueError(f"unknown persona: {persona}")
        distractor = (
            "\nIrrelevant fact: the notebook cover is violet; this does not affect any rule."
            if irrelevant_fact
            else ""
        )
        text = persona_line + "\n".join(manual) + "\n\nContext:\n" + context + distractor
        text += COMMON_TASK_SUFFIX
        return RenderedPrompt(text, observable, hashlib.sha256(text.encode()).hexdigest())

    def _render_context(self, observable: dict[str, Any], variant: str, *, renamed: bool) -> str:
        aliases = self.world.config.renamed_fields
        fields = (
            {
                "fluid": aliases["fluid"],
                "concentration": f"Composition {aliases['concentration']}",
                "pressure": f"Control {aliases['pressure']}",
                "sensor": f"Proxy {aliases['sensor']}",
                "nuisance": f"Nuisance {aliases['nuisance']}",
            }
            if renamed
            else {
                "fluid": "Fluid",
                "concentration": "Concentration M",
                "pressure": "Pressure P",
                "sensor": "Calibration reading R",
                "nuisance": "Nuisance Q",
            }
        )
        items = [
            (fields["fluid"], str(observable["fluid"])),
            (fields["concentration"], _number(float(observable["concentration"]))),
        ]
        if "pressure" in observable:
            items.append((fields["pressure"], _number(float(observable["pressure"]))))
        if "sensor_reading" in observable:
            items.append((fields["sensor"], _number(float(observable["sensor_reading"]))))
        if "irrelevant_q" in observable:
            items.append((fields["nuisance"], _number(float(observable["irrelevant_q"]))))
        if variant == "prose":
            return "; ".join(f"{key}: {value}" for key, value in items) + "."
        if variant == "bullets":
            return "\n".join(f"- {key}: {value}" for key, value in items)
        if variant == "json_like":
            return json.dumps(dict(items), sort_keys=True, separators=(",", ":"))
        if variant == "active_passive":
            assignments = "; ".join(f"{key} was recorded as {value}" for key, value in items)
            return f"In the synthetic log, {assignments}."
        if variant == "clause_order":
            return "; ".join(f"{key}: {value}" for key, value in reversed(items)) + "."
        if variant == "lexical_alias":
            return (
                "; ".join(
                    f"{LEXICAL_ALIAS_MAP.get(key, f'Alternate {key}')}: {value}"
                    for key, value in items
                )
                + "."
            )
        if variant.startswith("paraphrase_"):
            paraphrase_index = int(variant.rsplit("_", 1)[1])
            templates = (
                "Given ",
                "The specified synthetic state has ",
                "For this oracle query, use ",
            )
            prefix = templates[paraphrase_index % len(templates)]
            return prefix + ", ".join(f"{key.lower()} = {value}" for key, value in items) + "."
        raise ValueError(f"unknown renderer variant: {variant}")

    @staticmethod
    def chat_messages(prompt: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": prompt}]

    def token_ids(self, tokenizer: ChatTokenizer, prompt: str) -> list[int]:
        values: Any = tokenizer.apply_chat_template(
            self.chat_messages(prompt),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(values, Mapping):
            values = values["input_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def assert_common_anchor(
        self, tokenizer: ChatTokenizer, prompts: Sequence[str], suffix_tokens: int
    ) -> tuple[int, ...]:
        if not prompts:
            raise ValueError("at least one prompt is required for anchor validation")
        suffixes = [tuple(self.token_ids(tokenizer, prompt)[-suffix_tokens:]) for prompt in prompts]
        if any(suffix != suffixes[0] for suffix in suffixes[1:]):
            raise ValueError("prompt variants do not share the configured tokenized anchor suffix")
        return suffixes[0]
