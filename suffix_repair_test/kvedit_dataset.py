# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load KVEdit paired edits and prepare reproducible token-level inputs."""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from io import TextIOWrapper
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

KVEDIT_REVISION = "c24fbaf72a44b863d564d23310557cc66bd0b5b5"
KVEDIT_DATA_URL = (
    f"https://raw.githubusercontent.com/mingyangmao/KVEdit/{KVEDIT_REVISION}/data/"
)
DEFAULT_DATASET = KVEDIT_DATA_URL + "heldout_pairs.jsonl"
CONDITIONS = ("direct", "derived")

_OFFICIAL_MODELS = {
    "qwen/qwen3-8b": "qwen",
    "meta-llama/llama-3.1-8b-instruct": "llama",
    "mistralai/mistral-7b-instruct-v0.3": "mistral",
}
# Frozen formatting from KVEdit's tools/paired_edit.py.
_OFFICIAL_PROMPTS = {
    "qwen": (
        "<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_end|>",
    ),
    "llama": (
        (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        ),
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|eot_id|>",
    ),
    "mistral": ("<s>[INST] ", "[/INST]", "</s>"),
}
PROMPT_STYLES = ("auto", "plain", "chat", *_OFFICIAL_PROMPTS)


@dataclass(frozen=True)
class Case:
    pair_id: str
    condition: str
    distance: int | None
    question_block: str
    answer_old: str
    answer_new: str
    context_old: str
    context_new: str
    token_metadata: dict


@dataclass(frozen=True)
class PreparedCase:
    old_prefix: list[int]
    new_prefix: list[int]
    suffix: list[int]
    edit_span: tuple[int, int]
    replacement_ids: list[int]
    stop_token_ids: list[int]


def _read_cases(file, conditions, offset, limit):
    lines = (line for line in file if line.strip())
    for line in islice(lines, offset, None if limit is None else offset + limit):
        row = json.loads(line)
        for condition in conditions:
            selected = row["conditions"][condition]
            yield Case(
                pair_id=row["pair_id"],
                condition=condition,
                distance=row.get("d_target"),
                question_block=row["question_block"],
                answer_old=row["answer_old"],
                answer_new=row["answer_new"],
                context_old=selected["context_old"],
                context_new=selected["context_new"],
                token_metadata=selected.get("tok", {}),
            )


def load_cases(
    path: str | Path,
    conditions: Iterable[str] = CONDITIONS,
    offset: int = 0,
    limit: int | None = None,
) -> Iterator[Case]:
    """Yield selected conditions from a local JSONL file or public dataset URL.

    Offset and limit count source pairs before expanding their conditions.
    """
    conditions = tuple(conditions)
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError("Dataset offset and limit must be nonnegative.")
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"Conditions must be chosen from {CONDITIONS}.")
    source = str(path)
    if urlparse(source).scheme in ("http", "https"):
        with (
            urlopen(source, timeout=30) as response,  # noqa: S310
            TextIOWrapper(response, encoding="utf-8") as file,
        ):
            yield from _read_cases(file, conditions, offset, limit)
    else:
        with Path(path).open(encoding="utf-8") as file:
            yield from _read_cases(file, conditions, offset, limit)


def _edit_bounds(old: list[int], new: list[int]) -> tuple[int, int, int]:
    start = next(
        (i for i, (left, right) in enumerate(zip(old, new)) if left != right),
        min(len(old), len(new)),
    )
    old_end, new_end = len(old), len(new)
    while old_end > start and new_end > start and old[old_end - 1] == new[new_end - 1]:
        old_end -= 1
        new_end -= 1
    return start, old_end, new_end


def prepare(case: Case, tokenizer, prompt_style: str = "auto") -> PreparedCase:
    """Tokenize the two document prefixes and their shared question separately."""
    model = str(getattr(tokenizer, "name_or_path", "")).casefold()
    official_style = _OFFICIAL_MODELS.get(model)
    if prompt_style == "auto":
        prompt_style = official_style or (
            "chat" if getattr(tokenizer, "chat_template", None) else "plain"
        )

    if prompt_style in _OFFICIAL_PROMPTS:
        header, assistant, stop = _OFFICIAL_PROMPTS[prompt_style]
        suffix = case.question_block + assistant
        stop_token_ids = tokenizer.encode(stop, add_special_tokens=False)
        if len(stop_token_ids) != 1:
            raise ValueError(f"The {prompt_style} end-of-turn token must encode once.")
    elif prompt_style == "chat":
        # The marker leaves the document boundary independent of the tokenizer's
        # model-specific chat template.
        marker = "__KVEDIT_DOCUMENT_BOUNDARY__"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": marker + "\n\n" + case.question_block}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        header, _, suffix = rendered.partition(marker)
        if not suffix.startswith("\n\n"):
            raise ValueError("The chat template changed the KVEdit document boundary.")
        suffix = suffix[2:]
        stop_token_ids = []
    elif prompt_style == "plain":
        header, suffix, stop_token_ids = "", case.question_block, []
    else:
        raise ValueError(f"Unknown prompt style {prompt_style!r}.")

    old = tokenizer.encode(header + case.context_old + "\n\n", add_special_tokens=False)
    new = tokenizer.encode(header + case.context_new + "\n\n", add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    start, old_end, new_end = _edit_bounds(old, new)
    if start == old_end == new_end:
        raise ValueError(f"{case.pair_id}/{case.condition} has no tokenized edit.")

    metadata = case.token_metadata.get(prompt_style)
    if (
        prompt_style == official_style
        and metadata
        and (
            len(old) != len(new)
            or (len(old), start, old_end)
            != (metadata["prefix_len"], metadata["span_start"], metadata["span_end"])
        )
    ):
        raise ValueError(
            f"{case.pair_id}/{case.condition} differs from KVEdit offsets."
        )

    return PreparedCase(
        old_prefix=old,
        new_prefix=new,
        suffix=suffix_ids,
        edit_span=(start, old_end),
        replacement_ids=new[start:new_end],
        stop_token_ids=stop_token_ids,
    )
