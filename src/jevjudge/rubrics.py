"""Judging instructions and anchored rubrics.

Every judge (Jev and Claude) is given the same wording and the same information, so
differences in results come from the judge and not from the prompt. Jev receives these
as typed questions (``Noul`` / ``Score``); Claude receives the same text in a system
prompt and the same JSON state as the user message.
"""

from __future__ import annotations

import json
from typing import Any

from .schema import BinaryItem, RubricItem

# --------------------------------------------------------------------------- binary

BINARY_TRUE = "The candidate answer is correct."
BINARY_FALSE = (
    "The candidate answer is incorrect: it is wrong, contradicts the provided evidence, "
    "or does not do what the question asks."
)

_COMMON = " Judge correctness only, not style, length, or how confident the answer sounds."

BINARY_INSTRUCTIONS: dict[str, str] = {
    "gsm8k": "Is the candidate's final numeric answer to this math word problem correct?" + _COMMON,
    "mbpp": (
        "Is the candidate Python code a correct implementation of the described function, "
        "such that it would pass all of its unit tests? The example test is one of several."
        + _COMMON
    ),
    "halueval": (
        "Using the provided knowledge as ground truth, is the candidate answer to the "
        "question correct (not hallucinated)?" + _COMMON
    ),
    "private": "Is the candidate answer to the question correct?" + _COMMON,
}


def binary_instruction(item: BinaryItem) -> str:
    return BINARY_INSTRUCTIONS.get(item.source, BINARY_INSTRUCTIONS["private"])


def binary_state(item: BinaryItem) -> dict[str, Any]:
    """The information a judge sees for one binary item (identical for all judges)."""
    state: dict[str, Any] = {"question": item.question}
    if item.knowledge:
        state["knowledge"] = item.knowledge
    if item.example_test:
        state["example_test"] = item.example_test
    state["candidate_answer"] = item.candidate
    return state


# --------------------------------------------------------------------------- rubric

LEVELS = 5  # scores 0..4

RUBRIC: dict[str, dict[str, Any]] = {
    "helpfulness": {
        "instruction": "How helpful is the response to the user's prompt?",
        "anchors": [
            "Not helpful at all: ignores the request or is useless or misleading.",
            "Barely helpful: addresses the request only superficially or in a largely unusable way.",
            "Somewhat helpful: partly addresses the request, but important parts are missing or unclear.",
            "Helpful: addresses the request well, with only minor gaps.",
            "Very helpful: fully addresses the request; clear, actionable and complete.",
        ],
    },
    "correctness": {
        "instruction": "How factually and logically correct is the response?",
        "anchors": [
            "Completely incorrect: the main claims or content are wrong or fabricated.",
            "Mostly incorrect: some correct pieces, but major errors dominate.",
            "Partially correct: a mix of correct content and important errors or omissions.",
            "Mostly correct: only minor errors or omissions.",
            "Completely correct: accurate, with no meaningful errors or omissions.",
        ],
    },
    "coherence": {
        "instruction": "How coherent and clear is the response?",
        "anchors": [
            "Incoherent: contradictory, garbled, or impossible to follow.",
            "Poorly coherent: frequent contradictions, abrupt jumps, or confusing wording.",
            "Somewhat coherent: understandable, but with noticeable inconsistencies or awkward flow.",
            "Coherent: clear and consistent, with minor lapses.",
            "Perfectly coherent: clear, consistent and well organised, with no contradictions.",
        ],
    },
}

CRITERIA = list(RUBRIC)


def rubric_state(item: RubricItem) -> dict[str, Any]:
    return {"prompt": item.prompt, "response": item.response}


def rubric_scale_text(criterion: str) -> str:
    return "\n".join(f"{i}: {a}" for i, a in enumerate(RUBRIC[criterion]["anchors"]))


# ------------------------------------------------------------------ Claude prompts

CLAUDE_BINARY_SYSTEM_TEMPLATE = (
    "{instruction}\n\n"
    "YES means: {true}\n"
    "NO means: {false}\n\n"
    "The input is a JSON object. Reply with exactly one word, YES or NO. Do not explain."
)

CLAUDE_RUBRIC_SYSTEM_TEMPLATE = (
    "You are rating a response on one criterion: {criterion}.\n"
    "{instruction}\n\n"
    "Scale:\n{scale}\n\n"
    "The input is a JSON object. Reply with a single digit from 0 to 4 and nothing else."
)


def claude_binary_system(item: BinaryItem) -> str:
    return CLAUDE_BINARY_SYSTEM_TEMPLATE.format(
        instruction=binary_instruction(item), true=BINARY_TRUE, false=BINARY_FALSE
    )


def claude_rubric_system(criterion: str) -> str:
    return CLAUDE_RUBRIC_SYSTEM_TEMPLATE.format(
        criterion=criterion,
        instruction=RUBRIC[criterion]["instruction"],
        scale=rubric_scale_text(criterion),
    )


def state_text(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, ensure_ascii=False)
