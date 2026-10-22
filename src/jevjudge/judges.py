"""Judges: Jev (TypeSafe) and Claude, both called through LangChain.

All judges return raw probabilities / expected scores; a binary verdict is P(correct) >= 0.5.

Claude notes (Sonnet 5.5 / Haiku 5.5):
  * Non-default ``temperature`` is rejected (400), so every call uses the default
    sampling. Averaging several samples is the only way to get a continuous score
    because the API exposes no logprobs.
  * Thinking is on by default. The ``system1`` mode switches it off (matched compute
    with Jev: one forward pass, a one-token answer); ``reasoning`` mode turns adaptive
    thinking on at effort "max". Both modes use the same prompt and the same structured
    reply, so they differ only in whether the model thinks.
  * Refusals are recorded as errors. Server-side fallbacks are deliberately NOT
    enabled: they would silently swap a different model into the comparison.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .rubrics import (
    BINARY_FALSE,
    BINARY_TRUE,
    CRITERIA,
    LEVELS,
    RUBRIC,
    binary_instruction,
    binary_state,
    claude_binary_system,
    claude_rubric_system,
    rubric_state,
    state_text,
)
from .schema import BinaryItem, RubricItem

DEFAULT_JEV_MODEL = "jev-1.13.0"


# --------------------------------------------------------------------------- results


@dataclass
class CallStats:
    latency_s: float = 0.0  # wall clock for the whole item (parallel calls overlap)
    input_tokens: int | None = None
    output_tokens: int | None = None
    n_calls: int = 1
    retries: int = 0
    served_model: str | None = None
    error: str | None = None
    # One dict per API call (raw output etc.); the runner writes these to results/calls.csv
    calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BinaryResult(CallStats):
    p_correct: float | None = None  # P(candidate is correct)
    n_valid: int = 1  # parsed samples (Claude); 1 for Jev


@dataclass
class RubricResult(CallStats):
    scores: dict[str, float | None] = field(default_factory=dict)  # expected score 0..4
    probs: dict[str, list[float] | None] = field(default_factory=dict)  # per-level probs


class Judge:
    name: str

    def judge_binary(self, item: BinaryItem) -> BinaryResult:
        raise NotImplementedError

    def judge_rubric(self, item: RubricItem) -> RubricResult:
        raise NotImplementedError


def _call_row(
    criterion: str = "",
    sample: int = 0,
    raw: str = "",
    parsed: Any = "",
    stop_reason: str = "",
    input_tokens: int | None = 0,
    output_tokens: int | None = 0,
    latency_s: float = 0.0,
    error: str | None = None,
    served_model: str | None = None,
    thinking: str = "",
) -> dict[str, Any]:
    """One API call as a flat row for results/calls.csv (the raw output plus how it was parsed)."""
    return {
        "criterion": criterion,
        "sample": sample,
        "raw_output": raw,
        "parsed": "" if parsed is None else parsed,
        "stop_reason": stop_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": round(latency_s, 4),
        "error": error or "",
        "served_model": served_model or "",
        "thinking": thinking,
    }


def _answers_json(resp: Any) -> str:
    return json.dumps({k: v.model_dump() for k, v in resp.answers.items()})


# ------------------------------------------------------------------------------- Jev


class JevJudge(Judge):
    """TypeSafe Jev through ``langchain_typesafe.TypeSafeClassifier``."""

    max_attempts = 4

    def __init__(self, model: str = DEFAULT_JEV_MODEL, classifier: Any = None, timeout: float = 60.0):
        self.model = model
        self.name = model
        if classifier is None:
            from langchain_typesafe import TypeSafeClassifier

            classifier = TypeSafeClassifier(model=model, timeout=timeout)
        self.classifier = classifier

    @staticmethod
    def _retry_delay(error: Exception, attempt: int) -> float | None:
        """Seconds to wait before retrying, or None if the error is not retryable."""
        from langchain_typesafe.client import (
            TypeSafeAPIConnectionError,
            TypeSafeInternalServerError,
            TypeSafeRateLimitError,
        )

        if isinstance(error, TypeSafeRateLimitError):
            return (error.retry_after_ms / 1000) if error.retry_after_ms is not None else 2.0**attempt
        if isinstance(error, (TypeSafeInternalServerError, TypeSafeAPIConnectionError)):
            return 2.0**attempt
        return None  # 4xx and validation errors will not succeed on retry

    def _invoke(self, payload: dict[str, Any]) -> tuple[Any, float, int]:
        """Returns (response, latency of the successful attempt, number of retries)."""
        for attempt in range(self.max_attempts):
            t0 = time.perf_counter()
            try:
                resp = self.classifier.invoke(payload)
                return resp, time.perf_counter() - t0, attempt
            except Exception as e:  # noqa: BLE001
                delay = self._retry_delay(e, attempt)
                if delay is None or attempt == self.max_attempts - 1:
                    raise
                time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _usage(resp: Any) -> tuple[int | None, int | None]:
        usage = getattr(resp, "usage", None)
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)

    def judge_binary(self, item: BinaryItem) -> BinaryResult:
        from langchain_typesafe import Noul, NoulCriteria

        payload = {
            "state": binary_state(item),
            "questions": {
                "correct": Noul(
                    instructions=binary_instruction(item),
                    criteria=NoulCriteria(true=BINARY_TRUE, false=BINARY_FALSE),
                )
            },
        }
        try:
            resp, latency, retries = self._invoke(payload)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            return BinaryResult(error=err, n_valid=0, calls=[_call_row(criterion="correct", error=err)])
        in_tok, out_tok = self._usage(resp)
        p = float(resp.nouls["correct"].noul)
        return BinaryResult(
            p_correct=p,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            retries=retries,
            served_model=resp.model,
            calls=[_call_row(criterion="correct", raw=_answers_json(resp), parsed=p, input_tokens=in_tok,
                             output_tokens=out_tok, latency_s=latency, served_model=resp.model)],
        )

    def judge_rubric(self, item: RubricItem) -> RubricResult:
        from langchain_typesafe import Score

        payload = {
            "state": rubric_state(item),
            "questions": {
                c: Score(instructions=RUBRIC[c]["instruction"], criteria=list(RUBRIC[c]["anchors"]))
                for c in CRITERIA
            },
        }
        try:
            resp, latency, retries = self._invoke(payload)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            return RubricResult(error=err, calls=[_call_row(criterion="all", error=err)])
        in_tok, out_tok = self._usage(resp)
        scores: dict[str, float | None] = {}
        probs: dict[str, list[float] | None] = {}
        for c in CRITERIA:
            ans = resp.scores[c]
            scores[c] = float(ans.score)
            probs[c] = [float(ans.probabilities.get(i, 0.0)) for i in range(LEVELS)]
        return RubricResult(
            scores=scores,
            probs=probs,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            n_calls=1,
            retries=retries,
            served_model=resp.model,
            calls=[_call_row(criterion="all", raw=_answers_json(resp), parsed=json.dumps(scores), input_tokens=in_tok,
                             output_tokens=out_tok, latency_s=latency, served_model=resp.model)],
        )


# ----------------------------------------------------------------------------- Claude


@dataclass(frozen=True)
class ClaudeSpec:
    model_id: str
    thinking_off: dict[str, str]  # how to switch thinking off on this model


CLAUDE_SPECS: dict[str, ClaudeSpec] = {
    # Haiku 5.5 accepts thinking={"type": "disabled"} at effort high or below.
    "haiku": ClaudeSpec("claude-haiku-5-5", {"type": "disabled"}),
    # Sonnet 5.5 rejects "disabled" (400); "between_tools" is its off switch (effort <= high).
    "sonnet": ClaudeSpec("claude-sonnet-5-5", {"type": "between_tools"}),
}


# Both arms are forced into a tiny JSON object with structured outputs. Asking for "one word" in the
# prompt is not enough: without thinking, the models start solving the problem in the visible reply and
# run out of tokens (~35-60% of binary answers were unparseable). With thinking on, free text let Sonnet
# write its working in the reply, where it could steer the verdict. The schema keeps all reasoning in the
# hidden thinking blocks, so the two arms differ only in whether thinking is enabled.
BINARY_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["YES", "NO"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}
RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "string", "enum": [str(i) for i in range(LEVELS)]}},
    "required": ["score"],
    "additionalProperties": False,
}


def claude_chat_kwargs(spec: ClaudeSpec, mode: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    if mode not in ("system1", "reasoning"):
        raise ValueError(f"unknown mode {mode!r}")
    # Reasoning uses effort "max". Adaptive thinking lets the model decide how much to think, and the 5.5
    # models reject a fixed thinking budget, so effort is the only dial. In a small probe (8 binary items)
    # Sonnet 5.5 thought on 2 at "high", 5 at "xhigh" and all 8 at "max"; Haiku 5.5 thinks at "high" already.
    # "max" is used for both so the reasoning arm really reasons.
    output_config: dict[str, Any] = {"effort": "low" if mode == "system1" else "max"}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    if mode == "system1":
        return {"max_tokens": 32, "thinking": dict(spec.thinking_off), "output_config": output_config}
    # display "summarized" returns a readable summary of the thinking, which is logged to calls.csv
    return {"max_tokens": 16000, "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": output_config}


def build_claude_llm(spec: ClaudeSpec, mode: str, schema: dict[str, Any] | None = None) -> Any:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=spec.model_id,
        max_retries=6,
        timeout=180.0,
        **claude_chat_kwargs(spec, mode, schema),
    )


@dataclass
class _Call:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    served_model: str | None = None
    stop_reason: str = ""
    thinking: str = ""
    error: str | None = None


_YESNO = re.compile(r"\W*(YES|NO)\b", re.IGNORECASE)
_DIGIT = re.compile(r"\s*([0-4])\b")


def _json_field(text: str, key: str) -> str | None:
    try:
        return str(json.loads(text)[key])
    except (ValueError, KeyError, TypeError):
        return None


def parse_yes_no(text: str) -> int | None:
    """Reads {"verdict": "YES"} (system1) or a plain YES / NO reply (reasoning)."""
    m = _YESNO.match(_json_field(text, "verdict") or text)
    if not m:
        return None
    return 1 if m.group(1).upper() == "YES" else 0


def parse_digit(text: str) -> int | None:
    """Reads {"score": "3"} (system1) or a plain digit reply (reasoning)."""
    m = _DIGIT.match(_json_field(text, "score") or text)
    return int(m.group(1)) if m else None


def _message_text(msg: Any) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _message_thinking(msg: Any) -> str:
    """The thinking summary blocks of a reply (empty when thinking is off or was skipped)."""
    content = msg.content
    if isinstance(content, str):
        return ""
    return "\n".join(b.get("thinking", "") for b in content if isinstance(b, dict) and b.get("type") == "thinking")


class ClaudeJudge(Judge):
    def __init__(
        self,
        spec: ClaudeSpec,
        mode: str = "system1",
        binary_samples: int = 1,
        rubric_samples: int = 5,
        fanout: int = 64,
        llm: Any = None,
    ):
        self.spec = spec
        self.mode = mode
        self.binary_samples = binary_samples
        self.rubric_samples = rubric_samples
        self.name = f"{spec.model_id}:{mode}"
        if llm is not None:  # injected (tests): used for both tasks
            self.llm_binary = self.llm_rubric = llm
        else:
            self.llm_binary = build_claude_llm(spec, mode, BINARY_SCHEMA)
            self.llm_rubric = build_claude_llm(spec, mode, RUBRIC_SCHEMA)
        # Dedicated pool for sample fan-out; sized so calls never queue behind each other,
        # which would inflate measured latency.
        self._pool = ThreadPoolExecutor(max_workers=fanout)

    def _call(self, llm: Any, system: str, user: str) -> _Call:
        from langchain_core.messages import HumanMessage, SystemMessage

        t0 = time.perf_counter()
        try:
            msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as e:  # noqa: BLE001
            return _Call(latency_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
        latency = time.perf_counter() - t0
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = getattr(msg, "usage_metadata", None) or {}
        stop = meta.get("stop_reason")
        err = None
        if stop == "refusal":
            err = f"refusal:{(meta.get('stop_details') or {}).get('category')}"
        text = _message_text(msg)
        if not text and err is None:
            err = f"empty_response(stop={stop})"
        return _Call(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            latency_s=latency,
            served_model=meta.get("model_name") or meta.get("model"),
            stop_reason=stop or "",
            thinking=_message_thinking(msg),
            error=err,
        )

    def _fan_out(self, jobs: list[tuple[Any, str, str]]) -> tuple[list[_Call], float]:
        t0 = time.perf_counter()
        futures = [self._pool.submit(self._call, *job) for job in jobs]
        calls = [f.result() for f in futures]
        return calls, time.perf_counter() - t0

    @staticmethod
    def _stats(calls: list[_Call], wall: float) -> dict[str, Any]:
        served = next((c.served_model for c in calls if c.served_model), None)
        return {
            "latency_s": wall,
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "n_calls": len(calls),
            "served_model": served,
        }

    @staticmethod
    def _row(call: _Call, criterion: str, sample: int, parsed: Any) -> dict[str, Any]:
        return _call_row(criterion, sample, call.text, parsed, call.stop_reason, call.input_tokens,
                         call.output_tokens, call.latency_s, call.error, call.served_model, call.thinking)

    def judge_binary(self, item: BinaryItem) -> BinaryResult:
        system = claude_binary_system(item)
        user = state_text(binary_state(item))
        calls, wall = self._fan_out([(self.llm_binary, system, user)] * self.binary_samples)
        parsed = [parse_yes_no(c.text) if c.error is None else None for c in calls]
        rows = [self._row(c, "correct", n, v) for n, (c, v) in enumerate(zip(calls, parsed))]
        votes = [v for v in parsed if v is not None]
        stats = self._stats(calls, wall)
        if not votes:
            errs = [c.error for c in calls if c.error] or ["unparseable"]
            return BinaryResult(error=errs[0], n_valid=0, calls=rows, **stats)
        return BinaryResult(p_correct=sum(votes) / len(votes), n_valid=len(votes), calls=rows, **stats)

    def judge_rubric(self, item: RubricItem) -> RubricResult:
        user = state_text(rubric_state(item))
        jobs = [(self.llm_rubric, claude_rubric_system(c), user) for c in CRITERIA for _ in range(self.rubric_samples)]
        calls, wall = self._fan_out(jobs)
        stats = self._stats(calls, wall)
        scores: dict[str, float | None] = {}
        probs: dict[str, list[float] | None] = {}
        errors: list[str] = []
        rows: list[dict[str, Any]] = []
        for i, c in enumerate(CRITERIA):
            chunk = calls[i * self.rubric_samples : (i + 1) * self.rubric_samples]
            parsed = [parse_digit(x.text) if x.error is None else None for x in chunk]
            rows += [self._row(x, c, n, v) for n, (x, v) in enumerate(zip(chunk, parsed))]
            digits = [d for d in parsed if d is not None]
            if not digits:
                scores[c], probs[c] = None, None
                errors.append(f"{c}:{next((x.error for x in chunk if x.error), 'unparseable')}")
                continue
            scores[c] = sum(digits) / len(digits)
            probs[c] = [digits.count(k) / len(digits) for k in range(LEVELS)]
        return RubricResult(scores=scores, probs=probs, error="; ".join(errors) or None, calls=rows, **stats)


# ---------------------------------------------------------------------------- registry

JudgeFactory = Callable[..., Judge]


def make_judge(
    key: str,
    jev_model: str = DEFAULT_JEV_MODEL,
    binary_samples: int = 1,
    rubric_samples: int = 5,
    fanout: int = 64,
) -> Judge:
    """Build a judge from a CLI key: ``jev``, ``haiku``, ``sonnet``, ``haiku-reasoning`` ..."""
    if key == "jev":
        return JevJudge(model=jev_model)
    base, _, suffix = key.partition("-")
    if base in CLAUDE_SPECS and suffix in ("", "reasoning"):
        return ClaudeJudge(
            CLAUDE_SPECS[base],
            mode="reasoning" if suffix else "system1",
            binary_samples=binary_samples,
            rubric_samples=rubric_samples,
            fanout=fanout,
        )
    raise ValueError(f"unknown judge {key!r}")
