"""Judge tests: real LangChain/TypeSafe code paths against fake transports, no network."""

import json

import httpx2
import pytest
from langchain_core.messages import AIMessage
from langchain_typesafe import TypeSafeClassifier

from jevjudge.judges import (
    BINARY_SCHEMA,
    CLAUDE_SPECS,
    ClaudeJudge,
    JevJudge,
    claude_chat_kwargs,
    parse_digit,
    parse_yes_no,
)
from jevjudge.rubrics import CRITERIA, LEVELS, binary_state, state_text
from jevjudge.schema import BinaryItem, RubricItem


@pytest.fixture
def bitem():
    return BinaryItem(id="b1", group="g", source="halueval", question="Capital of France?",
                      candidate="Paris", label=1, knowledge="Paris is the capital of France.")


@pytest.fixture
def ritem():
    return RubricItem(id="r1", source="helpsteer2", prompt="Hi", response="Hello!",
                      labels={c: 3 for c in CRITERIA})


def make_jev(handler):
    """A real TypeSafeClassifier whose HTTP client is a mock transport."""
    client = httpx2.Client(transport=httpx2.MockTransport(handler), base_url="https://example.invalid")
    return JevJudge(model="jev-1.13.0", classifier=TypeSafeClassifier(model="jev-1.13.0", api_key="k", client=client))


def test_jev_binary_request_and_parse(bitem):
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={"model": "jev-1.13.0", "answers": {"correct": {"type": "noul", "noul": 0.93}},
                  "usage": {"input_tokens": 120, "output_tokens": 1}},
        )

    res = make_jev(handler).judge_binary(bitem)
    body = seen["body"]
    assert body["model"] == "jev-1.13.0"  # the version pin reaches the wire
    assert list(body["questions"]) == ["correct"]
    assert body["questions"]["correct"]["type"] == "noul"
    assert body["state"] == binary_state(bitem)  # same information Claude receives
    assert res.p_correct == pytest.approx(0.93)
    assert (res.input_tokens, res.output_tokens, res.served_model, res.error) == (120, 1, "jev-1.13.0", None)
    (row,) = res.calls  # one request -> one logged row holding Jev's full raw answer
    assert row["criterion"] == "correct" and row["parsed"] == pytest.approx(0.93)
    assert json.loads(row["raw_output"]) == {"correct": {"type": "noul", "noul": 0.93}}


def test_jev_rubric_one_request_three_scores(ritem):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        ans = {
            c: {"type": "score", "score": 2.5, "legend": {str(i): "x" for i in range(LEVELS)},
                "probabilities": {"2": 0.5, "3": 0.5}, "confidence": 0.5}
            for c in CRITERIA
        }
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": ans, "usage": {}})

    res = make_jev(handler).judge_rubric(ritem)
    assert set(seen["body"]["questions"]) == set(CRITERIA)
    assert all(len(q["criteria"]) == LEVELS for q in seen["body"]["questions"].values())
    assert res.scores == {c: 2.5 for c in CRITERIA}
    assert res.probs[CRITERIA[0]] == [0.0, 0.0, 0.5, 0.5, 0.0]
    (row,) = res.calls
    assert row["criterion"] == "all" and json.loads(row["raw_output"])[CRITERIA[0]]["probabilities"] == {"2": 0.5, "3": 0.5}


def test_jev_retries_5xx_but_not_4xx(bitem, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx2.Response(503, json={"error": "busy"})
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": {"correct": {"type": "noul", "noul": 0.1}}})

    res = make_jev(flaky).judge_binary(bitem)
    assert res.p_correct == pytest.approx(0.1) and res.retries == 2

    calls["n"] = 0
    bad = make_jev(lambda r: (calls.__setitem__("n", calls["n"] + 1), httpx2.Response(400, json={}))[1])
    res = bad.judge_binary(bitem)
    assert calls["n"] == 1 and res.error and res.p_correct is None


# ------------------------------------------------------------------------- Claude


class FakeLLM:
    def __init__(self, texts, stop_reason="end_turn"):
        self.texts, self.stop_reason, self.calls = list(texts), stop_reason, []

    def invoke(self, messages):
        self.calls.append(messages)
        text = self.texts.pop(0) if self.texts else "YES"
        return AIMessage(content=[{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
                         response_metadata={"stop_reason": self.stop_reason, "model_name": "claude-sonnet-5-5"},
                         usage_metadata={"input_tokens": 50, "output_tokens": 2, "total_tokens": 52})


def test_claude_binary_votes_to_probability(bitem):
    j = ClaudeJudge(CLAUDE_SPECS["sonnet"], binary_samples=3, fanout=1, llm=FakeLLM(["YES", "no.", "YES"]))
    res = j.judge_binary(bitem)
    assert res.p_correct == pytest.approx(2 / 3) and res.n_valid == 3
    assert res.input_tokens == 150 and res.output_tokens == 6 and res.n_calls == 3
    # every call is logged with its raw text and how it was parsed
    assert [(r["sample"], r["raw_output"], r["parsed"]) for r in res.calls] == [(0, "YES", 1), (1, "no.", 0), (2, "YES", 1)]
    assert all(r["criterion"] == "correct" and r["output_tokens"] == 2 for r in res.calls)
    sys_msg, human = j.llm_binary.calls[0]
    assert human.content == state_text(binary_state(bitem))  # identical state to Jev's


def test_claude_refusal_and_garbage_are_errors(bitem):
    r1 = ClaudeJudge(CLAUDE_SPECS["haiku"], fanout=1, llm=FakeLLM(["YES"], stop_reason="refusal")).judge_binary(bitem)
    assert r1.p_correct is None and r1.error.startswith("refusal")
    r2 = ClaudeJudge(CLAUDE_SPECS["haiku"], fanout=1, llm=FakeLLM(["I think it might be right"])).judge_binary(bitem)
    assert r2.p_correct is None and r2.error


def test_claude_rubric_expected_score(ritem):
    # 3 criteria x 5 samples; first criterion all 4s, second 2/3 split, third has one garbage reply
    texts = ["4"] * 5 + ["2", "2", "3", "3", "3"] + ["1", "x", "1", "1", "1"]
    res = ClaudeJudge(CLAUDE_SPECS["sonnet"], rubric_samples=5, fanout=1, llm=FakeLLM(texts)).judge_rubric(ritem)
    assert res.scores[CRITERIA[0]] == 4.0
    assert res.scores[CRITERIA[1]] == pytest.approx(2.6)
    assert res.scores[CRITERIA[2]] == 1.0  # garbage sample dropped, not counted as 0
    assert res.probs[CRITERIA[1]] == [0, 0, 0.4, 0.6, 0]
    assert res.error is None
    assert len(res.calls) == 15 and [r["criterion"] for r in res.calls[::5]] == CRITERIA
    assert res.calls[11]["raw_output"] == "x" and res.calls[11]["parsed"] == ""  # garbage kept raw, not parsed


def test_parsers():
    assert parse_yes_no("YES") == 1 and parse_yes_no("  No.") == 0 and parse_yes_no("Yesterday") is None
    assert parse_digit("3") == 3 and parse_digit(" 4\n") == 4 and parse_digit("5") is None and parse_digit("12") is None


def test_parsers_read_the_structured_output():
    assert parse_yes_no('{"verdict": "YES"}') == 1 and parse_yes_no('{"verdict":"NO"}') == 0
    assert parse_yes_no('{"verdict": "MAYBE"}') is None and parse_yes_no("Micah ran 3.5 × 8 = ") is None
    assert parse_digit('{"score": "3"}') == 3 and parse_digit('{"score":"0"}') == 0 and parse_digit('{"score": "7"}') is None


@pytest.mark.parametrize("key,mode,thinking,effort,max_tokens", [
    ("sonnet", "system1", {"type": "between_tools"}, "low", 32),
    ("haiku", "system1", {"type": "disabled"}, "low", 32),
    ("sonnet", "reasoning", {"type": "adaptive", "display": "summarized"}, "max", 16000),
    ("haiku", "reasoning", {"type": "adaptive", "display": "summarized"}, "max", 16000),
])
def test_claude_request_payload(monkeypatch, key, mode, thinking, effort, max_tokens):
    """The real ChatAnthropic payload: thinking/effort set, and no temperature (rejected by 5.5 models)."""
    from langchain_anthropic import ChatAnthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    spec = CLAUDE_SPECS[key]
    schema = BINARY_SCHEMA
    llm = ChatAnthropic(model=spec.model_id, **claude_chat_kwargs(spec, mode, schema))
    payload = llm._get_request_payload([("system", "s"), ("human", "h")])
    assert payload["model"] == spec.model_id
    assert payload["thinking"] == thinking
    assert payload["output_config"]["effort"] == effort
    assert payload["max_tokens"] == max_tokens
    assert "temperature" not in payload and "top_p" not in payload and "top_k" not in payload
    # both arms force a {"verdict": YES|NO} reply, so reasoning can only happen in the hidden thinking blocks
    assert payload["output_config"].get("format") == {"type": "json_schema", "schema": BINARY_SCHEMA}


def test_thinking_summary_is_logged_per_call(bitem):
    class ThinkingLLM(FakeLLM):
        def invoke(self, messages):
            msg = super().invoke(messages)
            msg.content = [{"type": "thinking", "thinking": "Paris is the capital, so YES."}, *msg.content[1:]]
            return msg

    res = ClaudeJudge(CLAUDE_SPECS["sonnet"], mode="reasoning", fanout=1, llm=ThinkingLLM(["YES"])).judge_binary(bitem)
    (row,) = res.calls
    assert row["thinking"] == "Paris is the capital, so YES." and row["raw_output"] == "YES" and row["parsed"] == 1
