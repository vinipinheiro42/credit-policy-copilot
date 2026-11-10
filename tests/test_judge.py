import json

from copilot.evals.judge import LLMJudge, detect_provider, parse_json


def fake(responses):
    calls = []

    def complete(system, user):
        calls.append((system, user))
        return responses.pop(0), {"input_tokens": 100, "output_tokens": 20}

    return complete, calls


def test_faithfulness_from_claims():
    complete, calls = fake([json.dumps({"claims": [{"supported": True}, {"supported": False}]})])
    j = LLMJudge(complete, "fake", "m")
    assert j.faithfulness("A [1]. B [2].", ["ctx a", "ctx b"]) == 0.5
    assert "[1] ctx a" in calls[0][1]
    assert j.usage.input_tokens == 100 and j.usage.calls == 1


def test_correctness_verdicts_and_wrapped_json():
    complete, _ = fake(
        ['Sure! {"verdict": "partial", "score": 0.5, "reason": "x"}', '{"verdict": "correct"}']
    )
    j = LLMJudge(complete, "fake", "m")
    assert j.correctness("q", "a", "ref") == 0.5
    assert j.correctness("q", "a", "ref") == 1.0


def test_judge_errors_are_recorded_not_raised():
    complete, _ = fake(["no json here"])
    j = LLMJudge(complete, "fake", "m")
    assert j.correctness("q", "a", "ref") is None
    assert j.usage.errors


def test_no_key_means_no_judge(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert detect_provider() is None
    assert LLMJudge.from_env() is None
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert detect_provider() == "openai"


def test_parse_json():
    assert parse_json('x {"a": 1} y') == {"a": 1}
