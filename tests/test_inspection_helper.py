import math

import pytest

from util.inspection_helper import parse_lambda, parse_params, parse_val, resolve_name


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12", 12),
        ("0.25", 0.25),
        ("True", True),
        ("None", None),
        ("[64, 32]", [64, 32]),
        ("{'a': 1}", {"a": 1}),
        ("MlpPolicy", "MlpPolicy"),
        ("[broken", "[broken"),
        (7, 7),
    ],
)
def test_parse_val(value, expected):
    assert parse_val(value) == expected


def test_learning_rate_schedule():
    schedule = parse_lambda("lambda progress: 0.001 * progress ** 2 + 0.0001")
    assert schedule(0) == pytest.approx(0.0001)
    assert schedule(0.5) == pytest.approx(0.00035)
    assert schedule(1) == pytest.approx(0.0011)


@pytest.mark.parametrize(
    "expression",
    [
        "lambda x: __import__('os')",
        "lambda x: x.real",
        "lambda x: x[0]",
        "lambda x: [v for v in x]",
        "lambda x: x if x else 1",
    ],
)
def test_lambda_rejects_unsupported_syntax(expression):
    with pytest.raises(ValueError, match="Unsupported expression"):
        parse_lambda(expression)


def test_resolves_qualified_names():
    assert resolve_name("math.sqrt") is math.sqrt
    assert parse_lambda("math.sqrt") is math.sqrt


def test_unresolved_name_remains_a_string():
    assert parse_lambda("auto") == "auto"


def test_nested_model_parameters_do_not_mutate_input():
    raw = {
        "learning_rate": "lambda p: p * 0.01",
        "policy_kwargs": {"activation_fn": "math.sqrt"},
        "seed": 42,
    }
    parsed = parse_params(raw)
    assert parsed["learning_rate"](0.5) == pytest.approx(0.005)
    assert parsed["policy_kwargs"]["activation_fn"] is math.sqrt
    assert parsed["seed"] == 42
    assert raw["policy_kwargs"]["activation_fn"] == "math.sqrt"
