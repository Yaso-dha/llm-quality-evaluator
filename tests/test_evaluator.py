
import json
from pathlib import Path

import pytest

from src.evaluator import evaluate_response


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_cases.json"


def load_evaluation_cases():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.mark.parametrize(
    "test_case",
    load_evaluation_cases(),
    ids=lambda test_case: test_case["id"]
)
def test_chatbot_response_quality(test_case):
    result = evaluate_response(test_case)

    assert result["passed"] == test_case["expected_pass"], (
        f"Test ID: {result['test_id']}\n"
        f"Missing facts: {result['missing_facts']}\n"
        f"Forbidden claims: "
        f"{result['detected_forbidden_claims']}"
    )
