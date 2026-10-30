
import json
from pathlib import Path

import pytest

from src.gemini_judge import ask_gemini_judge


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
def test_gemini_judge(test_case):
    judgement = ask_gemini_judge(
        context=test_case["context"],
        chatbot_response=test_case["actual_output"]
    )

    expected_verdict = (
        "PASS" if test_case["expected_pass"] else "FAIL"
    )

    assert judgement["verdict"] == expected_verdict, (
        f"Test ID: {test_case['id']}\n"
        f"Expected: {expected_verdict}\n"
        f"Actual: {judgement['verdict']}\n"
        f"Reason: {judgement['reason']}"
    )
