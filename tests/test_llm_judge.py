
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.llm_judge import ask_llm_judge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_cases.json"


def load_evaluation_cases():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

@pytest.mark.llm
@pytest.mark.parametrize(
    "test_case",
    load_evaluation_cases(),
    ids=lambda test_case: test_case["id"]
)
def test_llm_judge(test_case):
    judgement = ask_llm_judge(
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
@patch("src.llm_judge.requests.post")
def test_llm_judge_handles_invalid_json(mock_post):

    mock_response = Mock()
    mock_response.json.return_value = {
        "response": '{"verdict": "PASS", "reason": "Incomplete JSON"'
    }

    mock_post.return_value = mock_response

    judgement = ask_llm_judge(
        context="Refunds are available within 30 days.",
        chatbot_response="Refunds are available within 30 days."
    )

    assert judgement["verdict"] == "ERROR"
    assert judgement["reason"] == "Llama returned an invalid JSON response."
@patch("src.llm_judge.requests.post")
def test_llm_judge_handles_valid_json(mock_post):

    mock_response = Mock()
    mock_response.json.return_value = {
        "response": '{"verdict": "PASS", "reason": "Response is supported."}'
    }

    mock_post.return_value = mock_response

    judgement = ask_llm_judge(
        context="Refunds are available within 30 days.",
        chatbot_response="Refunds are available within 30 days."
    )

    assert judgement["verdict"] == "PASS"
    assert judgement["reason"] == "Response is supported."