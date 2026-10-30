
import json
from datetime import datetime, timezone
from pathlib import Path
from src.llm_judge import ask_llm_judge
from src.gemini_judge import ask_gemini_judge

from src.evaluator import evaluate_response


# Find the project folder and shared test dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_cases.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

REPORT_PATH = REPORTS_DIR / "comparison_results.json"






all_results = []


def record_result(
    evaluator_name,
    test_case,
    actual_verdict=None,
    reason=None,
    error=None
):
    expected_verdict = (
        "PASS" if test_case["expected_pass"] else "FAIL"
    )

    all_results.append({
        "test_id": test_case["id"],
        "evaluator": evaluator_name,
        "expected_verdict": expected_verdict,
        "actual_verdict": actual_verdict,
        "correct": (
            actual_verdict == expected_verdict
            if actual_verdict is not None
            else None
        ),
        "reason": reason,
        "error": error
    })


def run_evaluator(evaluator_name, test_case, evaluator_function):
    try:
        result = evaluator_function(test_case)
        if result["verdict"] == "ERROR":
            record_result(
                evaluator_name=evaluator_name,
                test_case=test_case,
                error=result.get("reason")
            )

            print(
                f'{test_case["id"]}: '
                f'ERROR - {result.get("reason")}'
            )
            return

        record_result(
            evaluator_name=evaluator_name,
            test_case=test_case,
            actual_verdict=result["verdict"],
            reason=result.get("reason")
        )

        print(
            f'{test_case["id"]}: '
            f'Expected={"PASS" if test_case["expected_pass"] else "FAIL"}, '
            f'Actual={result["verdict"]}'
        )

    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"

        record_result(
            evaluator_name=evaluator_name,
            test_case=test_case,
            error=error_message
        )

        print(f'{test_case["id"]}: ERROR - {error_message}')

def main():
    # Load the test cases from the JSON file.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        test_cases = json.load(file)


    print(f"Total test cases: {len(test_cases)}")
    # 1. Run the rule-based evaluator.
    print("\n--- Rule-Based Evaluator ---")

    for test_case in test_cases:
        result = evaluate_response(test_case)
        verdict = "PASS" if result["passed"] else "FAIL"

        record_result(
            evaluator_name="Rule-based",
            test_case=test_case,
            actual_verdict=verdict,
            reason=(
                f'Missing facts: {result["missing_facts"]}; '
                f'Forbidden claims: {result["detected_forbidden_claims"]}'
            )
        )

        print(f'{test_case["id"]}: Actual={verdict}')


    # 2. Run the local Llama judge.
    print("\n--- Llama Evaluator ---")

    for test_case in test_cases:
        run_evaluator(
            "Llama",
            test_case,
            lambda case: ask_llm_judge(
                context=case["context"],
                chatbot_response=case["actual_output"]
            )
        )


    # # 3. Run the Gemini judge.
    # print("\n--- Gemini Evaluator ---")

    # for test_case in test_cases:
    #     run_evaluator(
    #         "Gemini",
    #         test_case,
    #         lambda case: ask_gemini_judge(
    #             context=case["context"],
    #             chatbot_response=case["actual_output"]
    #         )
    #     )


    # 4. Save all results, including API errors.
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_test_cases": len(test_cases),
        "results": all_results
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nResults saved to: {REPORT_PATH}")
    print(f"Total recorded results: {len(all_results)}")
if __name__ == "__main__":
    main()