
import json
from pathlib import Path
from itertools import combinations


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = PROJECT_ROOT / "reports" / "comparison_results.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "comparison_summary.json"


# Load the saved evaluator results.
with RESULTS_PATH.open("r", encoding="utf-8") as file:
    report = json.load(file)

print(f"Loaded {len(report['results'])} recorded results.")

results = report["results"]
summary = {}

# Analyse each evaluator separately.
evaluator_names = sorted({
    result["evaluator"] for result in results
})

for evaluator_name in evaluator_names:

    evaluator_results = [
        result for result in results
        if result["evaluator"] == evaluator_name
    ]

    completed_results = [
        result for result in evaluator_results
        if result["actual_verdict"] is not None
    ]

    # Count correct and incorrect verdicts.
    correct_count = sum(
        result["correct"] is True
        for result in completed_results
    )

    incorrect_count = sum(
        result["correct"] is False
        for result in completed_results
    )

    # Count false passes and false fails.
    false_passes = sum(
        result["expected_verdict"] == "FAIL"
        and result["actual_verdict"] == "PASS"
        for result in completed_results
    )

    false_fails = sum(
        result["expected_verdict"] == "PASS"
        and result["actual_verdict"] == "FAIL"
        for result in completed_results
    )

    # Count API or runtime errors.
    error_count = sum(
        result["error"] is not None
        for result in evaluator_results
    )

    # Calculate accuracy.
    accuracy = (
        round(correct_count / len(completed_results) * 100, 2)
        if completed_results else None
    )

    # Calculate confusion matrix values.
    true_positives = correct_count - false_fails
    false_positives = false_passes
    false_negatives = false_fails

    # Calculate precision.
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0
    )

    # Calculate recall.
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0
    )

    # Calculate F1-score.
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    # Store evaluator metrics.
    summary[evaluator_name] = {
        "cases_attempted": len(evaluator_results),
        "completed_verdicts": len(completed_results),
        "correct_verdicts": correct_count,
        "incorrect_verdicts": incorrect_count,
        "accuracy_percent": accuracy,
        "false_passes": false_passes,
        "false_fails": false_fails,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_score_percent": round(f1_score * 100, 2),
        "api_or_runtime_errors": error_count
    }


# Compare evaluator verdicts pair by pair.
agreement_results = {}

for evaluator_a, evaluator_b in combinations(evaluator_names, 2):

    compared_cases = 0
    agreements = 0

    for test_case in report["results"]:

        if test_case["evaluator"] != evaluator_a:
            continue

        test_id = test_case["test_id"]

        matching_result = next(
            (
                result for result in report["results"]
                if result["evaluator"] == evaluator_b
                and result["test_id"] == test_id
            ),
            None
        )

        # Skip cases where either evaluator returned no verdict.
        if (
            test_case["actual_verdict"] is None
            or matching_result is None
            or matching_result["actual_verdict"] is None
        ):
            continue

        compared_cases += 1

        if (
            test_case["actual_verdict"]
            == matching_result["actual_verdict"]
        ):
            agreements += 1

    agreement_percent = (
        round(agreements / compared_cases * 100, 2)
        if compared_cases else None
    )

    pair_name = f"{evaluator_a} vs {evaluator_b}"

    agreement_results[pair_name] = {
        "compared_cases": compared_cases,
        "agreements": agreements,
        "disagreements": compared_cases - agreements,
        "agreement_percent": agreement_percent
    }


# Add evaluator agreement to the summary.
summary["evaluator_agreement"] = agreement_results


# Save the complete analysis.
with SUMMARY_PATH.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


# Display evaluator metrics in the terminal.
print("\n--- Evaluator Comparison Summary ---")

for evaluator_name, metrics in summary.items():

    if evaluator_name == "evaluator_agreement":
        continue

    print(f"\n{evaluator_name}")
    print(f"  Accuracy: {metrics['accuracy_percent']}%")
    print(f"  Correct verdicts: {metrics['correct_verdicts']}")
    print(f"  False passes: {metrics['false_passes']}")
    print(f"  False fails: {metrics['false_fails']}")
    print(f"  Precision: {metrics['precision_percent']}%")
    print(f"  Recall: {metrics['recall_percent']}%")
    print(f"  F1-score: {metrics['f1_score_percent']}%")
    print(f"  Errors: {metrics['api_or_runtime_errors']}")


print(f"\nSummary saved to: {SUMMARY_PATH}")


# Display evaluator agreement.
print("\n--- Evaluator Agreement ---")

for pair_name, metrics in agreement_results.items():

    print(
        f'{pair_name}: '
        f'{metrics["agreements"]}/{metrics["compared_cases"]} '
        f'agreements ({metrics["agreement_percent"]}%)'
    )
