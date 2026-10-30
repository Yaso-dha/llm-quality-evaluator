
def evaluate_response(test_case: dict) -> dict:
    """
    Check whether a chatbot response contains required facts
    and avoids known incorrect claims.
    """

    actual_output = test_case["actual_output"].casefold()

    required_facts = test_case["required_facts"]
    forbidden_claims = test_case["forbidden_claims"]

    missing_facts = [
        fact
        for fact in required_facts
        if fact.casefold() not in actual_output
    ]

    detected_forbidden_claims = [
        claim
        for claim in forbidden_claims
        if claim.casefold() in actual_output
    ]

    passed = (
        len(missing_facts) == 0
        and len(detected_forbidden_claims) == 0
    )

    return {
        "test_id": test_case["id"],
        "passed": passed,
        "missing_facts": missing_facts,
        "detected_forbidden_claims": detected_forbidden_claims
    }
