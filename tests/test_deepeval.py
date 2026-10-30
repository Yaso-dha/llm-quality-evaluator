
from deepeval.metrics import HallucinationMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase


def test_detect_hallucination_with_deepeval():

    # Connect DeepEval to our local Ollama model
    ollama_model = OllamaModel(
        model="llama3.2:1b",
        base_url="http://localhost:11434"
    )

    # Reference information: the source of truth
    context = [
        "Eligible products can be refunded within 30 days."
    ]

    # Intentionally incorrect chatbot response
    test_case = LLMTestCase(
        input="What is the refund policy?",
        actual_output=(
            "All products can be refunded within 90 days"),
        context=context
    )
    # Configure DeepEval's hallucination metric
    metric = HallucinationMetric(
        threshold=0.5,
        model=ollama_model,
        include_reason=True,
        async_mode=False,
        verbose_mode=True
    )

    # Evaluate the chatbot response
    metric.measure(test_case)

    print("\n--- DeepEval Hallucination Results ---")
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Passed: {metric.is_successful()}")

    # We expect this incorrect response to be detected
    assert not metric.is_successful(), (
        "DeepEval failed to detect the hallucinated response."
    )
    