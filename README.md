# AI Response Quality Evaluation Framework

A Python-based project exploring how to evaluate the factual accuracy and reliability of AI-generated chatbot responses using rule-based validation, LLM-as-a-Judge, and DeepEval.

## Project Overview

AI-generated responses can contain incorrect information, unsupported claims, or contradictions.

This project explores different approaches to evaluating chatbot responses against reference information and compares how effectively different evaluators identify factual errors.

The project uses a small, manually designed evaluation dataset containing both correct and deliberately incorrect chatbot responses.

## Key Features

- **Rule-based evaluation:** Checks required facts and identifies predefined forbidden claims.
- **LLM-as-a-Judge:** Uses a locally hosted Llama 3.2:1b model through Ollama to assess factual accuracy.
- **DeepEval integration:** Experiments with the HallucinationMetric using the local Ollama model.
- **Automated testing:** Uses pytest to validate evaluation functionality.
- **Evaluator comparison:** Compares evaluator verdicts against expected results.
- **Evaluation metrics:** Calculates accuracy, precision, recall, F1 score, false passes, and false fails.
- **CI/CD:** Uses GitHub Actions to automatically execute selected Python tests.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pytest | Test automation |
| Ollama | Local LLM execution |
| Llama 3.2:1b | LLM-as-a-Judge model |
| DeepEval | LLM evaluation framework |
| GitHub Actions | CI/CD automation |
| JSON | Evaluation dataset and reports |

## Project Structure

```text
llm-quality-evaluator/
│
├── .github/
│   └── workflows/
│       └── python-tests.yml
│
├── data/
│   └── evaluation_cases.json
│
├── reports/
│   ├── comparison_results.json
│   └── comparison_summary.json
│
├── src/
│   ├── evaluator.py
│   ├── llm_judge.py
│   ├── gemini_judge.py
│   ├── compare_evaluators.py
│   └── analyze_results.py
│
├── tests/
│   ├── test_evaluator.py
│   ├── test_llm_judge.py
│   ├── test_gemini_judge.py
│   ├── test_comparison.py
│   └── test_deepeval.py
│
├── pytest.ini
├── .gitignore
└── README.md
```

## Evaluation Approaches

### 1. Rule-Based Evaluation

Validates chatbot responses against predefined required facts and forbidden claims.

### 2. LLM-as-a-Judge

Uses Llama 3.2:1b through Ollama to compare chatbot responses against reference context and return a PASS or FAIL verdict with an explanation.

### 3. DeepEval

Experiments with DeepEval's HallucinationMetric to investigate whether an LLM judge can identify unsupported or contradictory information.

## Running the Project Locally

### Prerequisites

- Python
- Git
- Ollama
- Llama 3.2:1b model for local LLM evaluation

### Set up the environment

Clone the repository:

```bash
git clone https://github.com/Yaso-dha/llm-quality-evaluator.git
cd llm-quality-evaluator
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install pytest deepeval ollama
```

### Run rule-based tests

```bash
PYTHONPATH=. pytest -v tests/test_evaluator.py
```

### Run DeepEval experiment

Ensure Ollama is running and the Llama 3.2:1b model is available:

```bash
ollama pull llama3.2:1b
```

Then execute:

```bash
PYTHONPATH=. pytest -v -s tests/test_deepeval.py
```

### Run the evaluator comparison

Ensure the local Ollama model is running, then execute:

```bash
PYTHONPATH=. python src/compare_evaluators.py
```

Generate the evaluation summary:

```bash
PYTHONPATH=. python src/analyze_results.py
```

## CI/CD Pipeline

GitHub Actions is configured to run selected rule-based pytest tests automatically when code is pushed to the main branch or a pull request is created.

The current workflow:
1. Checks out the repository.
2. Sets up Python.
3. Installs pytest.
4. Executes the rule-based automated tests.

LLM evaluations requiring a locally hosted Ollama model are currently run separately.

## Current Findings and Limitations

Experiments with the small Llama 3.2:1b judge highlighted that LLM-based evaluation can produce incorrect PASS and FAIL verdicts.

DeepEval experiments also demonstrated that evaluation results can depend on the judge model and evaluation instructions.

The current dataset contains only 15 manually designed cases. Results are therefore exploratory and should not be interpreted as evidence of general evaluator reliability.

Further work will focus on:
- Expanding the evaluation dataset.
- Improving hallucination and factual consistency detection.
- Exploring contextual relevance and answer correctness metrics.
- Adding API testing with Python and pytest.
- Enhancing CI/CD reporting.
- Investigating bias and safety evaluation.

## Project Status

**In progress – Hands-on exploration of AI Quality Engineering and LLM evaluation.**