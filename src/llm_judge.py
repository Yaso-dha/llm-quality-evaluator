
import requests
import json
from unittest.mock import patch, Mock

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"


def ask_llm_judge(context: str, chatbot_response: str) -> dict:
    prompt = f"""
You are a strict factual accuracy evaluator.

Compare the chatbot response with the reference context.

Reference context:
{context}

Chatbot response:
{chatbot_response}


Evaluate whether the chatbot response is factually supported by
the reference context.

Rules:
1. Do not invent facts that are not in the context.
2. Treat equivalent wording as consistent when the meaning is
   reasonably the same.
3. Do not assume additional details that are not provided.
4. If the response makes an unsupported factual claim, use FAIL.
5. If the response is supported by the context, use PASS.
6. If the response changes a number, date, duration, price,
   or eligibility condition without support from the context,
   return FAIL.
7. Return PASS only when the response is supported by the
   context and contains no unsupported factual claims.
8. Check numbers and durations carefully. If the context says
   "30 days" but the response says "90 days", return FAIL.
   The fact that both numbers refer to a refund period does
   not make them equivalent.
9. Treat "30 days" and "one month" as approximately equivalent
   for this evaluation, unless the context requires exact
   calendar-month calculations.
Return only a valid JSON object in this exact format:
{{
  "verdict": "PASS or FAIL",
  "reason": "Brief explanation"

}}

"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    judge_output = response.json()["response"]

    try:
        return json.loads(judge_output)

    except json.JSONDecodeError:
        return {
            "verdict": "ERROR",
            "reason": "Llama returned an invalid JSON response."
        }


if __name__ == "__main__":
    context = (
        "Eligible products can be refunded within 30 days of purchase."
    )
    chatbot_response = (
        "Eligible products can be refunded within one month."
    )

    judgement = ask_llm_judge(context, chatbot_response)
    
    print("LLM Judge's assessment:")
    print(f"Verdict: {judgement['verdict']}")
    print(f"Reason: {judgement['reason']}")
