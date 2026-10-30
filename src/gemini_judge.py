
import os
import json
from google import genai

# Read the API key from the environment variable.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-3.8-flash"


def ask_gemini_judge(context: str, chatbot_response: str) -> dict:
    prompt = f"""
You are a chatbot quality evaluator.

Evaluate whether the chatbot response is supported by the reference context.

Reference context:
{context}

Chatbot response:
{chatbot_response}

Rules:
1. Return FAIL if the response contradicts the context.
2. Return FAIL if it adds unsupported factual claims.
3. Return PASS if the response is supported by the context.
4. Treat different wording as acceptable when the meaning is equivalent.
5. Do not invent facts or assumptions.

Return only valid JSON in this format:
{{
    "verdict": "PASS or FAIL",
    "reason": "Brief explanation"
}}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"temperature": 0}
    )

    judge_output = (response.text or "").strip()

    # Remove Markdown code fences if Gemini includes them.
    if judge_output.startswith("```"):
        judge_output = judge_output.removeprefix("```json")
        judge_output = judge_output.removeprefix("```")
        judge_output = judge_output.removesuffix("```")
        judge_output = judge_output.strip()

    if not judge_output:
        raise ValueError("Gemini returned an empty response.")

    return json.loads(judge_output)

if __name__ == "__main__":
    context = (
        "Eligible products can be refunded within 30 days of purchase."
    )
    chatbot_response = (
        "Eligible products can be refunded within 90 days of purchase."
    )

    judgement = ask_gemini_judge(context, chatbot_response)

    print("Gemini Judge's assessment:")
    print(f"Verdict: {judgement['verdict']}")
    print(f"Reason: {judgement['reason']}")
