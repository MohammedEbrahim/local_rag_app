import json
import ollama


def mistral_judge(
        question: str,
        context: str,
        answer: str,
        judge_model: str = "mistral:latest",
) -> dict:
    """
    Uses a local Ollama model (Mistral) as LLM-as-a-Judge.
    Returns JSON-like dict.
    """

    judge_prompt = f"""
You are an expert RAG evaluator.

You will be given:
- a QUESTION
- a CONTEXT (retrieved chunks)
- an ANSWER produced by the assistant

Score the ANSWER strictly based on the CONTEXT.

Return ONLY valid JSON with these fields:
{{
  "faithfulness": 0-10,
  "relevance": 0-10,
  "completeness": 0-10,
  "hallucination": 0-10,
  "overall": 0-10,
  "explanation": "short explanation"
}}

Rules:
- If the answer includes facts not present in context, faithfulness must be <= 3.
- hallucination = how much the answer invents (10 = none, 0 = severe).
- Keep explanation short.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""

    try:
        resp = ollama.chat(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a strict JSON-only evaluator."},
                {"role": "user", "content": judge_prompt},
            ],
        )

        text = (resp.get("message", {}).get("content", "") or "").strip()

    except Exception as e:
        return {"error": f"Ollama judge request failed: {e}"}

    # Best-effort JSON parse
    try:
        return json.loads(text)
    except Exception:
        return {"error": "Judge returned non-JSON", "raw": text}
