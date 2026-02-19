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
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Your task is to assess whether the provided ANSWER is relevant to the given QUESTION, based strictly on the provided CONTEXT. Do not use external knowledge. Base your evaluation only on the information explicitly available in the QUESTION and CONTEXT.

Definition:
Answer relevancy measures how well the ANSWER addresses the QUESTION using information supported by the CONTEXT.

An answer is considered irrelevant if it:

Fails to address the main intent of the QUESTION,

Includes substantial information unrelated to the QUESTION,

Omits key aspects required to properly answer the QUESTION (when such information exists in the CONTEXT), or

Focuses on tangential details instead of the core request.

Inputs:

QUESTION: {question}

CONTEXT (retrieved chunks): {context}

ANSWER (model-generated response): {answer}

Evaluation Instructions:

Identify the main intent of the QUESTION.

Determine whether the ANSWER directly and sufficiently addresses that intent.

Check whether the ANSWER stays focused on relevant information from the CONTEXT.

Ignore stylistic issues unless they impact relevance.

Output Format (strictly follow this structure):

{{
  "Hallucination": "Yes/No",
  "Explanation": "Brief explanation identifying specific unsupported or contradictory elements, if any",
  "Score": 0
}}

Scoring Guidelines:

0 = Completely irrelevant; does not address the QUESTION

1–3 = Mostly irrelevant; minimal connection to the QUESTION

4–6 = Partially relevant; addresses some aspects but misses key points

7–9 = Mostly relevant; addresses the main intent with minor gaps

10 = Fully relevant; directly and comprehensively answers the QUESTION using the CONTEXT
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
