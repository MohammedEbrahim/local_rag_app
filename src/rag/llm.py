import ollama

from src.config import SYSTEM_PROMPT


def stream_ollama_answer(llm_model: str, question: str, context: str):
    user_prompt = f"""CONTEXT:
{context}

QUESTION:
{question}

Answer using ONLY the context above.
"""

    stream = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]
