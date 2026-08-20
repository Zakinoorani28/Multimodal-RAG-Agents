import os
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from app.embedder import rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


def format_context(retrieved_chunks: list) -> str:
    """Formats retrieved chunks into structured context blocks."""
    formatted_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        rank = chunk.get("rank", i + 1)
        source_type = chunk["metadata"].get("type", "unknown")
        page = chunk["metadata"].get("page", "?")
        score = chunk.get("relevance_score", "?")

        header = f"[Source {rank} | Type: {source_type} | Page: {page} | Relevance: {score}]"
        content = chunk.get("content", "")
        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(formatted_blocks)


def generate_answer(query: str, retrieved_chunks: list) -> dict:
    """Generates grounded answers strictly from retrieved context using Gemini 2.5 Flash."""
    if not retrieved_chunks:
        return {
            "query": query,
            "answer": "No relevant context found.",
            "sources_used": 0,
            "source_types": [],
            "pages_referenced": [],
            "context": ""
        }

    context = format_context(retrieved_chunks)

    prompt = f"""You are an expert assistant on the research paper "Attention Is All You Need" (Vaswani et al., 2017), which introduced the Transformer architecture.

STRICT RULES:
- Answer ONLY from the retrieved context below
- If the answer is in a table, cite the exact numbers
- If the answer relates to a figure, describe what the figure shows
- If the context is insufficient, say: "The retrieved context does not contain enough information to answer this fully."
- Do not use your general training knowledge — ground every claim in the context provided

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Provide a thorough, accurate answer grounded in the context above:"""

    rate_limiter.wait_if_needed()
    primary_model_name = "gemini-2.5-flash"
    fallback_model_name = "gemini-3.6-flash"

    try:
        model = genai.GenerativeModel(primary_model_name)
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        err_str = str(e)
        if "404" in err_str or "not found" in err_str.lower() or "no longer available" in err_str.lower():
            logger.info(f"{primary_model_name} returned 404, using fallback {fallback_model_name}")
            try:
                fallback_model = genai.GenerativeModel(fallback_model_name)
                response = fallback_model.generate_content(prompt)
                answer = response.text
            except Exception as e2:
                err2_str = str(e2)
                if "429" in err2_str or "quota" in err2_str.lower():
                    print("Rate limit 429 hit. Sleeping 15 seconds before retrying...")
                    time.sleep(15)
                    try:
                        response = fallback_model.generate_content(prompt)
                        answer = response.text
                    except Exception as e3:
                        answer = f"Generation failed: {str(e3)}"
                else:
                    logger.error(f"Generation error: {e2}")
                    answer = f"Generation failed: {str(e2)}"
        elif "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower():
            print(f"Rate limit 429 hit for {primary_model_name}. Sleeping 15 seconds before retrying...")
            time.sleep(15)
            try:
                response = model.generate_content(prompt)
                answer = response.text
            except Exception as e_retry:
                logger.error(f"Generation error after retry with {primary_model_name}: {e_retry}")
                answer = f"Generation failed: {str(e_retry)}"
        else:
            logger.error(f"Generation error with {primary_model_name}: {e}")
            answer = f"Generation failed: {str(e)}"

    source_types = list(set(
        c["metadata"].get("type", "unknown")
        for c in retrieved_chunks
    ))

    pages_referenced = sorted(list(set(
        str(c["metadata"].get("page", "?"))
        for c in retrieved_chunks
    )))

    return {
        "query": query,
        "answer": answer,
        "sources_used": len(retrieved_chunks),
        "source_types": source_types,
        "pages_referenced": pages_referenced,
        "context": context
    }
