import os
import time
import hashlib
import logging
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY not found. Copy .env.example to .env and add your key."
    )

genai.configure(api_key=api_key)


class RateLimiter:
    """Tracks Gemini API calls and enforces free tier limits (max 12 calls/minute)."""
    def __init__(self, max_calls=12, per_seconds=60):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.call_times = []

    def wait_if_needed(self):
        now = time.time()
        # Remove timestamps older than per_seconds
        self.call_times = [t for t in self.call_times if now - t < self.per_seconds]
        
        if len(self.call_times) >= self.max_calls:
            oldest_call = self.call_times[0]
            sleep_time = self.per_seconds - (now - oldest_call) + 1
            if sleep_time > 0:
                print(f"Rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            now = time.time()
            self.call_times = [t for t in self.call_times if now - t < self.per_seconds]

        self.call_times.append(time.time())


# Singleton rate limiter instance
rate_limiter = RateLimiter()


def describe_images(images: list) -> list:
    """Uses Gemini 2.5 Flash Vision to generate detailed text descriptions of paper figures."""
    primary_model_name = "gemini-2.5-flash"
    fallback_model_name = "gemini-flash-latest"

    try:
        model = genai.GenerativeModel(primary_model_name)
    except Exception:
        model = genai.GenerativeModel(fallback_model_name)

    descriptions = []
    total = len(images)

    prompt = """You are analyzing a figure from the paper 
'Attention Is All You Need' (Vaswani et al., 2017).

Describe this figure thoroughly:
1. Figure type: architecture diagram / attention heatmap / 
   results table / equation / other
2. All visible text, labels, numbers, axis values
3. What the figure demonstrates about the Transformer model
4. Any flows, connections, or relationships shown
5. Specific values if it's a results table (BLEU scores, etc.)

Be detailed — this description is used for semantic search retrieval."""

    for i, img_dict in enumerate(images):
        rate_limiter.wait_if_needed()
        path = img_dict["path"]
        page = img_dict["metadata"].get("page", "?")

        try:
            pil_img = Image.open(path)
            try:
                response = model.generate_content([prompt, pil_img])
            except Exception as err:
                err_str = str(err)
                if "404" in err_str or "not found" in err_str.lower() or "no longer available" in err_str.lower():
                    logger.info(f"{primary_model_name} returned 404, using fallback {fallback_model_name}")
                    fallback_model = genai.GenerativeModel(fallback_model_name)
                    response = fallback_model.generate_content([prompt, pil_img])
                elif "429" in err_str or "quota" in err_str.lower():
                    print("Rate limit 429 hit. Sleeping 15 seconds before retrying...")
                    time.sleep(15)
                    response = model.generate_content([prompt, pil_img])
                else:
                    raise err

            desc_text = response.text.strip()
            
            descriptions.append({
                "content": desc_text,
                "metadata": {
                    **img_dict["metadata"],
                    "type": "image_description"
                }
            })
        except Exception as e:
            logger.warning(f"Failed to describe image at {path}: {e}")
            fallback_text = f"Figure from page {page} (description unavailable)"
            descriptions.append({
                "content": fallback_text,
                "metadata": {
                    **img_dict["metadata"],
                    "type": "image_description"
                }
            })

        time.sleep(1)  # Vision calls sleep 1s
        print(f"[{i+1}/{total}] Described figure from page {page}")

    return descriptions


def _call_embed(model_name: str, content):
    return genai.embed_content(model=model_name, content=content)["embedding"]


def get_embedding(text_or_list):
    """
    Generates embeddings using models/text-embedding-004 (with fallback to models/gemini-embedding-001 if 404).
    Accepts either a single text string or a list of text strings (batch).
    """
    rate_limiter.wait_if_needed()
    primary_model = "models/text-embedding-004"
    fallback_model = "models/gemini-embedding-001"

    try:
        if isinstance(text_or_list, list):
            truncated_batch = [t[:8000] for t in text_or_list]
            try:
                return _call_embed(primary_model, truncated_batch)
            except Exception as err:
                if "404" in str(err) or "not found" in str(err).lower():
                    return _call_embed(fallback_model, truncated_batch)
                raise err
        else:
            truncated = text_or_list[:8000]
            try:
                return _call_embed(primary_model, truncated)
            except Exception as err:
                if "404" in str(err) or "not found" in str(err).lower():
                    return _call_embed(fallback_model, truncated)
                raise err
    except Exception as e:
        logger.error(f"Error calling embed_content: {e}")
        return None


def sanitize_metadata(metadata: dict) -> dict:
    """Ensures metadata values are ChromaDB primitive types (str, int, float, bool)."""
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized


def build_vector_store(text_chunks: list, image_descriptions: list) -> None:
    """Indexes text chunks and image descriptions into persistent ChromaDB collection."""
    import chromadb

    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("attention_paper")
    except Exception:
        pass

    collection = client.create_collection(
        name="attention_paper",
        metadata={"hnsw:space": "cosine"}
    )

    all_docs = text_chunks + image_descriptions
    total = len(all_docs)
    ids, embeddings, documents, metadatas = [], [], [], []

    batch_size = 10
    for i in range(0, total, batch_size):
        batch_docs = all_docs[i:i + batch_size]
        batch_contents = [doc["content"] for doc in batch_docs]

        batch_embeds = get_embedding(batch_contents)
        time.sleep(0.5)

        for j, doc in enumerate(batch_docs):
            idx = i + j
            content_hash = hashlib.md5(doc["content"].encode()).hexdigest()

            # Handle embedding result from batch or fallback
            emb = None
            if batch_embeds and j < len(batch_embeds):
                emb = batch_embeds[j]
            else:
                emb = get_embedding(doc["content"])
                time.sleep(0.5)

            if emb is None:
                logger.warning(f"Skipping document index {idx} due to missing embedding")
                continue

            doc_id = f"doc_{idx}_{content_hash[:8]}"
            ids.append(doc_id)
            embeddings.append(emb)
            documents.append(doc["content"])

            meta = {**doc["metadata"], "content_hash": content_hash}
            metadatas.append(sanitize_metadata(meta))

        print(f"Embedded {min(i + batch_size, total)}/{total} documents...")

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print(f"Vector store ready: {len(ids)} documents indexed")


def load_collection():
    """Lazy loader for ChromaDB vector collection."""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        return client.get_collection("attention_paper")
    except Exception:
        raise RuntimeError("Knowledge base not found. Run: python ingest.py")
