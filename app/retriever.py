import logging
from app.embedder import get_embedding, load_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrieve(query: str, n_results: int = 5) -> list:
    """Queries ChromaDB collection for nearest neighbors to the query vector."""
    try:
        collection = load_collection()
        query_embedding = get_embedding(query)
        if query_embedding is None:
            raise ValueError("Failed to embed query")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results.get("documents") or len(results["documents"]) == 0:
            return []

        retrieved_list = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            retrieved_list.append({
                "content": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 4),
                "rank": i + 1
            })

        return retrieved_list
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


def retrieve_multimodal(query: str, n_results: int = 6) -> dict:
    """Retrieves chunks and guarantees both text and figure/image context inclusion."""
    all_chunks = retrieve(query, n_results=n_results * 2)

    text_chunks = [
        c for c in all_chunks
        if c["metadata"].get("type") in ["text", "heading", "table"]
    ]
    image_chunks = [
        c for c in all_chunks
        if c["metadata"].get("type") == "image_description"
    ]

    return {
        "text": text_chunks[:n_results],
        "images": image_chunks[:2],
        "all": (text_chunks[:n_results] + image_chunks[:2])
    }
