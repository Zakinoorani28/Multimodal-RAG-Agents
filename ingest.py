import os
import sys
import time
import shutil
from dotenv import load_dotenv

load_dotenv()

from app.extractor import extract_all
from app.embedder import describe_images, build_vector_store

PRIMARY_PDF = "data/Attention Is All You Need - Research Paper.pdf"
FALLBACK_PDF = "data/attention.pdf"

if os.path.exists(PRIMARY_PDF):
    PDF_PATH = PRIMARY_PDF
elif os.path.exists(FALLBACK_PDF):
    PDF_PATH = FALLBACK_PDF
else:
    PDF_PATH = PRIMARY_PDF

def main():
    print("=== Multimodal RAG Ingestion Pipeline ===")

    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF file not found at '{PDF_PATH}'.")
        print("Please place 'Attention Is All You Need - Research Paper.pdf' inside the data/ directory.")
        sys.exit(1)

    os.makedirs("data/figures", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("\n[Step 1/3] Extracting content from PDF...")
    t0 = time.time()
    extracted = extract_all(PDF_PATH)
    print(f"Extraction done in {time.time()-t0:.1f}s")

    print("\n[Step 2/3] Describing figures with Gemini Vision...")
    t1 = time.time()
    image_descriptions = describe_images(extracted["images"])
    print(f"Vision done in {time.time()-t1:.1f}s — {len(image_descriptions)} figures described")

    print("\n[Step 3/3] Building ChromaDB vector store...")
    t2 = time.time()
    build_vector_store(extracted["text_chunks"], image_descriptions)
    print(f"Indexing done in {time.time()-t2:.1f}s")

    total = time.time() - t0
    print(f"\n[SUCCESS] Ingestion complete in {total:.1f}s")
    print(f"  Text chunks: {len(extracted['text_chunks'])}")
    print(f"  Image descriptions: {len(image_descriptions)}")
    print(f"  Vector store: ./chroma_db/")

if __name__ == "__main__":
    main()
