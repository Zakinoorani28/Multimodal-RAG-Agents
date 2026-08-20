"""Ingestion script module alias for app.ingest."""
import sys
import os

# Allow running python -m app.ingest or importing app.ingest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ingest import PDF_PATH, extract_all, describe_images, build_vector_store

if __name__ == "__main__":
    from ingest import main
    main()
