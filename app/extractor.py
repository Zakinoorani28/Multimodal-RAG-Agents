import os
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import fitz
except ImportError:
    raise ImportError("Install pymupdf: pip install pymupdf")


def extract_all(pdf_path: str) -> dict:
    """
    Extract text chunks, headings, tables, and figures from the PDF.
    
    Returns:
        dict: {
            "text_chunks": [{"content": str, "metadata": dict}, ...],
            "images": [{"path": str, "metadata": dict}, ...]
        }
    """
    # Check for fallback PDF path if primary path does not exist
    if not os.path.exists(pdf_path):
        fallback_path = os.path.join(os.path.dirname(pdf_path), "Attention Is All You Need - Research Paper.pdf")
        if os.path.exists(fallback_path):
            pdf_path = fallback_path
        else:
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    text_chunks = []
    extracted_images = []
    
    seen_image_sizes = set()
    os.makedirs("data/figures", exist_ok=True)

    text_count = 0
    heading_count = 0
    table_count = 0
    figure_count = 0

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        try:
            page = doc[page_idx]

            # A) Extract Headings (font_size > 13)
            try:
                page_dict = page.get_text("dict")
                for block in page_dict.get("blocks", []):
                    if block.get("type") == 0:  # text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                font_size = span.get("size", 0)
                                heading_text = span.get("text", "").strip()
                                if font_size > 13 and len(heading_text) > 3:
                                    c_hash = hashlib.md5(heading_text.encode()).hexdigest()
                                    text_chunks.append({
                                        "content": heading_text,
                                        "metadata": {
                                            "type": "heading",
                                            "page": page_num,
                                            "source": "heading",
                                            "content_hash": c_hash
                                        }
                                    })
                                    heading_count += 1
            except Exception as e:
                logger.warning(f"Error extracting headings on page {page_num}: {e}")

            # B) Extract Full Text & Split into Chunks (600 chars, overlap 80 chars)
            try:
                full_text = page.get_text("text").strip()
                if full_text:
                    chunk_size = 600
                    overlap = 80
                    step = chunk_size - overlap
                    for start_i in range(0, len(full_text), step):
                        chunk = full_text[start_i:start_i + chunk_size].strip()
                        if chunk:
                            c_hash = hashlib.md5(chunk.encode()).hexdigest()
                            text_chunks.append({
                                "content": chunk,
                                "metadata": {
                                    "type": "text",
                                    "page": page_num,
                                    "source": "body_text",
                                    "content_hash": c_hash
                                }
                            })
                            text_count += 1
            except Exception as e:
                logger.warning(f"Error extracting text on page {page_num}: {e}")

            # C) Extract Tables
            try:
                if hasattr(page, "find_tables"):
                    tables = page.find_tables()
                    for tab in tables:
                        df = tab.extract()
                        if df and len(df) > 0:
                            headers = ", ".join([str(c) if c is not None else "" for c in df[0]])
                            rows_str = "\n".join([", ".join([str(cell) if cell is not None else "" for cell in row]) for row in df[1:]])
                            table_text = f"TABLE (Page {page_num}):\nHeaders: {headers}\nRows:\n{rows_str}"
                            c_hash = hashlib.md5(table_text.encode()).hexdigest()
                            text_chunks.append({
                                "content": table_text,
                                "metadata": {
                                    "type": "table",
                                    "page": page_num,
                                    "source": "table",
                                    "content_hash": c_hash
                                }
                            })
                            table_count += 1
            except Exception as e:
                logger.warning(f"Error extracting tables on page {page_num}: {e}")

            # D) Extract Figures/Images
            try:
                images = page.get_images(full=True)
                for img_idx, img_info in enumerate(images):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        byte_size = len(image_bytes)

                        # Filter out images smaller than 5000 bytes or duplicate file sizes
                        if byte_size < 5000:
                            continue
                        if byte_size in seen_image_sizes:
                            continue

                        seen_image_sizes.add(byte_size)
                        fig_path = f"data/figures/fig_p{page_num}_i{img_idx+1}.{image_ext}"
                        with open(fig_path, "wb") as f:
                            f.write(image_bytes)

                        extracted_images.append({
                            "path": fig_path,
                            "metadata": {
                                "type": "image",
                                "page": page_num,
                                "path": fig_path,
                                "source": "figure"
                            }
                        })
                        figure_count += 1
            except Exception as e:
                logger.warning(f"Error extracting images on page {page_num}: {e}")

        except Exception as e:
            logger.warning(f"Page {page_num} processing failed: {e}")
            continue

    print(f"Extracted: {text_count} text chunks, {heading_count} headings, {table_count} tables, {figure_count} figures")

    return {
        "text_chunks": text_chunks,
        "images": extracted_images
    }
