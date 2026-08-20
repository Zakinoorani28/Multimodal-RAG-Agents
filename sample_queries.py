import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from app.retriever import retrieve_multimodal
from app.generator import generate_answer

os.makedirs("outputs", exist_ok=True)

QUERIES = [
    {
        "id": 1,
        "query": "What BLEU score did the big Transformer achieve on "
                 "WMT 2014 English-German translation task?",
        "expected_modality": "table",
        "note": "Answer should come from Table 2 results"
    },
    {
        "id": 2,
        "query": "Describe the encoder and decoder architecture shown "
                 "in Figure 1. What are the main components?",
        "expected_modality": "figure",
        "note": "Answer should reference the architecture diagram"
    },
    {
        "id": 3,
        "query": "What is multi-head attention? How many heads were used "
                 "and what dimensions were used for each head?",
        "expected_modality": "text",
        "note": "Answer from Section 3.2.2"
    }
]

results = []
for item in QUERIES:
    print(f"\n[Query {item['id']}/3] {item['query'][:60]}...")

    chunks = retrieve_multimodal(item["query"], n_results=5)
    result = generate_answer(item["query"], chunks["all"])

    output = {
        "id": item["id"],
        "query": result["query"],
        "expected_modality": item["expected_modality"],
        "answer": result["answer"],
        "sources_used": result["sources_used"],
        "source_types": result["source_types"],
        "pages_referenced": result["pages_referenced"],
        "retrieved_context_preview": result["context"][:800] + "..."
    }
    results.append(output)

    print(f"Answer preview: {result['answer'][:200]}...")
    print(f"Sources: {result['source_types']} | Pages: {result['pages_referenced']}")

    time.sleep(2)

out_path = "outputs/sample_outputs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n[SUCCESS] Saved to {out_path}")
print("Submit this file as proof the pipeline runs end to end.")
