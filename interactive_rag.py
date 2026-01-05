# interactive_rag.py
# ------------------------------------------------------------
# Interactive RAG + logging
# - Ask question in terminal
# - Runs all configurations:
#     fixed+bm25, child+bm25, fixed+emb (if available), child+emb
# - Runs K values (default: 3,5,10)
# - Prints clickable links (file:///...) to ORIGINAL debate files
# - Saves everything to JSONL in evaluation/results/
# ------------------------------------------------------------

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from retrieval.bm25_retriever import load_jsonl, BM25Retriever
from retrieval.embedding_retriever import EmbeddingRetriever

# Must be your Ollama-based pipeline (not Gemini)
from rag.rag_pipeline import build_prompt, generate_with_ollama as generate_answer



# ---------------- CONFIG ----------------
K_VALUES = [3, 5, 8]
EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

FIXED_CHUNKS = Path("data/chunks/fixed_660w_3s.jsonl")
CHILD_CHUNKS = Path("data/chunks/child_chunks.jsonl")

FIXED_EMB = Path("data/embeddings/fixed_embeddings_mpnet.npy")   # optional
CHILD_EMB = Path("data/embeddings/child_embeddings_mpnet.npy")   # required for child+emb

# Where original debate files live (adjust if needed)
DATA_ROOT = Path("data/ex2_clean")  # data/ex2_clean/UK|US/filename.txt

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Helpers ----------------
def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def file_link(filename: str, parliament: str) -> Dict[str, str]:
    """
    Returns absolute path and file:// URI for the ORIGINAL debate file.
    Example:
      file:///C:/Users/.../data/ex2_clean/UK/debates2024-04-17.txt
    """
    p = (DATA_ROOT / parliament / filename).resolve()
    return {"abs_path": str(p), "file_uri": p.as_uri()}


def enrich_with_file_links(retrieved: List[Dict[str, Any]]) -> None:
    for c in retrieved:
        fn = c.get("filename", "unknown")
        parl = c.get("parliament", "UK")
        lk = file_link(fn, parl)
        c["file_abs_path"] = lk["abs_path"]
        c["file_uri"] = lk["file_uri"]


def print_retrieved_sources(retrieved: List[Dict[str, Any]], max_print: int = 5) -> None:
    print("Top sources (clickable file links):")
    for i, c in enumerate(retrieved[:max_print], start=1):
        print(f"  Doc {i}: {c.get('filename')} | {c.get('parliament')} | {c.get('date')} | {c.get('chunk_id')}")
        print(f"        score={c.get('score')}")
        print(f"        {c.get('file_uri')}")


def save_jsonl_append(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------- Main ----------------
def main():
    print("Loading chunks...")
    fixed_chunks = load_jsonl(FIXED_CHUNKS)
    child_chunks = load_jsonl(CHILD_CHUNKS)
    print(f"Fixed chunks: {len(fixed_chunks)}")
    print(f"Child chunks: {len(child_chunks)}")

    print("\nBuilding BM25 retrievers...")
    bm25_fixed = BM25Retriever(fixed_chunks)
    bm25_child = BM25Retriever(child_chunks)

    print("\nLoading embeddings...")
    child_emb = None
    if CHILD_EMB.exists():
        child_emb = np.load(CHILD_EMB)
        print("Child embeddings:", child_emb.shape)
    else:
        print(f"[WARN] Missing: {CHILD_EMB} -> child+emb will be skipped")

    fixed_emb = None
    if FIXED_EMB.exists():
        fixed_emb = np.load(FIXED_EMB)
        print("Fixed embeddings:", fixed_emb.shape)
    else:
        print(f"[WARN] Missing: {FIXED_EMB} -> fixed+emb will be skipped")

    print("\nBuilding Embedding retrievers...")
    emb_fixed: Optional[EmbeddingRetriever] = None
    emb_child: Optional[EmbeddingRetriever] = None

    if fixed_emb is not None:
        emb_fixed = EmbeddingRetriever(fixed_chunks, fixed_emb, EMB_MODEL_NAME)
    if child_emb is not None:
        emb_child = EmbeddingRetriever(child_chunks, child_emb, EMB_MODEL_NAME)

    # configs list (skip if not available)
    configs: List[Tuple[str, str, Any]] = [
        ("fixed", "bm25", bm25_fixed),
        ("child", "bm25", bm25_child),
    ]
    if emb_fixed is not None:
        configs.append(("fixed", "emb", emb_fixed))
    if emb_child is not None:
        configs.append(("child", "emb", emb_child))

    print("\nAvailable configs:")
    for a, b, _ in configs:
        print(f" - {a}+{b}")

    # output file for this interactive session
    out_path = RESULTS_DIR / f"interactive_{now_tag()}.jsonl"
    print(f"\nLogging to: {out_path}\n")

    while True:
        q = input("Tape ta question (ou 'exit'): ").strip()
        if q.lower() in ("exit", "quit"):
            print("Bye.")
            break
        if not q:
            continue

        for chunking_name, repr_name, retriever in configs:
            for k in K_VALUES:
                t0 = time.time()

                # retrieval
                retrieved = retriever.search(q, k)

                # add file links to original source
                enrich_with_file_links(retrieved)

                # build prompt + generate
                prompt, sources = build_prompt(q, retrieved)
                answer = generate_answer(prompt)

                dt = time.time() - t0

                # display
                print("\n" + "-" * 80)
                print(f"chunking={chunking_name} | repr={repr_name} | K={k} | time={dt:.2f}s")
                print_retrieved_sources(retrieved, max_print=min(5, k))
                print("\nANSWER:\n" + answer)

                # save record
                record = {
                    "suite": "interactive",
                    "query": q,
                    "chunking": chunking_name,
                    "representation": repr_name,
                    "k": k,
                    "elapsed_sec": round(dt, 4),
                    "retrieved": retrieved,         # includes file_uri + abs_path
                    "sources_unique": sources,      # whatever build_prompt returns
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                save_jsonl_append(out_path, record)


if __name__ == "__main__":
    main()
