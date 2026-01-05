# evaluation/run_all_queries.py
# evaluation/run_all_queries.py
# ------------------------------------------------------------
# STEP D - Systematic evaluation
# - Runs REQUIRED queries and CUSTOM queries separately
# - 2 chunking methods (fixed / child)
# - 2 retrieval methods (bm25 / emb)
# - 3 K values
# - Saves results to JSONL with different filenames:
#     run_required_YYYYMMDD_HHMMSS.jsonl
#     run_custom_YYYYMMDD_HHMMSS.jsonl
# - Stores clickable file links (file:///...) inside JSONL + optional terminal prints
#
# SPEED UPS INCLUDED:
# (3) BM25 cache on disk (pickle) -> avoids rebuilding BM25 every run
# (7) Much less terminal printing -> faster + readable
# ------------------------------------------------------------

from __future__ import annotations

import json
import time
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from rag.rag_pipeline import build_prompt, generate_with_ollama


import numpy as np

# ---------- OPTIONAL: Ollama generation ----------
# pip install ollama
try:
    import ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

# ---------- OPTIONAL: BM25 ----------
# pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

# ---------- SentenceTransformer for query embedding ----------
from sentence_transformers import SentenceTransformer


# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent  # exo3/
EVAL_DIR = Path(__file__).resolve().parent         # exo3/evaluation

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# (3) Cache directory
CACHE_DIR = EVAL_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BM25_CACHE_FIXED = CACHE_DIR / "bm25_fixed.pkl"
BM25_CACHE_CHILD = CACHE_DIR / "bm25_child.pkl"

# Where original debate files live
DATA_ROOT = BASE_DIR / "data" / "ex2_clean"  # data/ex2_clean/UK|US/xxx.txt

# Chunk files (created by 01_chunking.py)
CHUNKS_FIXED = BASE_DIR / "data" / "chunks" / "fixed_660w_3s.jsonl"
CHUNKS_CHILD = BASE_DIR / "data" / "chunks" / "child_chunks.jsonl"

# Embeddings files (child you already have; fixed if you computed it)
EMB_CHILD = BASE_DIR / "data" / "embeddings" / "child_embeddings_mpnet.npy"
EMB_FIXED = BASE_DIR / "data" / "embeddings" / "fixed_embeddings_mpnet.npy"

# Model used to embed queries (must match your embedding space)
EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Queries files
REQUIRED_QUERIES_PATH = EVAL_DIR / "queries_required.json"
CUSTOM_QUERIES_PATH = EVAL_DIR / "queries_custom.json"


# ---------------- CONFIG ----------------
K_VALUES = [3, 5, 8]

# Generation (Ollama)
USE_LLM = True
OLLAMA_MODEL = "qwen2.5:3b"    # << you installed this (lighter than mistral)
OLLAMA_TEMPERATURE = 0.0

# Limit how many chunks go into prompt (None => use K)
MAX_CONTEXT_CHUNKS: Optional[int] = None

# (7) Less terminal printing
VERBOSE = False
PRINT_TOP_SOURCES = 0          # 0 = none, 1 = top1, 3 = top3
PRINT_EVERY_N_RUNS = 10        # print one progress line every 10 runs


# ============================================================
# Utils
# ============================================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_queries(path: Path) -> Dict[str, List[Dict[str, str]]]:
    """
    Expected JSON format:
    {
      "factual": [{"id":"REQ_F1","text":"..."}, ...],
      "conceptual": [{"id":"REQ_C1","text":"..."}, ...]
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing queries file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    factual = obj.get("factual", [])
    conceptual = obj.get("conceptual", [])
    return {"factual": factual, "conceptual": conceptual}


def file_link(filename: str, parliament: str) -> Dict[str, str]:
    """
    Returns absolute path + file:// URI for the ORIGINAL debate file.
    """
    p = (DATA_ROOT / parliament / filename).resolve()
    return {"abs_path": str(p), "file_uri": p.as_uri()}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# Retrieval: BM25 (with disk cache)
# ============================================================
@dataclass
class BM25Index:
    chunks: List[Dict[str, Any]]
    bm25: Any  # BM25Okapi


def build_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Index:
    if not HAS_BM25:
        raise RuntimeError("rank-bm25 not installed. Run: pip install rank-bm25")

    tokenized = []
    for c in chunks:
        text = c.get("text", "")
        tokenized.append(text.split())

    bm25 = BM25Okapi(tokenized)
    return BM25Index(chunks=chunks, bm25=bm25)


def load_or_build_bm25_index(chunks: List[Dict[str, Any]], cache_path: Path) -> BM25Index:
    """
    Loads BM25 from cache if exists, otherwise builds and saves it.
    Cache valid ONLY if chunks order/length didn't change.
    """
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, dict) or obj.get("n_chunks") != len(chunks):
                raise ValueError("BM25 cache mismatch (different number of chunks).")
            bm25 = obj["bm25"]
            if VERBOSE:
                print(f"[CACHE] BM25 loaded: {cache_path.name}")
            return BM25Index(chunks=chunks, bm25=bm25)
        except Exception as e:
            print(f"[WARN] Failed to load BM25 cache {cache_path.name}: {e} -> rebuilding")

    idx = build_bm25_index(chunks)

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"n_chunks": len(chunks), "bm25": idx.bm25}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CACHE] BM25 saved: {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save BM25 cache {cache_path.name}: {e}")

    return idx


def retrieve_bm25(index: BM25Index, query: str, k: int) -> List[Dict[str, Any]]:
    q_tokens = query.split()
    scores = index.bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:k]

    out = []
    for i in top_idx:
        c = index.chunks[int(i)]
        out.append({
            "score": float(scores[int(i)]),
            "chunk_id": c.get("chunk_id"),
            "parent_id": c.get("parent_id"),
            "filename": c.get("filename"),
            "date": c.get("date"),
            "parliament": c.get("parliament"),
            "text": c.get("text", ""),
        })
    return out


# ============================================================
# Retrieval: Embeddings
# ============================================================
@dataclass
class EmbIndex:
    chunks: List[Dict[str, Any]]
    embeddings: np.ndarray
    embedder: SentenceTransformer


def build_emb_index(chunks: List[Dict[str, Any]], emb_path: Path, embedder: SentenceTransformer) -> EmbIndex:
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
    embeddings = np.load(emb_path)
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embeddings/chunks mismatch: embeddings={embeddings.shape[0]} chunks={len(chunks)}\n"
            f"-> Must be SAME order and SAME length."
        )
    return EmbIndex(chunks=chunks, embeddings=embeddings, embedder=embedder)


def retrieve_emb(index: EmbIndex, query: str, k: int) -> List[Dict[str, Any]]:
    q_emb = index.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = (index.embeddings @ q_emb.T).squeeze()
    top_idx = np.argsort(scores)[::-1][:k]

    out = []
    for i in top_idx:
        c = index.chunks[int(i)]
        out.append({
            "score": float(scores[int(i)]),
            "chunk_id": c.get("chunk_id"),
            "parent_id": c.get("parent_id"),
            "filename": c.get("filename"),
            "date": c.get("date"),
            "parliament": c.get("parliament"),
            "text": c.get("text", ""),
        })
    return out




# ============================================================
# Main runner
# ============================================================
def run_suite(suite_name: str, queries_path: Path) -> Path:
    """
    suite_name: "required" or "custom"
    Runs: 8 queries (4 factual + 4 conceptual) × 4 configs × 3 K
    Saves JSONL file with suite_name in filename.
    """
    queries_obj = load_queries(queries_path)
    factual = queries_obj["factual"]
    conceptual = queries_obj["conceptual"]

    # Flatten with type tags
    all_queries: List[Dict[str, str]] = []
    for q in factual:
        all_queries.append({"type": "factual", "id": q["id"], "text": q["text"]})
    for q in conceptual:
        all_queries.append({"type": "conceptual", "id": q["id"], "text": q["text"]})

    # Load chunks
    fixed_chunks = load_jsonl(CHUNKS_FIXED)
    child_chunks = load_jsonl(CHUNKS_CHILD)

    # Build embedder (for query embedding)
    embedder = SentenceTransformer(EMB_MODEL_NAME)

    # (3) BM25 indices with cache
    bm25_fixed = load_or_build_bm25_index(fixed_chunks, BM25_CACHE_FIXED)
    bm25_child = load_or_build_bm25_index(child_chunks, BM25_CACHE_CHILD)

    # Embeddings indices
    emb_child_index = build_emb_index(child_chunks, EMB_CHILD, embedder)

    emb_fixed_index: Optional[EmbIndex] = None
    if EMB_FIXED.exists():
        emb_fixed_index = build_emb_index(fixed_chunks, EMB_FIXED, embedder)
    else:
        print(f"[WARN] fixed embeddings not found: {EMB_FIXED} -> fixed+emb configs will be SKIPPED.")

    # Configs required by the assignment
    configs = [
        ("fixed", "bm25"),
        ("child", "bm25"),
        ("fixed", "emb"),
        ("child", "emb"),
    ]

    expected_runs = len(all_queries) * len(configs) * len(K_VALUES)
    print(f"Running {len(all_queries)} queries × {len(configs)} configs × {len(K_VALUES)} K = {expected_runs} runs")
    print(f"Suite: {suite_name} | queries: {queries_path.name}")

    results: List[Dict[str, Any]] = []
    run_count = 0

    for q in all_queries:
        q_type = q["type"]
        q_id = q["id"]
        query_text = q["text"]

        for (chunking, repr_name) in configs:
            # Skip fixed+emb if we don't have fixed embeddings
            if repr_name == "emb" and chunking == "fixed" and emb_fixed_index is None:
                continue

            for k in K_VALUES:
                run_count += 1
                t0 = time.time()

                # --- retrieval ---
                if repr_name == "bm25":
                    idx = bm25_fixed if chunking == "fixed" else bm25_child
                    retrieved = retrieve_bm25(idx, query_text, k)
                else:
                    idx2 = emb_fixed_index if chunking == "fixed" else emb_child_index
                    retrieved = retrieve_emb(idx2, query_text, k)  # type: ignore[arg-type]

                # Add file links per chunk
                for c in retrieved:
                    lk = file_link(c["filename"], c["parliament"])
                    c["file_abs_path"] = lk["abs_path"]
                    c["file_uri"] = lk["file_uri"]

                # (7) minimal terminal output
                if VERBOSE or (run_count % PRINT_EVERY_N_RUNS) == 0:
                    print(f"OK [{run_count}/{expected_runs}] | suite={suite_name} | {q_type} {q_id} | {chunking}/{repr_name} k={k}")

                # Optional: show only top-N sources sometimes
                if PRINT_TOP_SOURCES > 0 and (VERBOSE or (run_count % PRINT_EVERY_N_RUNS) == 0):
                    print("Top sources (clickable):")
                    for i, c in enumerate(retrieved[:PRINT_TOP_SOURCES], start=1):
                        print(f"  Doc {i}: {c['filename']} ({c['parliament']})")
                        print(f"        {c['file_uri']}")

                # --- RAG ---
                prompt, sources = build_prompt(query_text, retrieved)
                answer = generate_with_ollama(prompt)

                dt = time.time() - t0

                record = {
                    "suite": suite_name,                 # required/custom
                    "query_type": q_type,                # factual/conceptual
                    "query_id": q_id,
                    "query": query_text,
                    "chunking": chunking,                # fixed/child
                    "representation": repr_name,          # bm25/emb
                    "k": k,
                    "elapsed_sec": round(dt, 4),
                    "retrieved": retrieved,               # includes file_uri + abs_path
                    "sources_unique": sources,            # list w/ file_uri
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                results.append(record)

    out_path = RESULTS_DIR / f"run_{suite_name}_{now_tag()}.jsonl"
    save_jsonl(out_path, results)
    print(f"Saved: {out_path}")
    return out_path


def main():
    # Run REQUIRED suite
    if REQUIRED_QUERIES_PATH.exists():
        run_suite("required", REQUIRED_QUERIES_PATH)
    else:
        print(f"[WARN] Missing required queries file: {REQUIRED_QUERIES_PATH}")

    # Run CUSTOM suite
    if CUSTOM_QUERIES_PATH.exists():
        run_suite("custom", CUSTOM_QUERIES_PATH)
    else:
        print(f"[WARN] Missing custom queries file: {CUSTOM_QUERIES_PATH}")


if __name__ == "__main__":
    main()
