# 02_bm25.py
# ------------------------------------------------------------
# שלב ב' — Représentation vectorielle : BM25
# - Charge les chunks (méthode 1) depuis data/chunks/fixed_660w_3s.jsonl
# - Construit un index BM25
# - Permet de faire une requête et récupérer les Top-K chunks
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path("data/chunks/fixed_660w_3s.jsonl")


# ------------------------------------------------------------
# 1) Lecture JSONL
# ------------------------------------------------------------
def load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path.resolve()}")

    chunks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


# ------------------------------------------------------------
# 2) Tokenisation simple (compatible ex2_clean)
# ------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    # text déjà "clean" (tokens séparés). split suffit.
    return text.lower().split()


# ------------------------------------------------------------
# 3) Construction BM25
# ------------------------------------------------------------
def build_bm25_index(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[List[str]]]:
    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, corpus_tokens


# ------------------------------------------------------------
# 4) Recherche Top-K
# ------------------------------------------------------------
def bm25_search(
    bm25: BM25Okapi,
    chunks: List[Dict[str, Any]],
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    # top_k indices triés par score décroissant
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in ranked_idx:
        c = chunks[i]
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[i]),
            "chunk_id": c.get("chunk_id"),
            "doc_id": c.get("doc_id"),
            "parliament": c.get("parliament"),
            "date": c.get("date"),
            "text": c.get("text", "")
        })
    return results


# ------------------------------------------------------------
# 5) Affichage propre des résultats
# ------------------------------------------------------------
def print_results(results: List[Dict[str, Any]], max_chars: int = 350):
    for r in results:
        preview = r["text"][:max_chars].replace("\n", " ")
        print(f"\n#{r['rank']} | score={r['score']:.4f} | {r['parliament']} | {r['date']} | {r['chunk_id']}")
        print(preview)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== BM25 on fixed chunks (660w, overlap 3 sentences) ===")
    print("Loading chunks:", CHUNKS_PATH)

    chunks = load_chunks_jsonl(CHUNKS_PATH)
    print("Nb chunks:", len(chunks))

    bm25, _ = build_bm25_index(chunks)
    print("BM25 index built ✅")

    # ---------
    # Requête test (tu peux changer)
    # ---------
    query = "food security hunger crisis humanitarian assistance"
    top_k = 5

    print(f"\nQuery: {query!r} | top_k={top_k}")
    results = bm25_search(bm25, chunks, query=query, top_k=top_k)
    print_results(results)
