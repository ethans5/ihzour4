# retrieval/bm25_retriever.py
from __future__ import annotations
from typing import List, Dict, Any
import json
from pathlib import Path
from rank_bm25 import BM25Okapi


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        out = []
        for rank, i in enumerate(top_idx, start=1):
            c = self.chunks[i]
            out.append({
                "rank": rank,
                "score": float(scores[i]),
                "chunk_id": c.get("chunk_id"),
                "parent_id": c.get("parent_id"),
                "filename": c.get("filename"),
                "date": c.get("date"),
                "parliament": c.get("parliament"),
                "text": c.get("text", ""),
            })
        return out
