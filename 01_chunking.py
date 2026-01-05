# 01_chunking.py
# ------------------------------------------------------------
# שלב א – Chunking (VERSION FINALE CONFORME)
#
# Méthode 1 :
# - phrases entières
# - max 660 mots
# - overlap = 3 phrases
#
# Méthode 2 :
# - Father–Son (Parent–Child)
#
# Les chunks sont sauvegardés dans :
# data/chunks/
# ------------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import json
from pathlib import Path

from utils import read_token_docs_for_ex2


# ------------------------------------------------------------
# 0) Découpage en phrases
# ------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    try:
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]


def count_words(s: str) -> int:
    return len(s.split())


# ------------------------------------------------------------
# Sécurité : phrase trop longue
# ------------------------------------------------------------
def force_split_long_sentence(s: str, max_words: int) -> List[str]:
    if count_words(s) <= max_words:
        return [s]

    parts = re.split(r"(?<=[,;])\s+", s)
    out, cur, wc = [], [], 0

    for p in parts:
        w = count_words(p)
        if cur and wc + w > max_words:
            out.append(" ".join(cur))
            cur, wc = [], 0
        cur.append(p)
        wc += w

    if cur:
        out.append(" ".join(cur))

    final = []
    for o in out:
        if count_words(o) <= max_words:
            final.append(o)
        else:
            words = o.split()
            for i in range(0, len(words), max_words):
                final.append(" ".join(words[i:i + max_words]))
    return final


# ------------------------------------------------------------
# 1) MÉTHODE 1 — Fixed 660 mots + overlap 3 phrases
# ------------------------------------------------------------
def chunk_fixed_by_size_sentences(
    text: str,
    doc_id: str,
    max_words: int = 660,
    overlap_sentences: int = 3
) -> List[Dict[str, Any]]:
    raw_sents = split_sentences(text)

    sents: List[str] = []
    for s in raw_sents:
        sents.extend(force_split_long_sentence(s, max_words))

    chunks = []
    i = 0
    idx = 0

    while i < len(sents):
        cur, wc = [], 0
        j = i

        while j < len(sents):
            w = count_words(sents[j])
            if cur and wc + w > max_words:
                break
            cur.append(sents[j])
            wc += w
            j += 1

        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_F{idx}",
            "parent_id": None,
            "level": "fixed_660w_overlap3s",
            "text": " ".join(cur),
            "num_sentences": len(cur),
            "num_words": wc
        })

        idx += 1
        i = max(j - overlap_sentences, i + 1)

    return chunks


# ------------------------------------------------------------
# 2) MÉTHODE 2 — Father–Son
# ------------------------------------------------------------
def build_parent_chunks_sentences(
    text: str,
    doc_id: str,
    parent_max_words: int = 900,
    parent_overlap_sentences: int = 2
) -> List[Dict[str, Any]]:
    sents = split_sentences(text)
    parents = []
    i = 0
    idx = 0

    while i < len(sents):
        cur, wc = [], 0
        j = i

        while j < len(sents):
            w = count_words(sents[j])
            if cur and wc + w > parent_max_words:
                break
            cur.append(sents[j])
            wc += w
            j += 1

        parents.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_P{idx}",
            "parent_id": None,
            "level": "parent",
            "text": " ".join(cur),
            "num_sentences": len(cur),
            "num_words": wc
        })

        idx += 1
        i = max(j - parent_overlap_sentences, i + 1)

    return parents


def build_child_chunks_from_parent_sentences(
    parent: Dict[str, Any],
    child_max_words: int = 250,
    child_overlap_sentences: int = 1
) -> List[Dict[str, Any]]:
    sents = split_sentences(parent["text"])
    children = []
    i = 0
    idx = 0

    while i < len(sents):
        cur, wc = [], 0
        j = i

        while j < len(sents):
            w = count_words(sents[j])
            if cur and wc + w > child_max_words:
                break
            cur.append(sents[j])
            wc += w
            j += 1

        children.append({
            "doc_id": parent["doc_id"],
            "chunk_id": f"{parent['chunk_id']}_C{idx}",
            "parent_id": parent["chunk_id"],
            "level": "child",
            "text": " ".join(cur),
            "num_sentences": len(cur),
            "num_words": wc
        })

        idx += 1
        i = max(j - child_overlap_sentences, i + 1)

    return children


def father_son_chunking_sentences(
    text: str,
    doc_id: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    parents = build_parent_chunks_sentences(text, doc_id)
    children = []
    for p in parents:
        children.extend(build_child_chunks_from_parent_sentences(p))
    return parents, children


# ------------------------------------------------------------
# 3) PIPELINE EX2
# ------------------------------------------------------------
def build_all_chunks_ex2(mode: str = "Word"):
    docs, meta = read_token_docs_for_ex2(mode)

    fixed, parents, children = [], [], []

    for i, tokens in enumerate(docs):
        filename = meta[i]["filename"]
        doc_id = filename.replace(".txt", "")
        parliament = meta[i]["parliament"]
        date = meta[i]["date"]

        text = " ".join(tokens)
        date_header = f"[Document date: {date}]\n"
        text = date_header + text

        fc = chunk_fixed_by_size_sentences(text, doc_id)
        for c in fc:
            c.update({"filename": filename, "parliament": parliament, "date": date})
        fixed.extend(fc)

        ps, cs = father_son_chunking_sentences(text, doc_id)
        for p in ps:
            p.update({"filename": filename, "parliament": parliament, "date": date})
        for c in cs:
            c.update({"filename": filename, "parliament": parliament, "date": date})

        parents.extend(ps)
        children.extend(cs)

    return fixed, parents, children


# ------------------------------------------------------------
# 4) MAIN + SAUVEGARDE
# ------------------------------------------------------------
if __name__ == "__main__":
    fixed, parents, children = build_all_chunks_ex2(mode="Word")

    print("=== CHUNKING EX2 (CLEAN) ===")
    print("Fixed chunks  :", len(fixed))
    print("Parent chunks :", len(parents))
    print("Child chunks  :", len(children))
    print("Max words in fixed chunks:", max(c["num_words"] for c in fixed))

    out_dir = Path("data/chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(path: Path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    save_jsonl(out_dir / "fixed_660w_3s.jsonl", fixed)
    save_jsonl(out_dir / "parent_chunks.jsonl", parents)
    save_jsonl(out_dir / "child_chunks.jsonl", children)

    print("✔ Chunks sauvegardés dans data/chunks/")
