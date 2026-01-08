# 04_search_embeddings.py
# ------------------------------------------------------------
# Recherche sémantique avec embeddings PRE-CALCULÉS (MPNet)
# ------------------------------------------------------------

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# chemins
CHUNKS_PATH = Path("data/chunks/child_chunks.jsonl")
EMB_PATH = Path("data/embeddings/child_embeddings_mpnet.npy")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ------------------------------------------------------------
# Charger les chunks
# ------------------------------------------------------------
def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== Semantic search with MPNet embeddings ===")

    # 1) charger chunks
    chunks = load_chunks(CHUNKS_PATH)
    print("Nb chunks:", len(chunks))

    # 2) charger embeddings
    embeddings = np.load(EMB_PATH)
    print("Embeddings shape:", embeddings.shape)

    # 3) charger le modèle (pour encoder la requête seulement)
    model = SentenceTransformer(MODEL_NAME)

    # 4) requête test
    query = "food security hunger crisis humanitarian assistance"
    q_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 5) similarité cosinus (produit scalaire car normalisé)
    scores = (embeddings @ q_emb.T).squeeze()

    # 6) top-k
    top_k = 5
    top_idx = np.argsort(scores)[::-1][:top_k]

    # 7) afficher résultats
    for rank, idx in enumerate(top_idx, start=1):
        c = chunks[idx]
        print(
            f"\n#{rank} | score={float(scores[idx]):.4f} | "
            f"{c['parliament']} | {c['date']} | {c['chunk_id']}"
        )
        print(c["text"][:350])
