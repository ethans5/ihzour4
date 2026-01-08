import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

RESULTS_FILES = [
    Path("evaluation/results/run_required_20251223_174914.jsonl"),
    Path("evaluation/results/run_custom_20251223_182011.jsonl"),
]

EMB_OUT_PATH = Path("evaluation/results/answers_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_data(paths):
    rows = []
    texts = []
    
    for p in paths:
        if not p.exists():
            print(f"Missing file: {p}")
            continue

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                r = json.loads(line)
                ans = r.get("answer")
                
                if not ans:
                    continue

                chunking = r.get("chunking")
                repr_ = r.get("representation")
                
                if chunking and repr_:
                    r["config"] = f"{chunking}/{repr_}"
                    rows.append(r)
                    texts.append(ans)

    return rows, texts

def compute_embeddings(texts):
    print(f"Generating embeddings for {len(texts)} answers...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings

def cosine_matrix(emb):
    return emb @ emb.T

def group_key(r):
    return (r.get("query_id"), r.get("config"))

def main():
    # 1. Chargement des données
    rows, texts = load_data(RESULTS_FILES)
    print(f"Loaded {len(rows)} rows.")

    if not rows:
        print("No data found.")
        return

    # 2. Génération des embeddings (Partie activée)
    if EMB_OUT_PATH.exists():
        print(f"Loading existing embeddings from {EMB_OUT_PATH}")
        emb = np.load(EMB_OUT_PATH)
        if len(rows) != emb.shape[0]:
            print("Mismatch found, recomputing...")
            emb = compute_embeddings(texts)
            np.save(EMB_OUT_PATH, emb)
    else:
        emb = compute_embeddings(texts)
        np.save(EMB_OUT_PATH, emb)
        print(f"Saved embeddings to {EMB_OUT_PATH}")

    # 3. Analyse
    cos = cosine_matrix(emb)

    # A) Stabilité selon K
    by_qc = {}
    for i, r in enumerate(rows):
        key = group_key(r)
        by_qc.setdefault(key, []).append(i)

    stabilities = []
    for key, idxs in by_qc.items():
        if len(idxs) < 2:
            continue

        sims = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                sims.append(float(cos[idxs[a], idxs[b]]))

        stabilities.append((key, float(np.mean(sims)), len(idxs)))

    stabilities.sort(key=lambda x: x[1])

    print("\n=== Top 10 INSTABLE Configs (variance on K) ===")
    for (qid, cfg), s, n in stabilities[:10]:
        print(f"{qid} | {cfg} | n={n} | mean_cos={s:.3f}")

    print("\n=== Top 10 STABLE Configs (variance on K) ===")
    for (qid, cfg), s, n in stabilities[-10:]:
        print(f"{qid} | {cfg} | n={n} | mean_cos={s:.3f}")

    # B) Comparaison BM25 vs EMB
    index_map = {}
    for i, r in enumerate(rows):
        qid = r.get("query_id")
        cfg = r.get("config")
        k = r.get("k")
        if qid and cfg and k:
            index_map[(qid, cfg, k)] = i

    pairs = []
    for (qid, cfg, k), i_bm25 in index_map.items():
        if cfg.endswith("/bm25"):
            cfg_emb = cfg.replace("/bm25", "/emb")
            j = index_map.get((qid, cfg_emb, k))
            if j is not None:
                pairs.append((qid, cfg, cfg_emb, k, float(cos[i_bm25, j])))

    if pairs:
        pairs.sort(key=lambda x: x[4])

        print("\n=== Top 10 Different answers (BM25 vs EMB) ===")
        for qid, c1, c2, k, s in pairs[:10]:
            print(f"{qid} | {c1} vs {c2} | k={k} | cos={s:.3f}")

        print("\n=== Top 10 Similar answers (BM25 vs EMB) ===")
        for qid, c1, c2, k, s in pairs[-10:]:
            print(f"{qid} | {c1} vs {c2} | k={k} | cos={s:.3f}")
    else:
        print("\nNo pairs found for BM25 vs EMB comparison.")

if __name__ == "__main__":
    main()