import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# Paths
# ----------------------------
RUN_FILES = [
    Path("evaluation/results/run_required_20251223_174914.jsonl"),
    Path("evaluation/results/run_custom_20251223_182011.jsonl"),
]
EMB_PATH = Path("evaluation/results/answers_embeddings.npy")

OUT_DIR = Path("evaluation/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load runs (keep same order as embeddings!)
# ----------------------------
rows = []
for p in RUN_FILES:
    if not p.exists():
        print(f"[WARN] Missing run file: {p}")
        continue

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # Keep only rows with an answer (since embeddings were built from answers)
            if r.get("answer"):
                # config string: fixed/bm25, child/emb, etc.
                r["config"] = f"{r.get('chunking')}/{r.get('representation')}"
                rows.append(r)

if not rows:
    raise RuntimeError("No rows loaded. Check your JSONL paths.")

if not EMB_PATH.exists():
    raise FileNotFoundError(f"Missing embeddings file: {EMB_PATH}")

emb = np.load(EMB_PATH)  # embeddings normalized
if len(rows) != emb.shape[0]:
    raise RuntimeError(
        f"Mismatch rows vs embeddings: rows={len(rows)} emb={emb.shape[0]}.\n"
        "➡️ Ensure answers_embeddings.npy was generated from the SAME JSONL files in the SAME order."
    )

# Cosine similarity matrix (since normalized)
cos = emb @ emb.T


# ============================================================
# Helper: stability across K for same (query_id, config)
# ============================================================
def group_key_stability(r):
    return (r.get("query_id"), r.get("config"))


# Build index groups: (query_id, config) -> list of row indices
by_qc = defaultdict(list)
for i, r in enumerate(rows):
    by_qc[group_key_stability(r)].append(i)

# Compute stability list: one value per (query_id, config)
# store also query_type for Plot C
stability_items = []  # dicts with: query_id, config, mean_cos, query_type
for (qid, cfg), idxs in by_qc.items():
    if len(idxs) < 2:
        continue

    sims = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            sims.append(float(cos[idxs[a], idxs[b]]))

    # query_type should be the same for same query_id, but we take from first row safely
    qt = rows[idxs[0]].get("query_type", "unknown")
    stability_items.append({
        "query_id": qid,
        "config": cfg,
        "mean_cos": float(np.mean(sims)),
        "query_type": qt
    })

if not stability_items:
    raise RuntimeError("No stability items computed (maybe you only have one K per config?).")


# ============================================================
# Plot A: Mean stability by configuration
# ============================================================
cfg_vals = defaultdict(list)
for it in stability_items:
    cfg_vals[it["config"]].append(it["mean_cos"])

cfgs = sorted(cfg_vals.keys())
means = [float(np.mean(cfg_vals[c])) for c in cfgs]

plt.figure(figsize=(10, 5))
plt.bar(cfgs, means)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean cosine similarity (stability across K)")
plt.title("Plot A — Stability by configuration")
plt.tight_layout()

out_a = OUT_DIR / "plotA_stability_by_config.png"
plt.savefig(out_a, dpi=200, bbox_inches="tight")
print("Saved:", out_a.resolve())
plt.show()
plt.close()


# ============================================================
# Plot B: Histogram — BM25 vs EMB similarity (same query_id, same chunking, same K)
# We compare answers embeddings between bm25 and emb for matching tuples.
# ============================================================
index = {}
for i, r in enumerate(rows):
    qid = r.get("query_id")
    chunking = r.get("chunking")
    repr_ = r.get("representation")
    k = r.get("k")
    if qid is None or chunking is None or repr_ is None or k is None:
        continue
    index[(qid, chunking, int(k), repr_)] = i

pair_cos = []
pair_meta = []  # optional for debug
for (qid, chunking, k, repr_), i_bm25 in index.items():
    if repr_ != "bm25":
        continue
    j_emb = index.get((qid, chunking, k, "emb"))
    if j_emb is None:
        continue
    pair_cos.append(float(cos[i_bm25, j_emb]))
    pair_meta.append((qid, chunking, k))

if not pair_cos:
    print("[WARN] No BM25 vs EMB pairs found. Plot B will be skipped.")
else:
    plt.figure(figsize=(8, 5))
    plt.hist(pair_cos, bins=20)
    plt.xlabel("Cosine similarity between BM25-answer and EMB-answer")
    plt.ylabel("Count")
    plt.title("Plot B — BM25 vs Embeddings answer similarity (matched by query/chunking/K)")
    plt.tight_layout()

    out_b = OUT_DIR / "plotB_hist_bm25_vs_emb.png"
    plt.savefig(out_b, dpi=200, bbox_inches="tight")
    print("Saved:", out_b.resolve())
    plt.show()
    plt.close()


# ============================================================
# Plot C: Stability by query type (Factual vs Conceptual)
# We average stability_items over query_type.
# ============================================================
type_vals = defaultdict(list)
for it in stability_items:
    qt = it["query_type"] or "unknown"
    type_vals[qt].append(it["mean_cos"])

# We focus on these two if present
order = []
for t in ["factual", "conceptual"]:
    if t in type_vals:
        order.append(t)

# if other types exist, append them at the end
for t in sorted(type_vals.keys()):
    if t not in order:
        order.append(t)

type_means = [float(np.mean(type_vals[t])) for t in order]

plt.figure(figsize=(7, 5))
plt.bar(order, type_means)
plt.ylabel("Mean cosine similarity (stability across K)")
plt.title("Plot C — Stability by query type")
plt.tight_layout()

out_c = OUT_DIR / "plotC_stability_by_query_type.png"
plt.savefig(out_c, dpi=200, bbox_inches="tight")
print("Saved:", out_c.resolve())
plt.show()
plt.close()

print("\nDone. All plots saved in:", OUT_DIR.resolve())
