# ============================================================
# Embedding generation for RAG (Child chunks & Fixed chunks)
# ------------------------------------------------------------
# IMPORTANT NOTE:
# This script was executed on Google Colab with GPU enabled
# (Runtime → Change runtime type → GPU)
#
# Reason:
# Computing sentence embeddings for ~60k chunks is very slow
# on CPU. Using a GPU significantly reduces runtime.
#
# The generated .npy files are then downloaded and reused
# locally in the RAG pipeline.
# ============================================================

# ---------------------------
# 1. INSTALL DEPENDENCIES
# ---------------------------
# (Already installed on Colab in most cases)
!pip install -U sentence-transformers numpy tqdm

# ---------------------------
# 2. IMPORTS
# ---------------------------
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ---------------------------
# 3. CHECK GPU
# ---------------------------
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ---------------------------
# 4. LOAD MODEL
# ---------------------------
# MPNet is a strong bi-encoder model for semantic retrieval
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME, device="cuda")

# ---------------------------
# 5. UTILITY: LOAD JSONL
# ---------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# ============================================================
# 6. CHILD CHUNKS EMBEDDINGS
# ============================================================

print("\n=== Generating embeddings for CHILD chunks ===")

CHILD_CHUNKS_PATH = "/content/child_chunks.jsonl"
CHILD_EMB_OUT = "/content/child_embeddings_mpnet.npy"

child_chunks = load_jsonl(CHILD_CHUNKS_PATH)
child_texts = [c["text"] for c in child_chunks]

print("Number of child chunks:", len(child_texts))

child_embeddings = model.encode(
    child_texts,
    batch_size=64,               # GPU-friendly batch size
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(CHILD_EMB_OUT, child_embeddings)

print("Saved:", CHILD_EMB_OUT)
print("Shape:", child_embeddings.shape)

# ============================================================
# 7. FIXED CHUNKS EMBEDDINGS
# ============================================================

print("\n=== Generating embeddings for FIXED chunks ===")

FIXED_CHUNKS_PATH = "/content/fixed_660w_3s.jsonl"
FIXED_EMB_OUT = "/content/fixed_embeddings_mpnet.npy"

fixed_chunks = load_jsonl(FIXED_CHUNKS_PATH)
fixed_texts = [c["text"] for c in fixed_chunks]

print("Number of fixed chunks:", len(fixed_texts))

fixed_embeddings = model.encode(
    fixed_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(FIXED_EMB_OUT, fixed_embeddings)

print("Saved:", FIXED_EMB_OUT)
print("Shape:", fixed_embeddings.shape)

# ============================================================
# 8. DOWNLOAD RESULTS
# ============================================================
# After execution, download the files:
# - child_embeddings_mpnet.npy
# - fixed_embeddings_mpnet.npy
#
# These files are copied into:
# data/embeddings/
# and reused by the local RAG pipeline.
# ============================================================
