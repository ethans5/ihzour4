# retrieval/embedding_retriever.py
from __future__ import annotations

from typing import List, Dict, Any
import json
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# On pointe vers le bon utils, quitte à modifier le path si besoin
# pour importer convert_date_to_timestamp
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import convert_date_to_timestamp


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EmbeddingRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, model_name: str):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: chunks={len(chunks)} vs embeddings={embeddings.shape[0]} "
                "(il faut même ordre et même taille)."
            )

        self.chunks = chunks

        # Important: embeddings doivent être en float32 pour accélérer + réduire RAM
        # (ça ne change quasiment pas les résultats)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)
        self.embeddings = embeddings

        # Le modèle est chargé UNE fois ici (c'est déjà ton cas)
        self.model = SentenceTransformer(model_name)

        # Pré-calcul des timestamps pour chaque chunk (pour Mode A et B)
        # On stocke dans un array numpy float32 pour vectorisation
        self.chunk_timestamps = np.zeros(len(chunks), dtype=np.float32)
        self.chunk_years = np.zeros(len(chunks), dtype=np.int32)
        
        for i, c in enumerate(chunks):
            d_str = c.get("date", "1970-01-01")
            ts = convert_date_to_timestamp(d_str)
            self.chunk_timestamps[i] = ts
            # Pour filter_year
            # On parse vite fait l'année depuis le string (plus simple que datetime object complet)
            # format attendu YYYY-MM-DD
            try:
                y = int(d_str[:4])
            except:
                y = 1970
            self.chunk_years[i] = y

        # (1) Cache: évite de ré-encoder la même requête plusieurs fois
        # clé = texte de la requête, valeur = embedding normalisé shape (768,)
        self._query_cache: Dict[str, np.ndarray] = {}

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode + normalize query, with cache."""
        q = query.strip()
        if q in self._query_cache:
            return self._query_cache[q]

        emb = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.squeeze().astype(np.float32, copy=False)  # shape (768,)
        self._query_cache[q] = emb
        return emb

    def search(
        self,
        query: str,
        k: int,
        filter_year: int = None,
        filter_date_range: tuple = None,
        alpha: float = 0.0,
        lambda_decay: float = 0.0,
        query_date: float = None
    ) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        # Encode query (cached)
        q = self._encode_query(query)  # shape (768,)

        # Cosine similarity because embeddings are normalized
        # scores shape: (N,)
        cosine_sim = self.embeddings @ q
        
        # --- MODE A: Hard Filtering ---
        # On applique un masque booléen (-inf pour les chunks exclus)
        mask = np.ones(len(self.chunks), dtype=bool)
        
        if filter_year is not None:
            mask &= (self.chunk_years == filter_year)
            
        if filter_date_range is not None:
            start_date, end_date = filter_date_range
            # start/end peuvent être str ou float, supposons str -> convert ? 
            # Pour simplifier l'appel, acceptons float timestamp ou gérons string avant.
            # Ici on on suppose que l'appelant a déjà converti ou on compare simple.
            # MAIS le prompt demande "filter_date_range", restons simple sur timestamps
            # Si tuple de strings -> conversion
            if isinstance(start_date, str):
                start_date = convert_date_to_timestamp(start_date)
            if isinstance(end_date, str):
                end_date = convert_date_to_timestamp(end_date)
                
            mask &= (self.chunk_timestamps >= start_date)
            mask &= (self.chunk_timestamps <= end_date)
            
        # --- MODE B: Recency Weighting (Soft Decay) ---
        # Score = (1 - alpha) * Sim + alpha * (1 / (1 + lambda * delta_t))
        # delta_t = abs(query_date - chunk_date)
        # On suppose query_date fourni, sinon on prend "now" ou max chunk date
        
        scores = cosine_sim # base
        
        if alpha > 0 and lambda_decay > 0:
            if query_date is None:
                # Fallback: take max date in corpus or current time? 
                # Let's use max date in chunks to measure "recency" relative to dataset
                # query_date = self.chunk_timestamps.max()
                # Ou simplement utiliser time.time() si pertinent, mais max() reste cohérent avec le corpus
                query_date = self.chunk_timestamps.max()
            
            # Calcul delta_t (en années pour que lambda soit 'lisible' ex: 0.1)
            # 1 an env 31536000 sec
            SECONDS_PER_YEAR = 31536000.0
            delta_t_sec = np.abs(query_date - self.chunk_timestamps)
            delta_t_years = delta_t_sec / SECONDS_PER_YEAR
            
            decay_factor = 1.0 / (1.0 + lambda_decay * delta_t_years)
            
            # Combinaison
            # Note: cosine_sim est [-1, 1], decay est [0, 1].
            # On mixe.
            scores = (1 - alpha) * cosine_sim + alpha * decay_factor

        # Application du masque (Hard Filtering)
        # On met score = -9999 pour les items masqués
        scores[~mask] = -9999.0

        # --- FIX: Check if we have valid results ---
        # Si le score max est -9999.0, cela signifie que TOUT a été filtré.
        # On renvoie une liste vide pour éviter de renvoyer des "faux" résultats.
        if np.max(scores) == -9999.0:
            return []
        # -------------------------------------------

        # (2) Top-K faster: argpartition instead of full argsort
        n = scores.shape[0]
        k = min(k, n)

        # indices of top-k (unordered), then sort only those k by score desc
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        out: List[Dict[str, Any]] = []
        for rank, i in enumerate(top_idx, start=1):
            # Double sécurité (si k > nombre de résultats valides)
            if scores[int(i)] == -9999.0:
                continue

            c = self.chunks[int(i)]
            out.append({
                "rank": rank,
                "score": float(scores[int(i)]),
                "chunk_id": c.get("chunk_id"),
                "parent_id": c.get("parent_id"),
                "filename": c.get("filename"),
                "date": c.get("date"),
                "parliament": c.get("parliament"),
                "text": c.get("text", ""),
            })
        return out