import re, os, json, math, gzip, pathlib, logging
from typing import List, Iterable, Tuple, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PUNCT_TO_SEPARATE = r"""!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~—–“”‘’"""
# Keep hyphens separated but preserve things like "IDF" and acronym patterns by later rules if needed.

def ensure_dir(p:str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def extract_date_from_filename(name:str) -> str:
    # Expected: debatesYYYY-MM-DD?.xml (suffix letter optional like 'd')
    m = re.search(r"debates(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    # fallback: try ANY yyyy-mm-dd
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else "unknown"

def tokenize_with_punct_separated(text:str) -> List[str]:
    # Separate punctuation as standalone tokens while preserving internal apostrophes in words like don't
    # Strategy: Use regex to add spaces around "punct" except apostrophe in middle of word.
    # Then split on whitespace.
    punct = PUNCT_TO_SEPARATE.replace("'", "")  # handle apostrophe separately
    text = re.sub(f"([{re.escape(punct)}])", r" \1 ", text)
    # Handle apostrophes: split only if not between letters
    text = re.sub(r"(\A|[^A-Za-z])'|'([^A-Za-z]|\Z)", r" \1' \2", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def save_jsonl(path:str, rows:Iterable[Dict[str,Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def iter_files(root:str, ext=(".xml", ".txt")):
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(ext):
                yield os.path.join(dirpath, fn)


def read_token_docs_for_ex2(mode: str = "Lemm"):
    """
    Lit les documents de l'exercice 2 (UK + US).

    Structure attendue (exo3) :
    exo3/
      utils.py
      data/
        ex2_clean/
          UK/
          US/
        ex2_lemm/
          UK/
          US/

    mode = "Word"  -> ex2_clean (recommandé pour chunking par phrases)
    mode = "Lemm"  -> ex2_lemm  (recommandé pour BM25 / embeddings)

    Retourne :
    - docs : list[list[str]]  (tokens)
    - meta : list[dict]       (filename, parliament, date)
    """
    base_dir = pathlib.Path(__file__).resolve().parent   # exo3/
    data_dir = base_dir / "data"

    if mode.lower() == "word":
        root = data_dir / "ex2_clean"
    elif mode.lower() == "lemm":
        root = data_dir / "ex2_lemm"
    else:
        raise ValueError("mode must be 'Word' or 'Lemm'")

    docs: List[List[str]] = []
    meta: List[Dict[str, Any]] = []

    for parl in ["UK", "US"]:
        folder = root / parl
        if not folder.exists():
            raise FileNotFoundError(f"Dossier introuvable: {folder}")

        for path in sorted(folder.glob("*.txt")):
            with open(path, "r", encoding="utf-8") as f:
                tokens = f.read().strip().split()

            docs.append(tokens)
            meta.append({
                "filename": path.name,
                "parliament": parl,
                "date": extract_date_from_filename(path.name)
            })

    logging.info(f"[EX2] Loaded {len(docs)} documents (mode={mode})")
    return docs, meta

def parse_date_string(date_str: str) -> datetime:
    """Parses 'YYYY-MM-DD' string to datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # Fallback or error handling
        return datetime(1970, 1, 1)

def convert_date_to_timestamp(date_str: str) -> float:
    """Converts 'YYYY-MM-DD' string to Unix timestamp (float)."""
    dt = parse_date_string(date_str)
    return dt.timestamp()
