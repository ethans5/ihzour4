import json
from pathlib import Path
import csv
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluation/summarize_results.py evaluation/results/run_YYYYMMDD_HHMMSS.jsonl")
        return

    in_path = Path(sys.argv[1])
    out_path = in_path.with_name(in_path.stem.replace("run_", "summary_") + ".csv")

    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # colonnes utiles (adapte si tes clés diffèrent)
    fieldnames = [
        "query_type", "query_id", "query",
        "chunking", "repr", "k",
        "retrieved_files", "retrieved_chunk_ids",
        "answer_preview"
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "query_type": r.get("query_type"),
                "query_id": r.get("query_id"),
                "query": r.get("query"),
                "chunking": r.get("chunking"),
                "repr": r.get("repr"),
                "k": r.get("k"),
                "retrieved_files": " | ".join(r.get("retrieved_files", [])[:5]),
                "retrieved_chunk_ids": " | ".join(r.get("retrieved_chunk_ids", [])[:5]),
                "answer_preview": (r.get("answer","")[:180] + "...") if r.get("answer") else ""
            })

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
