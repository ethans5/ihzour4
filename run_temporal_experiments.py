import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import utils
from retrieval.embedding_retriever import EmbeddingRetriever
from rag.rag_pipeline import build_evolution_prompt, retrieve_oldest_newest, generate_with_ollama

# --- CONFIG ---
CHUNKS_PATH = Path("data/chunks/fixed_660w_3s.jsonl") # Adjust if needed
EMBEDDINGS_PATH = Path("data/embeddings/fixed_embeddings_mpnet.npy") # Adjust if needed
EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR = Path("evaluation/temporal_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- QUERIES ---
# 8 Queries: Point-in-time, Current, Evolution, Conflict
QUERIES = [
    # 1. Point-in-time (Constraint: Specific Year)
    {
        "type": "point_in_time",
        "query": "What was the debate about the NHS funding in 2015?",
        "filter_year": 2015,
        "desc": "NHS Funding (2023)"
    },
    {
        "type": "point_in_time",
        "query": "What were the concerns regarding Brexit in 2016?",
        "filter_year": 2016,
        "desc": "Brexit Concerns (2024)"
    },
    # 2. Conflict / Conflict (Constraint: Old vs New info)
    {
        "type": "conflict",
        "query": "Who is the Prime Minister?",
        "desc": "Prime Minister (Conflict 2015-2024)"
    },
    {
        "type": "conflict",
        "query": "What is the current inflation rate?",
        "desc": "Inflation Rate (Conflict)"
    },
    # 3. Current Status (Recency Bias needed)
    {
        "type": "current",
        "query": "What is the latest stance on AI regulation?",
        "desc": "AI Regulation (Latest)"
    },
    {
        "type": "current",
        "query": "What are the recent updates on Ukraine support?",
        "desc": "Ukraine Support (Recent)"
    },
    # 4. Evolution (Compare Old vs New)
    {
        "type": "evolution",
        "query": "How has the discussion on climate change targets evolved?",
        "desc": "Climate Change Evolution"
    },
    {
        "type": "evolution",
        "query": "How has the government's fiscal policy changed since 2010?",
        "desc": "Fiscal Policy Evolution"
    }
]

def load_data():
    print("Loading chunks...")
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH)
    
    return chunks, embeddings

def plot_histogram(chunks):
    print("Generating temporal histogram...")
    timestamps = []
    valid_chunks = 0
    for c in chunks:
        d_str = c.get("date")
        if d_str:
            try:
                # Use utils date parsing if available, or manual
                ts = utils.convert_date_to_timestamp(d_str)
                timestamps.append(datetime.fromtimestamp(ts).year)
                valid_chunks += 1
            except:
                pass
    
    if not timestamps:
        print("No valid dates found for histogram.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(timestamps, bins=range(min(timestamps), max(timestamps) + 2), edgecolor='black', alpha=0.7)
    plt.title(f"Temporal Distribution of Corpus ({valid_chunks} valid dates)")
    plt.xlabel("Year")
    plt.ylabel("Number of Chunks")
    plt.grid(axis='y', alpha=0.5)
    
    out_file = OUTPUT_DIR / "corpus_histogram.png"
    plt.savefig(out_file)
    print(f"Histogram saved to {out_file}")
    plt.close()

def run_baseline_failures(retriever, queries):
    print("\n--- RUNNING BASELINE (Standard Search) ---")
    results = []
    for q_obj in queries:
        q_text = q_obj["query"]
        # Standard search (no filters, no alpha)
        retrieved = retriever.search(q_text, k=3)
        
        # Log retrieved dates
        dates = [r.get('date', 'N/A') for r in retrieved]
        results.append({
            "desc": q_obj["desc"],
            "query": q_text,
            "retrieved_dates": dates,
            "top_doc": retrieved[0].get('text', '')[:100] if retrieved else "None"
        })
        print(f"[{q_obj['desc']}] Dates: {dates}")
    
    with open(OUTPUT_DIR / "baseline_failures.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def run_mode_a_tests(retriever):
    print("\n--- RUNNING MODE A (Hard Filtering) ---")
    # Test on Point-in-Time queries
    pit_queries = [q for q in QUERIES if q['type'] == 'point_in_time']
    
    for q_obj in pit_queries:
        filter_y = q_obj['filter_year']
        print(f"Query: {q_obj['query']} (Filter {filter_y})")
        
        retrieved = retriever.search(q_obj['query'], k=3, filter_year=filter_y)
        dates = [r.get('date', 'N/A') for r in retrieved]
        print(f"  -> Retrieved Dates: {dates}")
        
        # Verify
        all_match = all(str(filter_y) in d for d in dates if d != 'N/A')
        print(f"  -> Success: {all_match}")

def run_mode_b_tests(retriever):
    print("\n--- RUNNING MODE B (Recency Weighting) ---")
    # Test on Current/Conflict queries
    target_queries = [q for q in QUERIES if q['type'] in ('current', 'conflict')]
    
    # Params to test
    alphas = [0.1, 0.5, 0.9]
    lambdas = [0.1, 1.0] # 0.1 = slow decay, 1.0 = fast decay
    
    for q_obj in target_queries:
        print(f"\nQuery: {q_obj['query']}")
        
        # Baseline (Alpha=0)
        base = retriever.search(q_obj['query'], k=3)
        base_dates = [r.get('date') for r in base]
        print(f"  Baseline (A=0): {base_dates}")
        
        for a in alphas:
            for l in lambdas:
                input_date = utils.convert_date_to_timestamp("2025-01-01") # Simulate queries from "now"
                res = retriever.search(q_obj['query'], k=3, alpha=a, lambda_decay=l, query_date=input_date)
                dates = [r.get('date') for r in res]
                print(f"  A={a}, L={l}: {dates}")

def run_evolution_tests(retriever):
    print("\n--- RUNNING EVOLUTION TESTS ---")
    evo_queries = [q for q in QUERIES if q['type'] == 'evolution']
    
    for q_obj in evo_queries:
        print(f"\nQuery: {q_obj['query']}")
        oldest, newest = retrieve_oldest_newest(retriever, q_obj['query'], k=2)
        
        print("  Oldest Set:")
        for c in oldest:
            print(f"    - {c.get('date')}: {c.get('text', '')[:50]}...")
            
        print("  Newest Set:")
        for c in newest:
            print(f"    - {c.get('date')}: {c.get('text', '')[:50]}...")
            
        prompt = build_evolution_prompt(q_obj['query'], oldest, newest)
        # Uncomment to actually run LLM (optional for speed)
        # print("  Generating answer...")
        # ans = generate_with_ollama(prompt)
        # print(f"  LLM Answer: {ans[:200]}...")

def main():
    chunks, embeddings = load_data()
    retriever = EmbeddingRetriever(chunks, embeddings, EMB_MODEL_NAME)
    
    chunk_timestamps = retriever.chunk_timestamps
    
    # 1. Histogram
    plot_histogram(chunks)
    
    # 2. Baseline Failures
    run_baseline_failures(retriever, QUERIES)
    
    # 3. Mode A
    run_mode_a_tests(retriever)
    
    # 4. Mode B
    run_mode_b_tests(retriever)
    
    # 5. Evolution
    run_evolution_tests(retriever)
    
    print("\nExperiments complete. Results saved to evaluation/temporal_experiments/")


if __name__ == "__main__":
    with open("debug_log.txt", "w") as f:
        f.write("Starting script...\n")
    
    try:
        main()
        with open("debug_log.txt", "a") as f:
            f.write("Script finished successfully.\n")
    except Exception as e:
        import traceback
        with open("debug_log.txt", "a") as f:
            f.write(f"Script failed: {e}\n")
            f.write(traceback.format_exc())
