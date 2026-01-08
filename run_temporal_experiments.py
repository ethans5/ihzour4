import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import utils
from retrieval.embedding_retriever import EmbeddingRetriever
from retrieval.bm25_retriever import BM25Retriever, load_jsonl
from rag.rag_pipeline import build_evolution_prompt, retrieve_oldest_newest, generate_with_ollama

# --- CONFIG ---
EMB_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR = Path("evaluation/temporal_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- QUERIES COMPLÈTES (Exercise Requirements) ---
# 4 types × 2 queries each = 8 queries de base
# + 8 queries spécifiques Israel/Gaza/Hamas demandées dans l'exercice
QUERIES = [
    # ========================================
    # 1. POINT-IN-TIME QUERIES (Hard Filter)
    # ========================================
    {
        "type": "point_in_time",
        "query": "What was the debate about the NHS funding in 2015?",
        "filter_year": 2015,
        "desc": "NHS Funding (2015)"
    },
    {
        "type": "point_in_time",
        "query": "What was the specific budget allocated to security in 2024?",
        "filter_year": 2024,
        "desc": "Security Budget (2024)"
    },
    
    # ========================================
    # 2. CURRENT STATUS QUERIES (Recency)
    # ========================================
    {
        "type": "current",
        "query": "What is the current official position regarding the State of Israel?",
        "desc": "Israel Position (Current)"
    },
    {
        "type": "current",
        "query": "What is the current official position regarding Hamas/Gaza?",
        "desc": "Hamas/Gaza Position (Current)"
    },
    {
        "type": "current",
        "query": "Was the official position in the last quarter of 2023 supportive of the State of Israel?",
        "filter_date_range": ("2023-10-01", "2023-12-31"),
        "desc": "Israel Support Q4 2023"
    },
    {
        "type": "current",
        "query": "Was the official position in the last quarter of 2023 supportive of Hamas/Gaza?",
        "filter_date_range": ("2023-10-01", "2023-12-31"),
        "desc": "Hamas/Gaza Support Q4 2023"
    },
    
    # ========================================
    # 3. EVOLUTION QUERIES (Compare Old vs New)
    # ========================================
    {
        "type": "evolution",
        "query": "How did the Prime Minister's rhetoric regarding the war between Israel and Hamas/Gaza develop between his first and last speech?",
        "desc": "PM Rhetoric Evolution (Israel-Hamas)"
    },
    {
        "type": "evolution",
        "query": "How has the discussion on climate change targets evolved?",
        "desc": "Climate Change Evolution"
    },
    {
        "type": "evolution",
        "query": "Has the official position in the last quarter of 2023 changed relative to the official position in the last quarter of 2025?",
        "desc": "Position Change Q4 2023 vs Q4 2025"
    },
    
    # ========================================
    # 4. CONFLICT QUERIES (Ambiguity)
    # ========================================
    {
        "type": "conflict",
        "query": "Who is the Prime Minister?",
        "desc": "Prime Minister (Conflict)"
    },
    {
        "type": "conflict",
        "query": "Who is the Minister of Defense?",
        "desc": "Defense Minister (Conflict)"
    },
    {
        "type": "conflict",
        "query": "What is the current inflation rate?",
        "desc": "Inflation Rate (Conflict)"
    },
]

# --- 4 ADDITIONAL CUSTOM QUERIES (As required) ---
CUSTOM_QUERIES = [
    {
        "type": "current",
        "query": "What are the latest developments in UK-US trade relations?",
        "desc": "UK-US Trade (Latest)"
    },
    {
        "type": "evolution",
        "query": "How has the government's stance on immigration policy evolved since 2020?",
        "desc": "Immigration Policy Evolution"
    },
    {
        "type": "point_in_time",
        "query": "What was discussed about healthcare reform in 2023?",
        "filter_year": 2023,
        "desc": "Healthcare Reform 2023"
    },
    {
        "type": "conflict",
        "query": "What is the current unemployment rate?",
        "desc": "Unemployment Rate (Conflict)"
    }
]

# Combine all queries
ALL_QUERIES = QUERIES + CUSTOM_QUERIES


def load_corpus_data(corpus_name: str) -> Tuple[List[Dict], np.ndarray, List[Dict], np.ndarray]:
    """
    Load chunks and embeddings for a specific corpus (UK or US)
    Returns: (fixed_chunks, fixed_emb, child_chunks, child_emb)
    """
    base_path = Path("data")
    
    # Fixed chunks
    fixed_chunks_path = base_path / "chunks" / f"{corpus_name}_fixed_660w_3s.jsonl"
    fixed_emb_path = base_path / "embeddings" / f"{corpus_name}_fixed_embeddings_mpnet.npy"
    
    # Child chunks
    child_chunks_path = base_path / "chunks" / f"{corpus_name}_child_chunks.jsonl"
    child_emb_path = base_path / "embeddings" / f"{corpus_name}_child_embeddings_mpnet.npy"
    
    # Fallback to non-prefixed names if corpus-specific don't exist
    if not fixed_chunks_path.exists():
        fixed_chunks_path = base_path / "chunks" / "fixed_660w_3s.jsonl"
        fixed_emb_path = base_path / "embeddings" / "fixed_embeddings_mpnet.npy"
    
    if not child_chunks_path.exists():
        child_chunks_path = base_path / "chunks" / "child_chunks.jsonl"
        child_emb_path = base_path / "embeddings" / "child_embeddings_mpnet.npy"
    
    print(f"  Loading {corpus_name} fixed chunks from {fixed_chunks_path}")
    fixed_chunks = load_jsonl(fixed_chunks_path)
    fixed_emb = np.load(fixed_emb_path) if fixed_emb_path.exists() else None
    
    print(f"  Loading {corpus_name} child chunks from {child_chunks_path}")
    child_chunks = load_jsonl(child_chunks_path)
    child_emb = np.load(child_emb_path) if child_emb_path.exists() else None
    
    return fixed_chunks, fixed_emb, child_chunks, child_emb


def plot_corpus_histogram(chunks: List[Dict], corpus_name: str):
    """Generate temporal distribution histogram for a corpus"""
    print(f"  Generating temporal histogram for {corpus_name}...")
    timestamps = []
    valid_chunks = 0
    
    for c in chunks:
        d_str = c.get("date")
        if d_str:
            try:
                ts = utils.convert_date_to_timestamp(d_str)
                timestamps.append(datetime.fromtimestamp(ts).year)
                valid_chunks += 1
            except:
                pass
    
    if not timestamps:
        print(f"  No valid dates found for {corpus_name} histogram.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(timestamps, bins=range(min(timestamps), max(timestamps) + 2), 
             edgecolor='black', alpha=0.7)
    plt.title(f"Temporal Distribution of {corpus_name.upper()} Corpus ({valid_chunks} valid dates)")
    plt.xlabel("Year")
    plt.ylabel("Number of Chunks")
    plt.grid(axis='y', alpha=0.5)
    
    out_file = OUTPUT_DIR / f"{corpus_name}_corpus_histogram.png"
    plt.savefig(out_file)
    print(f"  Histogram saved to {out_file}")
    plt.close()


def run_baseline_failures(retriever, queries: List[Dict], corpus_name: str, 
                         chunking: str, representation: str):
    """Run baseline tests (no temporal awareness)"""
    print(f"\n  --- BASELINE ({chunking} + {representation}) ---")
    results = []
    
    for q_obj in queries:
        q_text = q_obj["query"]
        retrieved = retriever.search(q_text, k=3)
        
        dates = [r.get('date', 'N/A') for r in retrieved]
        results.append({
            "corpus": corpus_name,
            "chunking": chunking,
            "representation": representation,
            "type": q_obj["type"],
            "desc": q_obj["desc"],
            "query": q_text,
            "query_hebrew": q_obj.get("query_he", ""),
            "retrieved_dates": dates,
            "top_doc_preview": retrieved[0].get('text', '')[:150] if retrieved else "None"
        })
        print(f"    [{q_obj['desc']}] Dates: {dates}")
    
    return results


def run_hard_filtering_tests(retriever, queries: List[Dict], corpus_name: str,
                             chunking: str, representation: str):
    """Run Mode A: Hard Filtering tests"""
    print(f"\n  --- MODE A: HARD FILTERING ({chunking} + {representation}) ---")
    results = []
    
    # Test point-in-time and specific date range queries
    target_queries = [q for q in queries if q['type'] == 'point_in_time' or 
                     'filter_date_range' in q]
    
    for q_obj in target_queries:
        print(f"    Query: {q_obj['desc']}")
        
        # Apply filter
        if 'filter_year' in q_obj:
            retrieved = retriever.search(q_obj['query'], k=3, 
                                        filter_year=q_obj['filter_year'])
        elif 'filter_date_range' in q_obj:
            retrieved = retriever.search(q_obj['query'], k=3,
                                        filter_date_range=q_obj['filter_date_range'])
        else:
            retrieved = retriever.search(q_obj['query'], k=3)
        
        dates = [r.get('date', 'N/A') for r in retrieved]
        print(f"      -> Retrieved Dates: {dates}")
        
        # Verify correctness
        if 'filter_year' in q_obj:
            filter_val = q_obj['filter_year']
            success = all(str(filter_val) in d for d in dates if d != 'N/A') or len(dates) == 0
        else:
            success = len(dates) > 0
        
        results.append({
            "corpus": corpus_name,
            "chunking": chunking,
            "representation": representation,
            "type": q_obj["type"],
            "desc": q_obj["desc"],
            "query": q_obj["query"],
            "filter": q_obj.get('filter_year') or q_obj.get('filter_date_range'),
            "retrieved_dates": dates,
            "success": success,
            "num_results": len(dates)
        })
        print(f"      -> Success: {success} (Retrieved {len(dates)} results)")
    
    return results


def run_recency_weighting_tests(retriever, queries: List[Dict], corpus_name: str,
                                chunking: str, representation: str):
    """Run Mode B: Recency Weighting tests"""
    print(f"\n  --- MODE B: RECENCY WEIGHTING ({chunking} + {representation}) ---")
    results = []
    
    # Test current and conflict queries
    target_queries = [q for q in queries if q['type'] in ('current', 'conflict')]
    
    # Parameter grid
    alphas = [0.0, 0.3, 0.5, 0.7, 0.9]
    lambdas = [0.1, 0.5, 1.0]
    
    for q_obj in target_queries:
        print(f"    Query: {q_obj['desc']}")
        query_results = {
            "corpus": corpus_name,
            "chunking": chunking,
            "representation": representation,
            "type": q_obj["type"],
            "desc": q_obj["desc"],
            "query": q_obj["query"],
            "configurations": []
        }
        
        for alpha in alphas:
            for lambda_val in lambdas:
                if alpha == 0.0 and lambda_val > 0.1:
                    continue  # Skip redundant baselines
                
                query_date = utils.convert_date_to_timestamp("2025-01-05")
                retrieved = retriever.search(
                    q_obj['query'], k=3,
                    alpha=alpha, lambda_decay=lambda_val, query_date=query_date
                )
                
                dates = [r.get('date') for r in retrieved]
                config = {
                    "alpha": alpha,
                    "lambda": lambda_val,
                    "retrieved_dates": dates,
                    "avg_year": np.mean([int(d[:4]) for d in dates if d]) if dates else 0
                }
                query_results["configurations"].append(config)
                
                if alpha in [0.0, 0.5, 0.9] and lambda_val == 1.0:
                    print(f"      α={alpha}, λ={lambda_val}: {dates}")
        
        results.append(query_results)
    
    return results


def run_evolution_tests(retriever, queries: List[Dict], corpus_name: str,
                       chunking: str, representation: str):
    """Run Mode C: Evolution Analysis tests"""
    print(f"\n  --- MODE C: EVOLUTION ANALYSIS ({chunking} + {representation}) ---")
    results = []
    
    evo_queries = [q for q in queries if q['type'] == 'evolution']
    
    for q_obj in evo_queries:
        print(f"    Query: {q_obj['desc']}")
        
        oldest, newest = retrieve_oldest_newest(retriever, q_obj['query'], k=3)
        
        oldest_dates = [c.get('date') for c in oldest]
        newest_dates = [c.get('date') for c in newest]
        
        print(f"      Oldest: {oldest_dates}")
        print(f"      Newest: {newest_dates}")
        
        # Generate evolution answer
        if oldest and newest:
            prompt = build_evolution_prompt(q_obj['query'], oldest, newest)
            answer = generate_with_ollama(prompt)
        else:
            answer = "Insufficient temporal data for evolution analysis."
        
        results.append({
            "corpus": corpus_name,
            "chunking": chunking,
            "representation": representation,
            "type": q_obj["type"],
            "desc": q_obj["desc"],
            "query": q_obj["query"],
            "oldest_dates": oldest_dates,
            "newest_dates": newest_dates,
            "oldest_previews": [c.get('text', '')[:100] for c in oldest],
            "newest_previews": [c.get('text', '')[:100] for c in newest],
            "evolution_answer": answer[:500]  # Truncate for readability
        })
    
    return results


def run_full_experiment_suite(corpus_name: str):
    """Run complete experimental suite for one corpus"""
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENTS ON {corpus_name.upper()} CORPUS")
    print(f"{'='*60}")
    
    # Load data
    fixed_chunks, fixed_emb, child_chunks, child_emb = load_corpus_data(corpus_name)
    
    # Generate histograms
    plot_corpus_histogram(fixed_chunks, f"{corpus_name}_fixed")
    plot_corpus_histogram(child_chunks, f"{corpus_name}_child")
    
    # Build retrievers
    configurations = []
    
    # 1. Fixed + BM25
    if fixed_chunks:
        bm25_fixed = BM25Retriever(fixed_chunks)
        configurations.append(("fixed", "bm25", bm25_fixed))
    
    # 2. Fixed + Embeddings
    if fixed_chunks and fixed_emb is not None:
        emb_fixed = EmbeddingRetriever(fixed_chunks, fixed_emb, EMB_MODEL_NAME)
        configurations.append(("fixed", "embeddings", emb_fixed))
    
    # 3. Child + BM25
    if child_chunks:
        bm25_child = BM25Retriever(child_chunks)
        configurations.append(("child", "bm25", bm25_child))
    
    # 4. Child + Embeddings
    if child_chunks and child_emb is not None:
        emb_child = EmbeddingRetriever(child_chunks, child_emb, EMB_MODEL_NAME)
        configurations.append(("child", "embeddings", emb_child))
    
    print(f"\nConfigurations to test: {len(configurations)}")
    for chunking, repr_type, _ in configurations:
        print(f"  - {chunking} + {repr_type}")
    
    # Run experiments for each configuration
    all_results = {
        "corpus": corpus_name,
        "baseline": [],
        "hard_filtering": [],
        "recency_weighting": [],
        "evolution": []
    }
    
    for chunking, representation, retriever in configurations:
        print(f"\n--- Testing {chunking} + {representation} ---")
        
        # Baseline
        baseline_res = run_baseline_failures(retriever, ALL_QUERIES, corpus_name,
                                             chunking, representation)
        all_results["baseline"].extend(baseline_res)
        
        # Hard Filtering (only for embedding retrievers)
        if representation == "embeddings":
            hard_filter_res = run_hard_filtering_tests(retriever, ALL_QUERIES, 
                                                       corpus_name, chunking, representation)
            all_results["hard_filtering"].extend(hard_filter_res)
            
            # Recency Weighting
            recency_res = run_recency_weighting_tests(retriever, ALL_QUERIES,
                                                      corpus_name, chunking, representation)
            all_results["recency_weighting"].extend(recency_res)
            
            # Evolution
            evolution_res = run_evolution_tests(retriever, ALL_QUERIES,
                                               corpus_name, chunking, representation)
            all_results["evolution"].extend(evolution_res)
    
    # Save results
    output_file = OUTPUT_DIR / f"{corpus_name}_complete_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {output_file}")
    
    return all_results


def main():
    """Main execution: run experiments on both UK and US corpora"""
    print("="*70)
    print("TEMPORAL RAG COMPLETE EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Total queries: {len(ALL_QUERIES)}")
    print(f"  - Point-in-time: {len([q for q in ALL_QUERIES if q['type'] == 'point_in_time'])}")
    print(f"  - Current status: {len([q for q in ALL_QUERIES if q['type'] == 'current'])}")
    print(f"  - Evolution: {len([q for q in ALL_QUERIES if q['type'] == 'evolution'])}")
    print(f"  - Conflict: {len([q for q in ALL_QUERIES if q['type'] == 'conflict'])}")
    
    # Run for both corpora
    #results_uk = run_full_experiment_suite("UK")
    results_us = run_full_experiment_suite("US")
    
    # Generate comparative summary
    summary = {
        "experiment_date": datetime.now().isoformat(),
        "total_queries": len(ALL_QUERIES),
        "corpora_tested": ["UK", "US"],
        "configurations_per_corpus": 4,  # fixed+bm25, fixed+emb, child+bm25, child+emb
        # "uk_results_summary": {
        #     "baseline_tests": len(results_uk["baseline"]),
        #     "hard_filtering_tests": len(results_uk["hard_filtering"]),
        #     "recency_tests": len(results_uk["recency_weighting"]),
        #     "evolution_tests": len(results_uk["evolution"])
        # },
        "us_results_summary": {
            "baseline_tests": len(results_us["baseline"]),
            "hard_filtering_tests": len(results_us["hard_filtering"]),
            "recency_tests": len(results_us["recency_weighting"]),
            "evolution_tests": len(results_us["evolution"])
        }
    }
    
    summary_file = OUTPUT_DIR / "experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - UK results: UK_complete_results.json")
    print(f"  - US results: US_complete_results.json")
    print(f"  - Summary: experiment_summary.json")
    print(f"  - Histograms: UK/US corpus_histogram.png")

    # Generate comparative plots
    print("\nGenerating comparative graphs...")
    try:
        from evaluation.generate_comparison_plots import generate_comparative_graphs
        generate_comparative_graphs()
        print("✅ Comparative graphs generated in evaluation/temporal_experiments/")
    except ImportError:
        print("Warning: evaluation.generate_comparison_plots not found or failed to import.")
    except Exception as e:
        print(f"Error generating graphs: {e}")



if __name__ == "__main__":
    with open("debug_log.txt", "w") as f:
        f.write("Starting complete temporal experiments...\n")
    
    try:
        main()
        with open("debug_log.txt", "a") as f:
            f.write("Script finished successfully.\n")
    except Exception as e:
        import traceback
        with open("debug_log.txt", "a") as f:
            f.write(f"Script failed: {e}\n")
            f.write(traceback.format_exc())
        raise