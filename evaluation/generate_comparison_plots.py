import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("evaluation/temporal_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results(corpus_name):
    path = OUTPUT_DIR / f"{corpus_name}_complete_results.json"
    if not path.exists():
        print(f"File not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_baseline_comparison(data, corpus_name):
    """
    Compare all 4 configurations on Baseline Point-in-Time queries.
    Metric: % of retrieved documents matching the target year.
    """
    results = data.get("baseline", [])
    
    # Filter for point-in-time queries with a specific year filter goal
    # We infer the goal from the query metadata (which might not be explicitly in 'baseline' entries in full, 
    # but 'type'='point_in_time' is there. The 'filter_year' might not be in the baseline result dict,
    # but we can infer it from the description or query text if needed, OR look up in original queries.
    # Actually, the 'baseline' result objects in the JSON don't seem to have 'filter_year' preserved 
    # based on the file view (lines 1-200). 
    # However, we can deduce it: 
    # "NHS Funding (2015)" -> 2015
    # "Security Budget (2024)" -> 2024
    
    configs = ["fixed_bm25", "fixed_embeddings", "child_bm25", "child_embeddings"]
    readable = ["Fixed\nBM25", "Fixed\nEmb", "Child\nBM25", "Child\nEmb"]
    
    scores = {c: [] for c in configs}
    
    for r in results:
        if r["type"] != "point_in_time":
            continue
            
        # Determine target year
        target_year = None
        if "2015" in r["desc"] or "2015" in r["query"]:
            target_year = 2015
        elif "2024" in r["desc"] or "2024" in r["query"]:
            target_year = 2024
        elif "2023" in r["desc"] or "2023" in r["query"]:
            target_year = 2023
            
        if not target_year:
            continue
            
        retrieved_dates = r.get("retrieved_dates", [])
        # Count matches
        matches = 0
        for d in retrieved_dates:
            if d and str(target_year) in d:
                matches += 1
        
        # Precision @ K (usually K=3 here)
        precision = matches / len(retrieved_dates) if retrieved_dates else 0
        
        cfg_key = f"{r['chunking']}_{r['representation']}"
        if cfg_key in scores:
            scores[cfg_key].append(precision)
            
    # Calculate averages
    avg_scores = [np.mean(scores[c]) * 100 if scores[c] else 0 for c in configs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(readable, avg_scores, color=['#A9A9A9', '#4682B4', '#D3D3D3', '#5F9EA0'])
    plt.title(f"Baseline Temporal Precision (Point-in-Time Queries) - {corpus_name}\n(How often do they accidentally find the right year?)")
    plt.ylabel("Precision (%)")
    plt.ylim(0, 100)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
        
    out_path = OUTPUT_DIR / f"{corpus_name}_baseline_comparison.png"
    plt.savefig(out_path)
    print(f"Generated {out_path}")
    plt.close()

def plot_hard_filter_success(data, corpus_name):
    """
    Compare (Fixed vs Child) for Embeddings in Hard Filtering Mode.
    """
    hf_results = data.get("hard_filtering", [])
    if not hf_results:
        return

    # Fixed+Emb
    fixed_emb_success = [r["success"] for r in hf_results if r["chunking"]=="fixed" and r["representation"]=="embeddings"]
    fixed_emb_rate = np.mean(fixed_emb_success) * 100 if fixed_emb_success else 0
    
    # Child+Emb
    child_emb_success = [r["success"] for r in hf_results if r["chunking"]=="child" and r["representation"]=="embeddings"]
    child_emb_rate = np.mean(child_emb_success) * 100 if child_emb_success else 0
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Fixed + Emb", "Child + Emb"], [fixed_emb_rate, child_emb_rate], color=['skyblue', 'salmon'])
    plt.title(f"Hard Filtering Success Rate - {corpus_name}\n(Fixed vs Child)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 110)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom')
        
    out_path = OUTPUT_DIR / f"{corpus_name}_hard_filter_comparison.png"
    plt.savefig(out_path)
    print(f"Generated {out_path}")
    plt.close()

def plot_recency_effect(data, corpus_name):
    """
    Compare Fixed vs Child for Embeddings in Recency Mode.
    Metric: Average retrieved year for 'Current' queries at high alpha.
    """
    recency_results = data.get("recency_weighting", [])
    if not recency_results:
        return
        
    target_alpha = 0.9
    target_lambda = 1.0
    
    avg_years = {'fixed': [], 'child': []}
    
    for r in recency_results:
        if r["representation"] != "embeddings": continue
        
        chunking = r["chunking"]
        for config in r["configurations"]:
            if config["alpha"] == target_alpha and config["lambda"] == target_lambda:
                avg_years[chunking].append(config["avg_year"])
    
    mean_year_fixed = np.mean(avg_years['fixed']) if avg_years['fixed'] else 0
    mean_year_child = np.mean(avg_years['child']) if avg_years['child'] else 0
    
    if mean_year_fixed == 0 and mean_year_child == 0:
        return

    plt.figure(figsize=(8, 6))
    # Dynamic ylim to show valid range
    min_val = min(mean_year_fixed, mean_year_child)
    max_val = max(mean_year_fixed, mean_year_child)
    plt.ylim(min(2020, min_val-1), max(2026, max_val+1))
    
    bars = plt.bar(["Fixed + Emb", "Child + Emb"], [mean_year_fixed, mean_year_child], color=['lightgreen', 'orange'])
    plt.title(f"Avg. Retrieved Year for 'Current' Queries - {corpus_name}\n(α={target_alpha}, λ={target_lambda})")
    plt.ylabel("Average Year")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.1f}", ha='center', va='bottom')
    
    out_path = OUTPUT_DIR / f"{corpus_name}_recency_comparison.png"
    plt.savefig(out_path)
    print(f"Generated {out_path}")
    plt.close()



def generate_comparative_graphs():
    log_file = "plot_debug.txt"
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            log.write("Starting graph generation...\n")
    except Exception as e:
        print(f"Failed to open log file: {e}")
        
    print("Starting graph generation...", flush=True)
    
    for corpus in ["UK", "US"]:
        data = load_results(corpus)
        if not data:
            print(f"Skipping {corpus} (no data)", flush=True)
            try:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Skipping {corpus} (no data)\n")
            except: pass
            continue
            
        print(f"Processing {corpus}...", flush=True)
        try:
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"Processing {corpus}...\n")
        except: pass
            
        try:
            plot_baseline_comparison(data, corpus)
            try:
                with open(log_file, "a", encoding="utf-8") as log: log.write(f"  plot_baseline_comparison done for {corpus}\n")
            except: pass
            
            plot_hard_filter_success(data, corpus)
            try:
                with open(log_file, "a", encoding="utf-8") as log: log.write(f"  plot_hard_filter_success done for {corpus}\n")
            except: pass
            
            plot_recency_effect(data, corpus)
            try:
                with open(log_file, "a", encoding="utf-8") as log: log.write(f"  plot_recency_effect done for {corpus}\n")
            except: pass
            
        except Exception as e:
            print(f"Error processing {corpus}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            try:
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"Error processing {corpus}: {e}\n")
                    log.write(traceback.format_exc())
            except: pass
                
    print("Graph generation complete.", flush=True)
    try:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write("Graph generation complete.\n")
    except: pass




if __name__ == "__main__":
    generate_comparative_graphs()
