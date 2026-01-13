import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
RESULTS_FILE = Path("evaluation/temporal_experiments/US_complete_results.json")
QUERY_DESC = "Climate Change Evolution"
OUTPUT_PATH = "evaluation/temporal_experiments/US_k_comparison_proof.png"


def month_diff(d1, d2):
    """Calcule la différence en mois entre deux strings YYYY-MM-DD."""
    date1 = datetime.strptime(d1, "%Y-%m-%d")
    date2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)


def generate_proof():
    # 1. Charger les données réelles
    if not RESULTS_FILE.exists():
        print(f"Erreur : Le fichier {RESULTS_FILE} est introuvable.")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Extraire les données Baseline (K=3)
    baseline_entry = next(
        (q for q in data["baseline"] if q["desc"] == QUERY_DESC), None
    )
    if not baseline_entry:
        print(f"Erreur : Requête '{QUERY_DESC}' non trouvée dans la section baseline.")
        return

    baseline_dates = sorted(baseline_entry["retrieved_dates"])
    k3_months = month_diff(baseline_dates[0], baseline_dates[-1])
    k3_facts = len(set(baseline_dates))  # Nombre de dates/faits uniques

    # 3. Extraire les données Evolution (Représentant le pool large/K=50)
    evolution_entry = next(
        (q for q in data["evolution"] if q["desc"] == QUERY_DESC), None
    )
    if not evolution_entry:
        print(f"Erreur : Requête '{QUERY_DESC}' non trouvée dans la section evolution.")
        return

    # On prend la date la plus ancienne des 'oldest' et la plus récente des 'newest'
    all_evo_dates = sorted(
        evolution_entry["oldest_dates"] + evolution_entry["newest_dates"]
    )
    k_pool_months = month_diff(all_evo_dates[0], all_evo_dates[-1])
    k_pool_facts = len(set(all_evo_dates))

    # 4. Générer le graphique
    labels = ["Couverture Temporelle (mois)", "Faits/Dates uniques"]
    k3_data = [k3_months, k3_facts]
    k_pool_data = [k_pool_months, k_pool_facts]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width / 2, k3_data, width, label="K = 3 (Baseline)", color="#A9A9A9"
    )
    rects2 = ax.bar(
        x + width / 2,
        k_pool_data,
        width,
        label="Pool Evolution (K=50)",
        color="#44a1ad",
    )

    ax.set_ylabel("Valeur")
    ax.set_title(f"Preuve par les données US : {QUERY_DESC}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(k_pool_data) + 5)
    ax.legend()

    # Ajout des chiffres sur les barres
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Note explicative dynamique
    plt.text(
        0.5,
        -0.15,
        f"Preuve : La baseline s'arrête à {baseline_dates[0]}, alors que le pool large\n"
        f"récupère des documents dès {all_evo_dates[0]}, augmentant la fenêtre de {k_pool_months - k3_months} mois.",
        ha="center",
        fontsize=9,
        style="italic",
        transform=ax.transAxes,
        bbox={"facecolor": "orange", "alpha": 0.1, "pad": 5},
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Graphique généré avec succès : {OUTPUT_PATH}")


def generate_bm25_vs_emb_comparison():
    # 1. Charger les données
    if not RESULTS_FILE.exists():
        return
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- CALCUL MÉTRIQUE 1 : PRÉCISION POINT-IN-TIME (BASELINE) ---
    # On regarde combien de docs récupérés en baseline correspondent à l'année cible
    def get_baseline_precision(repr_type):
        matches, total = 0, 0
        for r in data["baseline"]:
            if r["type"] == "point_in_time" and r["representation"] == repr_type:
                target_year = (
                    "2015"
                    if "2015" in r["desc"]
                    else "2024" if "2024" in r["desc"] else "2023"
                )
                for d in r["retrieved_dates"]:
                    total += 1
                    if target_year in d:
                        matches += 1
        return (matches / total * 100) if total > 0 else 0

    # --- CALCUL MÉTRIQUE 2 : RÉCENCE (AVG YEAR POUR "CURRENT") ---
    # On compare BM25 Baseline vs Embedding Mode B (alpha=0.9, lambda=1.0)
    def get_avg_year_current_bm25():
        years = []
        for r in data["baseline"]:
            if r["type"] == "current" and r["representation"] == "bm25":
                years.extend([int(d[:4]) for d in r["retrieved_dates"] if d != "N/A"])
        return np.mean(years) if years else 0

    def get_avg_year_current_emb_optimized():
        years = []
        for r in data["recency_weighting"]:
            for cfg in r["configurations"]:
                if cfg["alpha"] == 0.9 and cfg["lambda"] == 1.0:
                    years.append(cfg["avg_year"])
        return np.mean(years) if years else 0

    # Données pour le graph
    # Basé sur tes fichiers : Child Emb est à 44.4% vs 11.1% pour BM25 en baseline US
    precision_bm25 = get_baseline_precision("bm25")
    precision_emb = get_baseline_precision("embeddings")

    avg_year_bm25 = get_avg_year_current_bm25()
    avg_year_emb = get_avg_year_current_emb_optimized()

    # --- GÉNÉRATION DU GRAPHIQUE ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graph 1 : Précision Point-in-Time
    ax1.bar(
        ["BM25", "Embeddings"],
        [precision_bm25, precision_emb],
        color=["#A9A9A9", "#4682B4"],
    )
    ax1.set_title(
        "Précision Baseline (Point-in-Time)\n% de documents de la bonne année"
    )
    ax1.set_ylabel("Précision (%)")
    ax1.set_ylim(0, 100)

    # Graph 2 : Recency Score
    ax2.bar(
        ["BM25 Baseline", "Embeddings Mode B"],
        [avg_year_bm25, avg_year_emb],
        color=["#A9A9A9", "#2E8B57"],
    )
    ax2.set_title("Capacité de Récence (Questions 'Current')\nAnnée moyenne récupérée")
    ax2.set_ylabel("Année")
    ax2.set_ylim(2023, 2026)

    # Annotations
    for ax in [ax1, ax2]:
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
                fontweight="bold",
            )

    plt.tight_layout()
    comp_path = "evaluation/temporal_experiments/BM25_vs_Embeddings_Comparison.png"
    plt.savefig(comp_path)
    print(f"Graphique comparatif généré : {comp_path}")


# Modifier le bloc final pour lancer les deux fonctions
if __name__ == "__main__":
    generate_proof()
    generate_bm25_vs_emb_comparison()
