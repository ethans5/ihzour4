# 📋 Temporal RAG Report - Quick Summary

## Document Overview

**Main Report**: [`TEMPORAL_RAG_REPORT.md`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/TEMPORAL_RAG_REPORT.md)

This is a quick reference guide to the main findings and structure of the Temporal RAG report.

---

## Key Findings at a Glance

### 🔴 Problem Identified

Standard RAG systems are **temporally blind**:
- Treat documents from 2010 and 2025 equally
- Create "temporal hallucinations" (mixing old and new information)
- Cannot answer "current" or "recent" questions accurately

### ✅ Solutions Implemented

Three complementary temporal strategies:

1. **Hard Filtering (Mode A)**: Binary date filtering
   - Example: "NHS funding in 2015" → Returns empty list (correct, no 2015 data)
   - Use case: Historical research, specific year queries

2. **Recency Weighting (Mode B)**: Temporal decay function
   - Formula: Score = (1-α)·Similarity + α·Recency
   - Example: "Who is the Prime Minister?" with α=0.9 → Only 2025 documents
   - Use case: "Current", "latest", "recent" queries

3. **Evolution Analysis (Mode C)**: Comparative retrieval
   - Retrieves K oldest + K newest documents
   - Example: "Fiscal policy evolution" → Compares 2023 vs 2025 stances
   - Use case: "How has X changed?" queries

---

## Experimental Results Summary

### Baseline Failures

| Query | Problem | Retrieved Dates |
|-------|---------|-----------------|
| "NHS funding in 2015" | Retrieved 2024 docs instead | 2024-11-19, 2024-03-06, 2024-11-14 |
| "Who is the Prime Minister?" | Mixed 2024 and 2025 | 2024-05-02, 2024-04-24, 2025-06-04 |
| "Current inflation rate" | Returned 2023 data first | 2023-11-22, 2025-05-21, 2024-11-20 |

### After Temporal RAG

| Query | Strategy | Result | Status |
|-------|----------|--------|--------|
| "NHS funding in 2015" | Hard Filter (year=2015) | Empty list | ✅ Correct (no 2015 data) |
| "Who is PM?" | Recency (α=0.9, λ=1.0) | Only 2025 docs | ✅ Current info only |
| "Fiscal policy evolution" | Evolution (K=50→3) | 2023 vs 2025 comparison | ✅ Temporal contrast |

---

## Technical Implementation

### Data Engineering

```python
# Date extraction from filenames
extract_date_from_filename("debates2024-11-19.xml") → "2024-11-19"

# Conversion to Unix timestamp
convert_date_to_timestamp("2024-11-19") → 1700352000.0

# Pre-computation in retriever
chunk_timestamps = np.array([ts1, ts2, ...])  # Fast vectorized operations
chunk_years = np.array([2024, 2024, 2025, ...])  # For year filtering
```

### Recency Formula

$$\text{Score} = (1-\alpha) \cdot \text{Similarity} + \alpha \cdot \frac{1}{1 + \lambda \cdot \Delta t}$$

**Parameter Guide**:

| Query Type | α (recency weight) | λ (decay rate) |
|------------|-------------------|----------------|
| "Current", "latest" | 0.7 - 0.9 | 1.0 |
| Ambiguous | 0.3 - 0.5 | 0.5 |
| Historical | 0.0 | - |

---

## Corpus Statistics

- **Total documents**: 1,383 chunks
- **Date range**: 2023-2025 (concentrated)
- **Missing years**: 2015-2022
- **Visualization**: [`corpus_histogram.png`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/evaluation/temporal_experiments/corpus_histogram.png)

---

## Key Insights

1. **Semantic similarity alone is insufficient** for time-dependent questions
2. **Different query types need different strategies**:
   - Point-in-time → Hard filtering
   - Current status → Recency weighting
   - Evolution → Comparative retrieval
3. **Parameter tuning is critical**: α and λ dramatically affect results
4. **Temporal metadata must be first-class**: Pre-compute and index dates

---

## Recommendations for Production

1. ✅ Always extract and store temporal metadata during indexing
2. ✅ Detect temporal intent in queries ("current", "recent", "in 2020")
3. ✅ Default to recency weighting for ambiguous queries (α=0.5, λ=1.0)
4. ✅ Use hard filtering only when user explicitly specifies dates
5. ✅ Implement evolution mode for comparative questions

---

## Report Structure

1. **Introduction** - Problem definition and objectives
2. **Baseline Analysis** - Experimental failures without temporal awareness
3. **Data Engineering** - Date extraction and timestamp conversion
4. **Retrieval Strategies** - Hard filtering, recency weighting, evolution
5. **Discussion** - Technical choices, parameter tuning, limitations
6. **Conclusion** - Key achievements and recommendations
7. **Appendices** - Full experimental logs and code references

---

## Files Referenced

| File | Purpose |
|------|---------|
| [`utils.py`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/utils.py) | Date extraction and conversion |
| [`embedding_retriever.py`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/retrieval/embedding_retriever.py) | Hard filtering and recency weighting |
| [`rag_pipeline.py`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/rag/rag_pipeline.py) | Evolution retrieval and prompts |
| [`run_temporal_experiments.py`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/run_temporal_experiments.py) | Experiment runner |
| [`baseline_failures.json`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/evaluation/temporal_experiments/baseline_failures.json) | Experimental results |

---

**For full details, mathematical derivations, and code examples, see**: [`TEMPORAL_RAG_REPORT.md`](file:///c:/Users/ethan/OneDrive/Bureau/exo4/TEMPORAL_RAG_REPORT.md)
