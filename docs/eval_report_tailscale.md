# Comprehensive Routing Evaluation Report

**Generated:** 2026-06-24 07:16:29  
**Random seed:** 789  
**Total queries:** 10 (5 medical, 5 general)  
**Endpoint:** `http://100.85.30.17:8765/chat`  
**NX Subserver:** http://100.85.30.17:8765  

---

## Confusion Matrix

| Actual \\ Predicted | **Medical** | **General** | **Total** |
|---|---|---|---|
| **Medical** | 5 (TP) | 0 (FN) | 5 |
| **General** | 1 (FP) | 4 (TN) | 5 |

## Classification Metrics

| Metric | Medical | General | Overall |
|---|---|---|---|
| **Precision** | 0.8333 | 1.0000 | — |
| **Recall** | 1.0000 | 0.8000 | — |
| **F1 Score** | 0.9091 | 0.8889 | — |
| **Support** | 5 | 5 | 10 |
| **Accuracy** | — | — | **0.9000** |
| **MCC** | — | — | **0.8165** |

- **Accuracy**: 90.00% (9/10)  
- **MCC (Matthews Correlation Coefficient)**: 0.8165 (+1 = perfect, 0 = random, -1 = inverse)  
- **False positives (general→medical)**: 1 — routed to NX instead of AGX  
- **False negatives (medical→general)**: 0 — routed to AGX instead of NX  

## Latency Breakdown

| Metric | All Queries | Medical (→NX) | General (→AGX) |
|---|---|---|---|
| **Count** | 10 | 5 | 5 |
| **Mean** | 27.46s
| **Median (P50)** | 27.36s
| **P95** | 34.33s
| **P99** | 34.33s
| **Min** | 19.76s
| **Max** | 40.92s
| **Mean** | — | 30.84s | 24.08s |

### Latency Distribution (All)

   15- 30s:   6 ( 60.0%) ██████████████████████████████
   30- 60s:   4 ( 40.0%) ████████████████████

## Classification Method Distribution

| Method | Total | Correct | Errors | Accuracy |
|---|---|---|---|---|
| **keyword** | 0 | 0 | 0 | — |
| **llm** | 0 | 0 | 0 | — |

## Per-Query Results

| # | Query | GT | Pred | Source | Correct | Latency | Confidence |
|---|---|---|---|---|---|---|---|
|   1 | What does insomnia indicate? | medical | medical |      NX | ✓ |   34.33s | 1.0000 |
|   2 | Tell me about basketball | general | general |     AGX | ✓ |   21.20s | 1.0000 |
|   3 | How long does anemia last? | medical | medical |      NX | ✓ |   40.92s | 1.0000 |
|   4 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   27.36s | 1.0000 |
|   5 | How many countries are in Africa? | general | general |     AGX | ✓ |   19.76s | 0.6667 |
|   6 | When should I see a doctor for numbness? | medical | medical |      NX | ✓ |   30.16s | 0.9500 |
|   7 | What causes rain? | general | general |     AGX | ✓ |   30.44s | 1.0000 |
|   8 | What medications are used for UTI? | medical | medical |      NX | ✓ |   21.40s | 0.9500 |
|   9 | How to relieve muscle cramps at home? | medical | medical |      NX | ✓ |   27.37s | 0.9500 |
|  10 | Can you explain the French Revolution simply? | general | general |     AGX | ✓ |   21.64s | 0.6667 |

---

## Query Generation Templates

**Medical** (25 templates × 20 diseases × 20 symptoms × 15 treatments × 10 parts)  
**General** (25 templates × 20 countries × 15 concepts × 10 people × 10 animals × 12 topics)  
