# Comprehensive Routing Evaluation Report

**Generated:** 2026-06-24 07:59:42  
**Random seed:** 999  
**Total queries:** 100 (50 medical, 50 general)  
**Endpoint:** `http://100.85.30.17:8765/chat`  
**NX Subserver:** http://100.85.30.17:8765  

---

## Confusion Matrix

| Actual \\ Predicted | **Medical** | **General** | **Total** |
|---|---|---|---|
| **Medical** | 49 (TP) | 1 (FN) | 50 |
| **General** | 16 (FP) | 34 (TN) | 50 |

## Classification Metrics

| Metric | Medical | General | Overall |
|---|---|---|---|
| **Precision** | 0.7538 | 0.9714 | — |
| **Recall** | 0.9800 | 0.6800 | — |
| **F1 Score** | 0.8522 | 0.8000 | — |
| **Support** | 50 | 50 | 100 |
| **Accuracy** | — | — | **0.8300** |
| **MCC** | — | — | **0.6919** |

- **Accuracy**: 83.00% (83/100)  
- **MCC (Matthews Correlation Coefficient)**: 0.6919 (+1 = perfect, 0 = random, -1 = inverse)  
- **False positives (general→medical)**: 16 — routed to NX instead of AGX  
- **False negatives (medical→general)**: 1 — routed to AGX instead of NX  

## Latency Breakdown

| Metric | All Queries | Medical (→NX) | General (→AGX) |
|---|---|---|---|
| **Count** | 100 | 50 | 50 |
| **Mean** | 25.30s
| **Median (P50)** | 23.84s
| **P95** | 42.80s
| **P99** | 54.90s
| **Min** | 8.44s
| **Max** | 55.13s
| **Mean** | — | 24.62s | 25.98s |

### Latency Distribution (All)

    8- 15s:  12 ( 12.0%) ██████
   15- 30s:  66 ( 66.0%) █████████████████████████████████
   30- 60s:  22 ( 22.0%) ███████████

## Classification Method Distribution

| Method | Total | Correct | Errors | Accuracy |
|---|---|---|---|---|
| **keyword** | 0 | 0 | 0 | — |
| **llm** | 0 | 0 | 0 | — |

## Per-Query Results

| # | Query | GT | Pred | Source | Correct | Latency | Confidence |
|---|---|---|---|---|---|---|---|
|   1 | What language is spoken in Kenya? | general | general |     AGX | ✓ |   31.01s | 1.0000 |
|   2 | What did Albert Einstein discover? | general | general |     AGX | ✓ |   30.49s | 0.6667 |
|   3 | Why do I have headache? | medical | medical |      NX | ✓ |   41.60s | 1.0000 |
|   4 | How long do dolphins live? | general | medical |      NX | ✗ |   27.84s | 1.0000 |
|   5 | What is the normal dosage for beta blockers? | medical | medical |      NX | ✓ |   11.72s | 0.9500 |
|   6 | Why do I have cough? | medical | medical |      NX | ✓ |   18.87s | 0.9500 |
|   7 | How is malaria diagnosed? | medical | medical |      NX | ✓ |   21.00s | 0.9500 |
|   8 | Why is origami important? | general | medical |      NX | ✗ |   42.80s | 1.0000 |
|   9 | Can shortness of breath be a sign of osteoporosis? | medical | medical |      NX | ✓ |   32.44s | 1.0000 |
|  10 | What is the largest butterflies species? | general | general |     AGX | ✓ |   20.34s | 0.6667 |
|  11 | Why do I have chest pain? | medical | medical |      NX | ✓ |   19.98s | 0.9500 |
|  12 | Can GERD be cured? | medical | medical |      NX | ✓ |   37.11s | 1.0000 |
|  13 | Why is jazz music important? | general | medical |      NX | ✗ |   40.49s | 1.0000 |
|  14 | What is South Korea known for? | general | general |     AGX | ✓ |   19.38s | 0.6667 |
|  15 | What is the recovery time for COPD treatment? | medical | medical |      NX | ✓ |   22.24s | 0.9500 |
|  16 | What is plate tectonics? | general | general |     AGX | ✓ |   21.99s | 1.0000 |
|  17 | What is the tallest mountain in the world? | general | general |     AGX | ✓ |   20.14s | 0.6667 |
|  18 | Why do I have swollen lymph nodes? | medical | medical |      NX | ✓ |   40.57s | 1.0000 |
|  19 | Is physical therapy effective for conjunctivitis? | medical | medical |      NX | ✓ |   11.52s | 0.9500 |
|  20 | Why do I have swollen lymph nodes? | medical | medical |      NX | ✓ |   37.28s | 1.0000 |
|  21 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   25.11s | 1.0000 |
|  22 | What is the recovery time for COPD treatment? | medical | medical |      NX | ✓ |   28.65s | 0.9500 |
|  23 | What is the normal dosage for antivirals? | medical | medical |      NX | ✓ |   21.61s | 0.9500 |
|  24 | What are the side effects of antivirals? | medical | medical |      NX | ✓ |   16.68s | 0.9500 |
|  25 | What is the speed of light? | general | general |     AGX | ✓ |   27.24s | 0.6667 |
|  26 | How long do chameleons live? | general | medical |      NX | ✗ |   27.21s | 1.0000 |
|  27 | What did Leonardo da Vinci discover? | general | general |     AGX | ✓ |   19.53s | 0.6667 |
|  28 | What is the treatment for eczema? | medical | medical |      NX | ✓ |   23.51s | 0.9500 |
|  29 | Where do bald eagles live? | general | general |     AGX | ✓ |   20.98s | 1.0000 |
|  30 | What is the speed of light? | general | general |     AGX | ✓ |   21.12s | 0.6667 |
|  31 | Is UTI contagious? | medical | medical |      NX | ✓ |   31.50s | 1.0000 |
|  32 | Does osteoporosis run in families? | medical | medical |      NX | ✓ |   27.87s | 1.0000 |
|  33 | What foods should I avoid with arthritis? | medical | medical |      NX | ✓ |   25.05s | 0.9500 |
|  34 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   28.69s | 1.0000 |
|  35 | How is malaria diagnosed? | medical | medical |      NX | ✓ |   24.31s | 0.9500 |
|  36 | Is weight loss serious? | medical | medical |      NX | ✓ |   13.71s | 0.9500 |
|  37 | How does gravity work? | general | general |     AGX | ✓ |   29.25s | 0.6667 |
|  38 | Can arthritis be cured? | medical | medical |      NX | ✓ |   12.08s | 0.9500 |
|  39 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   28.32s | 1.0000 |
|  40 | What medications are used for influenza? | medical | medical |      NX | ✓ |   21.76s | 0.9500 |
|  41 | How does WiFi work? | general | medical |      NX | ✗ |   31.46s | 1.0000 |
|  42 | What is the capital of Spain? | general | general |     AGX | ✓ |   20.94s | 1.0000 |
|  43 | What is the history of jazz music? | general | medical |      NX | ✗ |   55.13s | 1.0000 |
|  44 | How to relieve weight loss at home? | medical | medical |      NX | ✓ |   28.02s | 0.9500 |
|  45 | Tell me about the internet | general | general |     AGX | ✓ |   20.80s | 1.0000 |
|  46 | What is the largest octopuses species? | general | general |     AGX | ✓ |   29.69s | 0.6667 |
|  47 | How are mountains formed? | general | general |     AGX | ✓ |   24.63s | 0.6667 |
|  48 | What is the history of the stock market? | general | general |     AGX | ✓ |   29.41s | 0.6667 |
|  49 | How did climate change originate? | general | general |     AGX | ✓ |   25.80s | 0.6667 |
|  50 | How deep is the ocean? | general | general |     AGX | ✓ |   19.25s | 1.0000 |
|  51 | What is the recovery time for hepatitis treatment? | medical | medical |      NX | ✓ |   17.52s | 0.9500 |
|  52 | When was Marie Curie born? | general | medical |      NX | ✗ |   23.72s | 1.0000 |
|  53 | What foods should I avoid with osteoporosis? | medical | medical |      NX | ✓ |   54.90s | 1.0000 |
|  54 | What does muscle cramps indicate? | medical | medical |      NX | ✓ |   14.12s | 0.9500 |
|  55 | What is the population of Turkey? | general | general |     AGX | ✓ |   19.28s | 1.0000 |
|  56 | What are the risk factors for conjunctivitis? | medical | medical |      NX | ✓ |   50.92s | 1.0000 |
|  57 | What is the capital of Kenya? | general | general |     AGX | ✓ |   18.29s | 1.0000 |
|  58 | Why do I have cough? | medical | medical |      NX | ✓ |    8.44s | 0.9500 |
|  59 | How is COPD diagnosed? | medical | medical |      NX | ✓ |   13.26s | 0.9500 |
|  60 | Can hypertension be cured? | medical | medical |      NX | ✓ |   19.70s | 0.9500 |
|  61 | Tell me about origami | general | general |     AGX | ✓ |   29.40s | 0.6667 |
|  62 | How to relieve numbness at home? | medical | medical |      NX | ✓ |   38.29s | 1.0000 |
|  63 | Can malaria cause chest pain? | medical | medical |      NX | ✓ |   13.98s | 0.9500 |
|  64 | What does nausea indicate? | medical | medical |      NX | ✓ |   19.10s | 0.9500 |
|  65 | What is the capital of South Korea? | general | general |     AGX | ✓ |   18.39s | 1.0000 |
|  66 | How long do polar bears live? | general | medical |      NX | ✗ |   25.65s | 1.0000 |
|  67 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   27.05s | 1.0000 |
|  68 | How are mountains formed? | general | medical |      NX | ✗ |   43.31s | 1.0000 |
|  69 | What is the recovery time for bronchitis treatment | medical | medical |      NX | ✓ |   16.18s | 0.9500 |
|  70 | Can migraine be cured? | medical | medical |      NX | ✓ |   37.43s | 1.0000 |
|  71 | Can you explain WiFi simply? | general | general |     AGX | ✓ |   24.46s | 1.0000 |
|  72 | What is the largest elephants species? | general | general |     AGX | ✓ |   21.18s | 0.6667 |
|  73 | What is the population of Nigeria? | general | medical |      NX | ✗ |   25.08s | 1.0000 |
|  74 | What causes osteoporosis? | medical | medical |      NX | ✓ |   42.57s | 1.0000 |
|  75 | What is the speed of light? | general | general |     AGX | ✓ |   20.54s | 0.6667 |
|  76 | What is the population of France? | general | general |     AGX | ✓ |   24.90s | 1.0000 |
|  77 | How did the Renaissance originate? | general | general |     AGX | ✓ |   26.26s | 0.6667 |
|  78 | What is the treatment for hepatitis? | medical | medical |      NX | ✓ |   18.71s | 0.9500 |
|  79 | Can asthma be cured? | medical | medical |      NX | ✓ |   11.82s | 0.9500 |
|  80 | How to relieve fever at home? | medical | medical |      NX | ✓ |   20.15s | 0.9500 |
|  81 | What causes rain? | general | general |     AGX | ✓ |   19.48s | 1.0000 |
|  82 | What is the population of Vietnam? | general | general |     AGX | ✓ |   19.18s | 0.6667 |
|  83 | Does asthma run in families? | medical | medical |      NX | ✓ |   11.27s | 0.9500 |
|  84 | How to relieve chest pain at home? | medical | medical |      NX | ✓ |   23.84s | 0.9500 |
|  85 | How long do octopuses live? | general | medical |      NX | ✗ |   27.45s | 1.0000 |
|  86 | What do octopuses eat? | general | medical |      NX | ✗ |   28.45s | 1.0000 |
|  87 | Who was Leonardo da Vinci? | general | general |     AGX | ✓ |   29.49s | 1.0000 |
|  88 | Why do I have sore throat? | medical | medical |      NX | ✓ |   44.64s | 1.0000 |
|  89 | What is the largest bald eagles species? | general | general |     AGX | ✓ |   29.68s | 0.6667 |
|  90 | What is the speed of light? | general | general |     AGX | ✓ |   19.68s | 0.6667 |
|  91 | What is the capital of Canada? | general | general |     AGX | ✓ |   18.48s | 1.0000 |
|  92 | Is asthma contagious? | medical | medical |      NX | ✓ |   10.76s | 0.9500 |
|  93 | What is Egypt known for? | general | general |     AGX | ✓ |   20.61s | 1.0000 |
|  94 | How long does gastritis last? | medical | medical |      NX | ✓ |   29.17s | 1.0000 |
|  95 | Is anemia contagious? | medical | general |     AGX | ✗ |   18.62s | 0.6667 |
|  96 | What medications are used for osteoporosis? | medical | medical |      NX | ✓ |   32.27s | 0.9500 |
|  97 | Can hepatitis cause blurred vision? | medical | medical |      NX | ✓ |   30.31s | 1.0000 |
|  98 | When should I see a doctor for cough? | medical | medical |      NX | ✓ |    9.50s | 0.9500 |
|  99 | How is conjunctivitis diagnosed? | medical | medical |      NX | ✓ |   36.34s | 0.9500 |
| 100 | What is the recovery time for influenza treatment? | medical | medical |      NX | ✓ |   18.22s | 0.9500 |

---

## Query Generation Templates

**Medical** (25 templates × 20 diseases × 20 symptoms × 15 treatments × 10 parts)  
**General** (25 templates × 20 countries × 15 concepts × 10 people × 10 animals × 12 topics)  
