# Comprehensive Routing Evaluation Report

**Generated:** 2026-06-24 06:47:45  
**Random seed:** 123  
**Total queries:** 90 (45 medical, 45 general)  
**Endpoint:** `http://localhost:8765/chat`  
**NX Subserver:** http://localhost:8765  

---

## Confusion Matrix

| Actual \\ Predicted | **Medical** | **General** | **Total** |
|---|---|---|---|
| **Medical** | 45 (TP) | 0 (FN) | 45 |
| **General** | 19 (FP) | 26 (TN) | 45 |

## Classification Metrics

| Metric | Medical | General | Overall |
|---|---|---|---|
| **Precision** | 0.7031 | 1.0000 | — |
| **Recall** | 1.0000 | 0.5778 | — |
| **F1 Score** | 0.8257 | 0.7324 | — |
| **Support** | 45 | 45 | 90 |
| **Accuracy** | — | — | **0.7889** |
| **MCC** | — | — | **0.6374** |

- **Accuracy**: 78.89% (71/90)  
- **MCC (Matthews Correlation Coefficient)**: 0.6374 (+1 = perfect, 0 = random, -1 = inverse)  
- **False positives (general→medical)**: 19 — routed to NX instead of AGX  
- **False negatives (medical→general)**: 0 — routed to AGX instead of NX  

## Latency Breakdown

| Metric | All Queries | Medical (→NX) | General (→AGX) |
|---|---|---|---|
| **Count** | 90 | 45 | 45 |
| **Mean** | 32.66s
| **Median (P50)** | 28.52s
| **P95** | 68.80s
| **P99** | 81.74s
| **Min** | 9.03s
| **Max** | 86.40s
| **Mean** | — | 32.91s | 32.41s |

### Latency Distribution (All)

    8- 15s:   6 (  6.7%) ███
   15- 30s:  53 ( 58.9%) █████████████████████████████
   30- 60s:  24 ( 26.7%) █████████████
   60-120s:   7 (  7.8%) ███

## Classification Method Distribution

| Method | Total | Correct | Errors | Accuracy |
|---|---|---|---|---|
| **keyword** | 0 | 0 | 0 | — |
| **llm** | 0 | 0 | 0 | — |

## Per-Query Results

| # | Query | GT | Pred | Source | Correct | Latency | Confidence |
|---|---|---|---|---|---|---|---|
|   1 | How to relieve joint pain at home? | medical | medical |      NX | ✓ |   52.89s | 0.9500 |
|   2 | How are mountains formed? | general | general |     AGX | ✓ |   46.60s | 0.6667 |
|   3 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   64.10s | 1.0000 |
|   4 | What do penguins eat? | general | medical |      NX | ✗ |   73.34s | 1.0000 |
|   5 | How long does hepatitis last? | medical | medical |      NX | ✓ |   76.00s | 1.0000 |
|   6 | What causes pneumonia? | medical | medical |      NX | ✓ |   32.47s | 0.9500 |
|   7 | Tell me about democracy | general | general |     AGX | ✓ |   49.69s | 1.0000 |
|   8 | What is the recovery time for asthma treatment? | medical | medical |      NX | ✓ |   42.97s | 0.9500 |
|   9 | What foods should I avoid with anemia? | medical | medical |      NX | ✓ |   81.74s | 1.0000 |
|  10 | What is the normal dosage for insulin? | medical | medical |      NX | ✓ |   81.31s | 0.9500 |
|  11 | What medications are used for osteoporosis? | medical | medical |      NX | ✓ |   48.32s | 0.9500 |
|  12 | What is the recovery time for UTI treatment? | medical | medical |      NX | ✓ |   68.80s | 0.9500 |
|  13 | What is the recovery time for tuberculosis treatme | medical | medical |      NX | ✓ |   29.10s | 0.9500 |
|  14 | How are mountains formed? | general | medical |      NX | ✗ |   86.40s | 1.0000 |
|  15 | Does influenza run in families? | medical | medical |      NX | ✓ |   28.56s | 1.0000 |
|  16 | What is the largest dolphins species? | general | general |     AGX | ✓ |   18.31s | 1.0000 |
|  17 | What are the side effects of inhalers? | medical | medical |      NX | ✓ |   22.03s | 0.9500 |
|  18 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   27.73s | 1.0000 |
|  19 | What does shortness of breath indicate? | medical | medical |      NX | ✓ |   57.07s | 1.0000 |
|  20 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   26.63s | 1.0000 |
|  21 | What is gravity? | general | general |     AGX | ✓ |   18.50s | 1.0000 |
|  22 | How long does asthma last? | medical | medical |      NX | ✓ |   10.93s | 0.9500 |
|  23 | Why do I have swollen lymph nodes? | medical | medical |      NX | ✓ |   39.28s | 1.0000 |
|  24 | What is the recovery time for conjunctivitis treat | medical | medical |      NX | ✓ |   28.78s | 0.9500 |
|  25 | What does rash indicate? | medical | medical |      NX | ✓ |   24.60s | 0.9500 |
|  26 | What is the speed of light? | general | medical |      NX | ✗ |   26.23s | 1.0000 |
|  27 | Is ACE inhibitors effective for bronchitis? | medical | medical |      NX | ✓ |   28.84s | 1.0000 |
|  28 | Can pneumonia be cured? | medical | medical |      NX | ✓ |    9.97s | 0.9500 |
|  29 | How does osteoporosis affect the heart? | medical | medical |      NX | ✓ |   29.73s | 0.9500 |
|  30 | Tell me about volcanoes | general | general |     AGX | ✓ |   19.32s | 1.0000 |
|  31 | What is the largest chameleons species? | general | general |     AGX | ✓ |   30.32s | 0.6667 |
|  32 | How did the Renaissance originate? | general | general |     AGX | ✓ |   29.39s | 0.6667 |
|  33 | Is fever serious? | medical | medical |      NX | ✓ |    9.56s | 0.9500 |
|  34 | What is the capital of India? | general | general |     AGX | ✓ |   17.48s | 1.0000 |
|  35 | What language is spoken in Spain? | general | general |     AGX | ✓ |   29.87s | 1.0000 |
|  36 | Who was Shakespeare? | general | general |     AGX | ✓ |   24.50s | 1.0000 |
|  37 | Who was Alan Turing? | general | medical |      NX | ✗ |   54.09s | 1.0000 |
|  38 | What language is spoken in Japan? | general | general |     AGX | ✓ |   21.08s | 1.0000 |
|  39 | What medications are used for arthritis? | medical | medical |      NX | ✓ |   43.85s | 0.9500 |
|  40 | How long does influenza last? | medical | medical |      NX | ✓ |   27.48s | 1.0000 |
|  41 | How are mountains formed? | general | medical |      NX | ✗ |   54.26s | 1.0000 |
|  42 | What did Alan Turing discover? | general | medical |      NX | ✗ |   53.58s | 1.0000 |
|  43 | Is sore throat serious? | medical | medical |      NX | ✓ |   25.38s | 1.0000 |
|  44 | How long do butterflies live? | general | medical |      NX | ✗ |   27.47s | 1.0000 |
|  45 | Is swollen lymph nodes serious? | medical | medical |      NX | ✓ |   28.78s | 1.0000 |
|  46 | What language is spoken in Japan? | general | general |     AGX | ✓ |   17.70s | 1.0000 |
|  47 | What does fatigue indicate? | medical | medical |      NX | ✓ |   12.04s | 0.9500 |
|  48 | Why do I have abdominal pain? | medical | medical |      NX | ✓ |   28.52s | 0.9500 |
|  49 | How does quantum computing work? | general | medical |      NX | ✗ |   40.21s | 1.0000 |
|  50 | When should I see a doctor for swollen lymph nodes | medical | medical |      NX | ✓ |    9.46s | 0.9500 |
|  51 | What do chameleons eat? | general | medical |      NX | ✗ |   27.06s | 1.0000 |
|  52 | What causes rain? | general | general |     AGX | ✓ |   19.09s | 0.6667 |
|  53 | What are the risk factors for conjunctivitis? | medical | medical |      NX | ✓ |   54.43s | 1.0000 |
|  54 | How long do sharks live? | general | medical |      NX | ✗ |   34.46s | 1.0000 |
|  55 | What are the symptoms of arthritis? | medical | medical |      NX | ✓ |   28.55s | 0.9500 |
|  56 | Can numbness be a sign of UTI? | medical | medical |      NX | ✓ |   37.75s | 1.0000 |
|  57 | What does insomnia indicate? | medical | medical |      NX | ✓ |   25.09s | 1.0000 |
|  58 | How did photography originate? | general | general |     AGX | ✓ |   29.80s | 0.6667 |
|  59 | What are the side effects of inhalers? | medical | medical |      NX | ✓ |   21.93s | 0.9500 |
|  60 | Why is jazz music important? | general | general |     AGX | ✓ |   29.99s | 0.6667 |
|  61 | What are the symptoms of migraine? | medical | medical |      NX | ✓ |   29.97s | 0.9500 |
|  62 | What is the tallest mountain in the world? | general | medical |      NX | ✗ |   26.03s | 1.0000 |
|  63 | What are the symptoms of migraine? | medical | medical |      NX | ✓ |   19.61s | 0.9500 |
|  64 | When should I see a doctor for nausea? | medical | medical |      NX | ✓ |   19.28s | 0.9500 |
|  65 | What is the treatment for gastritis? | medical | medical |      NX | ✓ |   15.55s | 0.9500 |
|  66 | Who was Leonardo da Vinci? | general | general |     AGX | ✓ |   20.98s | 1.0000 |
|  67 | How to relieve numbness at home? | medical | medical |      NX | ✓ |   52.55s | 1.0000 |
|  68 | What is the population of South Korea? | general | general |     AGX | ✓ |   19.95s | 1.0000 |
|  69 | Can you explain supply and demand simply? | general | medical |      NX | ✗ |   48.81s | 1.0000 |
|  70 | What is the largest polar bears species? | general | general |     AGX | ✓ |   29.83s | 1.0000 |
|  71 | What language is spoken in Egypt? | general | general |     AGX | ✓ |   29.58s | 1.0000 |
|  72 | What is the French Revolution? | general | general |     AGX | ✓ |   20.55s | 1.0000 |
|  73 | How does GERD affect the brain? | medical | medical |      NX | ✓ |   25.31s | 0.9500 |
|  74 | What are the risk factors for influenza? | medical | medical |      NX | ✓ |   42.40s | 1.0000 |
|  75 | Is ibuprofen effective for bronchitis? | medical | medical |      NX | ✓ |   24.77s | 1.0000 |
|  76 | What blood tests check for hepatitis? | medical | medical |      NX | ✓ |   18.28s | 0.9500 |
|  77 | How are mountains formed? | general | medical |      NX | ✗ |   40.10s | 1.0000 |
|  78 | What did Shakespeare discover? | general | general |     AGX | ✓ |   21.56s | 1.0000 |
|  79 | What is the tallest mountain in the world? | general | general |     AGX | ✓ |   19.19s | 0.6667 |
|  80 | How are mountains formed? | general | general |     AGX | ✓ |   20.72s | 0.6667 |
|  81 | What is the largest penguins species? | general | general |     AGX | ✓ |   30.45s | 1.0000 |
|  82 | What is the largest dolphins species? | general | general |     AGX | ✓ |   30.33s | 0.6667 |
|  83 | What is the treatment for anemia? | medical | medical |      NX | ✓ |   22.35s | 0.9500 |
|  84 | How deep is the ocean? | general | general |     AGX | ✓ |   19.67s | 1.0000 |
|  85 | How long do chameleons live? | general | medical |      NX | ✗ |   27.63s | 1.0000 |
|  86 | How long do elephants live? | general | medical |      NX | ✗ |   25.30s | 1.0000 |
|  87 | How is influenza diagnosed? | medical | medical |      NX | ✓ |   27.38s | 0.9500 |
|  88 | Can you explain machine learning simply? | general | medical |      NX | ✗ |   30.66s | 1.0000 |
|  89 | What does fatigue indicate? | medical | medical |      NX | ✓ |    9.03s | 0.9500 |
|  90 | What are the symptoms of tuberculosis? | medical | medical |      NX | ✓ |   28.23s | 0.9500 |

---

## Query Generation Templates

**Medical** (25 templates × 20 diseases × 20 symptoms × 15 treatments × 10 parts)  
**General** (25 templates × 20 countries × 15 concepts × 10 people × 10 animals × 12 topics)  
