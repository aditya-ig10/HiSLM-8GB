#!/usr/bin/env python3
"""
test_comprehensive.py — Full end-to-end routing evaluation with random queries.

Generates 100+ random medical/general queries, sends them through the
subserver (which routes medical→NX, general→AGX), and produces a detailed
classification report with Accuracy, Precision, Recall, F1, MCC, Support.

Usage:
  python test_comprehensive.py                          # default 100 queries
  python test_comprehensive.py --count 200              # custom count
  python test_comprehensive.py --output docs/eval_report.md
  python test_comprehensive.py --subserver http://localhost:8765
"""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter

import requests

# ── Template-based random query generators ───────────────────────────

DISEASES = [
    "diabetes", "hypertension", "asthma", "arthritis", "pneumonia",
    "tuberculosis", "malaria", "hepatitis", "influenza", "anemia",
    "migraine", "eczema", "osteoporosis", "bronchitis", "gastritis",
    "thyroiditis", "conjunctivitis", "UTI", "GERD", "COPD",
]

SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "chest pain",
    "shortness of breath", "nausea", "dizziness", "weight loss",
    "joint pain", "rash", "swollen lymph nodes", "blurred vision",
    "sore throat", "abdominal pain", "muscle cramps", "insomnia",
    "loss of appetite", "back pain", "numbness",
]

TREATMENTS = [
    "insulin", "metformin", "ibuprofen", "antibiotics", "steroids",
    "antihistamines", "beta blockers", "ACE inhibitors", "inhalers",
    "dialysis", "chemotherapy", "physical therapy", "antivirals",
    "PPIs", "NSAIDs",
]

BODY_PARTS = [
    "heart", "lungs", "liver", "kidneys", "brain",
    "stomach", "thyroid", "pancreas", "intestines", "spine",
]

MEDICAL_TEMPLATES = [
    "What are the symptoms of {disease}?",
    "How is {disease} diagnosed?",
    "What causes {disease}?",
    "Is {disease} contagious?",
    "What is the treatment for {disease}?",
    "Can {disease} be cured?",
    "What are the risk factors for {disease}?",
    "How does {disease} affect the {part}?",
    "What medications are used for {disease}?",
    "Is {treatment} effective for {disease}?",
    "What are the side effects of {treatment}?",
    "How long does {disease} last?",
    "What is the normal dosage for {treatment}?",
    "Can {symptom} be a sign of {disease}?",
    "Why do I have {symptom}?",
    "How to relieve {symptom} at home?",
    "When should I see a doctor for {symptom}?",
    "What does {symptom} indicate?",
    "Is {symptom} serious?",
    "How to prevent {disease}?",
    "What foods should I avoid with {disease}?",
    "Does {disease} run in families?",
    "What blood tests check for {disease}?",
    "Can {disease} cause {symptom}?",
    "What is the recovery time for {disease} treatment?",
]

COUNTRIES = [
    "France", "Japan", "Brazil", "Canada", "Australia",
    "India", "Germany", "Egypt", "Mexico", "South Korea",
    "Italy", "Spain", "Argentina", "Sweden", "Vietnam",
    "Nigeria", "Turkey", "Thailand", "Kenya", "Norway",
]

CONCEPTS = [
    "quantum computing", "photosynthesis", "the Roman Empire",
    "machine learning", "plate tectonics", "the French Revolution",
    "the water cycle", "gravity", "evolution", "blockchain",
    "electricity", "DNA", "the solar system", "WiFi", "supply and demand",
]

PEOPLE = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci",
    "Isaac Newton", "Frida Kahlo", "Mozart", "Shakespeare",
    "Cleopatra", "Gandhi", "Alan Turing",
]

ANIMALS = [
    "elephants", "penguins", "dolphins", "octopuses", "butterflies",
    "sharks", "bees", "chameleons", "bald eagles", "polar bears",
]

TOPICS = [
    "climate change", "democracy", "the Renaissance",
    "basketball", "cooking pasta", "photography",
    "the Great Wall of China", "origami", "the internet",
    "volcanoes", "the stock market", "jazz music",
]

GENERAL_TEMPLATES = [
    "What is the capital of {country}?",
    "What language is spoken in {country}?",
    "What is {country} known for?",
    "What is the population of {country}?",
    "What is {concept}?",
    "How does {concept} work?",
    "Can you explain {concept} simply?",
    "Who was {person}?",
    "When was {person} born?",
    "What did {person} discover?",
    "Where do {animal} live?",
    "What do {animal} eat?",
    "How long do {animal} live?",
    "What is the largest {animal} species?",
    "Tell me about {topic}",
    "How did {topic} originate?",
    "What is the history of {topic}?",
    "Why is {topic} important?",
    "How many countries are in {continent}?",
    "What is the tallest mountain in the world?",
    "How deep is the ocean?",
    "What is the speed of light?",
    "How do airplanes fly?",
    "What causes rain?",
    "How are mountains formed?",
]

CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Australia", "Antarctica"]


def generate_medical(count):
    queries = []
    for _ in range(count):
        tmpl = random.choice(MEDICAL_TEMPLATES)
        disease = random.choice(DISEASES)
        symptom = random.choice(SYMPTOMS)
        treatment = random.choice(TREATMENTS)
        part = random.choice(BODY_PARTS)
        text = tmpl.format(disease=disease, symptom=symptom, treatment=treatment, part=part)
        queries.append({"text": text, "label": 1, "category": "medical"})
    return queries


def generate_general(count):
    queries = []
    continents = CONTINENTS
    for _ in range(count):
        tmpl = random.choice(GENERAL_TEMPLATES)
        country = random.choice(COUNTRIES)
        concept = random.choice(CONCEPTS)
        person = random.choice(PEOPLE)
        animal = random.choice(ANIMALS)
        topic = random.choice(TOPICS)
        continent = random.choice(continents)
        text = tmpl.format(country=country, concept=concept, person=person,
                           animal=animal, topic=topic, continent=continent)
        queries.append({"text": text, "label": 0, "category": "general"})
    return queries


# ── Metrics ───────────────────────────────────────────────────────────

def compute_metrics(tp, fp, fn, tn):
    eps = 1e-12
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision_med = tp / (tp + fp + eps)
    precision_gen = tn / (tn + fn + eps)
    recall_med = tp / (tp + fn + eps)
    recall_gen = tn / (tn + fp + eps)
    f1_med = 2 * precision_med * recall_med / (precision_med + recall_med + eps)
    f1_gen = 2 * precision_gen * recall_gen / (precision_gen + recall_gen + eps)

    # Matthews Correlation Coefficient
    numerator = tp * tn - fp * fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = numerator / denominator

    return {
        "accuracy": round(accuracy, 4),
        "precision_medical": round(precision_med, 4),
        "precision_general": round(precision_gen, 4),
        "recall_medical": round(recall_med, 4),
        "recall_general": round(recall_gen, 4),
        "f1_medical": round(f1_med, 4),
        "f1_general": round(f1_gen, 4),
        "mcc": round(mcc, 4),
        "support_medical": int(tp + fn),
        "support_general": int(tn + fp),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comprehensive routing evaluation")
    parser.add_argument("--count", type=int, default=100,
                        help="Total queries (half medical, half general)")
    parser.add_argument("--subserver", default="http://localhost:8765",
                        help="Subserver URL")
    parser.add_argument("--output", default="docs/eval_report.md",
                        help="Output report path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--target-method", choices=["chat", "classify"], default="chat",
                        help="Endpoint to hit: /chat (full pipeline) or /classify (routing only)")
    args = parser.parse_args()

    random.seed(args.seed)

    half = args.count // 2
    queries = generate_medical(half) + generate_general(args.count - half)
    random.shuffle(queries)

    print(f"Generated {len(queries)} queries ({half} medical, {len(queries)-half} general)")
    print(f"Subserver: {args.subserver}/{'chat' if args.target_method == 'chat' else 'classify'}")
    print()

    endpoint = f"{args.subserver}/{'chat' if args.target_method == 'chat' else 'classify'}"

    results = []
    errors = 0

    for i, q in enumerate(queries):
        gt = "medical" if q["label"] == 1 else "general"
        t0 = time.time()
        try:
            if args.target_method == "chat":
                r = requests.post(endpoint, json={"message": q["text"], "stream": False},
                                  timeout=300)
                body = r.json() if r.ok else {}
                pred_source = body.get("source", "?")
                pred_label = 1 if pred_source == "NX" else 0
                confidence = body.get("confidence")
                timing = body.get("timing_ms", {}) if isinstance(body.get("timing_ms"), dict) else {}
            else:
                r = requests.post(endpoint, json={"text": q["text"]}, timeout=300)
                body = r.json() if r.ok else {}
                pred_label = 1 if body.get("is_medical") else 0
                pred_source = "keyword" if body.get("method") == "keyword" else "llm"
                confidence = body.get("confidence")
                timing = {"classify_ms": body.get("classify_ms")}

            elapsed = time.time() - t0
            correct = pred_label == q["label"]

        except Exception as e:
            elapsed = time.time() - t0
            pred_label = -1
            pred_source = "error"
            confidence = None
            timing = {}
            correct = False
            errors += 1
            body = {}

        row = {
            "id": i + 1,
            "query": q["text"][:60],
            "ground_truth": gt,
            "category": q["category"],
            "predicted": "medical" if pred_label == 1 else ("general" if pred_label == 0 else "error"),
            "predicted_source": pred_source,
            "correct": correct,
            "latency_s": round(elapsed, 2),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "classify_ms": timing.get("classify_ms"),
            "inference_ms": timing.get("inference_ms"),
        }
        results.append(row)

        mark = "✓" if correct else "✗"
        print(f"  [{i+1:>3d}/{len(queries)}] {mark} GT={gt:>7s} → {row['predicted']:>7s}  "
              f"src={pred_source:>7s}  c={str(confidence):>7s}  "
              f"{elapsed:7.2f}s  \"{q['text'][:50]}...\"")

    # ── Compute metrics ─────────────────────────────────────────────
    tp = sum(1 for r in results if r["ground_truth"] == "medical" and r["correct"])
    fp = sum(1 for r in results if r["ground_truth"] == "general" and not r["correct"])
    fn = sum(1 for r in results if r["ground_truth"] == "medical" and not r["correct"])
    tn = sum(1 for r in results if r["ground_truth"] == "general" and r["correct"])

    metrics = compute_metrics(tp, fp, fn, tn)

    # Per-source stats
    method_counts = Counter(r["predicted_source"] for r in results if r["predicted_source"] != "error")
    method_ok = Counter(r["predicted_source"] for r in results if r["correct"] and r["predicted_source"] != "error")
    method_err = Counter(r["predicted_source"] for r in results if not r["correct"] and r["predicted_source"] != "error")

    # Latency stats
    ok_results = [r for r in results if r["latency_s"] > 0]
    lats = [r["latency_s"] for r in ok_results]
    med_lats = [r["latency_s"] for r in ok_results if r["ground_truth"] == "medical"]
    gen_lats = [r["latency_s"] for r in ok_results if r["ground_truth"] == "general"]

    def pctile(arr, p):
        if not arr:
            return 0
        s = sorted(arr)
        return s[int(p / 100 * (len(s) - 1))]

    # ── Generate Report ──────────────────────────────────────────────
    report = []
    report.append(f"# Comprehensive Routing Evaluation Report")
    report.append(f"")
    report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    report.append(f"**Random seed:** {args.seed}  ")
    report.append(f"**Total queries:** {len(queries)} ({len([q for q in queries if q['label']==1])} medical, "
                  f"{len([q for q in queries if q['label']==0])} general)  ")
    report.append(f"**Endpoint:** `{endpoint}`  ")
    report.append(f"**NX Subserver:** {args.subserver}  ")
    report.append(f"")
    report.append(f"---")
    report.append(f"")
    report.append(f"## Confusion Matrix")
    report.append(f"")
    report.append(f"| Actual \\\\ Predicted | **Medical** | **General** | **Total** |")
    report.append(f"|---|---|---|---|")
    report.append(f"| **Medical** | {tp} (TP) | {fn} (FN) | {tp+fn} |")
    report.append(f"| **General** | {fp} (FP) | {tn} (TN) | {tn+fp} |")
    report.append(f"")
    report.append(f"## Classification Metrics")
    report.append(f"")
    report.append(f"| Metric | Medical | General | Overall |")
    report.append(f"|---|---|---|---|")
    report.append(f"| **Precision** | {metrics['precision_medical']:.4f} | {metrics['precision_general']:.4f} | — |")
    report.append(f"| **Recall** | {metrics['recall_medical']:.4f} | {metrics['recall_general']:.4f} | — |")
    report.append(f"| **F1 Score** | {metrics['f1_medical']:.4f} | {metrics['f1_general']:.4f} | — |")
    report.append(f"| **Support** | {metrics['support_medical']} | {metrics['support_general']} | {metrics['support_medical']+metrics['support_general']} |")
    report.append(f"| **Accuracy** | — | — | **{metrics['accuracy']:.4f}** |")
    report.append(f"| **MCC** | — | — | **{metrics['mcc']:.4f}** |")
    report.append(f"")
    report.append(f"- **Accuracy**: {metrics['accuracy']*100:.2f}% ({tp+tn}/{tp+tn+fp+fn})  ")
    report.append(f"- **MCC (Matthews Correlation Coefficient)**: {metrics['mcc']:.4f} "
                  f"(+1 = perfect, 0 = random, -1 = inverse)  ")
    report.append(f"- **False positives (general→medical)**: {fp} — routed to NX instead of AGX  ")
    report.append(f"- **False negatives (medical→general)**: {fn} — routed to AGX instead of NX  ")
    report.append(f"")

    # Latency table
    report.append(f"## Latency Breakdown")
    report.append(f"")
    report.append(f"| Metric | All Queries | Medical (→NX) | General (→AGX) |")
    report.append(f"|---|---|---|---|")
    report.append(f"| **Count** | {len(lats)} | {len(med_lats)} | {len(gen_lats)} |")
    report.append(f"| **Mean** | {sum(lats)/len(lats):.2f}s" if lats else "| **Mean** | — |")
    report.append(f"| **Median (P50)** | {pctile(lats, 50):.2f}s" if lats else "| — |")
    report.append(f"| **P95** | {pctile(lats, 95):.2f}s" if lats else "| — |")
    report.append(f"| **P99** | {pctile(lats, 99):.2f}s" if lats else "| — |")
    report.append(f"| **Min** | {min(lats):.2f}s" if lats else "| — |")
    report.append(f"| **Max** | {max(lats):.2f}s" if lats else "| — |")
    medical_mean = sum(med_lats)/len(med_lats) if med_lats else 0
    general_mean = sum(gen_lats)/len(gen_lats) if gen_lats else 0
    if med_lats:
        report.append(f"| **Mean** | — | {medical_mean:.2f}s | {general_mean:.2f}s |")

    report.append(f"")
    report.append(f"### Latency Distribution (All)")
    report.append(f"")
    buckets = [(0,3),(3,8),(8,15),(15,30),(30,60),(60,120)]
    for lo, hi in buckets:
        c = sum(1 for l in lats if lo <= l < hi)
        if c:
            bar = "█" * max(1, int(c / max(1, len(lats)) * 50))
            report.append(f"  {lo:>3d}-{hi:>3d}s: {c:>3d} ({c/len(lats)*100:5.1f}%) {bar}")

    report.append(f"")
    report.append(f"## Classification Method Distribution")
    report.append(f"")
    report.append(f"| Method | Total | Correct | Errors | Accuracy |")
    report.append(f"|---|---|---|---|---|")
    for method in ["keyword", "llm"]:
        total = method_counts.get(method, 0)
        ok = method_ok.get(method, 0)
        err = method_err.get(method, 0)
        acc = f"{ok/total*100:.1f}%" if total else "—"
        report.append(f"| **{method}** | {total} | {ok} | {err} | {acc} |")

    report.append(f"")
    report.append(f"## Per-Query Results")
    report.append(f"")
    report.append(f"| # | Query | GT | Pred | Source | Correct | Latency | Confidence |")
    report.append(f"|---|---|---|---|---|---|---|---|")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        conf = f"{r['confidence']:.4f}" if r["confidence"] is not None else "—"
        report.append(f"| {r['id']:>3d} | {r['query'][:50]} | {r['ground_truth']:>7s} | "
                      f"{r['predicted']:>7s} | {r['predicted_source']:>7s} | {mark} | "
                      f"{r['latency_s']:7.2f}s | {conf} |")

    report.append(f"")
    report.append(f"---")
    report.append(f"")
    report.append(f"## Query Generation Templates")
    report.append(f"")
    report.append(f"**Medical** ({len(MEDICAL_TEMPLATES)} templates × {len(DISEASES)} diseases "
                  f"× {len(SYMPTOMS)} symptoms × {len(TREATMENTS)} treatments × {len(BODY_PARTS)} parts)  ")
    report.append(f"**General** ({len(GENERAL_TEMPLATES)} templates × {len(COUNTRIES)} countries "
                  f"× {len(CONCEPTS)} concepts × {len(PEOPLE)} people × {len(ANIMALS)} animals "
                  f"× {len(TOPICS)} topics)  ")
    report.append(f"")

    # Write
    report_str = "\n".join(report)
    with open(args.output, "w") as f:
        f.write(report_str)
    print(f"\nReport saved to {args.output}")

    # Also save raw results JSON
    results_path = "results/comprehensive_eval_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "config": {"count": args.count, "seed": args.seed, "subserver": args.subserver,
                       "endpoint": endpoint},
            "metrics": metrics,
            "results": results,
        }, f, indent=2)
    print(f"Raw results saved to {results_path}")

    # Print summary to stdout
    print(f"\n{'='*65}")
    print(f"  COMPREHENSIVE EVAL — SUMMARY")
    print(f"{'='*65}")
    print(f"  Total:    {len(queries)}")
    print(f"  Correct:  {tp+tn}")
    print(f"  Errors:   {errors}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: med={metrics['precision_medical']:.4f}, gen={metrics['precision_general']:.4f}")
    print(f"  Recall:   med={metrics['recall_medical']:.4f}, gen={metrics['recall_general']:.4f}")
    print(f"  F1:       med={metrics['f1_medical']:.4f}, gen={metrics['f1_general']:.4f}")
    print(f"  MCC:      {metrics['mcc']:.4f}")
    print(f"  Avg lat:  {sum(lats)/len(lats):.2f}s (med={medical_mean:.2f}s, gen={general_mean:.2f}s)")
    print(f"  FP:       {fp} (general→medical)")
    print(f"  FN:       {fn} (medical→general)")
    print(f"  Methods:  {dict(method_counts)}")


if __name__ == "__main__":
    main()
