#!/usr/bin/env python3
"""
Preprocess all 3 datasets into unified instruction/input/output format.

Output: dataset/training_data.jsonl  (one JSON object per line)
"""
import json
import os
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "dataset"
OUT_FILE = OUT_DIR / "training_data.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

# ── 1. Med_QA ──────────────────────────────────────────────────────

def process_medqa_us():
    path = BASE / "dataset/med_qa/questions/US"
    count = 0
    for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        fp = path / split
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                q = d.get("question", "")
                opts = d.get("options", {})
                ans = d.get("answer", "")
                meta = d.get("meta_info", "")
                opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(opts.items())])
                instruction = q
                input_text = f"Options:\n{opts_str}"
                output = ans
                records.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "source": f"med_qa_us_{split}_{meta}",
                })
                count += 1
    print(f"  Med_QA US: {count} records")
    return count

def process_medqa_mainland():
    path = BASE / "dataset/med_qa/questions/Mainland"
    count = 0
    for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        fp = path / split
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                q = d.get("question", "")
                opts = d.get("options", {})
                ans = d.get("answer", "")
                meta = d.get("meta_info", "")
                opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(opts.items())])
                records.append({
                    "instruction": q,
                    "input": f"选项：\n{opts_str}",
                    "output": ans,
                    "source": f"med_qa_mainland_{split}_{meta}",
                })
                count += 1
    print(f"  Med_QA Mainland: {count} records")
    return count

def process_medqa_taiwan():
    path = BASE / "dataset/med_qa/questions/Taiwan"
    count = 0
    for split in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        fp = path / split
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                q = d.get("question", "")
                opts = d.get("options", {})
                ans = d.get("answer", "")
                meta = d.get("meta_info", "")
                opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(opts.items())])
                records.append({
                    "instruction": q,
                    "input": f"Options:\n{opts_str}",
                    "output": ans,
                    "source": f"med_qa_taiwan_{split}_{meta}",
                })
                count += 1
    # Also process translated versions
    trans_path = path / "tw_translated_jsonl/en"
    for fname in ["train-2en.jsonl", "dev-2en.jsonl", "test-2en.jsonl"]:
        fp = trans_path / fname
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                q = d.get("question", d.get("Question", ""))
                opts = d.get("options", d.get("Options", {}))
                ans = d.get("answer", d.get("Answer", ""))
                opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(opts.items())])
                records.append({
                    "instruction": q,
                    "input": f"Options:\n{opts_str}",
                    "output": ans,
                    "source": f"med_qa_taiwan_en_{fname}",
                })
                count += 1
    print(f"  Med_QA Taiwan: {count} records")
    return count

def process_medqa_textbooks():
    """Convert textbook paragraphs into instruction-output pairs for knowledge."""
    count = 0
    langs = {
        "en": BASE / "dataset/med_qa/textbooks/en",
    }
    for lang, path in langs.items():
        if not path.exists():
            continue
        for fp in sorted(path.glob("*.txt")):
            subject = fp.stem
            text = fp.read_text(encoding="utf-8").strip()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                if len(para) < 100:
                    continue
                # Get first sentence as title
                title = para.split(".")[0] if "." in para else para[:80]
                records.append({
                    "instruction": f"Explain the following from {subject}: {title}",
                    "input": "",
                    "output": para,
                    "source": f"med_qa_textbook_{lang}_{subject}",
                })
                count += 1
    print(f"  Med_QA Textbooks: {count} records")
    return count

# Also process the 4-options variant
def process_medqa_4options():
    count = 0
    base = BASE / "dataset/med_qa/questions/US/4_options"
    if base.exists():
        for fname in ["phrases_no_exclude_train.jsonl", "phrases_no_exclude_dev.jsonl", "phrases_no_exclude_test.jsonl"]:
            fp = base / fname
            if not fp.exists():
                continue
            with open(fp) as f:
                for line in f:
                    d = json.loads(line)
                    q = d.get("question", "")
                    opts = d.get("options", {})
                    ans = d.get("answer", "")
                    opts_str = "\n".join([f"{k}. {v}" for k, v in sorted(opts.items())])
                    records.append({
                        "instruction": q,
                        "input": f"Options:\n{opts_str}",
                        "output": ans,
                        "source": f"med_qa_4opts_{fname}",
                    })
                    count += 1
    print(f"  Med_QA 4-Options: {count} records")
    return count

# ── 2. MT Samples (Medical Transcriptions) ─────────────────────────

def process_mt_samples():
    count = 0
    fp = BASE / "dataset/mt_samples/mtsamples.csv"
    if not fp.exists():
        print("  MT Samples CSV not found, skipping")
        return 0
    import csv
    with open(fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = row.get("description", "").strip()
            spec = row.get("medical_specialty", "").strip()
            sample = row.get("sample_name", "").strip()
            transcription = row.get("transcription", "").strip()
            keywords = row.get("keywords", "").strip()
            if not transcription:
                continue
            instruction = f"Generate a medical transcription for: {sample}" if sample else "Generate a medical transcription"
            if spec:
                instruction += f" (Specialty: {spec})"
            records.append({
                "instruction": instruction,
                "input": f"Description: {desc}\nKeywords: {keywords}" if desc or keywords else "",
                "output": transcription,
                "source": f"mt_samples_{spec}",
            })
            count += 1
    print(f"  MT Samples: {count} records")
    return count

# ── 3. Pub_Med_QA ──────────────────────────────────────────────────

def process_pubmedqa():
    count = 0
    fp = BASE / "dataset/pub_med_qa/train-00000-of-00001.parquet"
    if not fp.exists():
        print("  Pub_Med_QA parquet not found, skipping")
        return 0
    import pandas as pd
    import numpy as np
    df = pd.read_parquet(fp)
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        q = row_dict.get("question", row_dict.get("Question", ""))
        answer = row_dict.get("answer", row_dict.get("Answer", row_dict.get("long_answer", "")))
        context = row_dict.get("context", row_dict.get("Context", ""))
        # Convert numpy arrays to string
        if isinstance(context, np.ndarray):
            context = "\n".join([str(c) for c in context])
        context = context if isinstance(context, str) else str(context) if context else ""
        if not q or not answer:
            continue
        records.append({
            "instruction": str(q),
            "input": context if context else "",
            "output": str(answer),
            "source": "pubmed_qa",
        })
        count += 1
    print(f"  Pub_Med_QA: {count} records")
    return count

# ── MAIN ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Processing datasets...\n")

    print("1. Med_QA datasets:")
    process_medqa_us()
    process_medqa_mainland()
    process_medqa_taiwan()
    process_medqa_4options()
    process_medqa_textbooks()

    print("\n2. MT Samples:")
    process_mt_samples()

    print("\n3. Pub_Med_QA:")
    process_pubmedqa()

    # Shuffle and deduplicate by instruction
    print(f"\nTotal raw records: {len(records)}")
    seen = set()
    deduped = []
    for r in records:
        key = (r["instruction"], r["output"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    random.shuffle(deduped)

    # Split into train/val (95/5)
    split = int(len(deduped) * 0.95)
    train = deduped[:split]
    val = deduped[split:]

    train_file = OUT_DIR / "train.jsonl"
    val_file = OUT_DIR / "val.jsonl"
    full_file = OUT_FILE

    with open(train_file, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_file, "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(full_file, "w") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats by source
    from collections import Counter
    src_counts = Counter(r["source"].split("_")[0] for r in deduped)
    print(f"\nDataset splits:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"\nComposition:")
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src}: {cnt}")
    print(f"\nSaved to:")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Full:  {full_file}")
    print("Done!")
