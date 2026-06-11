#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
DATE=$(date +%Y%m%d_%H%M%S)
LOG="$DIR/output/train_$DATE.log"

mkdir -p "$DIR/output"

echo "=========================================="
echo " HiSLM A5000 Training — $DATE"
echo "=========================================="
echo ""

# Check for data
if [ ! -f "$DIR/data/train.jsonl" ]; then
    echo "ERROR: data/train.jsonl not found."
    echo "Copy dataset from NX:"
    echo "  scp <nx>:~/llama/HiSLM-8G/dataset/train.jsonl ./data/"
    echo "  scp <nx>:~/llama/HiSLM-8G/dataset/val.jsonl ./data/"
    exit 1
fi

echo "1. Training..."
python "$DIR/train_a5000.py" 2>&1 | tee "$LOG"

echo ""
echo "2. Merging + converting to GGUF..."
python "$DIR/merge_and_convert.py" --lora "$DIR/output/lora_adapter_final" \
    --output "$DIR/output/tinyllama-medical-q4_k_m.gguf" 2>&1 | tee -a "$LOG"

echo ""
echo "=========================================="
echo " Done! GGUF model: output/tinyllama-medical-q4_k_m.gguf"
echo "=========================================="
