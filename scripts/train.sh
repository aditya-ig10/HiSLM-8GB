#!/bin/bash
# Fine-tune TinyLlama-1.1B on medical datasets using QLoRA (manual loop)
source /home/nvidia/llama/HiSLM-8G/venv/bin/activate
export LD_LIBRARY_PATH="/home/nvidia/llama/HiSLM-8G/venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:/home/nvidia/llama/HiSLM-8G/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

python3 /home/nvidia/llama/HiSLM-8G/train.py "$@"
EXIT_CODE=$?
echo "[train.sh] $(date): Python exited with code $EXIT_CODE" >> /home/nvidia/llama/HiSLM-8G/output/train_exit.log
if [ $EXIT_CODE -ne 0 ]; then
    echo "[train.sh] Exit code $EXIT_CODE indicates failure (signal: $((EXIT_CODE-128)) if >128)" >> /home/nvidia/llama/HiSLM-8G/output/train_exit.log
fi
exit $EXIT_CODE
