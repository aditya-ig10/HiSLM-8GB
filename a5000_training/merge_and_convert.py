#!/usr/bin/env python3
"""
Merge LoRA adapter into base model, then convert to GGUF.

Usage:
  python merge_and_convert.py --help
  python merge_and_convert.py --lora ./output/lora_adapter_final --output ./output/tinyllama-medical-q4_k_m.gguf

Requirements on A5000:
  pip install torch transformers peft sentencepiece

  # Also need llama.cpp built:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && make -j

  # Or use the convert.py from llama.cpp directly.
"""
import argparse, os, subprocess, shutil, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def find_llama_cpp():
    """Try to find llama.cpp directory."""
    candidates = [
        Path.home() / "llama.cpp",
        Path.home() / "hislm_training" / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("/workspace/llama.cpp"),
    ]
    for c in candidates:
        if (c / "convert.py").exists():
            return c
    return None

def merge_lora(lora_path: Path, output_hf: Path):
    """Merge LoRA weights into base model and save as HF format."""
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_hf}")
    output_hf.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_hf, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(output_hf)
    print("Model saved (HF format).")
    return output_hf

def convert_to_gguf(hf_path: Path, output_gguf: Path, llama_cpp_dir: Path, quantize: str = "q4_k_m"):
    """Convert HF model to GGUF using llama.cpp convert.py, then quantize."""
    fp16_path = output_gguf.with_suffix(".fp16.gguf").name

    # Step 1: Convert to FP16 GGUF
    convert_py = llama_cpp_dir / "convert.py"
    if not convert_py.exists():
        print(f"ERROR: {convert_py} not found. Clone llama.cpp first.")
        sys.exit(1)

    cmd_convert = [
        sys.executable, str(convert_py),
        str(hf_path),
        "--outfile", str(output_gguf.parent / fp16_path),
        "--outtype", "f16",
    ]
    print(f"Running: {' '.join(cmd_convert)}")
    subprocess.run(cmd_convert, check=True)

    # Step 2: Quantize to Q4_K_M
    quantize_bin = llama_cpp_dir / "quantize"
    if not quantize_bin.exists():
        # Try building it
        print("quantize binary not found, building llama.cpp...")
        subprocess.run(["make", "-j", str(llama_cpp_dir)], check=True)

    if quantize_bin.exists():
        cmd_quant = [
            str(quantize_bin),
            str(output_gguf.parent / fp16_path),
            str(output_gguf),
            quantize,
        ]
        print(f"Running: {' '.join(cmd_quant)}")
        subprocess.run(cmd_quant, check=True)
        print(f"\nGGUF model saved: {output_gguf}")
    else:
        print(f"\nFP16 GGUF saved (no quantize binary): {output_gguf.parent / fp16_path}")
        print("Run quantize manually or copy the FP16 model.")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA + convert to GGUF")
    parser.add_argument("--lora", default=str(Path(__file__).resolve().parent / "output" / "lora_adapter_final"),
                        help="Path to LoRA adapter")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "output" / "tinyllama-medical-q4_k_m.gguf"),
                        help="Output GGUF path")
    parser.add_argument("--hf-output", default=str(Path(__file__).resolve().parent / "output" / "merged_hf"),
                        help="Temporary merged HF output")
    parser.add_argument("--quantize", default="q4_k_m", help="Quantization type")
    parser.add_argument("--no-gguf", action="store_true", help="Only merge, skip GGUF conversion")
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to llama.cpp directory")
    args = parser.parse_args()

    lora_path = Path(args.lora)
    if not lora_path.exists():
        print(f"LoRA adapter not found: {lora_path}")
        sys.exit(1)

    hf_output = Path(args.hf_output)
    gguf_output = Path(args.output)
    gguf_output.parent.mkdir(parents=True, exist_ok=True)

    merge_lora(lora_path, hf_output)

    if not args.no_gguf:
        llama_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else find_llama_cpp()
        if llama_dir and llama_dir.exists():
            convert_to_gguf(hf_output, gguf_output, llama_dir, args.quantize)
        else:
            print("\nllama.cpp not found. To convert to GGUF manually:")
            print(f"  1. Clone: git clone https://github.com/ggerganov/llama.cpp")
            print(f"  2. cd llama.cpp && pip install -r requirements.txt")
            print(f"  3. python convert.py {hf_output} --outfile model.fp16.gguf --outtype f16")
            print(f"  4. ./quantize model.fp16.gguf {gguf_output} q4_k_m")
    else:
        print(f"\nMerged model saved to {hf_output}")

    print("\nDone!")

if __name__ == "__main__":
    main()
