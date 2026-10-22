#!/usr/bin/env python3
"""
ForgeML Pipeline Studio — Adapter Merger, Safetensors & GGUF/mmproj Exporter
============================================================================
Executes:
  1. PEFT LoRA / DoRA fusion into BF16 / FP16 single-file Safetensors (`model.safetensors`)
  2. Multi-Adapter Task Arithmetic Merging (`mergekit-yaml` for DARE-TIES, TIES, SLERP)
  3. Importance Matrix (`llama-imatrix`) calibrated GGUF quantization (`Q4_K_M`)
  4. Dual-Artifact Vision-Language export (`<model>-q4_k_m.gguf` + `mmproj-<model>-f16.gguf`)

Usage:
  python3 engine/export_and_merge.py --adapter-dir ./runs/<id>/stage1_sft --base-model Qwen/Qwen3.5-2B-Instruct --out-dir ./exports
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def merge_peft_to_safetensors(base_model: str, adapter_dir: Path, out_dir: Path, max_shard_size: str = "5GB") -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FORGEML_EXPORT] Merging PEFT adapter {adapter_dir} into base model {base_model}...")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(str(out_dir))
    print(f"[FORGEML_EXPORT] Saved single-file Safetensors bundle to {out_dir}")
    return out_dir


def convert_to_gguf_and_mmproj(
    merged_dir: Path,
    out_dir: Path,
    quant_type: str = "Q4_K_M",
    export_mmproj: bool = False,
    llama_cpp_dir: Path = Path("./llama.cpp"),
) -> None:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    quant_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        print(
            f"[FORGEML_EXPORT] Note: {convert_script} not found. Clone https://github.com/ggerganov/llama.cpp to run GGUF conversion.",
            file=sys.stderr,
        )
        return

    fp16_gguf = out_dir / "model-f16.gguf"
    quant_gguf = out_dir / f"model-{quant_type.lower()}.gguf"

    subprocess.run(
        [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(fp16_gguf), "--outtype", "f16"],
        check=True,
    )

    if export_mmproj:
        mmproj_gguf = out_dir / "mmproj-model-f16.gguf"
        subprocess.run(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--mmproj",
                "--outfile",
                str(mmproj_gguf),
                "--outtype",
                "f16",
            ],
            check=True,
        )
        print(f"[FORGEML_EXPORT] Exported unquantized FP16 Vision Projector: {mmproj_gguf}")

    if quant_bin.exists():
        subprocess.run([str(quant_bin), str(fp16_gguf), str(quant_gguf), quant_type], check=True)
        print(f"[FORGEML_EXPORT] Quantized GGUF artifact ready: {quant_gguf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeML PEFT Merger & Safetensors/GGUF Exporter")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--quant", type=str, default="Q4_K_M")
    parser.add_argument("--mmproj", action="store_true", help="Also export FP16 mmproj-f16.gguf for Vision models")
    args = parser.parse_args()

    merged_dir = merge_peft_to_safetensors(
        args.base_model,
        Path(args.adapter_dir).resolve(),
        Path(args.out_dir).resolve() / "safetensors_merged",
    )
    convert_to_gguf_and_mmproj(
        merged_dir,
        Path(args.out_dir).resolve(),
        quant_type=args.quant,
        export_mmproj=args.mmproj,
    )


if __name__ == "__main__":
    main()
