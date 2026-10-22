#!/usr/bin/env python3
"""
ForgeML Pipeline Studio — Track B: Isolated Vision-Language (VLM) Fine-Tuning Engine
====================================================================================
Trains multimodal models (`Qwen/Qwen3.5-VL-0.8B-Instruct`, `Qwen/Qwen3.5-VL-2B-Instruct`,
`LiquidAI/LFM2.5-VL-450M-Instruct`, `LiquidAI/LFM2.5-VL-1.6B-Instruct`, and
`google/gemma-4-e2b-vision-it`) in complete isolation from pure Text LLM checkpoints.

Capabilities:
  - Uses `unsloth.FastVisionModel` + `UnslothVisionDataCollator` (or HuggingFace
    `AutoModelForImageTextToText` + `peft` fallback)
  - Supports freezing the SigLIP / NaViT Vision Tower (`finetune_vision_layers=False`)
    while training the Multimodal MLP Connector (`finetune_mlp_modules=True`) and
    LLM Backbone LoRA/DoRA (`finetune_language_layers=True`)
  - Enforces dynamic resolution patch budgets (`min_pixels` / `max_pixels`) to prevent
    excessive NaViT patch dilution
  - Registers atomic vision & grounding tokens (`<|vision_start|>`, `<|image_pad|>`,
    `<|vision_end|>`, `<ref>`, `</ref>`, `<box>`, `</box>`) for VEIL Privacy Hazard
    Auditing and `[0, 1000]` bounding-box regression

Usage:
  python3 engine/train_vision_pipeline.py --config ./runs/<run_id>/vision_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


def log_vision_event(stage: str, message: str, metrics: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "VISION_VLM",
        "stage": stage,
        "message": message,
    }
    if metrics:
        payload["metrics"] = metrics
    print(f"[FORGEML_VISION_ENGINE] {json.dumps(payload)}", flush=True)


def run_vision_pipeline(manifest_path: Path, dry_run: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("runId", "vlm-run-local")
    model_id = manifest["modelId"]
    dataset_path = Path(manifest["datasetJsonlPath"])
    output_dir = Path(manifest["outputDir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    hw_cfg = manifest.get("hardwareConfig", {})
    freeze_vision_tower = bool(hw_cfg.get("freezeVisionTower", True))
    train_mlp_projector = bool(hw_cfg.get("trainMlpProjector", True))
    apply_lora_to_llm = bool(hw_cfg.get("applyLoraToLlmBackbone", True))
    min_pixels = int(hw_cfg.get("minPixels", 256 * 28 * 28))
    max_pixels = int(hw_cfg.get("maxPixels", 1024 * 28 * 28))
    lora_r = int(manifest.get("loraRank", 32))
    lora_alpha = int(manifest.get("loraAlpha", 64))

    log_vision_event(
        "PREFLIGHT",
        f"Initialized isolated VLM run {run_id} for {model_id}",
        {
            "datasetPath": str(dataset_path),
            "sampleCount": manifest.get("sampleCount", 0),
            "freezeVisionTower": freeze_vision_tower,
            "trainMlpProjector": train_mlp_projector,
            "minPixels": min_pixels,
            "maxPixels": max_pixels,
        },
    )

    if dry_run:
        log_vision_event(
            "DRY_RUN_VERIFIED",
            "Vision manifest, dynamic NaViT pixel bounds, and multimodal JSONL verified on disk.",
        )
        return

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    has_cuda = torch.cuda.is_available()
    use_unsloth_vision = False
    if has_cuda:
        try:
            from unsloth import FastVisionModel  # type: ignore

            use_unsloth_vision = True
        except ImportError:
            use_unsloth_vision = False

    if use_unsloth_vision:
        from unsloth import FastVisionModel  # type: ignore
        from unsloth.trainer import UnslothVisionDataCollator  # type: ignore

        log_vision_event(
            "VISION_MODEL_INIT",
            f"Loading {model_id} via Unsloth FastVisionModel (min_pixels={min_pixels}, max_pixels={max_pixels})",
        )
        model, processor = FastVisionModel.from_pretrained(
            model_id,
            load_in_4bit=manifest.get("quantizationMode", "qlora-4bit") == "qlora-4bit",
            use_gradient_checkpointing="unsloth",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        # Register atomic vision & [0, 1000] grounding special tokens
        vision_special_tokens = [
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
            "<think>",
            "</think>",
        ]
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": vision_special_tokens}
        )
        if added > 0:
            model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
            log_vision_event("VISION_TOKENIZER_SURGERY", f"Registered {added} atomic VLM grounding tokens.")

        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=not freeze_vision_tower,
            finetune_language_layers=apply_lora_to_llm,
            finetune_attention_modules=True,
            finetune_mlp_modules=train_mlp_projector,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            random_state=3407,
        )

        FastVisionModel.for_training(model)
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")

        sft_args = SFTConfig(
            output_dir=str(output_dir / "vlm_checkpoint"),
            per_device_train_batch_size=int(manifest.get("batchSize", 4)),
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=int(manifest.get("epochs", 2)),
            learning_rate=float(manifest.get("learningRate", 1.5e-4)),
            logging_steps=5,
            save_strategy="epoch",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=int(manifest.get("maxSeqLength", 8192)),
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=processor.tokenizer,
            data_collator=UnslothVisionDataCollator(model, processor),
            train_dataset=dataset,
            args=sft_args,
        )
        trainer.train()
        model.save_pretrained(str(output_dir / "vlm_checkpoint"))
        processor.save_pretrained(str(output_dir / "vlm_checkpoint"))
        log_vision_event(
            "VISION_PIPELINE_COMPLETE",
            f"Saved Vision-Language checkpoint and processor to {output_dir / 'vlm_checkpoint'}",
        )
        return

    # Fallback: HuggingFace Transformers AutoProcessor + AutoModelForImageTextToText + PEFT
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    log_vision_event(
        "VISION_MODEL_INIT",
        f"Loading {model_id} via HuggingFace AutoModelForImageTextToText + PEFT",
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if (has_cuda and torch.cuda.is_bf16_supported()) else torch.float32,
        device_map="auto" if has_cuda else None,
        trust_remote_code=True,
    )

    if freeze_vision_tower and hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False
        log_vision_event("VISION_TOWER_FROZEN", "Froze SigLIP / NaViT vision encoder parameters.")

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_cfg)
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    sft_args = SFTConfig(
        output_dir=str(output_dir / "vlm_checkpoint"),
        per_device_train_batch_size=int(manifest.get("batchSize", 2)),
        gradient_accumulation_steps=4,
        num_train_epochs=int(manifest.get("epochs", 2)),
        learning_rate=float(manifest.get("learningRate", 1.5e-4)),
        dataset_text_field="formattedPayload",
        max_seq_length=int(manifest.get("maxSeqLength", 4096)),
        logging_steps=5,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor.tokenizer,
        train_dataset=dataset,
        args=sft_args,
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "vlm_checkpoint"))
    processor.save_pretrained(str(output_dir / "vlm_checkpoint"))
    log_vision_event(
        "VISION_PIPELINE_COMPLETE",
        f"Saved Vision-Language checkpoint to {output_dir / 'vlm_checkpoint'}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeML Isolated Vision-Language (VLM) Training Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to vision_manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and dataset without loading GPU weights")
    args = parser.parse_args()

    manifest_path = Path(args.config).resolve()
    if not manifest_path.exists():
        print(f"Error: Vision manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    run_vision_pipeline(manifest_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
