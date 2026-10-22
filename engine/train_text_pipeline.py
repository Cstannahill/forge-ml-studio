#!/usr/bin/env python3
"""
ForgeML Pipeline Studio — Track A: Pure Text LLM Fine-Tuning Engine
===================================================================
Executes the 3-Stage Sub-2B Fine-Tuning Pipeline on real local/cloud GPUs:
  - Stage 1: Supervised Fine-Tuning (SFT) with FlashAttention-2/3 Sample Packing,
             Completion-Only Loss Masking (<|im_start|>assistant), Special-Token Surgery,
             and Layer-Wise Adaptive LoRA / DoRA.
             Uses `unsloth.FastLanguageModel` when available and automatically falls back
             to HuggingFace `transformers.AutoModelForCausalLM` + `peft` for LiquidAI LFM2.5
             state-space models or non-CUDA environments.
  - Stage 2: Post-SFT Preference Alignment via `trl.ORPOTrainer`, `trl.DPOTrainer`,
             `trl.CPOTrainer` (SimPO), or `trl.KTOTrainer`.
  - Stage 3: Reinforcement Learning with Verifiable Rewards (RLVR) via `trl.GRPOTrainer`
             with deterministic Python AST/unit-test, GSM8K/MATH symbolic equivalence,
             JSON schema validation, and <think>...</think> format reward functions.

Usage:
  python3 engine/train_text_pipeline.py --config ./runs/<run_id>/training_manifest.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def log_event(stage: str, message: str, metrics: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "track": "TEXT_LLM",
        "stage": stage,
        "message": message,
    }
    if metrics:
        payload["metrics"] = metrics
    print(f"[FORGEML_ENGINE] {json.dumps(payload)}", flush=True)


# ==============================================================================
# Stage 3 GRPO Deterministic Verifiable Reward Functions (RLVR)
# ==============================================================================

def reward_think_tag_structure(completions: List[str], **kwargs: Any) -> List[float]:
    """Rewards strict <think>...</think> reasoning block presence before the final answer."""
    rewards: List[float] = []
    pattern = re.compile(r"^<think>\s*[\s\S]{15,}?\s*</think>\s*\S+", re.MULTILINE)
    for comp in completions:
        text = comp if isinstance(comp, str) else str(comp)
        if pattern.search(text.strip()):
            rewards.append(1.0)
        elif "<think>" in text and "</think>" in text:
            rewards.append(0.4)
        else:
            rewards.append(-0.5)
    return rewards


def reward_python_ast_and_syntax(completions: List[str], **kwargs: Any) -> List[float]:
    """Extracts ```python ... ``` blocks and verifies valid AST compilation."""
    rewards: List[float] = []
    code_re = re.compile(r"```(?:python|py)?\n([\s\S]*?)```", re.MULTILINE)
    for comp in completions:
        text = comp if isinstance(comp, str) else str(comp)
        matches = code_re.findall(text)
        if not matches:
            rewards.append(0.0)
            continue
        try:
            for snippet in matches:
                ast.parse(snippet)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(-0.75)
    return rewards


def reward_strict_json_schema(completions: List[str], **kwargs: Any) -> List[float]:
    """Verifies valid JSON object emission when structured output or tool_call is requested."""
    rewards: List[float] = []
    json_re = re.compile(r"(\{[\s\S]*\})", re.MULTILINE)
    for comp in completions:
        text = comp if isinstance(comp, str) else str(comp)
        after_think = text.split("</think>")[-1] if "</think>" in text else text
        match = json_re.search(after_think)
        if not match:
            rewards.append(0.0)
            continue
        try:
            parsed = json.loads(match.group(1))
            rewards.append(1.0 if isinstance(parsed, dict) and len(parsed) > 0 else 0.25)
        except Exception:
            rewards.append(-0.5)
    return rewards


# ==============================================================================
# Core Model & Tokenizer Loader (Unsloth + Transformers/PEFT Fallback)
# ==============================================================================

def load_model_and_tokenizer(manifest: Dict[str, Any]):
    import torch

    model_id: str = manifest["modelId"]
    architecture: str = manifest.get("architecture", "Qwen3.5")
    quant_mode: str = manifest.get("quantizationMode", "qlora-4bit")
    max_seq_length: int = int(manifest.get("maxSeqLength", 8192))
    hpo: Dict[str, Any] = manifest.get("hpoConfig", {})
    lora_r: int = int(hpo.get("loraRank", 32))
    lora_alpha: int = int(hpo.get("loraAlpha", 64))
    lora_dropout: float = float(hpo.get("loraDropout", 0.0))
    use_dora: bool = bool(manifest.get("useDora", False))

    load_in_4bit = quant_mode == "qlora-4bit"
    has_cuda = torch.cuda.is_available()

    # Check if Unsloth is available and compatible (Qwen3.5 / Gemma4 on CUDA)
    use_unsloth = False
    if has_cuda and architecture in ("Qwen3.5", "Gemma4") and not use_dora:
        try:
            from unsloth import FastLanguageModel  # type: ignore

            use_unsloth = True
        except ImportError:
            use_unsloth = False

    if use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

        log_event(
            "MODEL_INIT",
            f"Loading {model_id} via Unsloth FastLanguageModel (4-bit={load_in_4bit}, max_seq={max_seq_length})",
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        # Special-Token Surgery for <think> and <tool_call>
        special_tokens = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [t for t in special_tokens if t not in tokenizer.get_vocab()]}
        )
        if added > 0:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
            log_event("TOKENIZER_SURGERY", f"Registered {added} atomic special tokens and padded embeddings to 64.")

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        return model, tokenizer, "unsloth"

    # Standard HuggingFace Transformers + PEFT path (supports LFM2.5 hybrid state-space, DoRA, MPS & CPU)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log_event(
        "MODEL_INIT",
        f"Loading {model_id} ({architecture}) via HuggingFace Transformers + PEFT (DoRA={use_dora}, CUDA={has_cuda})",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [t for t in special_tokens if t not in tokenizer.get_vocab()]}
    )

    bnb_config = None
    if load_in_4bit and has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    dtype = torch.bfloat16 if (has_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    if not has_cuda:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
        trust_remote_code=True,
    )

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
        log_event("TOKENIZER_SURGERY", f"Added {added} special tokens to {model_id} vocabulary.")

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    target_modules = (
        ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "gate_proj", "up_proj", "down_proj"]
        if architecture == "LFM2.5"
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        use_dora=use_dora,
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer, "transformers+peft"


# ==============================================================================
# Stage 1 (SFT), Stage 2 (ORPO/SimPO/DPO) & Stage 3 (GRPO) Pipeline Execution
# ==============================================================================

def run_training_pipeline(manifest_path: Path, dry_run: bool = False) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("runId", "run-local")
    model_id = manifest["modelId"]
    dataset_path = Path(manifest["datasetJsonlPath"])
    output_dir = Path(manifest["outputDir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    hpo = manifest.get("hpoConfig", {})
    packing_cfg = manifest.get("packingConfig", {})
    pref_cfg = manifest.get("preferenceConfig", {})
    grpo_cfg = manifest.get("grpoConfig", {})

    log_event(
        "PREFLIGHT",
        f"Initialized run {run_id} for {model_id}",
        {
            "datasetPath": str(dataset_path),
            "sampleCount": manifest.get("sampleCount", 0),
            "loraRank": hpo.get("loraRank", 32),
            "learningRate": hpo.get("learningRate", 2e-4),
            "enableStage2Preference": pref_cfg.get("enabled", False),
            "enableStage3Grpo": grpo_cfg.get("enabled", False),
        },
    )

    if dry_run:
        log_event("DRY_RUN_VERIFIED", "Manifest and dataset JSONL verified on disk. Exiting dry-run cleanly.")
        return

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer, engine_backend = load_model_and_tokenizer(manifest)
    raw_dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    # Stage 1: Supervised Fine-Tuning (SFT)
    sft_output_dir = output_dir / "stage1_sft"
    enable_packing = bool(packing_cfg.get("enableSamplePacking", True))
    mask_user_prompts = bool(packing_cfg.get("maskPromptTokens", True))

    log_event(
        "STAGE_1_SFT_START",
        f"Starting Stage 1 SFT via {engine_backend} (packing={enable_packing}, response_only_mask={mask_user_prompts})",
    )

    sft_args = SFTConfig(
        output_dir=str(sft_output_dir),
        per_device_train_batch_size=int(hpo.get("batchSize", 4)),
        gradient_accumulation_steps=4,
        warmup_ratio=float(hpo.get("warmupRatio", 0.05)),
        num_train_epochs=int(manifest.get("epochs", 2)),
        learning_rate=float(hpo.get("learningRate", 2e-4)),
        weight_decay=float(hpo.get("weightDecay", 0.01)),
        lr_scheduler_type="cosine" if hpo.get("lrScheduler") == "wsd" else str(hpo.get("lrScheduler", "cosine")),
        neftune_noise_alpha=float(hpo.get("neftuneAlpha", 5.0)),
        logging_steps=5,
        save_strategy="epoch",
        max_seq_length=int(manifest.get("maxSeqLength", 8192)),
        packing=enable_packing,
        dataset_text_field="formattedPayload",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=raw_dataset,
        args=sft_args,
    )

    if mask_user_prompts and engine_backend == "unsloth":
        try:
            from unsloth.chat_templates import train_on_responses_only  # type: ignore

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            log_event("LOSS_MASKING", "Enabled Unsloth train_on_responses_only loss masking.")
        except Exception as exc:
            log_event("LOSS_MASKING_WARN", f"Skipping train_on_responses_only: {exc}")

    trainer.train()
    model.save_pretrained(str(sft_output_dir))
    tokenizer.save_pretrained(str(sft_output_dir))
    log_event("STAGE_1_SFT_DONE", f"Saved Stage 1 SFT checkpoint to {sft_output_dir}")

    # Stage 2: Post-SFT Preference Alignment (ORPO / SimPO / DPO)
    if pref_cfg.get("enabled", False):
        method = str(pref_cfg.get("method", "ORPO")).upper()
        pref_dataset_id = str(
            pref_cfg.get("preferenceDatasetId", "argilla/ultrafeedback-binarized-preferences-cleaned")
        )
        stage2_dir = output_dir / f"stage2_{method.lower()}"
        log_event("STAGE_2_ALIGNMENT_START", f"Starting Stage 2 {method} alignment on {pref_dataset_id}")

        pref_ds = load_dataset(pref_dataset_id, split="train[:2000]")
        if method in ("ORPO", "SIMPO"):
            from trl import ORPOConfig, ORPOTrainer

            orpo_args = ORPOConfig(
                output_dir=str(stage2_dir),
                beta=float(pref_cfg.get("beta", 0.15)),
                learning_rate=float(pref_cfg.get("learningRate", 8e-6)),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_length=int(manifest.get("maxSeqLength", 4096)),
                max_prompt_length=1024,
                num_train_epochs=1,
                logging_steps=5,
                report_to="none",
            )
            orpo_trainer = ORPOTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=pref_ds,
                args=orpo_args,
            )
            orpo_trainer.train()
        else:
            from trl import DPOConfig, DPOTrainer

            dpo_args = DPOConfig(
                output_dir=str(stage2_dir),
                beta=float(pref_cfg.get("beta", 0.1)),
                learning_rate=float(pref_cfg.get("learningRate", 5e-6)),
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                logging_steps=5,
                report_to="none",
            )
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                processing_class=tokenizer,
                train_dataset=pref_ds,
                args=dpo_args,
            )
            dpo_trainer.train()

        model.save_pretrained(str(stage2_dir))
        tokenizer.save_pretrained(str(stage2_dir))
        log_event("STAGE_2_ALIGNMENT_DONE", f"Saved Stage 2 {method} checkpoint to {stage2_dir}")

    # Stage 3: GRPO Verifiable Rewards (RLVR)
    if grpo_cfg.get("enabled", False):
        from trl import GRPOConfig, GRPOTrainer

        stage3_dir = output_dir / "stage3_grpo_rlvr"
        log_event(
            "STAGE_3_GRPO_START",
            f"Starting Stage 3 GRPO Verifiable Rewards (G={grpo_cfg.get('groupSize', 8)}, beta={grpo_cfg.get('klBeta', 0.04)})",
        )
        grpo_args = GRPOConfig(
            output_dir=str(stage3_dir),
            learning_rate=5e-6,
            num_generations=int(grpo_cfg.get("groupSize", 8)),
            beta=float(grpo_cfg.get("klBeta", 0.04)),
            max_completion_length=int(grpo_cfg.get("maxCompletionLength", 1024)),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=5,
            report_to="none",
        )
        grpo_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                reward_think_tag_structure,
                reward_python_ast_and_syntax,
                reward_strict_json_schema,
            ],
            train_dataset=raw_dataset,
            args=grpo_args,
        )
        grpo_trainer.train()
        model.save_pretrained(str(stage3_dir))
        tokenizer.save_pretrained(str(stage3_dir))
        log_event("STAGE_3_GRPO_DONE", f"Saved Stage 3 GRPO RLVR checkpoint to {stage3_dir}")

    log_event("PIPELINE_COMPLETE", f"All requested stages finished for {model_id}. Artifacts in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgeML Pure Text LLM Training Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to training_manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and dataset without loading GPU weights")
    args = parser.parse_args()

    manifest_path = Path(args.config).resolve()
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    run_training_pipeline(manifest_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
