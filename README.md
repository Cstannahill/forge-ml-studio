# ForgeML Pipeline Studio

**Multi-Model Sub-2B Fine-Tuning, Vision-Language (VLM) Grounding, VEIL Privacy Hazard Auditing & Agentic Orchestration Workbench**

ForgeML Pipeline Studio is an end-to-end engineering workbench for curating datasets, fine-tuning, aligning, evaluating, merging, quantizing, and deploying compact **0.3B–4B parameter models** across two strictly isolated modality tracks:

- **Track A — Pure Text Generation Pipeline (`TEXT_LLM`)**: Specialized for **LiquidAI LFM2.5** (`300M`, `1.2B`, `2.6B`), **Alibaba Qwen3.5** (`0.8B`, `2B`), and **Google Gemma 4** (`E2B`, `E4B`).
- **Track B — Isolated Vision-Language Pipeline (`VISION_VLM`)**: Specialized for **LiquidAI LFM2.5-VL** (`450M`, `1.6B`), **Alibaba Qwen3.5-VL** (`0.8B`, `2B`), and **Google Gemma 4 Vision** (`E2B-Vision`), featuring dynamic NaViT/SigLIP patch budgeting, `[0, 1000]` bounding-box grounding, and the **VEIL 5-Category Privacy Hazard Auditing Taxonomy**.

---

## Documentation Index

| Document | Purpose |
| :--- | :--- |
| **[`COMMANDS.md`](./COMMANDS.md)** | Complete CLI command reference: starting the server, running unit tests, `curl` API endpoints, Unsloth/Axolotl/TRL training commands, `mergekit`, `llama-imatrix`, `GGUF + mmproj` export, Ollama/vLLM/`llama-server`, and Docker commands |
| **[`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md)** | System architecture, dual-track modality isolation, 3-stage training pipeline (SFT → ORPO/SimPO → GRPO), state persistence, and backend agentic fallback design |
| **[`docs/VEIL_VISION_PIPELINE.md`](./docs/VEIL_VISION_PIPELINE.md)** | Deep-dive specification for the **VEIL Vision Privacy Hazard Auditing Taxonomy**, 10k–15k multi-turn split strategy, all **25 VLM dataset blends**, and the **Vision Tokenizer & Patch Density** profiler |

---

## Supported Sub-2B & Edge Model Registry

### Track A: Pure Text Generation Models (`TEXT_LLM`)
1. **`LiquidAI/LFM2.5-300M-Instruct`** (`0.3B` · Hybrid Liquid State-Space · 32k context)
2. **`LiquidAI/LFM2.5-1.2B-Instruct`** (`1.2B` · Hybrid Liquid State-Space · 64k context)
3. **`LiquidAI/LFM2.5-2.6B-Instruct`** (`2.6B` · Hybrid Liquid State-Space · 64k context)
4. **`Qwen/Qwen3.5-0.8B-Instruct`** (`0.8B` · Dense GQA + Dynamic NTK RoPE · 32k context)
5. **`Qwen/Qwen3.5-2B-Instruct`** (`2.0B` · Dense GQA + Dynamic NTK RoPE · 32k context)
6. **`google/gemma-4-E2B-it`** (`2.0B` · Interleaved Local/Global Attention · 64k context)
7. **`google/gemma-4-E4B-it`** (`4.0B` · Interleaved Local/Global Attention · 128k context)

### Track B: Isolated Vision-Language Models (`VISION_VLM`)
1. **`LiquidAI/LFM2.5-VL-450M-Instruct`** (`0.45B` total · `150M` SigLIP-Dynamic Tower + `0.3B` LFM Backbone)
2. **`LiquidAI/LFM2.5-VL-1.6B-Instruct`** (`1.6B` total · `400M` SigLIP-384 Tower + `1.2B` LFM Backbone)
3. **`Qwen/Qwen3.5-VL-0.8B-Instruct`** (`0.85B` total · `220M` NaViT 2D-RoPE Tower + `0.63B` Qwen Backbone)
4. **`Qwen/Qwen3.5-VL-2B-Instruct`** (`2.2B` total · `420M` NaViT M-RoPE Tower + `1.78B` Qwen Backbone)
5. **`google/gemma-4-e2b-vision-it`** (`2.4B` total · `400M` SigLIP Pan&Scan Tower + `2.0B` Gemma4 Backbone)

---

## Core Capabilities Across the 5 Workspace Studios

### 1. Command Center (`CommandCenterView` & `VisionCommandCenterPanel`)
- **Stage 1 SFT + Sequence Packing & Completion-Only Loss Masking**: FlashAttention-2/3 Multipack (`packing=True`) and assistant-only cross-entropy loss masking (`<|im_start|>assistant`), eliminating `<pad>` waste and improving sub-2B accuracy by +2–4%.
- **Layer-Wise Adaptive LoRA / DoRA**: Independent rank allocation across Attention (`q/k/v/o`), MLP Gate/Up/Down (`gate/up/down_proj`), and Embeddings (`embed_tokens`/`lm_head`) with DoRA magnitude/direction decomposition.
- **Stage 2 Post-SFT Preference Alignment**: Reference-free **ORPO** and **SimPO** (saving ~1.45 GB VRAM by avoiding a separate reference model) alongside **DPO** and **KTO**.
- **Stage 3 GRPO / RLVR (Reinforcement Learning with Verifiable Rewards)**: DeepSeek-R1 style group rollouts (`G=4..16`) scored by deterministic Python unit-test, symbolic math, and JSON schema verifiers.
- **Multi-Adapter Task Arithmetic Merger (`mergekit`)**: Combine 2–4 domain LoRA adapters via **DARE-TIES**, **TIES**, **SLERP**, or **Linear** interpolation.
- **Early-Stopping & Overfitting Watchdog**: Real-time validation-to-training divergence monitoring with automatic best-checkpoint locking.
- **Importance Matrix (`imatrix`) Calibration & Multi-Format Export**: Export single-file **Safetensors**, **`GGUF-Q4_K_M`** (plus separate **`mmproj-f16.gguf`** for Vision models), **GPTQ**, **ONNX**, and 1-click executable **Unsloth / Axolotl / TRL** script bundles.

### 2. Dataset Studio (`DatasetStudioView` & `VisionDatasetStudioView`)
- **24 Pure-Text Blends & 25 Vision-Language Blends**: Task-specific HuggingFace dataset blends weighted for sub-2B architectures.
- **VEIL 5-Category Vision Privacy Hazard Auditing Suite**: Complete taxonomy and 10,000–15,000 multi-turn split strategy across Residential/Geolocation Mail (25%), Medical/Travel/Financial PII (20%), Specular Mirror Crew Reflections (25%), Biometric Tattoos/Scars with `BLUR`/`MOSAIC` routing (10%), and Clean Studio Baselines with `<2%` false-positive rate (20%).
- **Vision Tokenizer & Patch Density Chart**: Profiles `visualPatchTokens : textTokens` ratios across the active corpus and caps excessive NaViT patch outliers with 1 click.
- **On-Policy Teacher Distillation & LLM-as-a-Judge Filtering**: Synthesizes verified `<think>...</think>` traces (and `[0, 1000]` grounded bounding boxes for VLM) from frontier teachers.
- **IFD / V-IFD Pruning & Benchmark Decontamination Gate**: Drops noisy/broken outliers and scans for 13-gram or perceptual hash (`pHash`) overlap against MMLU-Pro, GSM8K, HumanEval+, MMMU, and DocVQA.
- **3-Phase Curriculum Pacing & Special-Token Surgery**: Orders training samples from foundational alignment to complex reasoning and verifies atomic `<think>`, `<tool_call>`, `<|vision_start|>`, `<|image_pad|>`, and `<box>` token IDs.
- **Interactive `[0, 1000]` Bounding-Box Canvas & Immutable DVC Commits**: Live SVG bounding-box inspection, in-place sample editing, custom `.jsonl`/`.csv` upload normalizer, and SHA-256 dataset versioning.

### 3. Model Benchmarks & A/B Evaluation Arena (`ModelBenchmarksView`)
- **Base vs. Fine-Tuned Summary Table**: Highlights exact deltas across **MMLU-Pro**, **GSM8K**, **HumanEval**, **IFEval**, **MT-Bench**, **Perplexity**, and **Throughput (`tok/s`)**.
- **5-Axis Radar Chart & Plain-Language AI Explainer**: Translates benchmark tables into everyday strengths, trade-offs, and deployment verdicts.
- **Side-by-Side Interactive A/B Arena ("Vibe Check" Sandbox)**: Compare Base vs. Fine-Tuned responses with token-by-token diffing and 1-click **"Flag Bad Response → Send to Dataset Studio for Correction"** active learning loop.
- **Automated Red-Team & Safety Stress-Test Suite**: Evaluates Prompt Injection Resistance, JSON Schema Compliance, Context Needle-in-a-Haystack Recall, and Hallucination Refusal.
- **Experiment Run Diffing & Standardized HuggingFace Model Card Generator**: Compare any two historical runs and export spec-compliant `README.md` model cards with YAML frontmatter.

### 4. Agentic AI Orchestrator (`AgenticOrchestratorView`)
- Supports **Google Gemini** (`@google/genai`), **OpenRouter**, **Local `llama-server`**, and **Local Ollama** across four autonomous roles: Goal-to-Dataset Architect, Data Hygiene & PII Scrubber, Closed-Loop HPO Refiner, and Executive Benchmark Reporter.

### 5. Cloud & Docker Deployment (`CloudAndContainerView`)
- **Hardware & Edge Deployment Target Simulator**: Calculates training VRAM fit (`RTX 3060 12GB` through `A100 80GB`) and edge inference `tok/s` (`MacBook Air M2`, `iPhone 16 Pro ANE`, `Raspberry Pi 5`, `Jetson Orin Nano`).
- **Local Inference Server & Client SDK Generator**: Ready-to-run `ollama`, `llama-server`, `vllm`, Python `openai` SDK, TypeScript `fetch`, and `curl` snippets.

---

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Start the full-stack server on http://localhost:3000
npm run dev

# 3. Run the 33 unit tests
npm test

# 4. Run TypeScript validation
npm run lint
```

See **[`COMMANDS.md`](./COMMANDS.md)** for the full list of CLI commands.
