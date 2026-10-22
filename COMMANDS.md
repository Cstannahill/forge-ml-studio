# ForgeML Pipeline Studio — Complete CLI Commands Reference (`COMMANDS.md`)

This document provides the complete command-line reference for starting, building, testing, and operating **ForgeML Pipeline Studio**, as well as executing the generated fine-tuning scripts, quantized exporters, local inference servers, and container stacks for both **Track A (Pure Text LLM)** and **Track B (Isolated Vision-Language VLM)**.

---

## Table of Contents

1. [Quick Start & Application Lifecycle Commands](#1-quick-start--application-lifecycle-commands)
2. [Type-Checking, Linting & Unit Test Commands](#2-type-checking-linting--unit-test-commands)
3. [Backend REST API CLI Commands (`curl`)](#3-backend-rest-api-cli-commands-curl)
4. [Track A — Pure Text LLM Fine-Tuning CLI Commands](#4-track-a--pure-text-llm-fine-tuning-cli-commands)
5. [Track B — Vision-Language (VLM) & VEIL Audit CLI Commands](#5-track-b--vision-language-vlm--veil-audit-cli-commands)
6. [Adapter Merging, `imatrix` Calibration & Quantization Commands](#6-adapter-merging-imatrix-calibration--quantization-commands)
7. [Local Inference Server Commands (`Ollama`, `llama-server`, `vLLM`)](#7-local-inference-server-commands-ollama-llama-server-vllm)
8. [Docker & Multi-Container GPU Orchestration Commands](#8-docker--multi-container-gpu-orchestration-commands)
9. [Headless Browser & UI Verification Commands (`playwright-cli`)](#9-headless-browser--ui-verification-commands-playwright-cli)
10. [In-App Keyboard Shortcuts](#10-in-app-keyboard-shortcuts)

---

## 1. Quick Start & Application Lifecycle Commands

ForgeML Pipeline Studio runs as a unified Express + Vite full-stack TypeScript server on **Port `3000`**.

### Install Dependencies
```bash
npm install
```

### Environment Setup
Copy `.env.example` to `.env` and configure any optional provider keys or local inference URLs:
```bash
cp .env.example .env
```

Supported environment variables in `.env`:
```bash
# Google Gemini API key for server-side @google/genai agent & benchmark explainer routes
GEMINI_API_KEY="your-gemini-api-key"

# Optional OpenRouter key for frontier teacher distillation & agent orchestration
OPENROUTER_API_KEY="your-openrouter-api-key"

# Local inference endpoints for llama.cpp and Ollama
LLAMA_SERVER_URL="http://localhost:8080/v1"
OLLAMA_HOST="http://localhost:11434"

# Public application URL
APP_URL="http://localhost:3000"
```

### Start Development Server (Port 3000)
Starts Express (`server.ts`) via `tsx` with Vite middleware mounted in SPA mode:
```bash
npm run dev
```

### Build Production Frontend Bundle
Compiles TypeScript and bundles the production React SPA into `./dist`:
```bash
npm run build
```

### Start Production Server
Runs `server.ts` serving compiled static assets from `./dist` alongside `/api/*` routes on Port `3000`:
```bash
NODE_ENV=production npm run start
```

### Preview Static Vite Build
```bash
npm run preview
```

### Clean Build Artifacts
Removes `./dist` and any compiled server artifacts:
```bash
npm run clean
```

---

## 2. Type-Checking, Linting & Unit Test Commands

All pure domain logic in `/src/lib/pipelineLogic.ts` and `/src/lib/visionPipelineLogic.ts` is decoupled from React and tested via Node's native TypeScript test runner (`node:test`).

### Run Full TypeScript Type-Check (`lint`)
```bash
npm run lint
# Equivalent direct command:
npx tsc --noEmit
```

### Run Complete Unit Test Suite (33 Subtests)
```bash
npm test
# Equivalent direct command:
node --test src/lib/pipelineLogic.test.ts
```

### Run Individual / Filtered Unit Tests by Feature
Use `--test-name-pattern` to run specific domain test suites in isolation:

```bash
# 1. Test VEIL 5-Category Privacy Hazard Auditing Taxonomy & Seed Pack
node --test --test-name-pattern="VEIL_AUDIT_TAXONOMY" src/lib/pipelineLogic.test.ts

# 2. Test Vision Tokenizer & Patch Density Profiler + NaViT Outlier Capping
node --test --test-name-pattern="computeVisionTokenizerAndPatchDensity" src/lib/pipelineLogic.test.ts

# 3. Test 25 Curated Vision-Language Dataset Blends & 5 VLM Models
node --test --test-name-pattern="CURATED_VISION_BLENDS" src/lib/pipelineLogic.test.ts

# 4. Test Isolated Vision-Language VRAM & Dual-Artifact (mmproj-f16.gguf) Planner
node --test --test-name-pattern="computeVisionPipelineHardwarePlan" src/lib/pipelineLogic.test.ts

# 5. Test Vision Dataset Hygiene, V-IFD Alignment Pruning, Curriculum & Tokenizer Surgery
node --test --test-name-pattern="Vision Dataset Studio functions" src/lib/pipelineLogic.test.ts

# 6. Test 24 Curated Pure-Text Dataset Blends
node --test --test-name-pattern="CURATED_DATASET_BLENDS" src/lib/pipelineLogic.test.ts

# 7. Test Stage 2 Preference Alignment (ORPO / SimPO / DPO / KTO) VRAM Savings
node --test --test-name-pattern="computePreferenceAlignmentPlan" src/lib/pipelineLogic.test.ts

# 8. Test Stage 3 GRPO / RLVR Verifiable Reward Planner
node --test --test-name-pattern="computeGrpoVerifiableRewardPlan" src/lib/pipelineLogic.test.ts

# 9. Test Multi-Adapter Task Arithmetic Merging (DARE-TIES / TIES / SLERP)
node --test --test-name-pattern="computeMultiAdapterMergePlan" src/lib/pipelineLogic.test.ts

# 10. Test Importance Matrix (imatrix) Quantization Calibration
node --test --test-name-pattern="computeImatrixQuantizationCalibration" src/lib/pipelineLogic.test.ts

# 11. Test Safetensors, GGUF-Q4_K_M, GPTQ & ONNX Export Plans
node --test --test-name-pattern="buildSafetensorsExportPlan" src/lib/pipelineLogic.test.ts

# 12. Test Training Series LocalStorage Persistence Serialization
node --test --test-name-pattern="serializeTrainingSeriesMap" src/lib/pipelineLogic.test.ts
```

---

## 3. Backend REST API CLI Commands (`curl`)

While the server is running on `http://localhost:3000`, you can invoke all backend endpoints directly from the terminal.

### 3.1 Search HuggingFace Hub Datasets (`GET /api/hf/search`)
```bash
curl -s "http://localhost:3000/api/hf/search?q=qwen3+code" | jq .
```

### 3.2 Ingest & Normalize HuggingFace Dataset Rows (`POST /api/hf/ingest`)
```bash
curl -s -X POST http://localhost:3000/api/hf/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "datasetIds": [
      "Qwen/Qwen3-Code-Reasoning-110k",
      "HuggingFaceH4/ultrachat_200k"
    ]
  }' | jq .
```

### 3.3 Execute Autonomous Agentic Workflows (`POST /api/agent/execute`)

#### Role 1: Autonomous Goal-to-Pipeline Architect (`goal_to_pipeline`)
```bash
curl -s -X POST http://localhost:3000/api/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "modelName": "gemini-3.8-flash",
    "agentRole": "goal_to_pipeline",
    "payload": {
      "goal": "Fine-tune Qwen3.5-2B for strict TypeScript + SQL tool calling and JSON schema extraction",
      "modelId": "Qwen/Qwen3.5-2B-Instruct"
    }
  }' | jq .
```

#### Role 2: Agentic Data Hygiene & PII Scrubber (`clean_and_tag`)
```bash
curl -s -X POST http://localhost:3000/api/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "modelName": "gemini-3.8-flash",
    "agentRole": "clean_and_tag",
    "payload": {
      "sampleCount": 12,
      "targetFormat": "chatml"
    }
  }' | jq .
```

#### Role 3: Closed-Loop HPO & Loss Trajectory Refiner (`refine_hpo`)
```bash
curl -s -X POST http://localhost:3000/api/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "modelName": "gemini-3.8-flash",
    "agentRole": "refine_hpo",
    "payload": {
      "modelId": "LiquidAI/LFM2.5-1.2B-Instruct",
      "currentLr": 0.0002,
      "loraRank": 32
    }
  }' | jq .
```

#### Role 4: Executive Benchmark & Promotion Report Generator (`generate_report`)
```bash
curl -s -X POST http://localhost:3000/api/agent/execute \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "modelName": "gemini-3.8-flash",
    "agentRole": "generate_report",
    "payload": {
      "modelId": "Qwen/Qwen3.5-2B-Instruct"
    }
  }' | jq .
```

### 3.4 Plain-Language Benchmark Explainer (`POST /api/benchmarks/explain`)
```bash
curl -s -X POST http://localhost:3000/api/benchmarks/explain \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "providerModelName": "gemini-3.8-flash",
    "modelName": "Qwen3.5-2B-Instruct",
    "architecture": "Qwen3.5",
    "baseBenchmarks": {
      "mmluPro": 61.4,
      "gsm8k": 76.8,
      "humanEval": 71.2,
      "ifEval": 73.5,
      "mtBench": 7.82,
      "perplexity": 5.84,
      "tokensPerSec": 248.6
    },
    "fineTunedBenchmarks": {
      "mmluPro": 65.8,
      "gsm8k": 83.4,
      "humanEval": 79.1,
      "ifEval": 81.2,
      "mtBench": 8.36,
      "perplexity": 5.08,
      "tokensPerSec": 247.9
    }
  }' | jq .
```

### 3.5 Python Execution Engine Bridge (`/api/engine/*`)
Check Python & GPU runtime readiness (`torch`, `unsloth`, `transformers`, `trl`, `peft`, `bitsandbytes`, `nvidia-smi`):
```bash
curl -s http://localhost:3000/api/engine/preflight | jq .
```

Materialize the active UI manifest + JSONL corpus to `./runs/<run_id>/` and spawn the real Python execution engine (`--dry-run` or live GPU training):
```bash
curl -s -X POST http://localhost:3000/api/engine/materialize-and-run \
  -H "Content-Type: application/json" \
  -d '{
    "track": "TEXT_LLM",
    "dryRun": true,
    "manifest": {
      "track": "TEXT_LLM",
      "modelId": "Qwen/Qwen3.5-2B-Instruct",
      "architecture": "Qwen3.5",
      "maxSeqLength": 8192,
      "loadIn4Bit": true,
      "dtype": "bfloat16",
      "lora": { "r": 32, "loraAlpha": 64, "loraDropout": 0, "useDora": true, "useRsLora": true, "targetModules": ["q_proj", "k_proj", "v_proj", "o_proj"] },
      "sft": { "learningRate": 0.0002, "perDeviceTrainBatchSize": 8, "gradientAccumulationSteps": 2, "numTrainEpochs": 3, "warmupRatio": 0.05, "weightDecay": 0.01, "lrSchedulerType": "cosine", "neftuneNoiseAlpha": 5, "samplePacking": true, "trainOnResponsesOnly": true },
      "stage2Alignment": { "enabled": true, "method": "ORPO", "beta": 0.1, "datasetId": "argilla/ultrafeedback-binarized-preferences-cleaned", "maxSteps": 120 },
      "stage3Grpo": { "enabled": true, "numGenerations": 8, "klBeta": 0.04, "verifiers": ["json_schema", "python_pytest"], "maxSteps": 80 },
      "datasetPath": "./runs/demo/dataset_corpus.jsonl",
      "outputDir": "./runs/demo/checkpoints"
    }
  }' | jq .
```

---

## 4. Track A — Pure Text LLM Fine-Tuning CLI Commands

ForgeML includes a complete, production-ready Python execution engine inside **`/engine/`** (`engine/train_text_pipeline.py`, `engine/train_vision_pipeline.py`, `engine/export_and_merge.py`, and `engine/requirements.txt`).

### 4.0 Built-In `/engine/` Python Execution Layer (Direct Workstation & Docker CLI)
Install the pinned Python ML stack (`torch`, `unsloth`, `transformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `qwen-vl-utils`, `mergekit`):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r engine/requirements.txt
# Or via npm script:
npm run engine:install
```

Run Track A (Pure Text LLM: Stage 1 SFT → Stage 2 ORPO/SimPO/DPO → Stage 3 GRPO RLVR) directly from a materialized manifest:
```bash
# Dry-run preflight & manifest verification (works on CPU or GPU):
python3 engine/train_text_pipeline.py --config ./runs/latest_text/training_manifest.json --dry-run

# Full GPU training run (Unsloth FastLanguageModel + Transformers/PEFT fallback for LFM2.5/DoRA):
python3 engine/train_text_pipeline.py --config ./runs/latest_text/training_manifest.json
```

Run Track B (Isolated Vision-Language VLM: `FastVisionModel` / `AutoModelForImageTextToText` + Vision Tower freeze + MLP projector training):
```bash
# Dry-run VLM manifest & patch budget verification:
python3 engine/train_vision_pipeline.py --config ./runs/latest_vlm/vision_manifest.json --dry-run

# Full GPU VLM training run:
python3 engine/train_vision_pipeline.py --config ./runs/latest_vlm/vision_manifest.json
```

Merge LoRA/DoRA adapters into 16-bit Safetensors and quantize to `GGUF` (+ `mmproj-f16.gguf` for VLM):
```bash
# Text LLM merge & GGUF export:
python3 engine/export_and_merge.py \
  --base-model "Qwen/Qwen3.5-2B-Instruct" \
  --adapter-dir "./runs/latest_text/checkpoints" \
  --output-dir "./runs/latest_text/exported" \
  --quant "Q4_K_M"

# Vision VLM merge & dual-artifact GGUF + mmproj-f16.gguf export:
python3 engine/export_and_merge.py \
  --base-model "LiquidAI/LFM2.5-VL-450M-Instruct" \
  --adapter-dir "./runs/latest_vlm/checkpoints" \
  --output-dir "./runs/latest_vlm/exported" \
  --quant "Q4_K_M" \
  --mmproj
```

### 4.1 Stage 1: Standalone Unsloth QLoRA / DoRA Script (`train_unsloth.py`)
Includes FlashAttention-2/3 Sample Packing (`packing=True`) and Completion-Only Loss Masking (`train_on_responses_only`):
```bash
# Create isolated virtual environment & install Unsloth + TRL
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  trl peft accelerate bitsandbytes datasets

# Launch Stage 1 SFT
python3 train_unsloth.py \
  --model_id "Qwen/Qwen3.5-2B-Instruct" \
  --dataset_path "./dataset_blend_chatml.jsonl" \
  --output_dir "./checkpoints/qwen3.5-2b-sft-r32" \
  --lora_r 32 \
  --lora_alpha 64 \
  --learning_rate 2e-4 \
  --lr_scheduler_type "wsd" \
  --max_seq_length 8192 \
  --packing True \
  --neftune_noise_alpha 5.0
```

### 4.2 Stage 1 Alternative: Axolotl Declarative Training (`axolotl_config.yaml`)
```bash
# Preprocess tokenized & multipacked dataset shards
accelerate launch -m axolotl.cli.preprocess axolotl_config.yaml

# Run multi-GPU / single-GPU Axolotl fine-tuning
accelerate launch -m axolotl.cli.train axolotl_config.yaml
```

### 4.3 Stage 2: Post-SFT Preference Alignment (`train_orpo_trl.py`)
Runs reference-free **ORPO** or **SimPO** (zero extra reference-model VRAM) or **DPO / KTO**:
```bash
python3 train_orpo_trl.py \
  --sft_checkpoint "./checkpoints/qwen3.5-2b-sft-r32" \
  --preference_dataset "argilla/ultrafeedback-binarized-preferences-cleaned" \
  --method "ORPO" \
  --beta 0.15 \
  --learning_rate 8e-6 \
  --output_dir "./checkpoints/qwen3.5-2b-orpo-aligned"
```

### 4.4 Stage 3: GRPO / RLVR Verifiable Reward Post-Training (`train_grpo_verifier.py`)
Trains `<think>...</think>` reasoning chains with deterministic Python AST/unit-test, GSM8K/MATH symbolic equivalence, and JSON schema verifiers (zero Critic model VRAM overhead):
```bash
python3 train_grpo_verifier.py \
  --model_name_or_path "./checkpoints/qwen3.5-2b-orpo-aligned" \
  --group_size 8 \
  --kl_beta 0.04 \
  --max_completion_length 1536 \
  --verifiers "code_unit_tests,math_symbolic,json_schema,think_tag_format" \
  --output_dir "./checkpoints/qwen3.5-2b-grpo-rlvr"
```

---

## 5. Track B — Vision-Language (VLM) & VEIL Audit CLI Commands

The **Vision-Language Pipeline (`VISION_VLM`)** operates in complete isolation from Pure Text LLM checkpoints to prevent modality gradient interference.

### 5.1 Unsloth `FastVisionModel` Fine-Tuning (`train_vision_unsloth.py`)
Supports `LiquidAI/LFM2.5-VL-450M-Instruct`, `LiquidAI/LFM2.5-VL-1.6B-Instruct`, `Qwen/Qwen3.5-VL-0.8B-Instruct`, `Qwen/Qwen3.5-VL-2B-Instruct`, and `google/gemma-4-e2b-vision-it`:
```bash
python3 train_vision_unsloth.py \
  --model_id "Qwen/Qwen3.5-VL-2B-Instruct" \
  --dataset_manifest "./veil_master_12.5k_chatml.jsonl" \
  --freeze_vision_tower True \
  --train_mlp_projector True \
  --apply_lora_to_llm True \
  --min_pixels 200704 \
  --max_pixels 1003520 \
  --lora_r 32 \
  --lora_alpha 64 \
  --output_dir "./checkpoints/qwen3.5-vl-2b-veil-audit"
```

### 5.2 Dual-Artifact Vision Export (`Backbone GGUF` + `mmproj-f16.gguf`)
Vision-Language models require exporting the quantized language backbone alongside an **unquantized FP16 vision tower + multimodal projector (`mmproj-f16.gguf`)** so OCR and `[0, 1000]` bounding-box coordinates do not degrade:
```bash
# 1. Export FP16 Vision Encoder + Multimodal MLP Connector (mmproj)
python3 llama.cpp/convert_hf_to_gguf.py \
  ./checkpoints/qwen3.5-vl-2b-veil-audit-merged \
  --mmproj \
  --outfile ./exports/mmproj-qwen3.5-vl-2b-instruct-f16.gguf \
  --outtype f16

# 2. Export & Quantize Language Backbone to GGUF-Q4_K_M
python3 llama.cpp/convert_hf_to_gguf.py \
  ./checkpoints/qwen3.5-vl-2b-veil-audit-merged \
  --outfile ./exports/qwen3.5-vl-2b-instruct-f16.gguf \
  --outtype f16

./llama.cpp/build/bin/llama-quantize \
  ./exports/qwen3.5-vl-2b-instruct-f16.gguf \
  ./exports/qwen3.5-vl-2b-instruct-q4_k_m.gguf \
  Q4_K_M
```

---

## 6. Adapter Merging, `imatrix` Calibration & Quantization Commands

### 6.1 Multi-Adapter Task Arithmetic Merging (`mergekit-yaml`)
Combines 2–4 specialized LoRA checkpoints (e.g., Coding + Math + Chat) using **DARE-TIES**, **TIES**, **SLERP**, or **Linear Task Arithmetic**:
```bash
pip install mergekit

mergekit-yaml ./mergekit_config.yaml ./checkpoints/merged-multi-adapter-bf16 \
  --cuda \
  --low-cpu-memory
```

### 6.2 Importance Matrix (`imatrix`) Calibration + Salient-Layer Mixed-Precision GGUF
Recovers 75–82% of 4-bit quantization perplexity loss on sub-2B models while keeping `token_embd`, `output.weight`, and `attn_v` at `Q6_K`/`Q8_0`:
```bash
# Step 1: Compute activation importance matrix on domain calibration text
./llama.cpp/build/bin/llama-imatrix \
  -m ./exports/model-f16.gguf \
  -f ./calibration_domain_corpus.txt \
  -o ./exports/model.imatrix \
  -ngl 99 --chunks 128

# Step 2: Quantize with imatrix and salient tensor type overrides
./llama.cpp/build/bin/llama-quantize \
  --imatrix ./exports/model.imatrix \
  --token-embedding-type q6_K \
  --output-tensor-type q6_K \
  ./exports/model-f16.gguf \
  ./exports/model-imatrix-q4_k_m.gguf \
  Q4_K_M
```

### 6.3 Export to ONNX Runtime GenAI (`optimum-cli`)
```bash
optimum-cli export onnx \
  --model ./checkpoints/merged-bf16 \
  --task text-generation-with-past \
  --dtype fp16 \
  ./exports/onnx-fp16/
```

---

## 7. Local Inference Server Commands (`Ollama`, `llama-server`, `vLLM`)

### 7.1 Ollama Local Deployment (`Modelfile`)
```bash
# Build local Ollama model from exported GGUF + Modelfile
ollama create forgeml-qwen3.5-2b-custom -f ./Modelfile

# Run interactive terminal chat
ollama run forgeml-qwen3.5-2b-custom

# Test via Ollama local REST API
curl http://localhost:11434/api/chat -d '{
  "model": "forgeml-qwen3.5-2b-custom",
  "messages": [{ "role": "user", "content": "Write a lock-free ring buffer in Rust." }],
  "stream": false
}'
```

### 7.2 `llama.cpp` (`llama-server`) OpenAI-Compatible Server
#### Pure Text LLM:
```bash
./llama.cpp/build/bin/llama-server \
  -m ./exports/qwen3.5-2b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 32768 \
  -ngl 99 \
  --flash-attn
```

#### Multimodal Vision-Language VLM (`GGUF` + `--mmproj`):
```bash
./llama.cpp/build/bin/llama-server \
  -m ./exports/qwen3.5-vl-2b-instruct-q4_k_m.gguf \
  --mmproj ./exports/mmproj-qwen3.5-vl-2b-instruct-f16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 16384 \
  -ngl 99
```

### 7.3 `vLLM` High-Throughput Safetensors / GPTQ Server
```bash
vllm serve ./checkpoints/qwen3.5-2b-merged-safetensors \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enable-prefix-caching
```

---

## 8. Docker & Multi-Container GPU Orchestration Commands

The **Cloud & Docker** tab generates production `Dockerfile` and `docker-compose.yml` manifests with NVIDIA Container Toolkit passthrough.

```bash
# Build the ForgeML Training & Inference image
docker build -t forgeml-studio-runtime:latest .

# Launch GPU container with HuggingFace cache volume mount
docker run --gpus all --rm -it \
  -p 3000:3000 \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  forgeml-studio-runtime:latest

# Or launch the full multi-container stack (Studio + vLLM/llama-server + Eval Worker)
docker compose up -d --build
docker compose logs -f
```

---

## 9. Headless Browser & UI Verification Commands (`playwright-cli`)

To inspect the rendered application, verify zero console/page errors, or capture screenshots from the terminal:

```bash
# Open running app on port 3000, switch to Vision-Language Pipeline -> Dataset Studio, and verify zero console errors
playwright-cli open http://localhost:3000 && \
playwright-cli resize 1440 900 && \
playwright-cli snapshot && \
playwright-cli console error && \
playwright-cli close-all
```

---

## 10. In-App Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Cmd + K` / `Ctrl + K` | Open the **Global Command Palette** (jump to any view, switch models, trigger Auto-HPO, run Red-Team suite, or export bundles) |
| `Escape` | Close active modal or Command Palette |
