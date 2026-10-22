# ForgeML Pipeline Studio — System Architecture (`docs/ARCHITECTURE.md`)

This document describes the architectural design, modality isolation boundaries, pure functional domain engine, state persistence layer, and full-stack server topology of **ForgeML Pipeline Studio**.

---

## 1. Dual-Track Modality Isolation Architecture

Fine-tuning compact sub-2B models (`0.3B–2.6B` parameters) on mixed text-only and dense visual-patch sequences simultaneously frequently degrades pure text reasoning or skews attention heads toward `<|image_pad|>` tokens. ForgeML Pipeline Studio enforces strict architectural separation between two top-level modality tracks:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                         FORGEML MODALITY TRACK SWITCHER                           │
├─────────────────────────────────────────┬─────────────────────────────────────────┤
│   TRACK A: PURE TEXT LLM (TEXT_LLM)     │  TRACK B: VISION-LANGUAGE (VISION_VLM)  │
├─────────────────────────────────────────┼─────────────────────────────────────────┤
│ • Models: LFM2.5 (300M/1.2B/2.6B),      │ • Models: LFM2.5-VL (450M/1.6B),        │
│   Qwen3.5 (0.8B/2B), Gemma4 (E2B/E4B)   │   Qwen3.5-VL (0.8B/2B), Gemma4-Vision   │
│ • Logic: src/lib/pipelineLogic.ts       │ • Logic: src/lib/visionPipelineLogic.ts │
│ • Blends: 24 Pure-Text Task Blends      │ • Blends: 25 VLM + VEIL Audit Blends    │
│ • Stages: SFT -> ORPO/SimPO -> GRPO     │ • Controls: Freeze SigLIP/NaViT Tower,  │
│ • Export: Single-File Safetensors,      │   Train MLP Connector, Min/Max Pixels   │
│   GGUF-Q4_K_M (imatrix), GPTQ, ONNX     │ • Export: Dual-File Backbone GGUF +     │
│                                         │   Unquantized FP16 mmproj-f16.gguf      │
└─────────────────────────────────────────┴─────────────────────────────────────────┘
```

---

## 2. Directory & Module Topology

```text
/
├── COMMANDS.md                           # Complete CLI & operational command reference
├── README.md                             # Project overview & quick start
├── docs/
│   ├── ARCHITECTURE.md                   # System architecture & data flow (this file)
│   └── VEIL_VISION_PIPELINE.md           # VEIL Privacy Audit Taxonomy & VLM reference
├── server.ts                             # Express + Vite full-stack server & API routes
├── package.json                          # Scripts, dependencies, and test runner config
├── src/
│   ├── App.tsx                           # Root state container, track switcher & localStorage sync
│   ├── main.tsx                          # React 19 SPA entry point
│   ├── index.css                         # Gruvbox dark theme tokens & typography rules
│   ├── data/
│   │   └── initialPipelineData.ts        # Initial models, training series, HPO trials & DVC commits
│   ├── lib/
│   │   ├── pipelineLogic.ts              # Pure Text LLM domain engine & math functions
│   │   ├── visionPipelineLogic.ts        # Isolated VLM domain engine, 25 blends & VEIL taxonomy
│   │   └── pipelineLogic.test.ts         # 33 unit tests (node:test) covering both tracks
│   └── components/
│       ├── CommandCenterView.tsx         # Text LLM 3-stage pipeline, HPO, Merger & Exporter
│       ├── VisionCommandCenterPanel.tsx  # VLM Tower/Projector controls, NaViT budget & mmproj export
│       ├── DatasetStudioView.tsx         # Text LLM 24 blends, Distillation, IFD, Curriculum & DVC
│       ├── VisionDatasetStudioView.tsx   # VLM 25 blends, VEIL Taxonomy, Patch Density Chart & BBox SVG
│       ├── ModelBenchmarksView.tsx       # Summary Delta Table, Radar Chart, A/B Arena & Red-Team Suite
│       ├── AgenticOrchestratorView.tsx   # Multi-provider AI agents (Gemini, OpenRouter, Ollama, llama.cpp)
│       ├── CloudAndContainerView.tsx     # Hardware/Edge Simulator, Local SDKs & Docker/Compose generator
│       ├── CommandPaletteModal.tsx       # Cmd/Ctrl+K global action & navigation modal
│       └── InfoTooltip.tsx               # Interactive ML Glossary popover component
```

---

## 3. Pure Domain Logic Engine & Unit Test Contract

All quantitative calculations, dataset transformations, and configuration generators live in pure TypeScript modules with zero DOM or React dependencies:

### 3.1 Text LLM Engine (`src/lib/pipelineLogic.ts`)
- **`computeTrainingProgressTelemetry`**: Calculates epoch completion %, step throughput, dynamic ETA, and peak VRAM utilization.
- **`computePreferenceAlignmentPlan`**: Computes Stage 2 memory footprints for reference-free `ORPO`/`SimPO` vs. reference-model `DPO`/`KTO`.
- **`computePackingAndMaskingMetrics`**: Models FlashAttention-2/3 sample packing (`Multipack`) speedup and completion-only loss masking accuracy gains.
- **`computeLayerWiseAdaptiveLoraPlan`**: Allocates layer-specific LoRA/DoRA ranks across Attention, MLP projections, and token embeddings.
- **`computeGrpoVerifiableRewardPlan`**: Models Stage 3 GRPO group size (`G`), KL penalty (`beta`), Critic-free VRAM savings, and verifiable reward lifts.
- **`computeMultiAdapterMergePlan`**: Normalizes multi-adapter weights and generates `mergekit-yaml` and HuggingFace PEFT merge scripts (`DARE-TIES`, `TIES`, `SLERP`, `LINEAR`).
- **`computeImatrixQuantizationCalibration`**: Models perplexity recovery from importance-matrix calibration and salient tensor protection (`Q6_K`/`Q8_0`).
- **`buildSafetensorsExportPlan` & `createSafetensorsBinaryArtifact`**: Constructs spec-compliant single-file `.safetensors` binary headers (8-byte little-endian uint64 length + JSON header + aligned tensor buffer) as well as `GGUF-Q4_K_M`, `GPTQ`, and `ONNX` plans.
- **`serializeTrainingSeriesMap` & `deserializeTrainingSeriesMap`**: Persists per-model training telemetry across browser refreshes and tab switches via `localStorage`.

### 3.2 Vision-Language Engine (`src/lib/visionPipelineLogic.ts`)
- **`calculateVisualPatchTokens`**: Computes dynamic NaViT / SigLIP visual patch tokens from image dimensions `(widthPx, heightPx)` clamped to `[minPixels, maxPixels]` with effective patch stride `28x28`.
- **`computeVisionTokenizerAndPatchDensity` & `capExcessiveNavitPatches`**: Quantifies `visualPatchTokens : textTokens` ratios across the active dataset, categorizes samples into 4 ratio tiers (`<3.5x`, `3.5x–5.0x`, `5.0x–6.2x`, `>6.2x Excessive`), and down-tiles excessive NaViT outliers to prevent visual attention dilution.
- **`computeVisionPipelineHardwarePlan`**: Calculates separate VRAM budgets for the Vision Tower (frozen BF16 vs. unfrozen with AdamW states), Multimodal MLP Connector, LLM Backbone, and Patch Activations, plus dual-artifact `GGUF + mmproj-f16.gguf` export scripts.
- **`validateVisionDatasetCorpus` & `cleanAndNormalizeVisionSample`**: Audits aspect ratios, scrubs camera EXIF/OCR PII, and clamps `[0, 1000]` bounding boxes `[x1, y1, x2, y2]`.
- **`buildVeilPrivacyAuditSeedSamples`**: Synthesizes grounded multi-turn samples across all 5 categories of the **VEIL Vision Audit Taxonomy**.

---

## 4. Full-Stack Server & Resilient Agentic Fallback (`server.ts`)

`server.ts` mounts both REST API routes and Vite SPA middleware on port `3000`:
- **`GET /api/hf/search`**: Queries the live HuggingFace Hub Datasets API (`https://huggingface.co/api/datasets`) with a 3.5-second SLA timeout and automatic fallback to the curated catalog.
- **`POST /api/hf/ingest`**: Streams and normalizes sample records (`messages`, `sharegpt`, `alpaca`) for any HuggingFace dataset ID.
- **`POST /api/agent/execute`**: Executes autonomous ML engineering tasks via:
  1. **Google Gemini** (`@google/genai` SDK with strict `responseSchema` JSON validation),
  2. **OpenRouter** (`https://openrouter.ai/api/v1/chat/completions`),
  3. **Local `llama-server` / `Ollama`** (`http://localhost:8080/v1` or `http://localhost:11434`),
  4. **Deterministic Domain Synthesis Fallback** if no external API key or local daemon is reachable, ensuring zero broken UI workflows.
- **`POST /api/benchmarks/explain`**: Generates structured plain-English comparisons of Base vs. Fine-Tuned benchmark metrics.
