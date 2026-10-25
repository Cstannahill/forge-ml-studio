import React, { useState } from 'react';
import {
  Calculator,
  Check,
  Clock,
  Copy,
  Cpu,
  FileDiff,
  Sliders,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  ArchitectureFamily,
  ConstraintSolverQualityTarget,
  ExecutableRunBundle,
  HyperparameterConfig,
  PreferenceAlignmentMethod,
  TargetFormat,
  MixedPrecisionStrategy,
  computeLayerWiseMixedPrecisionPlan,
  computeSideBySideScriptDiff,
  generateExecutableRunBundle,
  solveHardwareAndTimeBudgetConstraints,
} from '../lib/pipelineLogic';

interface ConstraintSolverAndScriptDiffPanelProps {
  modelId: string;
  modelName: string;
  architecture: ArchitectureFamily;
  paramsBillions: number;
  contextWindow: number;
  activeHpoConfig: HyperparameterConfig;
  samplePackingEnabled: boolean;
  completionOnlyLossEnabled: boolean;
  stage2Enabled: boolean;
  alignmentMethod: PreferenceAlignmentMethod;
  alignmentBeta: number;
  preferenceDataset: string;
  datasetSources: string[];
  targetFormat: TargetFormat;
  currentBundle: ExecutableRunBundle;
  onApplySolvedConstraints: (solution: {
    hpoConfig: HyperparameterConfig;
    samplePackingEnabled: boolean;
    completionOnlyLossEnabled: boolean;
    summaryMessage: string;
  }) => void;
}

export const ConstraintSolverAndScriptDiffPanel: React.FC<
  ConstraintSolverAndScriptDiffPanelProps
> = ({
  modelId,
  modelName,
  architecture,
  paramsBillions,
  contextWindow,
  activeHpoConfig,
  samplePackingEnabled,
  completionOnlyLossEnabled,
  stage2Enabled,
  alignmentMethod,
  alignmentBeta,
  preferenceDataset,
  datasetSources,
  targetFormat,
  currentBundle,
  onApplySolvedConstraints,
}) => {
  // 1. Reverse Hardware & Time Budget Constraint Solver State
  const [maxVramGb, setMaxVramGb] = useState<number>(12); // Default: RTX 4070 (12 GB VRAM)
  const [maxMinutes, setMaxMinutes] = useState<number>(45); // Default: 45 minutes
  const [targetQuality, setTargetQuality] =
    useState<ConstraintSolverQualityTarget>('MAX_ACCURACY');
  const [datasetSampleCount, setDatasetSampleCount] = useState<number>(12000);
  const [solverAppliedBanner, setSolverAppliedBanner] = useState<string | null>(null);

  const gpuPresets: Array<{ label: string; vram: number; mins: number }> = [
    { label: 'RTX 4060 Laptop (8 GB · 30m)', vram: 8, mins: 30 },
    { label: 'RTX 4070 (12 GB · 45m)', vram: 12, mins: 45 },
    { label: 'RTX 4080 (16 GB · 40m)', vram: 16, mins: 40 },
    { label: 'RTX 3090 / 4090 (24 GB · 35m)', vram: 24, mins: 35 },
    { label: 'A100 / H100 (80 GB · 20m)', vram: 80, mins: 20 },
  ];

  const solvedPlan = solveHardwareAndTimeBudgetConstraints({
    maxVramGb,
    maxMinutes,
    targetQuality,
    paramsBillions,
    architecture,
    datasetSampleCount,
  });

  // 2. Side-by-Side Training Script Diff Viewer State
  const [diffMode, setDiffMode] = useState<
    'CONFIG_DELTA' | 'ARCH_QWEN_VS_GEMMA4_VL' | 'FRAMEWORK_UNSLOTH_VS_TRL_AXOLOTL'
  >('CONFIG_DELTA');
  const [frameworkPair, setFrameworkPair] = useState<'UNSLOTH_VS_TRL' | 'UNSLOTH_VS_AXOLOTL'>(
    'UNSLOTH_VS_TRL'
  );
  const [showChangedOnly, setShowChangedOnly] = useState<boolean>(false);
  const [copiedSide, setCopiedSide] = useState<'left' | 'right' | null>(null);

  // 3. Layer-Wise Mixed-Precision Bit-Allocation Planner State
  const [mixedPrecisionStrategy, setMixedPrecisionStrategy] =
    useState<MixedPrecisionStrategy>('HETEROGENEOUS_SALIENCE_KQUANT');
  const [protectVisionConnector, setProtectVisionConnector] = useState<boolean>(true);
  const [copiedHeteroCli, setCopiedHeteroCli] = useState<boolean>(false);

  const mixedPrecisionPlan = computeLayerWiseMixedPrecisionPlan({
    modelId,
    modelName,
    architecture,
    paramsBillions,
    modality: modelId.toLowerCase().includes('vl') || modelId.toLowerCase().includes('vision') ? 'VISION_VLM' : 'TEXT_LLM',
    strategy: mixedPrecisionStrategy,
    protectVisionConnector,
  });

  // Build baseline untuned bundle for Mode 1 (CONFIG_DELTA)
  const baselineUntunedBundle = generateExecutableRunBundle({
    modelId: 'Qwen/Qwen3.5-0.8B-Instruct',
    modelName: 'Qwen3.5-0.8B-Instruct (Default Baseline)',
    architecture: 'Qwen3.5',
    paramsBillions: 0.8,
    contextWindow: 4096,
    hpoConfig: {
      learningRate: 0.00015,
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0.0,
      warmupRatio: 0.03,
      weightDecay: 0.01,
      neftuneAlpha: 0,
      batchSize: 4,
      lrScheduler: 'cosine',
    },
    datasetSources: ['HuggingFaceH4/ultrachat_200k'],
    targetFormat: 'chatml',
    samplePackingEnabled: false,
    completionOnlyLossEnabled: false,
    alignmentStage: {
      enabled: false,
      method: 'ORPO',
      beta: 0.1,
      preferenceDataset: 'argilla/ultrafeedback-binarized-preferences-cleaned',
    },
  });

  // Build Qwen3.5 Text vs Gemma4-VL Vision Script for Mode 2
  const qwenTextScript = currentBundle.unslothTrainPy;
  const gemma4VlScript = `#!/usr/bin/env python3
# ForgeML Vision-Language Training Script: google/gemma-4-e4b-vision-it (Gemma4 Pan-&-Scan VLM)
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-4-e4b-vision-it"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# [VISION GUARD] Freeze Pretrained SigLIP-So400M Vision Tower; train Multimodal Projector + LLM DoRA
for name, param in model.named_parameters():
    if "vision_tower" in name:
        param.requires_grad = False

peft_config = LoraConfig(
    r=${activeHpoConfig.loraRank},
    lora_alpha=${activeHpoConfig.loraAlpha},
    lora_dropout=${activeHpoConfig.loraDropout},
    use_dora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mm_input_projection"],
)
model = get_peft_model(model, peft_config)

# Pan-&-Scan Adaptive 896x896 Tile Collator (<start_of_image><image_soft_token><end_of_image>)
sft_config = SFTConfig(
    output_dir="./checkpoints/gemma-4-e4b-vision-it-ft",
    per_device_train_batch_size=${activeHpoConfig.batchSize},
    gradient_accumulation_steps=${Math.max(1, Math.round(16 / activeHpoConfig.batchSize))},
    learning_rate=${activeHpoConfig.learningRate},
    lr_scheduler_type="${activeHpoConfig.lrScheduler}",
    warmup_ratio=${activeHpoConfig.warmupRatio},
    weight_decay=${activeHpoConfig.weightDecay},
    neftune_noise_alpha=${activeHpoConfig.neftuneAlpha},
    packing=False,  # VLM dynamic patch collator preserves contiguous <image_soft_token> spans
    dataset_kwargs={"skip_prepare_dataset": True},
    bf16=True,
)`;

  let leftTitle = 'Baseline Factory SFT (unsloth_train.py · r=16 · Cosine · Unpacked)';
  let leftCode = baselineUntunedBundle.unslothTrainPy;
  let rightTitle = `Active Tuned Config (${modelName} · r=${activeHpoConfig.loraRank} · ${activeHpoConfig.lrScheduler.toUpperCase()} · Multipack=${samplePackingEnabled ? 'ON' : 'OFF'})`;
  let rightCode = currentBundle.unslothTrainPy;

  if (diffMode === 'ARCH_QWEN_VS_GEMMA4_VL') {
    leftTitle = `Pure Text Track (${modelName} · unsloth_train.py)`;
    leftCode = qwenTextScript;
    rightTitle = `Vision-Language Track (Gemma4-E4B-Vision-IT · Pan-&-Scan VLM Trainer)`;
    rightCode = gemma4VlScript;
  } else if (diffMode === 'FRAMEWORK_UNSLOTH_VS_TRL_AXOLOTL') {
    leftTitle = `Unsloth FastLoRA Engine (unsloth_train.py)`;
    leftCode = currentBundle.unslothTrainPy;
    if (frameworkPair === 'UNSLOTH_VS_TRL') {
      rightTitle = `HuggingFace TRL Stage-2 ${alignmentMethod} Engine (trl_${alignmentMethod.toLowerCase()}.py)`;
      rightCode = currentBundle.trlSftOrpoPy;
    } else {
      rightTitle = `Axolotl Declarative YAML (axolotl_config.yaml)`;
      rightCode = currentBundle.axolotlConfigYaml;
    }
  }

  const diffResult = computeSideBySideScriptDiff({
    leftTitle,
    leftCode,
    rightTitle,
    rightCode,
  });

  const visibleDiffRows = showChangedOnly
    ? diffResult.diffRows.filter((row) => row.status !== 'UNCHANGED')
    : diffResult.diffRows;

  return (
    <div className="space-y-6">
      {/* PART A: Interactive VRAM & Training Time Budget Constraint Solver ("Reverse Calculator") */}
      <section
        aria-label="Interactive VRAM and Training Time Budget Constraint Solver"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">
                Reverse Hardware &amp; Time Budget Calculator
              </span>
              <span
                className={`text-[11px] font-mono px-2 py-0.5 rounded font-semibold ${
                  solvedPlan.feasibilityStatus === 'OPTIMAL_FIT'
                    ? 'bg-emerald-500/15 text-emerald-300'
                    : solvedPlan.feasibilityStatus === 'TIGHT_FIT_WITH_LISA'
                    ? 'bg-amber-500/15 text-amber-300'
                    : 'bg-red-500/20 text-red-300'
                }`}
              >
                {solvedPlan.feasibilityStatus === 'OPTIMAL_FIT'
                  ? `OPTIMAL FIT (${solvedPlan.vramHeadroomGb} GB Headroom)`
                  : solvedPlan.feasibilityStatus === 'TIGHT_FIT_WITH_LISA'
                  ? 'TIGHT FIT (LISA Layer Sampling Active)'
                  : 'EXCEEDS VRAM BUDGET'}
              </span>
            </div>
            <h2 className="text-base font-semibold text-slate-100 mt-1.5 flex items-center gap-2">
              <Calculator className="w-4 h-4 text-amber-400" />
              <span>
                Interactive VRAM &amp; Training Time Budget Constraint Solver (&ldquo;Reverse Calculator&rdquo;)
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Enter your exact GPU VRAM cap and wall-clock deadline (e.g., &ldquo;RTX 4070 12 GB in under 45 minutes&rdquo;) to backwards-solve quantization, LoRA rank, micro-batch size, sequence length, Multipack, and LISA layer sampling.
            </p>
          </div>

          <button
            type="button"
            onClick={() => {
              onApplySolvedConstraints({
                hpoConfig: solvedPlan.solvedHpoConfig,
                samplePackingEnabled: solvedPlan.samplePackingEnabled,
                completionOnlyLossEnabled: solvedPlan.completionOnlyLossEnabled,
                summaryMessage: `Constraint Solver applied for ${maxVramGb}GB VRAM / ${maxMinutes}m budget: ${solvedPlan.quantization.toUpperCase()}, r=${solvedPlan.loraRank}, batch=${solvedPlan.batchSize}, seq=${solvedPlan.seqLen}, LISA=${solvedPlan.enableLisaLayerSampling ? 'ON' : 'OFF'}.`,
              });
              setSolverAppliedBanner(
                `Applied Reverse Solver Config to ${modelName}: ${solvedPlan.quantization.toUpperCase()} · r=${solvedPlan.loraRank} (α=${solvedPlan.loraAlpha}) · Batch=${solvedPlan.batchSize} · Seq=${solvedPlan.seqLen} · Est. Peak VRAM=${solvedPlan.estimatedPeakVramGb} GB · Est. Time=${solvedPlan.estimatedTrainingMinutes} mins.`
              );
            }}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-mono font-semibold text-slate-950 bg-amber-400 hover:bg-amber-300 rounded-lg transition-colors shrink-0"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Apply Solved Constraints to Pipeline</span>
          </button>
        </div>

        {solverAppliedBanner && (
          <div
            role="status"
            className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300 flex items-center justify-between"
          >
            <span>{solverAppliedBanner}</span>
            <button
              type="button"
              onClick={() => setSolverAppliedBanner(null)}
              className="text-slate-400 hover:text-slate-200 text-[11px]"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Quick Hardware Budget Presets */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono text-slate-400">Quick GPU &amp; Time Presets:</span>
          {gpuPresets.map((preset) => {
            const isSelected = maxVramGb === preset.vram && maxMinutes === preset.mins;
            return (
              <button
                key={preset.label}
                type="button"
                onClick={() => {
                  setMaxVramGb(preset.vram);
                  setMaxMinutes(preset.mins);
                }}
                className={`px-2.5 py-1 text-xs font-mono rounded-lg border transition-colors ${
                  isSelected
                    ? 'bg-amber-500/20 border-amber-400 text-amber-200 font-semibold'
                    : 'bg-[#0B0F17] border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {preset.label}
              </button>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 5 cols: Hard Constraint Sliders & Target Priority */}
          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 space-y-4">
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="solver-vram-input" className="text-slate-300 font-medium">
                  1. Max GPU VRAM Ceiling (GB)
                </label>
                <span className="font-mono font-semibold text-amber-300">{maxVramGb} GB VRAM</span>
              </div>
              <input
                id="solver-vram-input"
                type="range"
                min={4}
                max={80}
                step={2}
                value={maxVramGb}
                onChange={(e) => setMaxVramGb(Number(e.target.value))}
                className="w-full accent-amber-400"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-0.5">
                <span>4 GB (Edge)</span>
                <span>12 GB (RTX 4070)</span>
                <span>24 GB (RTX 4090)</span>
                <span>80 GB (H100)</span>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <label htmlFor="solver-minutes-input" className="text-slate-300 font-medium">
                  2. Max Training Time Deadline (Minutes)
                </label>
                <span className="font-mono font-semibold text-emerald-400">
                  &le; {maxMinutes} mins
                </span>
              </div>
              <input
                id="solver-minutes-input"
                type="range"
                min={10}
                max={180}
                step={5}
                value={maxMinutes}
                onChange={(e) => setMaxMinutes(Number(e.target.value))}
                className="w-full accent-emerald-400"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-0.5">
                <span>10m Sprint</span>
                <span>45m Standard</span>
                <span>90m Deep</span>
                <span>180m Full</span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div>
                <label
                  htmlFor="solver-quality-select"
                  className="block text-xs text-slate-300 font-medium mb-1"
                >
                  3. Optimization Priority
                </label>
                <select
                  id="solver-quality-select"
                  value={targetQuality}
                  onChange={(e) =>
                    setTargetQuality(e.target.value as ConstraintSolverQualityTarget)
                  }
                  className="w-full bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value="MAX_ACCURACY">Max Downstream Accuracy</option>
                  <option value="BALANCED">Balanced Speed &amp; Quality</option>
                  <option value="FASTEST_PROTOTYPE">Fastest Turnaround</option>
                </select>
              </div>

              <div>
                <label
                  htmlFor="solver-samples-select"
                  className="block text-xs text-slate-300 font-medium mb-1"
                >
                  4. Dataset Corpus Size
                </label>
                <select
                  id="solver-samples-select"
                  value={datasetSampleCount}
                  onChange={(e) => setDatasetSampleCount(Number(e.target.value))}
                  className="w-full bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value={5000}>5,000 samples (Curated)</option>
                  <option value={12000}>12,000 samples (Standard)</option>
                  <option value={25000}>25,000 samples (Large Blend)</option>
                  <option value={50000}>50,000 samples (Full Scale)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Right 7 cols: Backwards-Solved Hyperparameter & Memory Allocation Plan */}
          <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono text-xs tabular-nums">
              <div className="bg-[#111827] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Solved Precision</div>
                <div className="text-amber-300 font-semibold mt-0.5">
                  {solvedPlan.quantization.toUpperCase()}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Grad Checkpointing: ON</div>
              </div>

              <div className="bg-[#111827] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Solved DoRA Rank</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  r={solvedPlan.loraRank} · α={solvedPlan.loraAlpha}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  LR: {solvedPlan.solvedHpoConfig.learningRate} (WSD)
                </div>
              </div>

              <div className="bg-[#111827] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Batch &amp; Context</div>
                <div className="text-blue-300 font-semibold mt-0.5">
                  Batch={solvedPlan.batchSize} · {solvedPlan.seqLen} ctx
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  Grad Accum: {solvedPlan.gradientAccumulationSteps} steps
                </div>
              </div>

              <div className="bg-[#111827] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">LISA &amp; Multipack</div>
                <div className="text-purple-300 font-semibold mt-0.5">
                  {solvedPlan.enableLisaLayerSampling
                    ? `LISA k=${solvedPlan.lisaActiveLayers} ON`
                    : 'Full LoRA Layers'}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Multipack + -100 Mask</div>
              </div>
            </div>

            {/* VRAM & Time Budget Bars */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 bg-[#111827] border border-slate-800 rounded-lg p-3 font-mono text-xs">
              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Peak VRAM vs. Ceiling:</span>
                  <span className="text-emerald-400 font-semibold">
                    {solvedPlan.estimatedPeakVramGb} GB / {maxVramGb} GB
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      solvedPlan.feasible ? 'bg-emerald-500' : 'bg-red-500'
                    }`}
                    style={{
                      width: `${Math.min(
                        100,
                        Math.round((solvedPlan.estimatedPeakVramGb / Math.max(1, maxVramGb)) * 100)
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-slate-400">Est. Runtime vs. Deadline:</span>
                  <span className="text-amber-300 font-semibold">
                    {solvedPlan.estimatedTrainingMinutes} mins / {maxMinutes} mins
                  </span>
                </div>
                <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-amber-400 rounded-full"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.round(
                          (solvedPlan.estimatedTrainingMinutes / Math.max(1, maxMinutes)) * 100
                        )
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Solver Step-by-Step Rationale */}
            <div className="space-y-1 text-[11px] font-mono text-slate-300">
              {solvedPlan.solverRationale.map((line, i) => (
                <div key={i} className="flex items-start gap-1.5">
                  <span className="text-amber-400">•</span>
                  <span>{line}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* PART B: Side-by-Side Diff Viewer for Generated Training Scripts (Unsloth vs. Axolotl vs. TRL & Architecture Diffs) */}
      <section
        aria-label="Side-by-Side Diff Viewer for Generated Training Scripts"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-4"
      >
        <div className="flex flex-col gap-3.5 pb-4 border-b border-slate-800/80">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2.5 py-0.5 rounded bg-blue-500/15 text-blue-300 border border-blue-500/30 whitespace-nowrap">
                Live Code Translation Inspector
              </span>
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded bg-slate-800 text-emerald-300 whitespace-nowrap">
                {diffResult.changedLinesCount} Modified · +{diffResult.addedLinesCount} Added · -
                {diffResult.removedLinesCount} Removed
              </span>
            </div>
            <label className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0B0F17] border border-slate-800 text-xs font-mono text-slate-300 cursor-pointer self-start sm:self-auto shrink-0">
              <input
                type="checkbox"
                checked={showChangedOnly}
                onChange={(e) => setShowChangedOnly(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-blue-600"
              />
              <span>Show Changed Lines Only (&Delta;)</span>
            </label>
          </div>

          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <FileDiff className="w-4 h-4 text-blue-400 shrink-0" />
              <span>
                Side-by-Side Diff Viewer for Generated Training Scripts (Unsloth vs. Axolotl vs. TRL &amp; Architecture Diffs)
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Inspect exact line-by-line Python/YAML code changes when tuning hyperparameters, switching from Qwen3.5 to Gemma4-VL, or comparing Unsloth, TRL, and Axolotl.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex flex-wrap items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800">
              {(
                [
                  { id: 'CONFIG_DELTA', label: '1. Hyperparameter Delta (Default vs. Live)' },
                  { id: 'ARCH_QWEN_VS_GEMMA4_VL', label: '2. Qwen3.5 Text vs. Gemma4-VL Vision' },
                  {
                    id: 'FRAMEWORK_UNSLOTH_VS_TRL_AXOLOTL',
                    label: '3. Unsloth vs. TRL / Axolotl',
                  },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setDiffMode(tab.id)}
                  className={`px-3 py-1.5 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                    diffMode === tab.id
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {diffMode === 'FRAMEWORK_UNSLOTH_VS_TRL_AXOLOTL' && (
              <select
                aria-label="Select Framework Comparison Pair"
                value={frameworkPair}
                onChange={(e) =>
                  setFrameworkPair(e.target.value as 'UNSLOTH_VS_TRL' | 'UNSLOTH_VS_AXOLOTL')
                }
                className="bg-[#0B0F17] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-200"
              >
                <option value="UNSLOTH_VS_TRL">Unsloth Python vs. TRL ORPO Python</option>
                <option value="UNSLOTH_VS_AXOLOTL">Unsloth Python vs. Axolotl YAML</option>
              </select>
            )}
          </div>
        </div>

        {/* Key Takeaways Strip */}
        <div className="bg-[#0B0F17] border border-slate-800/90 rounded-lg px-3.5 py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <div className="flex flex-wrap items-center gap-3 text-slate-300">
            {diffResult.keyTakeaways.map((note, idx) => (
              <span key={idx} className="inline-flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span>{note}</span>
              </span>
            ))}
          </div>
          <span className="text-[11px] text-slate-400">
            Showing {visibleDiffRows.length} of {diffResult.totalLines} lines
          </span>
        </div>

        {/* Side-by-Side Code Headers */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3.5 py-2 flex items-center justify-between">
            <span className="text-xs font-mono font-semibold text-amber-300 truncate">
              LEFT: {diffResult.leftTitle}
            </span>
            <button
              type="button"
              onClick={() => {
                navigator.clipboard?.writeText(leftCode);
                setCopiedSide('left');
                setTimeout(() => setCopiedSide(null), 1500);
              }}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-mono text-slate-300 bg-slate-800 hover:bg-slate-700 rounded shrink-0"
            >
              {copiedSide === 'left' ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Left</span>
                </>
              )}
            </button>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3.5 py-2 flex items-center justify-between">
            <span className="text-xs font-mono font-semibold text-emerald-300 truncate">
              RIGHT: {diffResult.rightTitle}
            </span>
            <button
              type="button"
              onClick={() => {
                navigator.clipboard?.writeText(rightCode);
                setCopiedSide('right');
                setTimeout(() => setCopiedSide(null), 1500);
              }}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-mono text-slate-300 bg-slate-800 hover:bg-slate-700 rounded shrink-0"
            >
              {copiedSide === 'right' ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Right</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Unified / Side-by-Side Line Diff Table */}
        <div className="bg-[#0B0F17] border border-slate-800 rounded-lg overflow-x-auto max-h-96 overflow-y-auto font-mono text-[11px]">
          <table className="w-full border-collapse">
            <tbody>
              {visibleDiffRows.map((row, idx) => {
                const isChanged = row.status !== 'UNCHANGED';
                return (
                  <tr
                    key={idx}
                    className={`border-b border-slate-900/80 ${
                      row.status === 'MODIFIED'
                        ? 'bg-amber-500/10'
                        : row.status === 'ADDED'
                        ? 'bg-emerald-500/10'
                        : row.status === 'REMOVED'
                        ? 'bg-red-500/10'
                        : ''
                    }`}
                  >
                    {/* Left Line Number */}
                    <td className="w-9 px-2 py-1 text-right text-slate-500 select-none border-r border-slate-800/80">
                      {row.lineNumberLeft ?? ''}
                    </td>
                    {/* Left Code */}
                    <td
                      className={`w-1/2 px-3 py-1 whitespace-pre overflow-x-auto border-r border-slate-800/80 ${
                        row.status === 'MODIFIED' || row.status === 'REMOVED'
                          ? 'text-red-300'
                          : 'text-slate-300'
                      }`}
                    >
                      {row.status === 'MODIFIED' || row.status === 'REMOVED' ? '- ' : '  '}
                      {row.leftText}
                    </td>

                    {/* Right Line Number */}
                    <td className="w-9 px-2 py-1 text-right text-slate-500 select-none border-r border-slate-800/80">
                      {row.lineNumberRight ?? ''}
                    </td>
                    {/* Right Code + Category Badge */}
                    <td
                      className={`w-1/2 px-3 py-1 whitespace-pre overflow-x-auto ${
                        row.status === 'MODIFIED' || row.status === 'ADDED'
                          ? 'text-emerald-300 font-medium'
                          : 'text-slate-300'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span>
                          {row.status === 'MODIFIED' || row.status === 'ADDED' ? '+ ' : '  '}
                          {row.rightText}
                        </span>
                        {isChanged && row.highlightCategory && (
                          <span className="px-1.5 py-0.2 text-[9px] rounded bg-slate-800 text-amber-300 shrink-0">
                            {row.highlightCategory}
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* PART C: Layer-Wise Mixed-Precision Bit-Allocation Planner (AWQ / Heterogeneous GGUF K-Quants) */}
      <section
        aria-label="Layer-Wise Mixed-Precision Bit-Allocation Planner"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
      >
        <div className="flex flex-col gap-3 pb-4 border-b border-slate-800/80">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                Hessian Salience &amp; Super-Weight Bit Allocation
              </span>
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded bg-slate-800 text-amber-300 font-semibold">
                {mixedPrecisionPlan.effectiveBitsPerWeight} Effective bpw · +{mixedPrecisionPlan.perplexityRecoveredPct}% Perplexity Recovery vs. Uniform Q4
              </span>
            </div>

            <label className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-[#0B0F17] border border-slate-800 text-xs font-mono text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={protectVisionConnector}
                onChange={(e) => setProtectVisionConnector(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-emerald-500"
              />
              <span>Pin Vision Projector (mmproj) at FP16</span>
            </label>
          </div>

          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                Layer-Wise Mixed-Precision Bit-Allocation Planner (AWQ / Heterogeneous GGUF K-Quants)
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Protects high-salience boundary blocks (<code className="font-mono text-slate-300">blk.0</code>, <code className="font-mono text-slate-300">lm_head</code>, <code className="font-mono text-slate-300">attn_v</code>, <code className="font-mono text-slate-300">ffn_down</code>) at 6-bit/8-bit precision while compressing flat-spectrum interior layers to 3-bit/4-bit.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 bg-[#0B0F17] p-1 rounded-lg border border-slate-800 self-start">
            {(
              [
                {
                  id: 'HETEROGENEOUS_SALIENCE_KQUANT',
                  label: '1. Salience-Guided Heterogeneous K-Quant (4.62 bpw)',
                },
                {
                  id: 'ULTRA_EDGE_3_5_BPW',
                  label: '2. Ultra-Compact Edge Profile (3.68 bpw)',
                },
                {
                  id: 'HIGH_FIDELITY_5_2_BPW',
                  label: '3. High-Fidelity Reasoning Profile (5.35 bpw)',
                },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setMixedPrecisionStrategy(opt.id)}
                className={`px-3 py-1.5 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                  mixedPrecisionStrategy === opt.id
                    ? 'bg-emerald-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* 4-Card Telemetry Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs tabular-nums">
          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">Effective Bits-per-Weight</div>
            <div className="text-base font-bold text-emerald-400 mt-1">
              {mixedPrecisionPlan.effectiveBitsPerWeight} bpw
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              -{mixedPrecisionPlan.vramSavedVsFp16Pct}% footprint vs FP16
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">Artifact Footprint</div>
            <div className="text-base font-bold text-blue-300 mt-1">
              {mixedPrecisionPlan.mixedPrecisionSizeGb} GB
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Uniform Q4_K_M: {mixedPrecisionPlan.uniformQ4SizeGb} GB
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">Perplexity Degradation (&Delta;PPL)</div>
            <div className="text-base font-bold text-amber-300 mt-1">
              +{mixedPrecisionPlan.mixedPrecisionPerplexityDelta} PPL
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Uniform Q4: +{mixedPrecisionPlan.uniformQ4PerplexityDelta} PPL
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">PPL Degradation Recovered</div>
            <div className="text-base font-bold text-purple-300 mt-1">
              {mixedPrecisionPlan.perplexityRecoveredPct}% Recovered
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Super-weight outlier channels protected
            </div>
          </div>
        </div>

        {/* Layer-Block Bit Allocation Table + CLI */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-2.5">
            <div className="text-xs font-semibold text-slate-200">
              Per-Block Hessian Sensitivity &amp; Bitwidth Allocation Matrix
            </div>
            <div className="space-y-2">
              {mixedPrecisionPlan.layerAllocations.map((layer) => (
                <div
                  key={layer.blockId}
                  className="p-2.5 rounded bg-[#111827] border border-slate-800/90 space-y-1"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                    <span className="font-semibold text-slate-100">{layer.blockLabel}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400 text-[11px]">
                        Hessian: <strong className="text-amber-300">{layer.hessianSensitivity}/10</strong>
                      </span>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 font-semibold">
                        {layer.assignedPrecision} ({layer.bitsPerWeight} bpw)
                      </span>
                    </div>
                  </div>
                  <div className="text-[11px] font-mono text-blue-300">
                    Tensors: {layer.tensorPattern}
                  </div>
                  <div className="text-[11px] text-slate-400">{layer.rationale}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-semibold text-slate-200">
                  Heterogeneous llama-quantize Tensor-Type CLI
                </span>
                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard?.writeText(
                      mixedPrecisionPlan.llamaQuantizeHeterogeneousCli
                    );
                    setCopiedHeteroCli(true);
                    setTimeout(() => setCopiedHeteroCli(false), 1500);
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded"
                >
                  {copiedHeteroCli ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span>Copied CLI</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy CLI</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="p-3 rounded bg-[#111827] border border-slate-800 text-[11px] font-mono text-emerald-300 overflow-x-auto leading-relaxed whitespace-pre">
                {mixedPrecisionPlan.llamaQuantizeHeterogeneousCli}
              </pre>
            </div>

            <div className="p-3 rounded bg-[#111827] border border-slate-800 text-[11px] font-mono text-slate-300 space-y-1">
              <div className="text-amber-300 font-semibold">
                Why Heterogeneous K-Quants Beat Uniform Q4:
              </div>
              <div>
                • Keeps <code className="text-emerald-300">token_embd</code> and{' '}
                <code className="text-emerald-300">output.weight</code> at Q8_0 so tool-call JSON delimiters never suffer logit rounding drift.
              </div>
              <div>
                • Protects <code className="text-emerald-300">blk.0</code> and{' '}
                <code className="text-emerald-300">ffn_down</code> super-weights while saving VRAM on interior Q/K matrices.
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

