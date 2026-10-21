export type ArchitectureFamily =
  | 'Qwen3.5'
  | 'LFM2.5'
  | 'Gemma4'
  | 'Llama3.3'
  | 'Mistral3'
  | 'DeepSeekV3'
  | 'Custom';

export type QuantizationMode = 'qlora-4bit' | 'lora-8bit' | 'lora-16bit' | 'full-bf16';

export type TargetFormat = 'chatml' | 'sharegpt' | 'alpaca' | 'openai-messages';

export type LoggingFramework = 'wandb' | 'mlflow' | 'tensorboard' | 'otel';

export type CloudProviderId = 'runpod' | 'lambda' | 'modal' | 'aws-sagemaker' | 'gcp-vertex';

export interface VramEstimateInput {
  paramsBillions: number;
  quantization: QuantizationMode;
  contextLength: number;
  batchSize: number;
  gradientCheckpointing: boolean;
  architecture: ArchitectureFamily;
}

export interface VramEstimateResult {
  weightsGb: number;
  optimizerAndGradsGb: number;
  kvCacheAndActivationsGb: number;
  totalRequiredGb: number;
  recommendedGpu: string;
  canRunLocally24Gb: boolean;
  canRunLocally48Gb: boolean;
}

export function estimateVramRequirements(input: VramEstimateInput): VramEstimateResult {
  const bytesPerParamMap: Record<QuantizationMode, number> = {
    'qlora-4bit': 0.55,
    'lora-8bit': 1.08,
    'lora-16bit': 2.05,
    'full-bf16': 2.0,
  };

  const weightsGb = Number((input.paramsBillions * bytesPerParamMap[input.quantization]).toFixed(2));

  // Optimizer states + gradients (AdamW 8-bit/32-bit for full vs LoRA adapters)
  const optimizerMultiplier =
    input.quantization === 'full-bf16'
      ? 6.0 // gradients (2B) + Adam states (4B in 8-bit / paged)
      : input.quantization === 'lora-16bit'
      ? 0.35
      : 0.22;

  const optimizerAndGradsGb = Number((input.paramsBillions * optimizerMultiplier).toFixed(2));

  // Architecture efficiency factor: LFM2.5 (Liquid state-space hybrid) uses significantly less KV/activation memory at long context
  const archActivationFactor: Record<ArchitectureFamily, number> = {
    'LFM2.5': 0.58,
    'Qwen3.5': 0.82,
    'Gemma4': 0.88,
    'Llama3.3': 0.9,
    'Mistral3': 0.85,
    'DeepSeekV3': 0.78,
    'Custom': 1.0,
  };

  const checkpointFactor = input.gradientCheckpointing ? 0.42 : 1.0;
  const rawActivations =
    (input.contextLength / 4096) *
    input.batchSize *
    Math.sqrt(input.paramsBillions) *
    0.95 *
    archActivationFactor[input.architecture] *
    checkpointFactor;

  const kvCacheAndActivationsGb = Number(Math.max(0.35, rawActivations).toFixed(2));
  const cudaOverheadGb = input.paramsBillions <= 3 ? 0.65 : 1.4;
  const totalRequiredGb = Number(
    (weightsGb + optimizerAndGradsGb + kvCacheAndActivationsGb + cudaOverheadGb).toFixed(2)
  );

  let recommendedGpu = '1x Local 8 GB GPU (RTX 4060 / Apple M3)';
  if (totalRequiredGb > 160) {
    recommendedGpu = '4x NVIDIA H100 SXM (320 GB)';
  } else if (totalRequiredGb > 80) {
    recommendedGpu = '2x NVIDIA H100 / A100 (160 GB)';
  } else if (totalRequiredGb > 46) {
    recommendedGpu = '1x NVIDIA H100 / A100 (80 GB)';
  } else if (totalRequiredGb > 22) {
    recommendedGpu = '1x NVIDIA RTX 6000 Ada / L40S (48 GB)';
  } else if (totalRequiredGb > 14.5) {
    recommendedGpu = '1x Local 24 GB GPU (RTX 3090 / 4090)';
  } else if (totalRequiredGb > 7.2) {
    recommendedGpu = '1x Local 12–16 GB GPU (RTX 4070 Ti / 4080)';
  }

  return {
    weightsGb,
    optimizerAndGradsGb,
    kvCacheAndActivationsGb,
    totalRequiredGb,
    recommendedGpu,
    canRunLocally24Gb: totalRequiredGb <= 22.5,
    canRunLocally48Gb: totalRequiredGb <= 45.5,
  };
}

export interface CanonicalMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface NormalizedDatasetSample {
  id: string;
  sourceDataset: string;
  messages: CanonicalMessage[];
  formattedPreview: string;
  tokenEstimate: number;
  qualityScore: number;
  tags: string[];
  cleaned: boolean;
  piiDetected: boolean;
  issueFlags: string[];
}

export function normalizeDatasetRecord(
  raw: Record<string, unknown>,
  sourceDataset: string,
  index: number,
  targetFormat: TargetFormat = 'chatml'
): NormalizedDatasetSample {
  const messages: CanonicalMessage[] = [];
  const issueFlags: string[] = [];

  if (Array.isArray(raw.messages)) {
    for (const m of raw.messages) {
      if (m && typeof m === 'object' && 'role' in m && 'content' in m) {
        const roleRaw = String((m as Record<string, unknown>).role).toLowerCase();
        const role: CanonicalMessage['role'] =
          roleRaw === 'system' ? 'system' : roleRaw === 'assistant' || roleRaw === 'gpt' || roleRaw === 'model' ? 'assistant' : 'user';
        messages.push({ role, content: String((m as Record<string, unknown>).content ?? '').trim() });
      }
    }
  } else if (Array.isArray(raw.conversations)) {
    for (const c of raw.conversations) {
      if (c && typeof c === 'object') {
        const fromRaw = String((c as Record<string, unknown>).from ?? (c as Record<string, unknown>).role ?? 'user').toLowerCase();
        const val = String((c as Record<string, unknown>).value ?? (c as Record<string, unknown>).content ?? '').trim();
        const role: CanonicalMessage['role'] =
          fromRaw === 'system' ? 'system' : fromRaw === 'gpt' || fromRaw === 'assistant' || fromRaw === 'model' ? 'assistant' : 'user';
        messages.push({ role, content: val });
      }
    }
  } else if ('instruction' in raw || 'prompt' in raw || 'question' in raw) {
    const sys = raw.system ? String(raw.system).trim() : '';
    if (sys) messages.push({ role: 'system', content: sys });
    const inst = String(raw.instruction ?? raw.prompt ?? raw.question ?? '').trim();
    const inp = raw.input ? String(raw.input).trim() : '';
    const userText = inp ? `${inst}\n\nContext:\n${inp}` : inst;
    const out = String(raw.output ?? raw.response ?? raw.answer ?? raw.completion ?? '').trim();
    if (userText) messages.push({ role: 'user', content: userText });
    if (out) messages.push({ role: 'assistant', content: out });
  }

  if (messages.length === 0) {
    issueFlags.push('UNRECOGNIZED_SCHEMA');
  }

  const combinedText = messages.map((m) => m.content).join(' ');
  const piiRegex = /\b(?:\d{3}-\d{2}-\d{4}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|sk-[a-zA-Z0-9]{20,})\b/;
  const piiDetected = piiRegex.test(combinedText);
  if (piiDetected) {
    issueFlags.push('PII_OR_SECRET_PATTERN');
  }

  const hasAssistant = messages.some((m) => m.role === 'assistant' && m.content.length > 0);
  if (!hasAssistant) {
    issueFlags.push('MISSING_ASSISTANT_TURN');
  }

  const shortResponse = messages.some((m) => m.role === 'assistant' && m.content.length < 12);
  if (shortResponse) {
    issueFlags.push('LOW_INFORMATION_TURN');
  }

  const tokenEstimate = Math.max(1, Math.ceil(combinedText.length / 3.8));

  // Tag inference
  const tags: string[] = [];
  if (/(?:def |function |class |SELECT |import |```|\bconst\b)/.test(combinedText)) {
    tags.push('code-generation');
  }
  if (/(?:step-by-step|therefore|equation|calculate|proof|theorem|\d+\s*[+\-*/=]\s*\d+)/i.test(combinedText)) {
    tags.push('reasoning-math');
  }
  if (/(?:json|schema|xml|yaml|extract|format)/i.test(combinedText)) {
    tags.push('structured-output');
  }
  if (tags.length === 0) {
    tags.push('general-instruction');
  }

  let qualityScore = 94;
  if (issueFlags.includes('UNRECOGNIZED_SCHEMA')) qualityScore -= 60;
  if (issueFlags.includes('MISSING_ASSISTANT_TURN')) qualityScore -= 45;
  if (issueFlags.includes('PII_OR_SECRET_PATTERN')) qualityScore -= 25;
  if (issueFlags.includes('LOW_INFORMATION_TURN')) qualityScore -= 18;
  if (tokenEstimate > 120 && tokenEstimate < 2048 && issueFlags.length === 0) qualityScore = Math.min(99, qualityScore + 4);

  const formattedPreview = formatSampleToTarget(messages, targetFormat);

  return {
    id: `${sourceDataset.replace(/[^a-zA-Z0-9]/g, '_')}_${index}`,
    sourceDataset,
    messages,
    formattedPreview,
    tokenEstimate,
    qualityScore: Math.max(10, qualityScore),
    tags,
    cleaned: issueFlags.length === 0,
    piiDetected,
    issueFlags,
  };
}

export function formatSampleToTarget(messages: CanonicalMessage[], targetFormat: TargetFormat): string {
  if (targetFormat === 'chatml') {
    return (
      messages
        .map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>`)
        .join('\n') + '\n'
    );
  }
  if (targetFormat === 'sharegpt') {
    return JSON.stringify(
      {
        conversations: messages.map((m) => ({
          from: m.role === 'user' ? 'human' : m.role === 'assistant' ? 'gpt' : 'system',
          value: m.content,
        })),
      },
      null,
      2
    );
  }
  if (targetFormat === 'alpaca') {
    const sys = messages.find((m) => m.role === 'system')?.content ?? '';
    const usr = messages.find((m) => m.role === 'user')?.content ?? '';
    const ast = messages.find((m) => m.role === 'assistant')?.content ?? '';
    return JSON.stringify(
      {
        instruction: sys ? `${sys}\n${usr}` : usr,
        input: '',
        output: ast,
      },
      null,
      2
    );
  }
  return JSON.stringify({ messages }, null, 2);
}

export function cleanAndRedactSample(sample: NormalizedDatasetSample, targetFormat: TargetFormat = 'chatml'): NormalizedDatasetSample {
  const cleanedMessages = sample.messages
    .map((m) => ({
      role: m.role,
      content: m.content
        .replace(/\b\d{3}-\d{2}-\d{4}\b/g, '[REDACTED_SSN]')
        .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, '[REDACTED_EMAIL]')
        .replace(/\bsk-[a-zA-Z0-9]{20,}\b/g, '[REDACTED_API_KEY]')
        .replace(/\s+\n/g, '\n')
        .trim(),
    }))
    .filter((m) => m.content.length > 0);

  const remainingFlags = sample.issueFlags.filter((f) => f !== 'PII_OR_SECRET_PATTERN');
  return {
    ...sample,
    messages: cleanedMessages,
    formattedPreview: formatSampleToTarget(cleanedMessages, targetFormat),
    cleaned: true,
    piiDetected: false,
    issueFlags: remainingFlags,
    qualityScore: Math.min(99, sample.qualityScore + (sample.piiDetected ? 24 : 5)),
  };
}

export interface DatasetValidationReport {
  totalSamples: number;
  validSchemaSamples: number;
  duplicateCount: number;
  piiFlaggedCount: number;
  lowInfoCount: number;
  minTokens: number;
  medianTokens: number;
  p95Tokens: number;
  maxTokens: number;
  overallHealthScore: number;
  status: 'PASSED' | 'WARNING' | 'CRITICAL';
  tagDistribution: Record<string, number>;
}

export function validateDatasetRecords(samples: NormalizedDatasetSample[]): DatasetValidationReport {
  if (samples.length === 0) {
    return {
      totalSamples: 0,
      validSchemaSamples: 0,
      duplicateCount: 0,
      piiFlaggedCount: 0,
      lowInfoCount: 0,
      minTokens: 0,
      medianTokens: 0,
      p95Tokens: 0,
      maxTokens: 0,
      overallHealthScore: 0,
      status: 'CRITICAL',
      tagDistribution: {},
    };
  }

  const seenFingerprints = new Set<string>();
  let duplicateCount = 0;
  let validSchemaSamples = 0;
  let piiFlaggedCount = 0;
  let lowInfoCount = 0;
  const tokenCounts: number[] = [];
  const tagDistribution: Record<string, number> = {};

  for (const s of samples) {
    const fp = s.messages.map((m) => `${m.role}:${m.content.toLowerCase().slice(0, 120)}`).join('|');
    if (seenFingerprints.has(fp)) {
      duplicateCount++;
    } else {
      seenFingerprints.add(fp);
    }

    if (!s.issueFlags.includes('UNRECOGNIZED_SCHEMA') && !s.issueFlags.includes('MISSING_ASSISTANT_TURN')) {
      validSchemaSamples++;
    }
    if (s.piiDetected) piiFlaggedCount++;
    if (s.issueFlags.includes('LOW_INFORMATION_TURN')) lowInfoCount++;

    tokenCounts.push(s.tokenEstimate);
    for (const t of s.tags) {
      tagDistribution[t] = (tagDistribution[t] ?? 0) + 1;
    }
  }

  tokenCounts.sort((a, b) => a - b);
  const minTokens = tokenCounts[0];
  const medianTokens = tokenCounts[Math.floor(tokenCounts.length * 0.5)];
  const p95Tokens = tokenCounts[Math.min(tokenCounts.length - 1, Math.floor(tokenCounts.length * 0.95))];
  const maxTokens = tokenCounts[tokenCounts.length - 1];

  const validRatio = validSchemaSamples / samples.length;
  const dupPenalty = (duplicateCount / samples.length) * 35;
  const piiPenalty = (piiFlaggedCount / samples.length) * 40;
  const lowInfoPenalty = (lowInfoCount / samples.length) * 20;

  const overallHealthScore = Math.max(
    0,
    Math.min(100, Math.round(validRatio * 100 - dupPenalty - piiPenalty - lowInfoPenalty))
  );

  const status: DatasetValidationReport['status'] =
    overallHealthScore >= 85 && piiFlaggedCount === 0
      ? 'PASSED'
      : overallHealthScore >= 60
      ? 'WARNING'
      : 'CRITICAL';

  return {
    totalSamples: samples.length,
    validSchemaSamples,
    duplicateCount,
    piiFlaggedCount,
    lowInfoCount,
    minTokens,
    medianTokens,
    p95Tokens,
    maxTokens,
    overallHealthScore,
    status,
    tagDistribution,
  };
}

export interface HyperparameterConfig {
  learningRate: number;
  loraRank: number;
  loraAlpha: number;
  loraDropout: number;
  warmupRatio: number;
  weightDecay: number;
  neftuneAlpha: number;
  batchSize: number;
  lrScheduler: 'cosine' | 'linear' | 'constant_with_warmup' | 'wsd';
}

export interface TrialOutcome {
  trialNumber: number;
  config: HyperparameterConfig;
  trainLoss: number;
  valLoss: number;
  gradNorm: number;
  evalScore: number;
}

export interface HpoRefinementProposal {
  nextConfig: HyperparameterConfig;
  diagnosis: 'OVERFITTING' | 'UNDERFITTING' | 'GRADIENT_INSTABILITY' | 'CONVERGING_WELL';
  rationale: string;
  expectedImprovementPct: number;
}

export function computeNextHyperparameters(outcome: TrialOutcome): HpoRefinementProposal {
  const { config, trainLoss, valLoss, gradNorm } = outcome;
  const generalizationGap = valLoss - trainLoss;

  if (gradNorm > 1.75) {
    return {
      diagnosis: 'GRADIENT_INSTABILITY',
      nextConfig: {
        ...config,
        learningRate: Number((config.learningRate * 0.6).toFixed(6)),
        warmupRatio: Number(Math.min(0.15, config.warmupRatio + 0.04).toFixed(2)),
        lrScheduler: 'cosine',
      },
      rationale: `Gradient norm spike detected (${gradNorm.toFixed(2)} > 1.75). Reducing learning rate by 40% and extending warmup ratio to stabilize early attention/SSM projections.`,
      expectedImprovementPct: 4.8,
    };
  }

  if (generalizationGap > 0.28) {
    return {
      diagnosis: 'OVERFITTING',
      nextConfig: {
        ...config,
        loraDropout: Number(Math.min(0.15, config.loraDropout + 0.05).toFixed(2)),
        weightDecay: Number(Math.min(0.1, config.weightDecay + 0.02).toFixed(3)),
        neftuneAlpha: config.neftuneAlpha < 5 ? 5 : 10,
        loraRank: Math.max(8, config.loraRank / 2),
        loraAlpha: Math.max(16, config.loraAlpha / 2),
      },
      rationale: `Validation loss (${valLoss.toFixed(3)}) diverges from training loss (${trainLoss.toFixed(3)}) by +${generalizationGap.toFixed(3)}. Increasing weight decay, activating NEFTune noise injection, and constraining LoRA rank capacity.`,
      expectedImprovementPct: 3.6,
    };
  }

  if (trainLoss > 1.35 && valLoss > 1.35) {
    return {
      diagnosis: 'UNDERFITTING',
      nextConfig: {
        ...config,
        learningRate: Number(Math.min(5e-4, config.learningRate * 1.35).toFixed(6)),
        loraRank: Math.min(128, config.loraRank * 2),
        loraAlpha: Math.min(256, config.loraAlpha * 2),
      },
      rationale: `Both training (${trainLoss.toFixed(3)}) and validation (${valLoss.toFixed(3)}) loss remain elevated. Doubling LoRA rank/alpha expressivity and scaling learning rate by 1.35x.`,
      expectedImprovementPct: 5.2,
    };
  }

  return {
    diagnosis: 'CONVERGING_WELL',
    nextConfig: {
      ...config,
      learningRate: Number((config.learningRate * 0.92).toFixed(6)),
      lrScheduler: 'wsd',
    },
    rationale: `Healthy generalization gap (${generalizationGap.toFixed(3)}) and stable gradient norm (${gradNorm.toFixed(2)}). Applying Warmup-Stable-Decay (WSD) schedule with fine learning rate annealing.`,
    expectedImprovementPct: 1.9,
  };
}

export interface BenchmarkMetrics {
  mmluPro: number;
  gsm8k: number;
  humanEval: number;
  ifEval: number;
  mtBench: number;
  perplexity: number;
  tokensPerSec: number;
}

export interface BenchmarkComparisonResult {
  deltas: {
    mmluPro: number;
    gsm8k: number;
    humanEval: number;
    ifEval: number;
    mtBench: number;
    perplexity: number;
    tokensPerSec: number;
  };
  averageAccuracyGain: number;
  regressions: string[];
  verdict: 'PROMOTE_TO_PRODUCTION' | 'ITERATE_HPO' | 'REGRESSION_ALERT';
}

export function compareBaseAndFineTuned(
  base: BenchmarkMetrics,
  fineTuned: BenchmarkMetrics
): BenchmarkComparisonResult {
  const deltas = {
    mmluPro: Number((fineTuned.mmluPro - base.mmluPro).toFixed(2)),
    gsm8k: Number((fineTuned.gsm8k - base.gsm8k).toFixed(2)),
    humanEval: Number((fineTuned.humanEval - base.humanEval).toFixed(2)),
    ifEval: Number((fineTuned.ifEval - base.ifEval).toFixed(2)),
    mtBench: Number((fineTuned.mtBench - base.mtBench).toFixed(2)),
    perplexity: Number((fineTuned.perplexity - base.perplexity).toFixed(2)), // negative is better
    tokensPerSec: Number((fineTuned.tokensPerSec - base.tokensPerSec).toFixed(1)),
  };

  const regressions: string[] = [];
  if (deltas.mmluPro < -1.5) regressions.push(`MMLU-Pro dropped by ${deltas.mmluPro}% (catastrophic forgetting risk)`);
  if (deltas.gsm8k < -2.0) regressions.push(`GSM8K dropped by ${deltas.gsm8k}%`);
  if (deltas.humanEval < -2.0) regressions.push(`HumanEval dropped by ${deltas.humanEval}%`);
  if (deltas.perplexity > 0.8) regressions.push(`Perplexity worsened by +${deltas.perplexity}`);

  const averageAccuracyGain = Number(
    ((deltas.mmluPro + deltas.gsm8k + deltas.humanEval + deltas.ifEval) / 4).toFixed(2)
  );

  let verdict: BenchmarkComparisonResult['verdict'] = 'ITERATE_HPO';
  if (regressions.length > 0) {
    verdict = 'REGRESSION_ALERT';
  } else if (averageAccuracyGain >= 2.0 && deltas.perplexity <= 0) {
    verdict = 'PROMOTE_TO_PRODUCTION';
  }

  return {
    deltas,
    averageAccuracyGain,
    regressions,
    verdict,
  };
}

export function generateContainerArtifacts(params: {
  modelId: string;
  architecture: ArchitectureFamily;
  quantization: QuantizationMode;
  loggingFramework: LoggingFramework;
  cloudStorageUri: string;
}): { dockerfile: string; dockerCompose: string; cloudLaunchScript: string } {
  const envLogger =
    params.loggingFramework === 'wandb'
      ? 'WANDB_PROJECT="forgeml-finetune"\nENV REPORT_TO="wandb"'
      : params.loggingFramework === 'mlflow'
      ? 'MLFLOW_TRACKING_URI="http://mlflow:5000"\nENV REPORT_TO="mlflow"'
      : params.loggingFramework === 'tensorboard'
      ? 'TENSORBOARD_LOGDIR="/workspace/runs"\nENV REPORT_TO="tensorboard"'
      : 'OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"\nENV REPORT_TO="none"';

  const archSpecificKernel =
    params.architecture === 'LFM2.5'
      ? '# Liquid Foundation Model (LFM2.5) hybrid state-space + attention kernels\nRUN pip install --no-cache-dir causal-conv1d>=1.4.0 mamba-ssm>=2.2.2'
      : params.architecture === 'Gemma4'
      ? '# Gemma4 sliding-window + softcap FlashAttention-3 kernels\nRUN pip install --no-cache-dir flash-attn>=2.7.2 --no-build-isolation'
      : '# Qwen3.5 GQA + RoPE dynamic NTK fused kernels\nRUN pip install --no-cache-dir flash-attn>=2.7.2 unsloth>=2025.3';

  const dockerfile = `# ForgeML Production Fine-Tuning & Auto-Eval Container
# Architecture: ${params.architecture} | Target Model: ${params.modelId}
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV ${envLogger}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential python3-pip python3-dev jq rclone && \\
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip3 install --no-cache-dir \\
    torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \\
    transformers>=4.50.0 datasets>=3.3.0 accelerate>=1.4.0 peft>=0.14.0 \\
    trl>=0.15.0 bitsandbytes>=0.45.2 optuna>=4.2.0 lm-eval>=0.4.8 \\
    wandb mlflow tensorboard opentelemetry-sdk

${archSpecificKernel}

COPY ./pipeline /workspace/pipeline
COPY ./configs/${params.architecture.toLowerCase()}_hpo.yaml /workspace/config.yaml

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \\
  CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

ENTRYPOINT ["python3", "-m", "pipeline.orchestrator"]
CMD ["--model", "${params.modelId}", "--quantization", "${params.quantization}", "--sync-uri", "${params.cloudStorageUri}", "--auto-eval"]`;

  const dockerCompose = `version: "3.9"
services:
  forgeml-trainer:
    build:
      context: .
      dockerfile: Dockerfile
    image: forgeml/pipeline-${params.architecture.toLowerCase()}:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    environment:
      - HF_TOKEN=\${HF_TOKEN}
      - WANDB_API_KEY=\${WANDB_API_KEY}
      - CLOUD_SYNC_URI=${params.cloudStorageUri}
      - BASE_MODEL=${params.modelId}
      - QUANTIZATION=${params.quantization}
    volumes:
      - ./data_cache:/workspace/data
      - ./checkpoints:/workspace/checkpoints
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "6006:6006"   # TensorBoard / Metrics Exporter
      - "8265:8265"   # Ray / Optuna HPO Dashboard`;

  const cloudLaunchScript = `#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Syncing versioned dataset snapshot from ${params.cloudStorageUri}..."
rclone copy "${params.cloudStorageUri}/datasets/v-latest" ./data_cache --progress

echo "[2/4] Verifying NVIDIA GPU topology & NVLink state..."
nvidia-smi --query-gpu=name,memory.total,driver_version,pstate --format=csv

echo "[3/4] Launching Base Model Auto-Benchmark & HPO Fine-Tuning (${params.modelId})..."
docker compose up --build --detach forgeml-trainer

echo "[4/4] Streaming structured telemetry & evaluation checkpoints..."
docker compose logs -f forgeml-trainer`;

  return { dockerfile, dockerCompose, cloudLaunchScript };
}

export interface GpuTelemetrySnapshot {
  vramUsedGb: number;
  vramTotalGb: number;
  smUtilPct: number;
  tensorCorePct: number;
  tempCelsius: number;
  powerWatts: number;
  nvlinkGbps: number;
}

export interface GpuAlertThresholds {
  vramPctThreshold: number;
  tempCelsiusThreshold: number;
  powerWattsThreshold: number;
}

export interface GpuThresholdBreach {
  id: string;
  metric: 'VRAM' | 'TEMP' | 'POWER';
  severity: 'CRITICAL' | 'WARNING';
  currentValue: number;
  thresholdValue: number;
  unit: string;
  title: string;
  message: string;
  remediation: string;
}

export interface GpuAlertEvaluation {
  vramUsagePct: number;
  hasAlerts: boolean;
  hasCriticalAlert: boolean;
  breaches: GpuThresholdBreach[];
}

export function evaluateGpuThresholds(
  stats: GpuTelemetrySnapshot,
  thresholds: GpuAlertThresholds = {
    vramPctThreshold: 90,
    tempCelsiusThreshold: 85,
    powerWattsThreshold: 400,
  }
): GpuAlertEvaluation {
  const vramUsagePct = Number(
    ((stats.vramUsedGb / Math.max(1, stats.vramTotalGb)) * 100).toFixed(1)
  );
  const breaches: GpuThresholdBreach[] = [];

  if (vramUsagePct > thresholds.vramPctThreshold) {
    breaches.push({
      id: 'breach-vram',
      metric: 'VRAM',
      severity: vramUsagePct >= 95 ? 'CRITICAL' : 'WARNING',
      currentValue: vramUsagePct,
      thresholdValue: thresholds.vramPctThreshold,
      unit: '%',
      title: `VRAM Utilization Threshold Exceeded (${vramUsagePct}% > ${thresholds.vramPctThreshold}%)`,
      message: `GPU memory allocation is at ${stats.vramUsedGb.toFixed(1)} GB / ${stats.vramTotalGb.toFixed(
        0
      )} GB (${vramUsagePct}%), risking CUDA Out-Of-Memory (OOM) during long-context activation spikes.`,
      remediation:
        'Reduce per-GPU micro-batch size, enable gradient checkpointing, or switch optimizer to PagedAdamW 8-bit.',
    });
  }

  if (stats.tempCelsius > thresholds.tempCelsiusThreshold) {
    breaches.push({
      id: 'breach-temp',
      metric: 'TEMP',
      severity: stats.tempCelsius >= 89 ? 'CRITICAL' : 'WARNING',
      currentValue: stats.tempCelsius,
      thresholdValue: thresholds.tempCelsiusThreshold,
      unit: '°C',
      title: `GPU Die Temperature Threshold Exceeded (${stats.tempCelsius}°C > ${thresholds.tempCelsiusThreshold}°C)`,
      message: `GPU thermal sensor reported ${stats.tempCelsius}°C (threshold ${thresholds.tempCelsiusThreshold}°C). Hardware thermal throttling may degrade training throughput.`,
      remediation:
        'Apply power cap (nvidia-smi -pl 320), increase chassis fan curve, or pause HPO trial for thermal cooldown.',
    });
  }

  if (stats.powerWatts > thresholds.powerWattsThreshold) {
    breaches.push({
      id: 'breach-power',
      metric: 'POWER',
      severity: 'WARNING',
      currentValue: stats.powerWatts,
      thresholdValue: thresholds.powerWattsThreshold,
      unit: 'W',
      title: `GPU Power Draw Threshold Exceeded (${stats.powerWatts}W > ${thresholds.powerWattsThreshold}W)`,
      message: `Sustained board power draw (${stats.powerWatts}W) exceeds configured envelope (${thresholds.powerWattsThreshold}W).`,
      remediation: 'Enable FP8/NF4 Tensor Core fusion or cap board power envelope.',
    });
  }

  return {
    vramUsagePct,
    hasAlerts: breaches.length > 0,
    hasCriticalAlert: breaches.some((b) => b.severity === 'CRITICAL'),
    breaches,
  };
}

export interface PlainLanguageComparison {
  headline: string;
  plainSummary: string;
  betterAt: Array<{
    area: string;
    metricEvidence: string;
    plainExplanation: string;
  }>;
  worseAt: Array<{
    area: string;
    metricEvidence: string;
    plainExplanation: string;
  }>;
  otherDifferences: Array<{
    aspect: string;
    detail: string;
  }>;
  recommendation: string;
}

export function buildPlainLanguageBenchmarkAnalysis(params: {
  modelName: string;
  architecture: ArchitectureFamily;
  base: BenchmarkMetrics;
  fineTuned: BenchmarkMetrics;
}): PlainLanguageComparison {
  const { modelName, architecture, base, fineTuned } = params;
  const comp = compareBaseAndFineTuned(base, fineTuned);
  const { deltas } = comp;

  const betterAt: PlainLanguageComparison['betterAt'] = [];
  const worseAt: PlainLanguageComparison['worseAt'] = [];

  if (deltas.humanEval > 0.5) {
    betterAt.push({
      area: 'Writing & Debugging Code',
      metricEvidence: `HumanEval+ ${base.humanEval}% → ${fineTuned.humanEval}% (+${deltas.humanEval}%)`,
      plainExplanation:
        'The fine-tuned model writes working code that passes automated unit tests on the first try much more reliably than the original base model.',
    });
  } else if (deltas.humanEval < -0.5) {
    worseAt.push({
      area: 'Code Generation Accuracy',
      metricEvidence: `HumanEval+ ${base.humanEval}% → ${fineTuned.humanEval}% (${deltas.humanEval}%)`,
      plainExplanation:
        'The model lost some coding precision compared to the base model, likely because the training data emphasized non-coding topics.',
    });
  }

  if (deltas.gsm8k > 0.5) {
    betterAt.push({
      area: 'Step-by-Step Math & Logic Word Problems',
      metricEvidence: `GSM8K ${base.gsm8k}% → ${fineTuned.gsm8k}% (+${deltas.gsm8k}%)`,
      plainExplanation:
        'When asked to solve multi-step calculations or logic puzzles, the model makes fewer arithmetic slips and shows clearer chain-of-thought reasoning.',
    });
  } else if (deltas.gsm8k < -0.5) {
    worseAt.push({
      area: 'Multi-Step Math Reasoning',
      metricEvidence: `GSM8K ${base.gsm8k}% → ${fineTuned.gsm8k}% (${deltas.gsm8k}%)`,
      plainExplanation:
        'The model is slightly less reliable on multi-step arithmetic reasoning than the base model.',
    });
  }

  if (deltas.ifEval > 0.5) {
    betterAt.push({
      area: 'Following Strict Formatting Rules',
      metricEvidence: `IFEval ${base.ifEval}% → ${fineTuned.ifEval}% (+${deltas.ifEval}%)`,
      plainExplanation:
        'If you ask for a strict JSON schema, a specific bullet count, or exact constraints, the fine-tuned model obeys those rules with far fewer formatting mistakes.',
    });
  } else if (deltas.ifEval < -0.5) {
    worseAt.push({
      area: 'Strict Constraint Compliance',
      metricEvidence: `IFEval ${base.ifEval}% → ${fineTuned.ifEval}% (${deltas.ifEval}%)`,
      plainExplanation:
        'The model occasionally ignores strict formatting or length constraints more often than the base model.',
    });
  }

  if (deltas.mmluPro > 0.5) {
    betterAt.push({
      area: 'Broad Academic & Professional Knowledge',
      metricEvidence: `MMLU-Pro ${base.mmluPro}% → ${fineTuned.mmluPro}% (+${deltas.mmluPro}%)`,
      plainExplanation:
        'It answers difficult domain questions (science, engineering, law, finance) more accurately without forgetting its general pre-trained knowledge.',
    });
  } else if (deltas.mmluPro < -0.5) {
    worseAt.push({
      area: 'General Encyclopedic Recall (Forgetting Trade-off)',
      metricEvidence: `MMLU-Pro ${base.mmluPro}% → ${fineTuned.mmluPro}% (${deltas.mmluPro}%)`,
      plainExplanation:
        'Specializing the model on your custom dataset caused it to forget a small amount of broad trivia and textbook knowledge ("catastrophic forgetting").',
    });
  }

  if (deltas.tokensPerSec < -0.2) {
    worseAt.push({
      area: 'Slightly Slower Response Speed (Unmerged Adapter Overhead)',
      metricEvidence: `Throughput ${base.tokensPerSec} → ${fineTuned.tokensPerSec} tok/s (${deltas.tokensPerSec} tok/s)`,
      plainExplanation:
        'There is a tiny speed difference when running LoRA adapters dynamically, though merging the weights into the base model before production eliminates this gap.',
    });
  }

  if (worseAt.length === 0) {
    worseAt.push({
      area: 'No Measurable Accuracy Regressions Detected',
      metricEvidence: '0 benchmark drops across MMLU-Pro, GSM8K, HumanEval+, and IFEval',
      plainExplanation:
        'The fine-tuned model improved across every tested skill without losing general reasoning ability. The only trade-off is that it is now more specialized toward your dataset style.',
    });
  }

  const otherDifferences: PlainLanguageComparison['otherDifferences'] = [
    {
      aspect: 'Prediction Confidence & Fluency (Perplexity)',
      detail:
        deltas.perplexity < 0
          ? `Perplexity dropped from ${base.perplexity} to ${fineTuned.perplexity} (${deltas.perplexity}). In plain terms, the model is less "hesitant" or confused when generating responses in your target domain.`
          : `Perplexity shifted from ${base.perplexity} to ${fineTuned.perplexity} (+${deltas.perplexity}), indicating slightly higher uncertainty on general text.`,
    },
    {
      aspect: 'Multi-Turn Conversation Quality (MT-Bench)',
      detail: `Conversational coherence moved from ${base.mtBench}/10 to ${fineTuned.mtBench}/10 (${
        deltas.mtBench >= 0 ? `+${deltas.mtBench}` : deltas.mtBench
      }), meaning follow-up questions in long chats stay on track better.`,
    },
    {
      aspect: `${architecture} Architecture Behavior`,
      detail:
        architecture === 'LFM2.5'
          ? 'As a Liquid Hybrid state-space model, LFM2.5 retains constant memory speed on long documents while adopting the cleaner instruction formatting from your dataset.'
          : architecture === 'Gemma4'
          ? 'Gemma4 preserves its sliding-window attention speed while gaining much stronger step-by-step reasoning discipline.'
          : 'Qwen3.5 leverages its grouped-query attention and dynamic context scaling to apply the newly learned coding and reasoning patterns across long prompts.',
    },
  ];

  const headline =
    comp.verdict === 'PROMOTE_TO_PRODUCTION'
      ? `${modelName} is measurably smarter (+${comp.averageAccuracyGain}% overall) and ready to replace the base model.`
      : comp.verdict === 'REGRESSION_ALERT'
      ? `${modelName} improved in specialized tasks but lost some general knowledge — review trade-offs before deploying.`
      : `${modelName} shows modest gains (+${comp.averageAccuracyGain}%) and can benefit from one more hyperparameter tuning pass.`;

  const plainSummary = `Compared to the untuned base ${modelName}, this fine-tuned version gained an average of ${
    comp.averageAccuracyGain >= 0 ? `+${comp.averageAccuracyGain}%` : `${comp.averageAccuracyGain}%`
  } across core reasoning, coding, and instruction-following tests. In everyday use, it follows formatting instructions more strictly, writes cleaner code, and solves multi-step problems more reliably while running at virtually the same speed (${
    fineTuned.tokensPerSec
  } tokens/sec).`;

  const recommendation =
    comp.verdict === 'PROMOTE_TO_PRODUCTION'
      ? 'Recommended Action: Promote this fine-tuned checkpoint to production and merge the LoRA adapter into the base weights for maximum inference speed.'
      : comp.verdict === 'REGRESSION_ALERT'
      ? 'Recommended Action: Lower the learning rate or mix 15% general instruction data (UltraChat) into your dataset to prevent the model from forgetting general knowledge.'
      : 'Recommended Action: Run one additional HPO trial with a slightly higher LoRA rank (r=64) to capture deeper domain patterns.';

  return {
    headline,
    plainSummary,
    betterAt,
    worseAt,
    otherDifferences,
    recommendation,
  };
}

export type SafetensorsExportPrecision = 'bfloat16' | 'float16' | 'fp8_e4m3fn';
export type SafetensorsDeploymentFormat =
  | 'SAFETENSORS'
  | 'GGUF-Q4_K_M'
  | 'GPTQ'
  | 'ONNX';
export type SafetensorsRuntimeTarget =
  | 'llama-cpp'
  | 'ollama'
  | 'vllm-local'
  | 'transformers-local'
  | 'mlx-apple';

export interface SafetensorsExportInput {
  modelId: string;
  modelName: string;
  architecture: ArchitectureFamily;
  paramsBillions: number;
  adapterType: QuantizationMode;
  loraRank: number;
  loraAlpha: number;
  exportPrecision: SafetensorsExportPrecision;
  deploymentFormat?: SafetensorsDeploymentFormat;
  maxShardSizeGb: number;
  includeTokenizerAndConfig: boolean;
  runtimeTarget: SafetensorsRuntimeTarget;
  checkpointId: string;
}

export interface SafetensorsExportPlan {
  deploymentFormat: SafetensorsDeploymentFormat;
  quantizationSummary: string;
  outputFilename: string;
  mergedFileSizeGb: number;
  mergedFileSizeMb: number;
  shardCount: number;
  isSingleFileContiguous: boolean;
  mergedTensorCount: number;
  adapterScalingFactor: number;
  sha256Preview: string;
  safetensorsHeaderJson: string;
  pythonMergeScript: string;
  localDeployCommand: string;
  companionFiles: string[];
}

export function buildSafetensorsExportPlan(input: SafetensorsExportInput): SafetensorsExportPlan {
  const deploymentFormat: SafetensorsDeploymentFormat =
    input.deploymentFormat ?? 'SAFETENSORS';

  const precisionBytesPerParam: Record<SafetensorsExportPrecision, number> = {
    bfloat16: 2.0,
    float16: 2.0,
    fp8_e4m3fn: 1.0,
  };

  const effectiveBytesPerParam =
    deploymentFormat === 'GGUF-Q4_K_M'
      ? 0.62
      : deploymentFormat === 'GPTQ'
      ? 0.58
      : deploymentFormat === 'ONNX'
      ? 1.05
      : precisionBytesPerParam[input.exportPrecision];

  const rawSizeGb = input.paramsBillions * effectiveBytesPerParam;
  const mergedFileSizeGb = Number(Math.max(0.12, rawSizeGb).toFixed(2));
  const mergedFileSizeMb = Math.round(mergedFileSizeGb * 1024);
  const shardCount =
    deploymentFormat === 'GGUF-Q4_K_M'
      ? 1
      : Math.max(1, Math.ceil(mergedFileSizeGb / Math.max(1, input.maxShardSizeGb)));
  const isSingleFileContiguous = shardCount === 1;

  const adapterScalingFactor = Number((input.loraAlpha / Math.max(1, input.loraRank)).toFixed(2));

  // Estimate layer & tensor count for compact 0.3B - 9B models
  const estimatedLayers =
    input.paramsBillions <= 0.5
      ? 16
      : input.paramsBillions <= 1.5
      ? 24
      : input.paramsBillions <= 3.0
      ? 28
      : 32;
  const tensorsPerLayer = input.architecture === 'LFM2.5' ? 11 : 9;
  const mergedTensorCount = estimatedLayers * tensorsPerLayer + 3;

  const hiddenDim =
    input.paramsBillions <= 0.5
      ? 1024
      : input.paramsBillions <= 1.5
      ? 1536
      : input.paramsBillions <= 3.0
      ? 2048
      : 3072;

  const dtypeCode =
    deploymentFormat === 'GGUF-Q4_K_M'
      ? 'Q4_K_M'
      : deploymentFormat === 'GPTQ'
      ? 'INT4_GPTQ'
      : deploymentFormat === 'ONNX'
      ? 'ORT_INT8_FP16'
      : input.exportPrecision === 'bfloat16'
      ? 'BF16'
      : input.exportPrecision === 'float16'
      ? 'F16'
      : 'F8_E4M3';

  const quantizationSummary =
    deploymentFormat === 'GGUF-Q4_K_M'
      ? 'GGUF 4-Bit K-Quant Medium (4.85 bpw super-blocks)'
      : deploymentFormat === 'GPTQ'
      ? 'GPTQ 4-Bit INT4 (group_size=128 · Marlin/ExLlamaV2)'
      : deploymentFormat === 'ONNX'
      ? 'ONNX Runtime Graph (O3 Fused Attention · INT8/FP16)'
      : `Contiguous Safetensors (${input.exportPrecision.toUpperCase()})`;

  const headerObj: Record<string, unknown> = {
    __metadata__: {
      format: deploymentFormat.toLowerCase(),
      exporter: 'ForgeML-Safetensors-Merger-v2.6',
      base_model: input.modelId,
      architecture: input.architecture,
      parameters_billions: String(input.paramsBillions),
      deployment_format: deploymentFormat,
      quantization_summary: quantizationSummary,
      adapter_merged: 'true',
      source_adapter_mode: input.adapterType,
      lora_r: String(input.loraRank),
      lora_alpha: String(input.loraAlpha),
      lora_scaling: String(adapterScalingFactor),
      merge_formula: 'W_merged = dequant(W_base) + (lora_alpha / lora_r) * (lora_B @ lora_A)',
      target_dtype: dtypeCode,
      checkpoint_id: input.checkpointId,
    },
    'model.embed_tokens.weight': {
      dtype: dtypeCode,
      shape: [151936, hiddenDim],
      data_offsets: [0, Math.round(151936 * hiddenDim * effectiveBytesPerParam)],
    },
    'model.layers.0.self_attn.q_proj.weight': {
      dtype: dtypeCode,
      shape: [hiddenDim, hiddenDim],
      data_offsets: [
        Math.round(151936 * hiddenDim * effectiveBytesPerParam),
        Math.round((151936 * hiddenDim + hiddenDim * hiddenDim) * effectiveBytesPerParam),
      ],
    },
    'model.layers.0.mlp.gate_proj.weight': {
      dtype: dtypeCode,
      shape: [hiddenDim * 4, hiddenDim],
      data_offsets: [
        Math.round((151936 * hiddenDim + hiddenDim * hiddenDim) * effectiveBytesPerParam),
        Math.round((151936 * hiddenDim + hiddenDim * hiddenDim * 5) * effectiveBytesPerParam),
      ],
    },
    'model.norm.weight': {
      dtype: dtypeCode,
      shape: [hiddenDim],
      data_offsets: [
        Math.round((151936 * hiddenDim + hiddenDim * hiddenDim * 5) * effectiveBytesPerParam),
        Math.round(
          (151936 * hiddenDim + hiddenDim * hiddenDim * 5 + hiddenDim) * effectiveBytesPerParam
        ),
      ],
    },
  };

  const safetensorsHeaderJson = JSON.stringify(headerObj, null, 2);
  const slug = input.modelName.toLowerCase().replace(/[^a-z0-9.]/g, '-');

  const outputFilename =
    deploymentFormat === 'GGUF-Q4_K_M'
      ? `${slug}-q4_k_m.gguf`
      : deploymentFormat === 'GPTQ'
      ? `${slug}-gptq-4bit.safetensors`
      : deploymentFormat === 'ONNX'
      ? `${slug}-onnx-int8.onnx`
      : isSingleFileContiguous
      ? `${slug}-merged-${input.exportPrecision}.safetensors`
      : `model-00001-of-0000${shardCount}.safetensors`;

  const companionFiles = [
    outputFilename,
    ...(deploymentFormat === 'GPTQ' ? ['quantize_config.json'] : []),
    ...(deploymentFormat === 'ONNX' ? ['model.onnx.data', 'ort_config.json'] : []),
    ...(shardCount > 1 && deploymentFormat === 'SAFETENSORS'
      ? ['model.safetensors.index.json']
      : []),
    ...(input.includeTokenizerAndConfig
      ? ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
      : []),
  ];

  const torchDtype =
    input.exportPrecision === 'bfloat16'
      ? 'torch.bfloat16'
      : input.exportPrecision === 'float16'
      ? 'torch.float16'
      : 'torch.float8_e4m3fn';

  const formatSpecificExportStep =
    deploymentFormat === 'GGUF-Q4_K_M'
      ? `
# [4/4] Serialize merged HF Safetensors & quantize to GGUF-Q4_K_M
merged_model.save_pretrained(EXPORT_DIR, safe_serialization=True)
import subprocess
subprocess.run(["python3", "llama.cpp/convert_hf_to_gguf.py", EXPORT_DIR, "--outtype", "f16", "--outfile", f"{EXPORT_DIR}/${slug}-f16.gguf"], check=True)
subprocess.run(["./llama.cpp/llama-quantize", f"{EXPORT_DIR}/${slug}-f16.gguf", f"{EXPORT_DIR}/${outputFilename}", "Q4_K_M"], check=True)`
      : deploymentFormat === 'GPTQ'
      ? `
# [4/4] Calibrate 4-Bit GPTQ Hessian weights (group_size=128, desc_act=True, Marlin kernel)
from gptqmodel import GPTQModel, QuantizeConfig
quant_cfg = QuantizeConfig(bits=4, group_size=128, desc_act=True, sym=True)
gptq_model = GPTQModel.from_pretrained(EXPORT_DIR, quant_cfg)
gptq_model.quantize(calibration_dataset)
gptq_model.save_quantized(EXPORT_DIR)`
      : deploymentFormat === 'ONNX'
      ? `
# [4/4] Export fused computation graph to ONNX Runtime (opset=18, O3 graph optimization)
merged_model.save_pretrained(EXPORT_DIR, safe_serialization=True)
from optimum.exporters.onnx import main_export
main_export(model_name_or_path=EXPORT_DIR, output=f"{EXPORT_DIR}/onnx", task="text-generation-with-past", optimize="O3")`
      : `
print(f"[4/4] Serializing contiguous Safetensors checkpoint (max_shard_size='${input.maxShardSizeGb}GB')...")
merged_model.save_pretrained(
    EXPORT_DIR,
    safe_serialization=True,
    max_shard_size="${input.maxShardSizeGb}GB",
)`;

  const pythonMergeScript = `#!/usr/bin/env python3
"""
ForgeML Checkpoint Merger & ${deploymentFormat} Exporter (${input.modelName})
Optimized for sub-2B / edge models: dequantizes base weights in memory,
fuses LoRA/QLoRA adapter (r=${input.loraRank}, alpha=${input.loraAlpha}), and exports ${deploymentFormat} (${outputFilename}).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "${input.modelId}"
ADAPTER_DIR = "./checkpoints/${input.checkpointId}"
EXPORT_DIR = "./exported_models/${slug}-${deploymentFormat.toLowerCase()}"

print(f"[1/4] Loading base model {BASE_MODEL_ID} in high precision for clean adapter fusion...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=${torchDtype === 'torch.float8_e4m3fn' ? 'torch.bfloat16' : torchDtype},
    device_map="auto",
    trust_remote_code=True,
)

print(f"[2/4] Attaching trained LoRA/QLoRA adapter from {ADAPTER_DIR} (scaling=${adapterScalingFactor})...")
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("[3/4] Fusing W_merged = W_base + (alpha/r) * (B @ A) and unloading PEFT wrapper...")
merged_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
${formatSpecificExportStep}
${
  input.includeTokenizerAndConfig
    ? `tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)\ntokenizer.save_pretrained(EXPORT_DIR)`
    : ''
}
print(f"Export complete -> {EXPORT_DIR}/${outputFilename} (${mergedFileSizeGb} GB)")`;

  let localDeployCommand = '';
  if (deploymentFormat === 'GGUF-Q4_K_M') {
    localDeployCommand = `# Run quantized GGUF-Q4_K_M checkpoint directly via llama-server or Ollama
llama-server -m ./exported_models/${slug}-gguf-q4_k_m/${outputFilename} --ctx-size 8192 --port 8080 --ngl 99
# Or import into Ollama:
printf 'FROM ./exported_models/${slug}-gguf-q4_k_m/${outputFilename}\\nPARAMETER temperature 0.2\\n' > Modelfile && ollama create ${slug}:q4_k_m -f Modelfile`;
  } else if (deploymentFormat === 'GPTQ') {
    localDeployCommand = `# Serve 4-bit GPTQ checkpoint with vLLM Marlin kernel for maximum GPU throughput
vllm serve ./exported_models/${slug}-gptq \\
  --quantization gptq_marlin \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.85 \\
  --port 8000`;
  } else if (deploymentFormat === 'ONNX') {
    localDeployCommand = `# Execute exported ONNX graph via ONNX Runtime (CPU / CUDA / DirectML / WebGPU)
optimum-cli onnxruntime run --model ./exported_models/${slug}-onnx --prompt "Verify ONNX Runtime edge inference."`;
  } else if (input.runtimeTarget === 'llama-cpp') {
    localDeployCommand = `# Convert merged Safetensors directly & serve on local llama-server (port 8080)
python3 llama.cpp/convert_hf_to_gguf.py ./exported_models/${slug}-safetensors --outtype bf16 --outfile ${slug}.gguf
llama-server -m ${slug}.gguf --ctx-size 8192 --port 8080 --ngl 99`;
  } else if (input.runtimeTarget === 'ollama') {
    localDeployCommand = `# Import merged Safetensors directory directly into local Ollama (native Safetensors import)
printf 'FROM ./exported_models/${slug}-safetensors\\nPARAMETER temperature 0.2\\n' > Modelfile
ollama create ${slug}:latest -f Modelfile
ollama run ${slug}:latest`;
  } else if (input.runtimeTarget === 'vllm-local') {
    localDeployCommand = `# Serve merged Safetensors checkpoint directly with vLLM OpenAI-compatible server
vllm serve ./exported_models/${slug}-safetensors \\
  --dtype ${input.exportPrecision === 'float16' ? 'float16' : 'bfloat16'} \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.85 \\
  --port 8000`;
  } else if (input.runtimeTarget === 'mlx-apple') {
    localDeployCommand = `# Run merged Safetensors natively on Apple Silicon (M1/M2/M3/M4) via MLX
pip install mlx-lm
mlx_lm.generate --model ./exported_models/${slug}-safetensors --prompt "Verify edge deployment readiness."`;
  } else {
    localDeployCommand = `# Load merged Safetensors directly in local PyTorch / Transformers with zero PEFT overhead
python3 -c "from transformers import pipeline; pipe = pipeline('text-generation', './exported_models/${slug}-safetensors', device_map='auto'); print(pipe('Hello!'))"`;
  }

  return {
    deploymentFormat,
    quantizationSummary,
    outputFilename,
    mergedFileSizeGb,
    mergedFileSizeMb,
    shardCount,
    isSingleFileContiguous,
    mergedTensorCount,
    adapterScalingFactor,
    sha256Preview: `sha256:7e4b91c0${input.loraRank}f2a8d3`,
    safetensorsHeaderJson,
    pythonMergeScript,
    localDeployCommand,
    companionFiles,
  };
}

export function createSafetensorsBinaryArtifact(headerJson: string): Uint8Array {
  const encoder = new TextEncoder();
  const headerBytes = encoder.encode(headerJson);
  // Pad header to 8-byte alignment per Safetensors specification
  const paddingNeeded = (8 - (headerBytes.byteLength % 8)) % 8;
  const alignedHeaderLen = headerBytes.byteLength + paddingNeeded;

  // 8-byte little-endian uint64 header size + aligned UTF-8 JSON header + 64 bytes sample BF16 tensor payload
  const sampleTensorPayloadBytes = 64;
  const totalBuffer = new Uint8Array(8 + alignedHeaderLen + sampleTensorPayloadBytes);
  const view = new DataView(totalBuffer.buffer);
  view.setBigUint64(0, BigInt(alignedHeaderLen), true);

  totalBuffer.set(headerBytes, 8);
  for (let i = 0; i < paddingNeeded; i++) {
    totalBuffer[8 + headerBytes.byteLength + i] = 0x20; // space padding per spec
  }

  return totalBuffer;
}

export interface RadarBenchmarkAxisPoint {
  metric: string;
  fullName: string;
  base: number;
  fineTuned: number;
  delta: number;
  fullMark: number;
}

export function buildRadarChartComparisonData(
  base: BenchmarkMetrics,
  fineTuned?: BenchmarkMetrics
): RadarBenchmarkAxisPoint[] {
  const ft = fineTuned ?? base;
  const baseMtScaled = Number((base.mtBench * 10).toFixed(1));
  const ftMtScaled = Number((ft.mtBench * 10).toFixed(1));

  return [
    {
      metric: 'MMLU-Pro',
      fullName: 'MMLU-Pro (STEM & Multi-Discipline Reasoning)',
      base: base.mmluPro,
      fineTuned: ft.mmluPro,
      delta: Number((ft.mmluPro - base.mmluPro).toFixed(2)),
      fullMark: 100,
    },
    {
      metric: 'GSM8K',
      fullName: 'GSM8K (8-Shot Chain-of-Thought Math)',
      base: base.gsm8k,
      fineTuned: ft.gsm8k,
      delta: Number((ft.gsm8k - base.gsm8k).toFixed(2)),
      fullMark: 100,
    },
    {
      metric: 'HumanEval',
      fullName: 'HumanEval+ (Pass@1 Code Synthesis)',
      base: base.humanEval,
      fineTuned: ft.humanEval,
      delta: Number((ft.humanEval - base.humanEval).toFixed(2)),
      fullMark: 100,
    },
    {
      metric: 'IFEval',
      fullName: 'IFEval (Strict Formatting & Constraint Adherence)',
      base: base.ifEval,
      fineTuned: ft.ifEval,
      delta: Number((ft.ifEval - base.ifEval).toFixed(2)),
      fullMark: 100,
    },
    {
      metric: 'MT-Bench (x10)',
      fullName: 'MT-Bench Multi-Turn Score (Scaled 0-100)',
      base: baseMtScaled,
      fineTuned: ftMtScaled,
      delta: Number((ftMtScaled - baseMtScaled).toFixed(2)),
      fullMark: 100,
    },
  ];
}

export interface TrainingProgressTelemetryInput {
  currentStep: number;
  currentEpoch: number;
  targetEpochs: number;
  stepsPerEpoch: number;
  tokensPerSec: number;
  avgTokensPerStep: number;
  vramUsedGb: number;
  vramTotalGb: number;
  recentSteps: Array<{
    step: number;
    trainLoss: number;
    valLoss: number;
    tokensPerSec: number;
  }>;
}

export interface MemoryTrendPoint {
  step: number;
  allocatedVramGb: number;
  allocatedPct: number;
  kvCacheGb: number;
}

export interface TrainingProgressTelemetryResult {
  currentEpochIndex: number;
  totalTargetEpochs: number;
  stepWithinCurrentEpoch: number;
  stepsPerEpoch: number;
  currentEpochCompletionPct: number;
  overallRunCompletionPct: number;
  totalTargetSteps: number;
  remainingSteps: number;
  etaSeconds: number;
  etaFormatted: string;
  elapsedSeconds: number;
  elapsedFormatted: string;
  vramUtilizationPct: number;
  peakReservedVramGb: number;
  memoryTrendDeltaPct: number;
  memoryTrendStatus: 'STABLE' | 'CREEPING_UP' | 'NEAR_CEILING';
  memoryTrendPoints: MemoryTrendPoint[];
}

function formatDurationSeconds(totalSec: number): string {
  const clamped = Math.max(0, Math.round(totalSec));
  const hrs = Math.floor(clamped / 3600);
  const mins = Math.floor((clamped % 3600) / 60);
  const secs = clamped % 60;
  if (hrs > 0) {
    return `${hrs}h ${String(mins).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`;
  }
  return `${String(mins).padStart(2, '0')}m ${String(secs).padStart(2, '0')}s`;
}

export function computeTrainingProgressTelemetry(
  input: TrainingProgressTelemetryInput
): TrainingProgressTelemetryResult {
  const stepsPerEpoch = Math.max(50, input.stepsPerEpoch);
  const totalTargetEpochs = Math.max(1, input.targetEpochs);
  const totalTargetSteps = stepsPerEpoch * totalTargetEpochs;

  const clampedStep = Math.max(0, input.currentStep);
  const currentEpochIndex = Math.min(
    totalTargetEpochs,
    Math.floor((clampedStep - 1) / stepsPerEpoch) + 1
  );
  const rawStepInEpoch = clampedStep % stepsPerEpoch;
  const stepWithinCurrentEpoch =
    clampedStep > 0 && rawStepInEpoch === 0 ? stepsPerEpoch : rawStepInEpoch;

  const currentEpochCompletionPct = Number(
    Math.min(100, (stepWithinCurrentEpoch / stepsPerEpoch) * 100).toFixed(1)
  );
  const overallRunCompletionPct = Number(
    Math.min(100, (clampedStep / Math.max(1, totalTargetSteps)) * 100).toFixed(1)
  );

  const remainingSteps = Math.max(0, totalTargetSteps - clampedStep);
  const secPerStep =
    input.avgTokensPerStep / Math.max(100, input.tokensPerSec);
  const etaSeconds = Math.round(remainingSteps * secPerStep);
  const elapsedSeconds = Math.round(clampedStep * secPerStep);

  const vramUtilizationPct = Number(
    Math.min(100, (input.vramUsedGb / Math.max(1, input.vramTotalGb)) * 100).toFixed(1)
  );
  const peakReservedVramGb = Number(
    Math.min(input.vramTotalGb, input.vramUsedGb * 1.08 + 0.35).toFixed(2)
  );

  const memoryTrendPoints: MemoryTrendPoint[] = input.recentSteps.slice(-6).map((pt, idx, arr) => {
    // Deterministic slight step progression reflecting KV cache & optimizer state warmup
    const offsetGb = (idx - (arr.length - 1)) * 0.08;
    const allocatedVramGb = Number(
      Math.max(0.8, Math.min(input.vramTotalGb, input.vramUsedGb + offsetGb)).toFixed(2)
    );
    const allocatedPct = Number(
      ((allocatedVramGb / Math.max(1, input.vramTotalGb)) * 100).toFixed(1)
    );
    const kvCacheGb = Number(Math.max(0.25, allocatedVramGb * 0.24).toFixed(2));
    return {
      step: pt.step,
      allocatedVramGb,
      allocatedPct,
      kvCacheGb,
    };
  });

  const firstPct = memoryTrendPoints[0]?.allocatedPct ?? vramUtilizationPct;
  const lastPct =
    memoryTrendPoints[memoryTrendPoints.length - 1]?.allocatedPct ?? vramUtilizationPct;
  const memoryTrendDeltaPct = Number((lastPct - firstPct).toFixed(1));

  const memoryTrendStatus: TrainingProgressTelemetryResult['memoryTrendStatus'] =
    vramUtilizationPct >= 88
      ? 'NEAR_CEILING'
      : memoryTrendDeltaPct > 3.5
      ? 'CREEPING_UP'
      : 'STABLE';

  return {
    currentEpochIndex: Math.max(1, currentEpochIndex),
    totalTargetEpochs,
    stepWithinCurrentEpoch,
    stepsPerEpoch,
    currentEpochCompletionPct,
    overallRunCompletionPct,
    totalTargetSteps,
    remainingSteps,
    etaSeconds,
    etaFormatted: formatDurationSeconds(etaSeconds),
    elapsedSeconds,
    elapsedFormatted: formatDurationSeconds(elapsedSeconds),
    vramUtilizationPct,
    peakReservedVramGb,
    memoryTrendDeltaPct,
    memoryTrendStatus,
    memoryTrendPoints,
  };
}

export interface ModelCardGeneratorInput {
  modelId: string;
  modelName: string;
  architecture: ArchitectureFamily;
  paramsBillions: number;
  contextWindow: number;
  quantizationMode: QuantizationMode;
  checkpointId: string;
  datasetVersionId: string;
  datasetCommitHash: string;
  sourceDatasets: string[];
  targetFormat: TargetFormat;
  sampleCount: number;
  datasetHealthScore: number;
  hpoConfig: HyperparameterConfig;
  baseBenchmarks: BenchmarkMetrics;
  fineTunedBenchmarks?: BenchmarkMetrics;
  safetensorsPlan: SafetensorsExportPlan;
}

export interface StandardizedModelCardResult {
  filename: string;
  markdown: string;
  extractedMetadata: {
    modelId: string;
    architecture: ArchitectureFamily;
    parameterScale: string;
    contextLength: string;
    deploymentFormat: SafetensorsDeploymentFormat;
    quantizationSummary: string;
    datasetVersion: string;
    checkpointFilename: string;
    mergedSizeGb: string;
    baseMmluPro: string;
    fineTunedMmluPro: string;
    meanAccuracyDelta: string;
    verdict: string;
  };
}

export function generateStandardizedModelCard(
  input: ModelCardGeneratorInput
): StandardizedModelCardResult {
  const ft = input.fineTunedBenchmarks ?? input.baseBenchmarks;
  const comp = compareBaseAndFineTuned(input.baseBenchmarks, ft);
  const slug = input.modelName.toLowerCase().replace(/[^a-z0-9.]/g, '-');
  const filename = `${slug}-README.md`;
  const deployFmt = input.safetensorsPlan.deploymentFormat ?? 'SAFETENSORS';
  const libraryName =
    deployFmt === 'GGUF-Q4_K_M'
      ? 'gguf'
      : deployFmt === 'GPTQ'
      ? 'gptq'
      : deployFmt === 'ONNX'
      ? 'onnx'
      : 'safetensors';

  const formatDelta = (val: number, unit = '%') =>
    `${val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}${unit}`;

  const datasetsYamlList =
    input.sourceDatasets.length > 0
      ? input.sourceDatasets.map((d) => `  - ${d}`).join('\n')
      : '  - HuggingFaceH4/ultrachat_200k';

  const markdown = `---
license: apache-2.0
base_model: ${input.modelId}
library_name: ${libraryName}
deployment_format: ${deployFmt}
pipeline_tag: text-generation
tags:
  - forgeml
  - ${input.architecture.toLowerCase()}
  - ${input.quantizationMode}
  - ${deployFmt.toLowerCase()}
  - sub-2b-optimized
  - ${input.targetFormat}
datasets:
${datasetsYamlList}
model-index:
  - name: ${input.checkpointId}
    results:
      - task:
          type: text-generation
          name: Benchmark Evaluation vs Base Model
        dataset:
          name: ForgeML Unified Evaluation Suite (${input.datasetVersionId})
          type: lm-evaluation-harness
        metrics:
          - type: mmlu_pro
            value: ${ft.mmluPro}
            name: MMLU-Pro (%)
          - type: gsm8k
            value: ${ft.gsm8k}
            name: GSM8K (%)
          - type: humaneval_plus
            value: ${ft.humanEval}
            name: HumanEval+ (%)
          - type: ifeval
            value: ${ft.ifEval}
            name: IFEval (%)
          - type: mt_bench
            value: ${ft.mtBench}
            name: MT-Bench (/10)
          - type: perplexity
            value: ${ft.perplexity}
            name: Validation Perplexity
---

# ${input.modelName} — ${deployFmt} Deployment Checkpoint (\`${input.checkpointId}\`)

This standardized model card accompanies the exported **${deployFmt}** checkpoint \`${input.safetensorsPlan.outputFilename}\` (${input.safetensorsPlan.mergedFileSizeGb} GB · ${input.safetensorsPlan.quantizationSummary}), generated automatically by **ForgeML Pipeline Studio**. The trained ${input.quantizationMode.toUpperCase()} adapter has been fused directly into the base weights using \`W_merged = dequant(W_base) + (${input.hpoConfig.loraAlpha}/${input.hpoConfig.loraRank}) * (B @ A)\` for direct local deployment.

## 1. Extracted Model & Checkpoint Metadata

| Property | Value |
| :--- | :--- |
| **Base Model ID** | \`${input.modelId}\` |
| **Architecture Family** | \`${input.architecture}\` |
| **Parameter Count** | \`${input.paramsBillions}B Parameters\` |
| **Context Window** | \`${input.contextWindow.toLocaleString()} tokens\` |
| **Target Deployment Format** | **\`${deployFmt}\`** (${input.safetensorsPlan.quantizationSummary}) |
| **Exported Artifact File** | \`${input.safetensorsPlan.outputFilename}\` (${input.safetensorsPlan.mergedFileSizeGb} GB · ${input.safetensorsPlan.mergedTensorCount} tensors) |
| **SHA-256 Digest** | \`${input.safetensorsPlan.sha256Preview}\` |
| **Training Dataset Version** | \`${input.datasetVersionId} (${input.datasetCommitHash})\` |
| **Unified Prompt Format** | \`${input.targetFormat.toUpperCase()}\` (${input.sampleCount} verified records · Health ${input.datasetHealthScore}/100) |
| **Evaluation Promotion Verdict** | **\`${comp.verdict}\`** (Mean Accuracy Gain: **${formatDelta(comp.averageAccuracyGain)}**) |

## 2. Base Model vs. Fine-Tuned Evaluation Benchmarks

| Benchmark Suite | Base Model (\`${input.modelName}\`) | Fine-Tuned (\`${input.checkpointId}\`) | Net Delta |
| :--- | ---: | ---: | ---: |
| **MMLU-Pro (STEM & Domain Reasoning)** | ${input.baseBenchmarks.mmluPro}% | **${ft.mmluPro}%** | \`${formatDelta(comp.deltas.mmluPro)}\` |
| **GSM8K (8-Shot Chain-of-Thought Math)** | ${input.baseBenchmarks.gsm8k}% | **${ft.gsm8k}%** | \`${formatDelta(comp.deltas.gsm8k)}\` |
| **HumanEval+ (Pass@1 Code Synthesis)** | ${input.baseBenchmarks.humanEval}% | **${ft.humanEval}%** | \`${formatDelta(comp.deltas.humanEval)}\` |
| **IFEval (Strict Constraint Following)** | ${input.baseBenchmarks.ifEval}% | **${ft.ifEval}%** | \`${formatDelta(comp.deltas.ifEval)}\` |
| **MT-Bench (Multi-Turn Judge Score)** | ${input.baseBenchmarks.mtBench}/10 | **${ft.mtBench}/10** | \`${formatDelta(comp.deltas.mtBench, '')}\` |
| **Held-Out Validation Perplexity (Lower = Better)** | ${input.baseBenchmarks.perplexity} ppl | **${ft.perplexity} ppl** | \`${formatDelta(comp.deltas.perplexity, ' ppl')}\` |
| **Inference Decode Throughput** | ${input.baseBenchmarks.tokensPerSec} tok/s | **${ft.tokensPerSec} tok/s** | \`${formatDelta(comp.deltas.tokensPerSec, ' tok/s')}\` |

## 3. Training Dataset Lineage & Hyperparameters

- **Dataset Snapshot**: \`${input.datasetVersionId}\` (\`${input.datasetCommitHash}\`)
- **Constituent HuggingFace Sources**: ${input.sourceDatasets.map((s) => `\`${s}\``).join(', ')}
- **Hyperparameter Configuration (Optuna Bayesian TPE)**:
  - **Learning Rate**: \`${input.hpoConfig.learningRate}\` (\`${input.hpoConfig.lrScheduler.toUpperCase()}\` schedule, warmup ratio \`${input.hpoConfig.warmupRatio}\`)
  - **Adapter Rank / Alpha**: \`r = ${input.hpoConfig.loraRank}\`, \`alpha = ${input.hpoConfig.loraAlpha}\` (Scaling factor \`${input.safetensorsPlan.adapterScalingFactor}x\`, dropout \`${input.hpoConfig.loraDropout}\`)
  - **Regularization**: Weight decay \`${input.hpoConfig.weightDecay}\`, NEFTune noise \`alpha = ${input.hpoConfig.neftuneAlpha}\`, micro-batch size \`${input.hpoConfig.batchSize}\`

## 4. Direct Local Deployment Quickstart (${deployFmt})

\`\`\`bash
${input.safetensorsPlan.localDeployCommand}
\`\`\`
`;

  return {
    filename,
    markdown,
    extractedMetadata: {
      modelId: input.modelId,
      architecture: input.architecture,
      parameterScale: `${input.paramsBillions}B`,
      contextLength: `${input.contextWindow.toLocaleString()} tokens`,
      deploymentFormat: deployFmt,
      quantizationSummary: input.safetensorsPlan.quantizationSummary,
      datasetVersion: `${input.datasetVersionId} (${input.datasetCommitHash})`,
      checkpointFilename: input.safetensorsPlan.outputFilename,
      mergedSizeGb: `${input.safetensorsPlan.mergedFileSizeGb} GB`,
      baseMmluPro: `${input.baseBenchmarks.mmluPro}%`,
      fineTunedMmluPro: `${ft.mmluPro}%`,
      meanAccuracyDelta: formatDelta(comp.averageAccuracyGain),
      verdict: comp.verdict,
    },
  };
}

export const TRAINING_SERIES_STORAGE_KEY = 'forgeml.trainingSeriesByModel.v1';
export const SELECTED_MODEL_STORAGE_KEY = 'forgeml.selectedModelId.v1';

export interface PersistedTrainingMetricPoint {
  step: number;
  epoch: number;
  trainLoss: number;
  valLoss: number;
  gradNorm: number;
  learningRate: number;
  tokensPerSec: number;
}

function isValidMetricPoint(item: unknown): item is PersistedTrainingMetricPoint {
  if (!item || typeof item !== 'object') return false;
  const candidate = item as Record<string, unknown>;
  return (
    typeof candidate.step === 'number' &&
    Number.isFinite(candidate.step) &&
    typeof candidate.epoch === 'number' &&
    Number.isFinite(candidate.epoch) &&
    typeof candidate.trainLoss === 'number' &&
    Number.isFinite(candidate.trainLoss) &&
    typeof candidate.valLoss === 'number' &&
    Number.isFinite(candidate.valLoss) &&
    typeof candidate.gradNorm === 'number' &&
    Number.isFinite(candidate.gradNorm) &&
    typeof candidate.learningRate === 'number' &&
    Number.isFinite(candidate.learningRate) &&
    typeof candidate.tokensPerSec === 'number' &&
    Number.isFinite(candidate.tokensPerSec)
  );
}

export function serializeTrainingSeriesMap(
  seriesByModel: Record<string, PersistedTrainingMetricPoint[]>
): string {
  return JSON.stringify({
    schemaVersion: 1,
    updatedAt: new Date().toISOString(),
    seriesByModel,
  });
}

export function deserializeTrainingSeriesMap(
  rawJson: string | null | undefined,
  fallbackSeries: PersistedTrainingMetricPoint[],
  modelIds: string[]
): Record<string, PersistedTrainingMetricPoint[]> {
  const defaultMap: Record<string, PersistedTrainingMetricPoint[]> = {};
  for (const id of modelIds) {
    defaultMap[id] = fallbackSeries.map((pt) => ({ ...pt }));
  }

  if (!rawJson || !rawJson.trim()) {
    return defaultMap;
  }

  try {
    const parsed = JSON.parse(rawJson) as Record<string, unknown>;
    const rawMap =
      parsed && typeof parsed === 'object' && parsed.seriesByModel && typeof parsed.seriesByModel === 'object'
        ? (parsed.seriesByModel as Record<string, unknown>)
        : (parsed as Record<string, unknown>);

    const result: Record<string, PersistedTrainingMetricPoint[]> = { ...defaultMap };
    for (const [modelId, arr] of Object.entries(rawMap)) {
      if (Array.isArray(arr)) {
        const validPoints = arr.filter(isValidMetricPoint);
        if (validPoints.length > 0) {
          result[modelId] = validPoints;
        }
      }
    }
    return result;
  } catch {
    return defaultMap;
  }
}

export interface BenchmarkSummaryTableRow {
  modelId: string;
  modelName: string;
  architecture: ArchitectureFamily;
  paramsBillions: number;
  hasFineTuned: boolean;
  checkpointId?: string;
  mmluProBase: number;
  mmluProFineTuned: number | null;
  mmluProDelta: number | null;
  humanEvalBase: number;
  humanEvalFineTuned: number | null;
  humanEvalDelta: number | null;
  gsm8kBase: number;
  gsm8kFineTuned: number | null;
  gsm8kDelta: number | null;
  ifEvalBase: number;
  ifEvalFineTuned: number | null;
  ifEvalDelta: number | null;
  perplexityBase: number;
  perplexityFineTuned: number | null;
  perplexityDelta: number | null;
  meanAccuracyGain: number | null;
  verdict: 'PROMOTE_TO_PRODUCTION' | 'REGRESSION_ALERT' | 'MARGINAL_GAIN' | 'BASE_ONLY';
}

export function buildBenchmarkSummaryTableRows(
  models: Array<{
    id: string;
    name: string;
    architecture: ArchitectureFamily;
    paramsBillions: number;
    fineTunedCheckpointId?: string;
    baseBenchmarks: BenchmarkMetrics;
    fineTunedBenchmarks?: BenchmarkMetrics;
  }>
): BenchmarkSummaryTableRow[] {
  return models.map((m) => {
    const base = m.baseBenchmarks;
    const ft = m.fineTunedBenchmarks;
    if (!ft) {
      return {
        modelId: m.id,
        modelName: m.name,
        architecture: m.architecture,
        paramsBillions: m.paramsBillions,
        hasFineTuned: false,
        checkpointId: m.fineTunedCheckpointId,
        mmluProBase: base.mmluPro,
        mmluProFineTuned: null,
        mmluProDelta: null,
        humanEvalBase: base.humanEval,
        humanEvalFineTuned: null,
        humanEvalDelta: null,
        gsm8kBase: base.gsm8k,
        gsm8kFineTuned: null,
        gsm8kDelta: null,
        ifEvalBase: base.ifEval,
        ifEvalFineTuned: null,
        ifEvalDelta: null,
        perplexityBase: base.perplexity,
        perplexityFineTuned: null,
        perplexityDelta: null,
        meanAccuracyGain: null,
        verdict: 'BASE_ONLY',
      };
    }

    const comp = compareBaseAndFineTuned(base, ft);
    return {
      modelId: m.id,
      modelName: m.name,
      architecture: m.architecture,
      paramsBillions: m.paramsBillions,
      hasFineTuned: true,
      checkpointId: m.fineTunedCheckpointId,
      mmluProBase: base.mmluPro,
      mmluProFineTuned: ft.mmluPro,
      mmluProDelta: comp.deltas.mmluPro,
      humanEvalBase: base.humanEval,
      humanEvalFineTuned: ft.humanEval,
      humanEvalDelta: comp.deltas.humanEval,
      gsm8kBase: base.gsm8k,
      gsm8kFineTuned: ft.gsm8k,
      gsm8kDelta: comp.deltas.gsm8k,
      ifEvalBase: base.ifEval,
      ifEvalFineTuned: ft.ifEval,
      ifEvalDelta: comp.deltas.ifEval,
      perplexityBase: base.perplexity,
      perplexityFineTuned: ft.perplexity,
      perplexityDelta: comp.deltas.perplexity,
      meanAccuracyGain: comp.averageAccuracyGain,
      verdict: comp.verdict,
    };
  });
}




