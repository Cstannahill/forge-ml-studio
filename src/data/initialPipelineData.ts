import {
  ArchitectureFamily,
  BenchmarkMetrics,
  HyperparameterConfig,
  LoggingFramework,
  NormalizedDatasetSample,
  QuantizationMode,
  TargetFormat,
  normalizeDatasetRecord,
} from '../lib/pipelineLogic';

export interface RegisteredModel {
  id: string;
  name: string;
  architecture: ArchitectureFamily;
  paramsBillions: number;
  contextWindow: number;
  quantizationDefault: QuantizationMode;
  status: 'BASE_BENCHMARKED' | 'FINE_TUNED_READY' | 'TRAINING';
  detectedAt: string;
  baseBenchmarks: BenchmarkMetrics;
  fineTunedBenchmarks?: BenchmarkMetrics;
  fineTunedCheckpointId?: string;
  datasetVersionUsed?: string;
  hpoConfigUsed?: HyperparameterConfig;
}

export interface DatasetVersionCommit {
  versionId: string;
  commitHash: string;
  timestamp: string;
  sourceDatasets: string[];
  targetFormat: TargetFormat;
  sampleCount: number;
  healthScore: number;
  cloudSynced: boolean;
  cloudUri: string;
  notes: string;
}

export interface TrainingMetricPoint {
  step: number;
  epoch: number;
  trainLoss: number;
  valLoss: number;
  gradNorm: number;
  learningRate: number;
  tokensPerSec: number;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'METRIC' | 'WARN' | 'ERROR';
  component: 'TRAINER' | 'EVALUATOR' | 'DATA_PIPELINE' | 'HPO_OPTUNA' | 'AGENT_ORCHESTRATOR';
  framework: LoggingFramework;
  message: string;
  metadata?: string;
}

export interface CloudProviderGuide {
  id: string;
  name: string;
  tagline: string;
  recommendedInstance: string;
  vramTotalGb: number;
  hourlyRateUsd: number;
  bestFor: string;
  setupSteps: Array<{
    stepTitle: string;
    description: string;
    command: string;
  }>;
}

export const INITIAL_MODELS: RegisteredModel[] = [
  {
    id: 'Qwen/Qwen3.5-2B-Instruct',
    name: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    contextWindow: 32768,
    quantizationDefault: 'qlora-4bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-18 09:15 UTC',
    baseBenchmarks: {
      mmluPro: 61.4,
      gsm8k: 76.8,
      humanEval: 71.2,
      ifEval: 73.5,
      mtBench: 7.82,
      perplexity: 5.84,
      tokensPerSec: 248.6,
    },
    fineTunedBenchmarks: {
      mmluPro: 65.8,
      gsm8k: 83.4,
      humanEval: 79.1,
      ifEval: 81.2,
      mtBench: 8.36,
      perplexity: 5.08,
      tokensPerSec: 247.9,
    },
    fineTunedCheckpointId: 'ft-qwen3.5-2b-r32-wsd-v4',
    datasetVersionUsed: 'v2.4.0 (sha256:9f82c1e)',
    hpoConfigUsed: {
      learningRate: 0.0002,
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0.05,
      warmupRatio: 0.05,
      weightDecay: 0.01,
      neftuneAlpha: 5,
      batchSize: 8,
      lrScheduler: 'wsd',
    },
  },
  {
    id: 'LiquidAI/LFM2.5-1.2B-Instruct',
    name: 'LFM2.5-1.2B-Instruct (Liquid Edge)',
    architecture: 'LFM2.5',
    paramsBillions: 1.2,
    contextWindow: 32768,
    quantizationDefault: 'qlora-4bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-18 11:10 UTC',
    baseBenchmarks: {
      mmluPro: 57.2,
      gsm8k: 71.4,
      humanEval: 66.8,
      ifEval: 72.1,
      mtBench: 7.54,
      perplexity: 6.12,
      tokensPerSec: 342.0,
    },
    fineTunedBenchmarks: {
      mmluPro: 61.9,
      gsm8k: 78.6,
      humanEval: 73.5,
      ifEval: 80.4,
      mtBench: 8.12,
      perplexity: 5.34,
      tokensPerSec: 341.2,
    },
    fineTunedCheckpointId: 'ft-lfm2.5-1.2b-r32-wsd-v2',
    datasetVersionUsed: 'v2.4.0 (sha256:9f82c1e)',
    hpoConfigUsed: {
      learningRate: 0.00022,
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0.03,
      warmupRatio: 0.04,
      weightDecay: 0.01,
      neftuneAlpha: 0,
      batchSize: 8,
      lrScheduler: 'wsd',
    },
  },
  {
    id: 'google/gemma-4-E2B-it',
    name: 'Gemma4-E2B-IT (Effective 2B)',
    architecture: 'Gemma4',
    paramsBillions: 2.0,
    contextWindow: 32768,
    quantizationDefault: 'qlora-4bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-18 14:20 UTC',
    baseBenchmarks: {
      mmluPro: 62.8,
      gsm8k: 78.2,
      humanEval: 72.4,
      ifEval: 75.0,
      mtBench: 7.91,
      perplexity: 5.62,
      tokensPerSec: 236.4,
    },
    fineTunedBenchmarks: {
      mmluPro: 66.9,
      gsm8k: 84.7,
      humanEval: 79.8,
      ifEval: 82.6,
      mtBench: 8.45,
      perplexity: 4.91,
      tokensPerSec: 235.8,
    },
    fineTunedCheckpointId: 'ft-gemma4-e2b-r32-cosine-v1',
    datasetVersionUsed: 'v2.4.0 (sha256:9f82c1e)',
    hpoConfigUsed: {
      learningRate: 0.00018,
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0.05,
      warmupRatio: 0.05,
      weightDecay: 0.01,
      neftuneAlpha: 5,
      batchSize: 8,
      lrScheduler: 'cosine',
    },
  },
  {
    id: 'LiquidAI/LFM2.5-300M-Instruct',
    name: 'LFM2.5-300M-Instruct (Ultra-Compact)',
    architecture: 'LFM2.5',
    paramsBillions: 0.3,
    contextWindow: 16384,
    quantizationDefault: 'lora-16bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-18 16:05 UTC',
    baseBenchmarks: {
      mmluPro: 44.6,
      gsm8k: 54.2,
      humanEval: 51.0,
      ifEval: 63.8,
      mtBench: 6.75,
      perplexity: 7.45,
      tokensPerSec: 518.4,
    },
    fineTunedBenchmarks: {
      mmluPro: 49.8,
      gsm8k: 63.5,
      humanEval: 59.4,
      ifEval: 74.2,
      mtBench: 7.42,
      perplexity: 6.38,
      tokensPerSec: 517.9,
    },
    fineTunedCheckpointId: 'ft-lfm2.5-300m-r16-cosine-v3',
    datasetVersionUsed: 'v2.3.0 (sha256:4a19b0d)',
    hpoConfigUsed: {
      learningRate: 0.0003,
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0.03,
      warmupRatio: 0.03,
      weightDecay: 0.01,
      neftuneAlpha: 0,
      batchSize: 16,
      lrScheduler: 'cosine',
    },
  },
  {
    id: 'Qwen/Qwen3.5-0.8B-Instruct',
    name: 'Qwen3.5-0.8B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 0.8,
    contextWindow: 32768,
    quantizationDefault: 'qlora-4bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-19 07:30 UTC',
    baseBenchmarks: {
      mmluPro: 53.4,
      gsm8k: 67.5,
      humanEval: 63.2,
      ifEval: 68.9,
      mtBench: 7.28,
      perplexity: 6.48,
      tokensPerSec: 384.2,
    },
    fineTunedBenchmarks: {
      mmluPro: 58.1,
      gsm8k: 75.2,
      humanEval: 71.0,
      ifEval: 77.4,
      mtBench: 7.89,
      perplexity: 5.66,
      tokensPerSec: 383.5,
    },
    fineTunedCheckpointId: 'ft-qwen3.5-0.8b-r16-wsd-v2',
    datasetVersionUsed: 'v2.4.0 (sha256:9f82c1e)',
    hpoConfigUsed: {
      learningRate: 0.00025,
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0.05,
      warmupRatio: 0.04,
      weightDecay: 0.01,
      neftuneAlpha: 0,
      batchSize: 16,
      lrScheduler: 'wsd',
    },
  },
  {
    id: 'LiquidAI/LFM2.5-2.6B-Instruct',
    name: 'LFM2.5-2.6B-Instruct (Liquid Long-Ctx)',
    architecture: 'LFM2.5',
    paramsBillions: 2.6,
    contextWindow: 65536,
    quantizationDefault: 'qlora-4bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-19 09:45 UTC',
    baseBenchmarks: {
      mmluPro: 63.9,
      gsm8k: 79.4,
      humanEval: 73.8,
      ifEval: 76.2,
      mtBench: 8.04,
      perplexity: 5.48,
      tokensPerSec: 218.5,
    },
    fineTunedBenchmarks: {
      mmluPro: 68.4,
      gsm8k: 85.9,
      humanEval: 80.6,
      ifEval: 83.8,
      mtBench: 8.58,
      perplexity: 4.79,
      tokensPerSec: 217.8,
    },
    fineTunedCheckpointId: 'ft-lfm2.5-2.6b-r32-wsd-v1',
    datasetVersionUsed: 'v2.4.0 (sha256:9f82c1e)',
    hpoConfigUsed: {
      learningRate: 0.00018,
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0.05,
      warmupRatio: 0.05,
      weightDecay: 0.01,
      neftuneAlpha: 5,
      batchSize: 4,
      lrScheduler: 'wsd',
    },
  },
  {
    id: 'google/gemma-4-E4B-it',
    name: 'Gemma4-E4B-IT (Effective 4B)',
    architecture: 'Gemma4',
    paramsBillions: 4.0,
    contextWindow: 65536,
    quantizationDefault: 'qlora-4bit',
    status: 'FINE_TUNED_READY',
    detectedAt: '2026-10-19 10:15 UTC',
    baseBenchmarks: {
      mmluPro: 68.5,
      gsm8k: 83.6,
      humanEval: 77.2,
      ifEval: 79.4,
      mtBench: 8.28,
      perplexity: 5.12,
      tokensPerSec: 156.0,
    },
    fineTunedBenchmarks: {
      mmluPro: 72.4,
      gsm8k: 89.1,
      humanEval: 83.5,
      ifEval: 85.7,
      mtBench: 8.76,
      perplexity: 4.49,
      tokensPerSec: 155.4,
    },
    fineTunedCheckpointId: 'ft-gemma4-e4b-r32-cosine-v1',
    datasetVersionUsed: 'v2.4.0 (sha256:9f82c1e)',
    hpoConfigUsed: {
      learningRate: 0.00016,
      loraRank: 32,
      loraAlpha: 64,
      loraDropout: 0.05,
      warmupRatio: 0.05,
      weightDecay: 0.01,
      neftuneAlpha: 5,
      batchSize: 4,
      lrScheduler: 'cosine',
    },
  },
  {
    id: 'Qwen/Qwen3.5-9B-Instruct',
    name: 'Qwen3.5-9B-Instruct (Heavy Ceiling)',
    architecture: 'Qwen3.5',
    paramsBillions: 9.0,
    contextWindow: 131072,
    quantizationDefault: 'qlora-4bit',
    status: 'BASE_BENCHMARKED',
    detectedAt: '2026-10-19 11:40 UTC',
    baseBenchmarks: {
      mmluPro: 72.1,
      gsm8k: 86.4,
      humanEval: 80.5,
      ifEval: 81.8,
      mtBench: 8.48,
      perplexity: 4.82,
      tokensPerSec: 114.2,
    },
  },
];

export const INITIAL_TRAINING_SERIES: TrainingMetricPoint[] = [
  { step: 50, epoch: 0.25, trainLoss: 1.94, valLoss: 1.98, gradNorm: 1.42, learningRate: 0.00007, tokensPerSec: 9420 },
  { step: 100, epoch: 0.5, trainLoss: 1.62, valLoss: 1.68, gradNorm: 1.18, learningRate: 0.00014, tokensPerSec: 9680 },
  { step: 150, epoch: 0.75, trainLoss: 1.38, valLoss: 1.45, gradNorm: 0.96, learningRate: 0.00014, tokensPerSec: 9740 },
  { step: 200, epoch: 1.0, trainLoss: 1.19, valLoss: 1.27, gradNorm: 0.89, learningRate: 0.000135, tokensPerSec: 9810 },
  { step: 250, epoch: 1.25, trainLoss: 1.04, valLoss: 1.12, gradNorm: 0.82, learningRate: 0.00012, tokensPerSec: 9790 },
  { step: 300, epoch: 1.5, trainLoss: 0.91, valLoss: 1.01, gradNorm: 0.77, learningRate: 0.000098, tokensPerSec: 9850 },
  { step: 350, epoch: 1.75, trainLoss: 0.82, valLoss: 0.93, gradNorm: 0.71, learningRate: 0.000064, tokensPerSec: 9890 },
  { step: 400, epoch: 2.0, trainLoss: 0.76, valLoss: 0.87, gradNorm: 0.68, learningRate: 0.000028, tokensPerSec: 9910 },
];

export const INITIAL_RAW_SAMPLES: Array<{ source: string; record: Record<string, unknown> }> = [
  {
    source: 'Qwen/Qwen3-Code-Reasoning-110k',
    record: {
      instruction: 'Implement a PyTorch custom rotary position embedding (RoPE) frequency scaling helper for Qwen3.5 dynamic NTK context extension.',
      input: 'dim=128, max_seq_len=32768, base=1000000.0, scaling_factor=4.0',
      output:
        '```python\nimport torch\n\ndef compute_dynamic_ntk_rope(dim: int, max_seq_len: int, base: float = 1000000.0, scaling_factor: float = 4.0) -> torch.Tensor:\n    scaled_base = base * ((scaling_factor * max_seq_len / 8192) - (scaling_factor - 1)) ** (dim / (dim - 2))\n    inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))\n    return inv_freq\n```',
    },
  },
  {
    source: 'HuggingFaceH4/ultrachat_200k',
    record: {
      messages: [
        {
          role: 'user',
          content: 'Please review ticket from user ops-lead@enterprise.internal (SSN 412-88-9012) regarding CUDA OOM on node-04.',
        },
        {
          role: 'assistant',
          content:
            'To resolve the CUDA out-of-memory error on node-04, enable PagedAdamW 8-bit optimizer, activate gradient checkpointing (`use_reentrant=False`), and reduce per-device micro-batch size from 8 to 2 while increasing gradient accumulation steps to 16.',
        },
      ],
    },
  },
  {
    source: 'Open-Orca/SlimOrca-Dedup',
    record: {
      conversations: [
        { from: 'system', value: 'You are an AI assistant that explains reasoning step-by-step before giving the final answer.' },
        {
          from: 'human',
          value: 'A cluster of 4 GPUs processes 12,800 tokens/sec total with 85% scaling efficiency when doubling to 8 GPUs. What is the 8-GPU throughput?',
        },
        {
          from: 'gpt',
          value:
            'Step-by-step calculation:\n1. Ideal linear doubling throughput = 12,800 * 2 = 25,600 tokens/sec.\n2. Applying the 85% scaling efficiency factor: 25,600 * 0.85 = 21,760 tokens/sec.\nTherefore, the 8-GPU cluster achieves 21,760 tokens/sec.',
        },
      ],
    },
  },
  {
    source: 'LiquidAI/LFM-LongContext-Synthesis-64k',
    record: {
      messages: [
        { role: 'system', content: 'Extract structured technical invariants and latency SLAs from long architecture specifications.' },
        {
          role: 'user',
          content: 'Synthesize the memory complexity advantage of Liquid Foundation Models (LFM2.5) over dense quadratic attention during 64k token prefill.',
        },
        {
          role: 'assistant',
          content:
            'LFM2.5 replaces quadratic O(L^2) self-attention blocks with input-dependent linear state-space operators interleaved with local gated convolutions. During 64k token prefill, activation memory scales linearly O(L) and inference KV state remains constant O(1) per recurrent layer, reducing VRAM usage by ~42% compared to standard GQA transformers.',
        },
      ],
    },
  },
  {
    source: 'google/gemma-4-it-math-olympiad',
    record: {
      instruction: 'Determine the eigenvalues of the symmetric matrix A = [[4, 2], [2, 1]] and verify the trace invariant.',
      input: '',
      output:
        'Step-by-step derivation:\n1. Characteristic polynomial: det(A - λI) = (4 - λ)(1 - λ) - (2)(2) = λ^2 - 5λ + 4 - 4 = λ(λ - 5) = 0.\n2. The eigenvalues are λ_1 = 5 and λ_2 = 0.\n3. Trace verification: Tr(A) = 4 + 1 = 5 = λ_1 + λ_2. Determinant det(A) = 0 = λ_1 * λ_2.',
    },
  },
];

export function buildInitialNormalizedSamples(format: TargetFormat = 'chatml'): NormalizedDatasetSample[] {
  return INITIAL_RAW_SAMPLES.map((item, idx) => normalizeDatasetRecord(item.record, item.source, idx, format));
}

export const INITIAL_DATASET_VERSIONS: DatasetVersionCommit[] = [
  {
    versionId: 'v2.4.0',
    commitHash: 'sha256:9f82c1e',
    timestamp: '2026-10-19 15:42 UTC',
    sourceDatasets: ['Qwen/Qwen3-Code-Reasoning-110k', 'HuggingFaceH4/ultrachat_200k', 'Open-Orca/SlimOrca-Dedup'],
    targetFormat: 'chatml',
    sampleCount: 5,
    healthScore: 88,
    cloudSynced: true,
    cloudUri: 's3://forgeml-datasets/snapshots/v2.4.0.parquet',
    notes: 'Merged multi-source HF code + reasoning blend in unified ChatML format.',
  },
  {
    versionId: 'v2.3.0',
    commitHash: 'sha256:4a19b0d',
    timestamp: '2026-10-18 11:20 UTC',
    sourceDatasets: ['LiquidAI/LFM-LongContext-Synthesis-64k', 'HuggingFaceH4/ultrachat_200k'],
    targetFormat: 'chatml',
    sampleCount: 3,
    healthScore: 95,
    cloudSynced: true,
    cloudUri: 's3://forgeml-datasets/snapshots/v2.3.0.parquet',
    notes: 'Long-context synthesis corpus prepared for LFM2.5-8B state-space fine-tuning.',
  },
];

export const INITIAL_LOGS: LogEntry[] = [
  {
    id: 'log-1',
    timestamp: '16:28:04.112',
    level: 'INFO',
    component: 'DATA_PIPELINE',
    framework: 'wandb',
    message: 'Mounted unified ChatML dataset v2.4.0 (sha256:9f82c1e) with zero schema violations.',
    metadata: 'rows=5 | format=chatml | packing=FlashAttention-3-varlen',
  },
  {
    id: 'log-2',
    timestamp: '16:28:19.450',
    level: 'INFO',
    component: 'EVALUATOR',
    framework: 'mlflow',
    message: 'Base model Qwen/Qwen3.5-2B-Instruct baseline fingerprint verified in artifact registry.',
    metadata: 'mmlu_pro=61.4% | gsm8k=76.8% | humaneval=71.2% | tok_s=248.6',
  },
  {
    id: 'log-3',
    timestamp: '16:29:02.804',
    level: 'METRIC',
    component: 'HPO_OPTUNA',
    framework: 'wandb',
    message: 'Bayesian TPE Trial #4 converged: train_loss=0.760, val_loss=0.870, grad_norm=0.680.',
    metadata: 'lr=2.0e-4 | lora_r=32 | lora_alpha=64 | neftune=5.0 | scheduler=wsd',
  },
  {
    id: 'log-4',
    timestamp: '16:29:41.019',
    level: 'WARN',
    component: 'DATA_PIPELINE',
    framework: 'otel',
    message: 'Detected 1 unredacted PII pattern in raw HuggingFaceH4/ultrachat_200k sample #1; queued for Agentic Hygiene pass.',
    metadata: 'rule=PII_OR_SECRET_PATTERN | action=redact_on_clean',
  },
  {
    id: 'log-5',
    timestamp: '16:30:10.602',
    level: 'INFO',
    component: 'EVALUATOR',
    framework: 'tensorboard',
    message: 'Post-training benchmark complete for ft-qwen3.5-2b-r32-wsd-v4: +6.65% mean accuracy delta over base.',
    metadata: 'verdict=PROMOTE_TO_PRODUCTION | regressions=0',
  },
];

export const CLOUD_PROVIDER_GUIDES: CloudProviderGuide[] = [
  {
    id: 'runpod',
    name: 'RunPod Secure Cloud',
    tagline: 'Instant containerized GPU pods with persistent network volumes and sub-minute cold start.',
    recommendedInstance: '1x NVIDIA H100 PCIe (80 GB VRAM) or 2x RTX 4090 (48 GB VRAM)',
    vramTotalGb: 80,
    hourlyRateUsd: 2.39,
    bestFor: 'Interactive QLoRA / LoRA HPO sweeps on Qwen3.5-14B, LFM2.5-8B, and Gemma4-27B',
    setupSteps: [
      {
        stepTitle: '01. Authenticate RunPod CLI & Provision Persistent Volume',
        description: 'Install runpodctl and allocate a high-IOPS NVMe network volume in the same datacenter region to cache HuggingFace weights and DVC dataset snapshots.',
        command: `pip install runpod\nrunpodctl config --apiKey "$RUNPOD_API_KEY"\nrunpodctl create volume --name "forgeml-cache" --size 200 --dataCenterId "US-OR-1"`,
      },
      {
        stepTitle: '02. Launch ForgeML Container Pod with CUDA 12.6 & FlashAttention-3',
        description: 'Deploy the generated ForgeML Docker image directly onto an H100 80GB node with port forwarding for TensorBoard and Optuna.',
        command: `runpodctl create pod \\\n  --name "forgeml-qwen35-hpo" \\\n  --gpuType "NVIDIA H100 PCIe" \\\n  --gpuCount 1 \\\n  --imageName "forgeml/pipeline-qwen3.5:latest" \\\n  --volumePath "/workspace" \\\n  --ports "6006/http,8265/http"`,
      },
      {
        stepTitle: '03. Sync Dataset Snapshot & Stream Telemetry Back to Local Dashboard',
        description: 'Establish an SSH tunnel to stream live metrics and checkpoint evaluations back to your local ForgeML web console.',
        command: `ssh -L 6006:localhost:6006 -L 8265:localhost:8265 root@<RUNPOD_PUBLIC_IP> -p <SSH_PORT>`,
      },
    ],
  },
  {
    id: 'lambda',
    name: 'Lambda Labs GPU Cloud',
    tagline: 'Bare-metal NVIDIA H100 SXM5 clusters with 3.2 Tbps InfiniBand for multi-GPU DeepSpeed ZeRO-3.',
    recommendedInstance: '2x or 8x NVIDIA H100 SXM5 (80 GB per GPU)',
    vramTotalGb: 160,
    hourlyRateUsd: 5.98,
    bestFor: 'Full BF16 fine-tuning and large-batch multi-node runs on Gemma4-27B and Qwen3.5-72B',
    setupSteps: [
      {
        stepTitle: '01. Provision Lambda Instance via Cloud API',
        description: 'Launch a multi-GPU H100 instance pre-configured with Lambda Stack (CUDA 12.6, NCCL, PyTorch).',
        command: `curl -u "$LAMBDA_API_KEY:" https://cloud.lambdalabs.com/api/v1/instance-operations/launch \\\n  -H "Content-Type: application/json" \\\n  -d '{"region_name":"us-west-1","instance_type_name":"gpu_2x_h100_sxm5","ssh_key_names":["forgeml-key"]}'`,
      },
      {
        stepTitle: '02. Clone Pipeline & Mount Cloud Dataset Bucket',
        description: 'Sync versioned Parquet datasets using rclone and start the Docker Compose stack with NCCL P2P enabled.',
        command: `export NCCL_P2P_LEVEL=NVL\nrclone copy s3://forgeml-datasets/snapshots/v2.4.0.parquet ./data_cache/\ndocker compose up --build -d forgeml-trainer`,
      },
    ],
  },
  {
    id: 'modal',
    name: 'Modal Serverless GPU',
    tagline: 'Zero-idle cost serverless A100/H100 execution triggered directly from Python decorators.',
    recommendedInstance: 'Serverless H100 / A100-80GB Auto-Scaling Worker Pool',
    vramTotalGb: 80,
    hourlyRateUsd: 3.15,
    bestFor: 'Automated burst HPO trials and asynchronous Base vs. Fine-Tuned lm-eval benchmarking jobs',
    setupSteps: [
      {
        stepTitle: '01. Configure Modal Token & Persistent Volumes',
        description: 'Create serverless volumes for model checkpoints and tokenized HuggingFace caches.',
        command: `pip install modal\nmodal setup\nmodal volume create forgeml-checkpoints\nmodal volume create forgeml-hf-cache`,
      },
      {
        stepTitle: '02. Dispatch Parallel Bayesian HPO Sweep',
        description: 'Spawn parallel serverless GPU containers for each Optuna trial and automatically terminate on completion.',
        command: `modal run pipeline/modal_hpo_worker.py --model "Qwen/Qwen3.5-14B-Instruct" --trials 8 --gpu "H100"`,
      },
    ],
  },
  {
    id: 'aws-sagemaker',
    name: 'AWS SageMaker HyperPod / EC2 P5',
    tagline: 'Enterprise VPC isolation, S3 FastFile mounting, and managed spot checkpoint recovery.',
    recommendedInstance: 'ml.p4d.24xlarge (8x A100 40GB) or ml.g6e.12xlarge (4x L40S 192GB)',
    vramTotalGb: 192,
    hourlyRateUsd: 7.24,
    bestFor: 'Production compliance workloads requiring IAM role-based S3 encryption and automatic spot resumption',
    setupSteps: [
      {
        stepTitle: '01. Push ForgeML Image to Amazon ECR',
        description: 'Build and push the containerized pipeline to your private ECR repository.',
        command: `aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com\ndocker build -t forgeml/pipeline:latest .\ndocker tag forgeml/pipeline:latest <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com/forgeml:latest\ndocker push <ACCOUNT>.dkr.ecr.us-west-2.amazonaws.com/forgeml:latest`,
      },
      {
        stepTitle: '02. Launch Managed Training Job with S3 FastFile Channel',
        description: 'Stream Parquet shards directly from S3 into GPU memory without local disk bottleneck.',
        command: `aws sagemaker create-training-job --cli-input-json file://configs/sagemaker_job.json`,
      },
    ],
  },
  {
    id: 'gcp-vertex',
    name: 'Google Cloud Vertex AI / A3 Mega',
    tagline: 'Native GCS Cloud Storage FUSE integration and high-bandwidth GPUDirect-TCPX networking.',
    recommendedInstance: 'a3-highgpu-1g (1x H100 80GB) or g2-standard-48 (4x L4 96GB)',
    vramTotalGb: 80,
    hourlyRateUsd: 2.95,
    bestFor: 'Gemma4 & Qwen3.5 pipelines backed by Google Cloud Storage buckets and Vertex Experiment Tracking',
    setupSteps: [
      {
        stepTitle: '01. Push Container to Google Artifact Registry',
        description: 'Authenticate gcloud Docker credential helper and upload the pipeline image.',
        command: `gcloud auth configure-docker us-central1-docker.pkg.dev\ndocker tag forgeml/pipeline:latest us-central1-docker.pkg.dev/$PROJECT_ID/forgeml/trainer:latest\ndocker push us-central1-docker.pkg.dev/$PROJECT_ID/forgeml/trainer:latest`,
      },
      {
        stepTitle: '02. Submit Vertex AI Custom Training Job',
        description: 'Mount gs://forgeml-datasets automatically at /gcs/forgeml-datasets and execute fine-tuning + auto-eval.',
        command: `gcloud ai custom-jobs create \\\n  --region=us-central1 \\\n  --display-name=forgeml-gemma4-hpo \\\n  --worker-pool-spec=machine-type=a3-highgpu-1g,replica-count=1,accelerator-type=NVIDIA_H100_80GB,accelerator-count=1,container-image-uri=us-central1-docker.pkg.dev/$PROJECT_ID/forgeml/trainer:latest`,
      },
    ],
  },
];
