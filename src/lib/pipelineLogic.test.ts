import test from 'node:test';
import assert from 'node:assert/strict';
import {
  estimateVramRequirements,
  normalizeDatasetRecord,
  cleanAndRedactSample,
  validateDatasetRecords,
  computeNextHyperparameters,
  compareBaseAndFineTuned,
  generateContainerArtifacts,
  evaluateGpuThresholds,
  buildPlainLanguageBenchmarkAnalysis,
  buildSafetensorsExportPlan,
  createSafetensorsBinaryArtifact,
  computeTrainingProgressTelemetry,
  generateStandardizedModelCard,
  buildRadarChartComparisonData,
  serializeTrainingSeriesMap,
  deserializeTrainingSeriesMap,
  buildBenchmarkSummaryTableRows,
} from './pipelineLogic.ts';
import { CURATED_DATASET_BLENDS } from '../data/curatedDatasetBlends.ts';
import { ML_GLOSSARY } from './glossaryData.ts';

test('estimateVramRequirements calculates lower activation footprint for LFM2.5 than standard architectures', () => {
  const qwenEst = estimateVramRequirements({
    paramsBillions: 14,
    quantization: 'qlora-4bit',
    contextLength: 8192,
    batchSize: 4,
    gradientCheckpointing: true,
    architecture: 'Qwen3.5',
  });
  const lfmEst = estimateVramRequirements({
    paramsBillions: 14,
    quantization: 'qlora-4bit',
    contextLength: 8192,
    batchSize: 4,
    gradientCheckpointing: true,
    architecture: 'LFM2.5',
  });

  assert.equal(qwenEst.weightsGb, lfmEst.weightsGb);
  assert.ok(lfmEst.kvCacheAndActivationsGb <= qwenEst.kvCacheAndActivationsGb);
  assert.equal(qwenEst.canRunLocally24Gb, true);
});

test('normalizeDatasetRecord converts Alpaca, ShareGPT, and Messages into canonical ChatML and flags PII', () => {
  const alpacaRaw = {
    instruction: 'Write a SQL query to select active users.',
    input: 'Table: users(id, email, active)',
    output: 'SELECT id, email FROM users WHERE active = 1;',
  };
  const sample1 = normalizeDatasetRecord(alpacaRaw, 'Open-Orca/SlimOrca', 0, 'chatml');
  assert.equal(sample1.messages.length, 2);
  assert.equal(sample1.cleaned, true);
  assert.ok(sample1.tags.includes('code-generation'));
  assert.ok(sample1.formattedPreview.includes('<|im_start|>user'));

  const piiRaw = {
    messages: [
      { role: 'user', content: 'Contact admin@corp.example.com or SSN 123-45-6789' },
      { role: 'assistant', content: 'I have logged your ticket details carefully.' },
    ],
  };
  const sample2 = normalizeDatasetRecord(piiRaw, 'custom/support-logs', 1, 'chatml');
  assert.equal(sample2.piiDetected, true);
  assert.ok(sample2.issueFlags.includes('PII_OR_SECRET_PATTERN'));

  const redacted = cleanAndRedactSample(sample2, 'chatml');
  assert.equal(redacted.piiDetected, false);
  assert.ok(redacted.formattedPreview.includes('[REDACTED_EMAIL]'));
  assert.ok(redacted.formattedPreview.includes('[REDACTED_SSN]'));
});

test('validateDatasetRecords detects duplicates, PII, and computes token distribution', () => {
  const s1 = normalizeDatasetRecord(
    { instruction: 'Calculate 25 * 4 step-by-step', output: 'Step-by-step: 25 * 4 = 100.' },
    'math/gsm8k',
    0
  );
  const s2 = normalizeDatasetRecord(
    { instruction: 'Calculate 25 * 4 step-by-step', output: 'Step-by-step: 25 * 4 = 100.' },
    'math/gsm8k',
    1
  );
  const report = validateDatasetRecords([s1, s2]);
  assert.equal(report.totalSamples, 2);
  assert.equal(report.duplicateCount, 1);
  assert.ok(report.overallHealthScore > 50);
});

test('computeNextHyperparameters diagnoses overfitting, gradient spikes, and underfitting accurately', () => {
  const overfitProposal = computeNextHyperparameters({
    trialNumber: 3,
    config: {
      learningRate: 2e-4,
      loraRank: 64,
      loraAlpha: 128,
      loraDropout: 0.0,
      warmupRatio: 0.03,
      weightDecay: 0.01,
      neftuneAlpha: 0,
      batchSize: 4,
      lrScheduler: 'cosine',
    },
    trainLoss: 0.52,
    valLoss: 0.94,
    gradNorm: 0.85,
    evalScore: 71.2,
  });
  assert.equal(overfitProposal.diagnosis, 'OVERFITTING');
  assert.ok(overfitProposal.nextConfig.loraRank < 64);
  assert.ok(overfitProposal.nextConfig.neftuneAlpha >= 5);

  const gradSpikeProposal = computeNextHyperparameters({
    trialNumber: 4,
    config: overfitProposal.nextConfig,
    trainLoss: 1.1,
    valLoss: 1.2,
    gradNorm: 2.4,
    evalScore: 64.0,
  });
  assert.equal(gradSpikeProposal.diagnosis, 'GRADIENT_INSTABILITY');
  assert.ok(gradSpikeProposal.nextConfig.learningRate < overfitProposal.nextConfig.learningRate);
});

test('compareBaseAndFineTuned flags regressions and promotes strong accuracy improvements', () => {
  const base = {
    mmluPro: 68.4,
    gsm8k: 81.2,
    humanEval: 74.0,
    ifEval: 76.5,
    mtBench: 8.1,
    perplexity: 5.42,
    tokensPerSec: 94.0,
  };
  const ftGood = {
    mmluPro: 71.8,
    gsm8k: 86.5,
    humanEval: 79.2,
    ifEval: 81.0,
    mtBench: 8.65,
    perplexity: 4.88,
    tokensPerSec: 93.5,
  };
  const compGood = compareBaseAndFineTuned(base, ftGood);
  assert.equal(compGood.verdict, 'PROMOTE_TO_PRODUCTION');
  assert.equal(compGood.regressions.length, 0);

  const ftRegressed = {
    ...ftGood,
    mmluPro: 64.0,
  };
  const compBad = compareBaseAndFineTuned(base, ftRegressed);
  assert.equal(compBad.verdict, 'REGRESSION_ALERT');
  assert.ok(compBad.regressions.length > 0);
});

test('generateContainerArtifacts outputs architecture-specific kernels and logging variables', () => {
  const artifacts = generateContainerArtifacts({
    modelId: 'LiquidAI/LFM2.5-8B-Instruct',
    architecture: 'LFM2.5',
    quantization: 'qlora-4bit',
    loggingFramework: 'wandb',
    cloudStorageUri: 's3://forgeml-artifacts/prod',
  });
  assert.ok(artifacts.dockerfile.includes('mamba-ssm'));
  assert.ok(artifacts.dockerfile.includes('WANDB_PROJECT'));
  assert.ok(artifacts.dockerCompose.includes('LiquidAI/LFM2.5-8B-Instruct'));
});

test('evaluateGpuThresholds detects VRAM > 90% and Temp > 85C breaches accurately', () => {
  const nominal = evaluateGpuThresholds({
    vramUsedGb: 18.4,
    vramTotalGb: 24.0,
    smUtilPct: 90,
    tensorCorePct: 85,
    tempCelsius: 68,
    powerWatts: 340,
    nvlinkGbps: 56.0,
  });
  assert.equal(nominal.hasAlerts, false);
  assert.equal(nominal.breaches.length, 0);

  const overloaded = evaluateGpuThresholds({
    vramUsedGb: 22.8, // 95% of 24GB (> 90%)
    vramTotalGb: 24.0,
    smUtilPct: 99,
    tensorCorePct: 96,
    tempCelsius: 88, // > 85C
    powerWatts: 425, // > 400W
    nvlinkGbps: 58.2,
  });
  assert.equal(overloaded.hasAlerts, true);
  assert.equal(overloaded.hasCriticalAlert, true);
  assert.equal(overloaded.breaches.length, 3);
  assert.ok(overloaded.breaches.some((b) => b.metric === 'VRAM'));
  assert.ok(overloaded.breaches.some((b) => b.metric === 'TEMP'));
});

test('buildPlainLanguageBenchmarkAnalysis explains strengths, regressions, and other differences clearly', () => {
  const analysis = buildPlainLanguageBenchmarkAnalysis({
    modelName: 'Qwen3.5-14B-Instruct',
    architecture: 'Qwen3.5',
    base: {
      mmluPro: 71.4,
      gsm8k: 84.2,
      humanEval: 78.6,
      ifEval: 79.1,
      mtBench: 8.35,
      perplexity: 5.18,
      tokensPerSec: 112.4,
    },
    fineTuned: {
      mmluPro: 74.9,
      gsm8k: 89.6,
      humanEval: 85.4,
      ifEval: 85.2,
      mtBench: 8.84,
      perplexity: 4.52,
      tokensPerSec: 111.8,
    },
  });

  assert.ok(analysis.headline.includes('Qwen3.5-14B-Instruct'));
  assert.ok(analysis.betterAt.length >= 3);
  assert.ok(analysis.worseAt.length >= 1);
  assert.ok(analysis.otherDifferences.length >= 2);
});

test('buildSafetensorsExportPlan and createSafetensorsBinaryArtifact produce spec-compliant single-file Safetensors for sub-2B models', () => {
  const plan = buildSafetensorsExportPlan({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    adapterType: 'qlora-4bit',
    loraRank: 32,
    loraAlpha: 64,
    exportPrecision: 'bfloat16',
    maxShardSizeGb: 5,
    includeTokenizerAndConfig: true,
    runtimeTarget: 'ollama',
    checkpointId: 'ft-qwen3.5-2b-r32-wsd-v4',
  });

  assert.equal(plan.mergedFileSizeGb, 4.0);
  assert.equal(plan.shardCount, 1);
  assert.equal(plan.isSingleFileContiguous, true);
  assert.equal(plan.adapterScalingFactor, 2.0);
  assert.ok(plan.outputFilename.endsWith('.safetensors'));
  assert.ok(plan.pythonMergeScript.includes('merge_and_unload'));
  assert.ok(plan.localDeployCommand.includes('ollama create'));

  const binary = createSafetensorsBinaryArtifact(plan.safetensorsHeaderJson);
  const view = new DataView(binary.buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  assert.equal(headerLen % 8, 0);
  assert.equal(binary.byteLength, 8 + headerLen + 64);
});

test('CURATED_DATASET_BLENDS contains 24 task-specific blends covering LFM2.5 (300M, 1.2B, 2.6B), Qwen3.5 (0.8B, 2B), and Gemma4 (E2B, E4B)', () => {
  assert.equal(CURATED_DATASET_BLENDS.length, 24);
  for (const blend of CURATED_DATASET_BLENDS) {
    const totalWeight = blend.datasets.reduce((acc, d) => acc + d.weightPct, 0);
    assert.equal(totalWeight, 100, `Blend ${blend.id} weights must sum to 100%`);
    assert.ok(blend.datasets.length >= 2);
  }
});

test('computeTrainingProgressTelemetry calculates current epoch completion %, ETA, and memory utilization trends', () => {
  const telemetry = computeTrainingProgressTelemetry({
    currentStep: 350,
    currentEpoch: 1.75,
    targetEpochs: 3,
    stepsPerEpoch: 200,
    tokensPerSec: 9890,
    avgTokensPerStep: 32768,
    vramUsedGb: 3.4,
    vramTotalGb: 24.0,
    recentSteps: [
      { step: 200, trainLoss: 1.19, valLoss: 1.27, tokensPerSec: 9810 },
      { step: 250, trainLoss: 1.04, valLoss: 1.12, tokensPerSec: 9790 },
      { step: 300, trainLoss: 0.91, valLoss: 1.01, tokensPerSec: 9850 },
      { step: 350, trainLoss: 0.82, valLoss: 0.93, tokensPerSec: 9890 },
    ],
  });

  assert.equal(telemetry.currentEpochIndex, 2);
  assert.equal(telemetry.stepWithinCurrentEpoch, 150);
  assert.equal(telemetry.currentEpochCompletionPct, 75);
  assert.equal(telemetry.remainingSteps, 250);
  assert.ok(telemetry.etaSeconds > 0);
  assert.ok(telemetry.etaFormatted.includes('m'));
  assert.equal(telemetry.memoryTrendPoints.length, 4);
  assert.equal(telemetry.vramUtilizationPct, 14.2);
});

test('generateStandardizedModelCard extracts size, architecture, base vs fine-tuned metrics, and dataset version into standardized Markdown', () => {
  const plan = buildSafetensorsExportPlan({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    adapterType: 'qlora-4bit',
    loraRank: 32,
    loraAlpha: 64,
    exportPrecision: 'bfloat16',
    maxShardSizeGb: 5,
    includeTokenizerAndConfig: true,
    runtimeTarget: 'ollama',
    checkpointId: 'ft-qwen3.5-2b-r32-wsd-v4',
  });

  const card = generateStandardizedModelCard({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    contextWindow: 32768,
    quantizationMode: 'qlora-4bit',
    checkpointId: 'ft-qwen3.5-2b-r32-wsd-v4',
    datasetVersionId: 'v2.4.0',
    datasetCommitHash: 'sha256:9f82c1e',
    sourceDatasets: ['Qwen/Qwen3-Code-Reasoning-110k', 'HuggingFaceH4/ultrachat_200k'],
    targetFormat: 'chatml',
    sampleCount: 5,
    datasetHealthScore: 98,
    hpoConfig: {
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
    safetensorsPlan: plan,
  });

  assert.equal(card.filename, 'qwen3.5-2b-instruct-README.md');
  assert.ok(card.markdown.includes('library_name: safetensors'));
  assert.ok(card.markdown.includes('v2.4.0 (sha256:9f82c1e)'));
  assert.ok(card.markdown.includes('2B Parameters'));
  assert.ok(card.markdown.includes('61.4%'));
  assert.ok(card.markdown.includes('65.8%'));
  assert.equal(card.extractedMetadata.parameterScale, '2B');
  assert.equal(card.extractedMetadata.architecture, 'Qwen3.5');
});

test('buildSafetensorsExportPlan and generateStandardizedModelCard reflect GGUF-Q4_K_M, GPTQ, and ONNX deployment formats', () => {
  const ggufPlan = buildSafetensorsExportPlan({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    adapterType: 'qlora-4bit',
    loraRank: 32,
    loraAlpha: 64,
    exportPrecision: 'bfloat16',
    deploymentFormat: 'GGUF-Q4_K_M',
    maxShardSizeGb: 5,
    includeTokenizerAndConfig: true,
    runtimeTarget: 'ollama',
    checkpointId: 'ft-qwen3.5-2b-r32-wsd-v4',
  });
  assert.equal(ggufPlan.deploymentFormat, 'GGUF-Q4_K_M');
  assert.ok(ggufPlan.outputFilename.endsWith('-q4_k_m.gguf'));
  assert.ok(ggufPlan.mergedFileSizeGb < 2.0);
  assert.ok(ggufPlan.pythonMergeScript.includes('llama-quantize'));

  const ggufCard = generateStandardizedModelCard({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    contextWindow: 32768,
    quantizationMode: 'qlora-4bit',
    checkpointId: 'ft-qwen3.5-2b-r32-wsd-v4',
    datasetVersionId: 'v2.4.0',
    datasetCommitHash: 'sha256:9f82c1e',
    sourceDatasets: ['Qwen/Qwen3-Code-Reasoning-110k'],
    targetFormat: 'chatml',
    sampleCount: 5,
    datasetHealthScore: 98,
    hpoConfig: {
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
    baseBenchmarks: {
      mmluPro: 61.4,
      gsm8k: 76.8,
      humanEval: 71.2,
      ifEval: 73.5,
      mtBench: 7.82,
      perplexity: 5.84,
      tokensPerSec: 248.6,
    },
    safetensorsPlan: ggufPlan,
  });
  assert.equal(ggufCard.extractedMetadata.deploymentFormat, 'GGUF-Q4_K_M');
  assert.ok(ggufCard.markdown.includes('library_name: gguf'));
  assert.ok(ggufCard.markdown.includes('deployment_format: GGUF-Q4_K_M'));

  const gptqPlan = buildSafetensorsExportPlan({
    modelId: 'LiquidAI/LFM2.5-1.2B-Instruct',
    modelName: 'LFM2.5-1.2B-Instruct',
    architecture: 'LFM2.5',
    paramsBillions: 1.2,
    adapterType: 'qlora-4bit',
    loraRank: 16,
    loraAlpha: 32,
    exportPrecision: 'float16',
    deploymentFormat: 'GPTQ',
    maxShardSizeGb: 5,
    includeTokenizerAndConfig: true,
    runtimeTarget: 'vllm-local',
    checkpointId: 'ft-lfm2.5-1.2b-r16',
  });
  assert.equal(gptqPlan.deploymentFormat, 'GPTQ');
  assert.ok(gptqPlan.outputFilename.includes('gptq-4bit'));
  assert.ok(gptqPlan.localDeployCommand.includes('gptq_marlin'));

  const onnxPlan = buildSafetensorsExportPlan({
    modelId: 'google/gemma-4-E2B-it',
    modelName: 'Gemma-4-E2B-it',
    architecture: 'Gemma4',
    paramsBillions: 2.0,
    adapterType: 'qlora-4bit',
    loraRank: 16,
    loraAlpha: 32,
    exportPrecision: 'float16',
    deploymentFormat: 'ONNX',
    maxShardSizeGb: 5,
    includeTokenizerAndConfig: true,
    runtimeTarget: 'transformers-local',
    checkpointId: 'ft-gemma4-e2b-r16',
  });
  assert.equal(onnxPlan.deploymentFormat, 'ONNX');
  assert.ok(onnxPlan.outputFilename.endsWith('.onnx'));
  assert.ok(onnxPlan.localDeployCommand.includes('onnxruntime'));
});

test('buildRadarChartComparisonData produces 5 side-by-side radar axes including MMLU-Pro, GSM8K, HumanEval, and IFEval', () => {
  const axes = buildRadarChartComparisonData(
    {
      mmluPro: 61.4,
      gsm8k: 76.8,
      humanEval: 71.2,
      ifEval: 73.5,
      mtBench: 7.82,
      perplexity: 5.84,
      tokensPerSec: 248.6,
    },
    {
      mmluPro: 65.8,
      gsm8k: 83.4,
      humanEval: 79.1,
      ifEval: 81.2,
      mtBench: 8.36,
      perplexity: 5.08,
      tokensPerSec: 247.9,
    }
  );
  assert.equal(axes.length, 5);
  assert.equal(axes[0].metric, 'MMLU-Pro');
  assert.equal(axes[0].delta, 4.4);
  assert.equal(axes[1].metric, 'GSM8K');
  assert.equal(axes[2].metric, 'HumanEval');
  assert.equal(axes[3].metric, 'IFEval');
});

test('ML_GLOSSARY contains comprehensive instructive definitions (whatItIs and whatItTellsUs) for PPL, benchmarks, hyperparameters, and formats', () => {
  const keys = Object.keys(ML_GLOSSARY);
  assert.ok(keys.length >= 18);
  assert.ok(ML_GLOSSARY.ppl.whatItIs.length > 20);
  assert.ok(ML_GLOSSARY.ppl.whatItTellsUs.length > 20);
  assert.ok(ML_GLOSSARY.ggufQ4KM.whatItTellsUs.length > 20);
  assert.ok(ML_GLOSSARY.gptq.whatItTellsUs.length > 20);
  assert.ok(ML_GLOSSARY.onnx.whatItTellsUs.length > 20);
});

test('serializeTrainingSeriesMap and deserializeTrainingSeriesMap persist and restore per-model trainingSeries accurately', () => {
  const fallback = [
    {
      step: 50,
      epoch: 0.25,
      trainLoss: 1.98,
      valLoss: 2.04,
      gradNorm: 1.42,
      learningRate: 0.0002,
      tokensPerSec: 9420,
    },
  ];
  const customSeries = [
    ...fallback,
    {
      step: 400,
      epoch: 2.0,
      trainLoss: 0.75,
      valLoss: 0.86,
      gradNorm: 0.65,
      learningRate: 0.0002,
      tokensPerSec: 9925,
    },
  ];

  const serialized = serializeTrainingSeriesMap({
    'Qwen/Qwen3.5-2B-Instruct': customSeries,
  });

  const restored = deserializeTrainingSeriesMap(serialized, fallback, [
    'Qwen/Qwen3.5-2B-Instruct',
    'LiquidAI/LFM2.5-1.2B-Instruct',
  ]);

  assert.equal(restored['Qwen/Qwen3.5-2B-Instruct'].length, 2);
  assert.equal(restored['Qwen/Qwen3.5-2B-Instruct'][1].step, 400);
  assert.equal(restored['LiquidAI/LFM2.5-1.2B-Instruct'].length, 1);
  assert.equal(restored['LiquidAI/LFM2.5-1.2B-Instruct'][0].step, 50);

  // Verify graceful fallback on corrupted JSON
  const corrupted = deserializeTrainingSeriesMap('{invalid_json', fallback, [
    'Qwen/Qwen3.5-2B-Instruct',
  ]);
  assert.equal(corrupted['Qwen/Qwen3.5-2B-Instruct'].length, 1);
});

test('buildBenchmarkSummaryTableRows computes base vs fine-tuned MMLU-Pro and HumanEval deltas accurately', () => {
  const rows = buildBenchmarkSummaryTableRows([
    {
      id: 'Qwen/Qwen3.5-2B-Instruct',
      name: 'Qwen3.5-2B-Instruct',
      architecture: 'Qwen3.5',
      paramsBillions: 2.0,
      fineTunedCheckpointId: 'ft-qwen3.5-2b-r32',
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
    },
  ]);

  assert.equal(rows.length, 1);
  assert.equal(rows[0].hasFineTuned, true);
  assert.equal(rows[0].mmluProDelta, 4.4);
  assert.equal(rows[0].humanEvalDelta, 7.9);
  assert.equal(rows[0].verdict, 'PROMOTE_TO_PRODUCTION');
});
