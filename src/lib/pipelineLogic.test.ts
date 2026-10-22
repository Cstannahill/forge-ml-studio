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
  computePreferenceAlignmentPlan,
  computePackingAndMaskingMetrics,
  computeMultiAdapterMergePlan,
  evaluateEarlyStoppingWatchdog,
  generateExecutableRunBundle,
  scoreAndPruneDatasetByIfdAndDecontamination,
  generateSyntheticDistillationBatch,
  computeTokenLengthHistogramAndTruncation,
  generateArenaComparisonResponse,
  diffExperimentSnapshots,
  computeImatrixQuantizationCalibration,
  computeGrpoVerifiableRewardPlan,
  buildCurriculumPacingPlan,
  computeLayerWiseAdaptiveLoraPlan,
  inspectTokenizerAndChatTemplate,
  parseCustomDatasetUploadText,
  runAutomatedRedTeamStressSuite,
  generateLocalInferenceAndSdkBundle,
  simulateHardwareAndEdgeTargets,
  filterCommandPaletteActions,
  buildTextPythonEngineManifest,
  buildVisionPythonEngineManifest,
  computeSpinAndBestOfNAlignmentPlan,
  computeWhiteBoxLogitDistillationPlan,
  computeLoftQAndQatEdgeHardeningPlan,
  computeContinualReplayAndLisaPlan,
  buildWebGpuAndTransformersJsExportBundle,
  runPipelineDoctorAudit,
  parseEngineMetricLine,
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

test('computePreferenceAlignmentPlan calculates reference-free VRAM savings for ORPO/SimPO vs DPO/KTO', () => {
  const orpo = computePreferenceAlignmentPlan({
    method: 'ORPO',
    beta: 0.1,
    preferenceDataset: 'argilla/ultrafeedback-binarized-preferences-cleaned',
    paramsBillions: 2.0,
    baseIfEval: 73.5,
    baseMtBench: 7.82,
  });
  const dpo = computePreferenceAlignmentPlan({
    method: 'DPO',
    beta: 0.1,
    preferenceDataset: 'argilla/ultrafeedback-binarized-preferences-cleaned',
    paramsBillions: 2.0,
    baseIfEval: 73.5,
    baseMtBench: 7.82,
  });

  assert.equal(orpo.requiresReferenceModel, false);
  assert.equal(orpo.vramSavingsPct, 44);
  assert.equal(dpo.requiresReferenceModel, true);
  assert.ok(orpo.estimatedVramGb < dpo.estimatedVramGb);
  assert.ok(orpo.projectedIfEvalGain > 3.0);
});

test('computePackingAndMaskingMetrics computes padding waste reduction and sub-2B accuracy lift', () => {
  const packed = computePackingAndMaskingMetrics({
    samplePackingEnabled: true,
    completionOnlyLossEnabled: true,
    maxSeqLength: 4096,
    avgSampleTokens: 1200,
    baseTokensPerSec: 9500,
  });
  const unpacked = computePackingAndMaskingMetrics({
    samplePackingEnabled: false,
    completionOnlyLossEnabled: false,
    maxSeqLength: 4096,
    avgSampleTokens: 1200,
    baseTokensPerSec: 9500,
  });

  assert.ok(packed.paddingWastePct < unpacked.paddingWastePct);
  assert.ok(packed.effectiveThroughputTokSec > unpacked.effectiveThroughputTokSec);
  assert.equal(packed.estimatedSub2bAccuracyGainPct, 3.3);
});

test('computeMultiAdapterMergePlan normalizes adapter weights and generates mergekit YAML and PEFT scripts', () => {
  const mergePlan = computeMultiAdapterMergePlan({
    adapters: [
      { id: 'code-r32', name: 'Code Adapter', domain: 'CODING', weight: 0.6, loraRank: 32, enabled: true },
      { id: 'math-r32', name: 'Math Adapter', domain: 'REASONING_MATH', weight: 0.4, loraRank: 32, enabled: true },
    ],
    algorithm: 'DARE-TIES',
    density: 0.8,
    baseModelId: 'Qwen/Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
  });

  assert.equal(mergePlan.activeAdapterCount, 2);
  assert.equal(mergePlan.normalizedWeights[0].normalizedWeight, 0.6);
  assert.equal(mergePlan.normalizedWeights[1].normalizedWeight, 0.4);
  assert.ok(mergePlan.mergekitYaml.includes('dare_ties'));
  assert.ok(mergePlan.peftMergeScript.includes('add_weighted_adapter'));
});

test('evaluateEarlyStoppingWatchdog detects overfitting divergence and identifies best checkpoint', () => {
  const report = evaluateEarlyStoppingWatchdog({
    trainingSeries: [
      { step: 100, epoch: 0.5, trainLoss: 1.4, valLoss: 1.45, gradNorm: 1.1, learningRate: 0.0002, tokensPerSec: 9500 },
      { step: 150, epoch: 0.75, trainLoss: 0.92, valLoss: 0.96, gradNorm: 0.8, learningRate: 0.0002, tokensPerSec: 9600 },
      { step: 200, epoch: 1.0, trainLoss: 0.74, valLoss: 1.12, gradNorm: 1.2, learningRate: 0.0002, tokensPerSec: 9600 },
    ],
    patienceSteps: 1,
    minDelta: 0.005,
  });

  assert.equal(report.bestCheckpoint.step, 150);
  assert.equal(report.bestIndex, 1);
  assert.equal(report.shouldTriggerEarlyStop, true);
  assert.equal(report.statusLabel, 'EARLY_STOP_TRIGGERED');
});

test('generateExecutableRunBundle outputs Axolotl YAML, Unsloth Python, TRL ORPO script, Blend JSON, and Modelfile', () => {
  const bundle = generateExecutableRunBundle({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    contextWindow: 32768,
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
    datasetSources: ['Qwen/Qwen3-Code-Reasoning-110k'],
    targetFormat: 'chatml',
    samplePackingEnabled: true,
    completionOnlyLossEnabled: true,
    alignmentStage: {
      enabled: true,
      method: 'ORPO',
      beta: 0.1,
      preferenceDataset: 'argilla/ultrafeedback-binarized-preferences-cleaned',
    },
  });

  assert.ok(bundle.axolotlConfigYaml.includes('sample_packing: true'));
  assert.ok(bundle.axolotlConfigYaml.includes('train_on_inputs: false'));
  assert.ok(bundle.unslothTrainPy.includes('FastLanguageModel.get_peft_model'));
  assert.ok(bundle.trlSftOrpoPy.includes('ORPOTrainer'));
  assert.ok(bundle.ollamaModelfile.includes('FROM ./qwen3.5-2b-instruct-merged.safetensors'));
});

test('generateSyntheticDistillationBatch, scoreAndPruneDatasetByIfdAndDecontamination, and computeTokenLengthHistogramAndTruncation work end-to-end', () => {
  const distilled = generateSyntheticDistillationBatch({
    seedPrompt: 'Implement a thread-safe LRU cache in Python',
    domainTag: 'code-python',
    teacherModel: 'gemini-3-flash-preview',
    minRubricScore: 8.5,
    targetFormat: 'chatml',
  });

  assert.equal(distilled.length, 3);
  const passing = distilled.filter((d) => d.passedRubric);
  assert.equal(passing.length, 2);

  const samples = distilled.map((d) => d.normalizedSample);
  const pruned = scoreAndPruneDatasetByIfdAndDecontamination({
    samples,
    minIfd: 0.2,
    maxIfd: 0.95,
    enableDecontamination: true,
  });
  assert.ok(pruned.meanIfdScore > 0);
  assert.equal(pruned.benchmarkDecontaminationSummary.length, 4);

  const hist = computeTokenLengthHistogramAndTruncation({
    samples,
    contextCutoffTokens: 64,
  });
  assert.equal(hist.buckets.length, 5);
  assert.ok(hist.maxSampleTokens > 0);
});

test('generateArenaComparisonResponse and diffExperimentSnapshots produce accurate side-by-side comparisons', () => {
  const arena = generateArenaComparisonResponse({
    prompt: 'Extract strict JSON',
    presetCategory: 'JSON_SCHEMA',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    stage2Aligned: true,
  });
  assert.equal(arena.constraintPassedBase, false);
  assert.equal(arena.constraintPassedFineTuned, true);
  assert.ok(arena.fineTunedOutput.includes('<think>'));

  const snapA = {
    id: 'a',
    name: 'Run A',
    createdAt: '2026-10-20',
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5' as const,
    paramsBillions: 2.0,
    datasetVersionId: 'v2.3.0',
    hpoConfig: {
      learningRate: 0.00015,
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0.05,
      warmupRatio: 0.05,
      weightDecay: 0.01,
      neftuneAlpha: 0,
      batchSize: 4,
      lrScheduler: 'cosine' as const,
    },
    samplePackingEnabled: false,
    completionOnlyLossEnabled: false,
    stage2Method: 'NONE' as const,
    finalStep: 300,
    finalTrainLoss: 0.95,
    finalValLoss: 1.05,
    mmluPro: 62.0,
    humanEval: 73.0,
    gsm8k: 78.0,
    ifEval: 75.0,
  };
  const snapB = {
    ...snapA,
    id: 'b',
    name: 'Run B',
    hpoConfig: { ...snapA.hpoConfig, loraRank: 32, loraAlpha: 64 },
    samplePackingEnabled: true,
    stage2Method: 'ORPO' as const,
    finalValLoss: 0.88,
    mmluPro: 65.8,
    humanEval: 79.1,
    gsm8k: 83.4,
    ifEval: 81.2,
  };

  const diff = diffExperimentSnapshots(snapA, snapB);
  assert.equal(diff.mmluProDelta, 3.8);
  assert.equal(diff.humanEvalDelta, 6.1);
  assert.ok(diff.changedHyperparameters.length >= 3);
});

test('computeImatrixQuantizationCalibration recovers quantization perplexity degradation and protects salient layers', () => {
  const cal = computeImatrixQuantizationCalibration({
    modelId: 'LiquidAI/LFM2.5-1.2B-Instruct',
    modelName: 'LFM2.5-1.2B-Instruct',
    architecture: 'LFM2.5',
    paramsBillions: 1.2,
    basePerplexity: 4.82,
    quantType: 'Q4_K_M',
    calibrationSamplesCount: 256,
    protectSalientLayers: true,
    datasetSourceLabel: 'Qwen/Qwen3-Code-Reasoning-110k',
  });

  assert.ok(cal.imatrixQuantPerplexity < cal.naiveQuantPerplexity);
  assert.ok(cal.imatrixQuantPerplexity > cal.fp16Perplexity);
  assert.ok(cal.perplexityRecoveryPct >= 70);
  assert.ok(cal.calibrationCliPipeline.includes('llama-imatrix'));
  assert.ok(cal.calibrationCliPipeline.includes('--output-tensor-type q8_0'));
});

test('computeGrpoVerifiableRewardPlan calculates zero Critic VRAM overhead and generates TRL GRPO script', () => {
  const grpo = computeGrpoVerifiableRewardPlan({
    enabled: true,
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    modelName: 'Qwen3.5-2B-Instruct',
    paramsBillions: 2.0,
    groupRolloutSize: 8,
    klBeta: 0.04,
    activeVerifiers: ['json_schema', 'python_pytest', 'math_symbolic'],
  });

  assert.equal(grpo.criticVramOverheadGb, 0);
  assert.ok(grpo.projectedHumanEvalGain > 3.0);
  assert.ok(grpo.projectedGsm8kGain > 3.5);
  assert.equal(grpo.rewardTrajectory.length, 5);
  assert.ok(grpo.trlGrpoScriptPy.includes('GRPOTrainer'));
});

test('buildCurriculumPacingPlan, computeLayerWiseAdaptiveLoraPlan, and inspectTokenizerAndChatTemplate work accurately', () => {
  const s1 = normalizeDatasetRecord(
    { instruction: 'Short prompt', input: '', output: 'Short answer' },
    'test/short',
    0,
    'chatml'
  );
  const s2 = normalizeDatasetRecord(
    {
      instruction: 'Complex multi-step problem',
      input: '',
      output: '<think>\nStep 1... Step 2... Step 3...\n</think>\ndef solve():\n    return 42',
    },
    'test/long',
    1,
    'chatml'
  );

  const curriculum = buildCurriculumPacingPlan({
    samples: [s2, s1],
    strategy: 'hybrid_ifd_length',
    totalSteps: 600,
  });
  assert.equal(curriculum.phases.length, 3);
  assert.equal(curriculum.orderedSamples[0].id, s1.id);

  const dora = computeLayerWiseAdaptiveLoraPlan({
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    baseRank: 32,
    baseAlpha: 64,
    useDora: true,
    useRsLora: true,
    allocationPolicy: 'svd_salience',
  });
  assert.equal(dora.layerQuartiles.length, 4);
  assert.ok(dora.peftConfigSnippet.includes('use_dora=True'));
  assert.ok(dora.projectedMmluGainPct >= 2.5);

  const tokDiag = inspectTokenizerAndChatTemplate({
    architecture: 'Qwen3.5',
    modelName: 'Qwen3.5-2B-Instruct',
    targetFormat: 'chatml',
    injectReasoningTokens: true,
    injectToolTokens: true,
  });
  assert.equal(tokDiag.isEosAligned, true);
  assert.equal(tokDiag.requiresResizeTokenEmbeddings, true);
  assert.ok(tokDiag.surgeryScriptPy.includes('resize_token_embeddings'));
});

test('parseCustomDatasetUploadText, runAutomatedRedTeamStressSuite, generateLocalInferenceAndSdkBundle, simulateHardwareAndEdgeTargets, and filterCommandPaletteActions work end-to-end', () => {
  const jsonlInput = `{"instruction": "Convert 10C to F", "output": "50F"}\n{"instruction": "Write JSON", "output": "{\\"ok\\": true}"}`;
  const parsed = parseCustomDatasetUploadText({
    rawText: jsonlInput,
    sourceLabel: 'UserUpload/custom.jsonl',
    targetFormat: 'chatml',
  });
  assert.equal(parsed.detectedFormat, 'JSONL');
  assert.equal(parsed.parsedSamples.length, 2);

  const redTeam = runAutomatedRedTeamStressSuite({
    modelName: 'Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    paramsBillions: 2.0,
    stage2Aligned: true,
  });
  assert.equal(redTeam.totalProbeCount, 4);
  assert.equal(redTeam.passedProbeCount, 4);
  assert.ok(redTeam.fineTunedResilienceScorePct > redTeam.baseResilienceScorePct);

  const sdk = generateLocalInferenceAndSdkBundle({
    modelName: 'Qwen3.5-2B-Instruct',
    checkpointTag: 'ft-qwen3.5-2b-r32',
    exportFormat: 'GGUF-Q4_K_M',
    contextWindow: 8192,
    port: 8080,
    temperature: 0.2,
    repeatPenalty: 1.08,
    runtime: 'llama_cpp',
  });
  assert.ok(sdk.serverLaunchCommand.includes('llama-server'));
  assert.ok(sdk.pythonSdkSnippet.includes('OpenAI'));

  const hwSim = simulateHardwareAndEdgeTargets({
    paramsBillions: 1.2,
    architecture: 'LFM2.5',
    quantization: 'qlora-4bit',
    contextLength: 4096,
    trainingGpuId: 'rtx3060-12gb',
    edgeDeviceId: 'macbook-air-m2-8gb',
  });
  assert.equal(hwSim.trainingProfile.fitsInVram, true);
  assert.ok(hwSim.edgeProfile.tokPerSecQ4Km > hwSim.edgeProfile.tokPerSecFp16);

  const filteredActions = filterCommandPaletteActions(
    [
      {
        id: 'nav-datasets',
        title: 'Go to Dataset Studio',
        category: 'NAVIGATION',
        keywords: ['dataset', 'curriculum', 'upload'],
      },
      {
        id: 'act-grpo',
        title: 'Run Stage 3 GRPO Verifier Pass',
        category: 'PIPELINE_ACTION',
        keywords: ['grpo', 'rlvr', 'reward'],
      },
    ],
    'grpo'
  );
  assert.equal(filteredActions.length, 1);
  assert.equal(filteredActions[0].id, 'act-grpo');
});

import {
  CURATED_VISION_BLENDS,
  INITIAL_VISION_MODELS,
  VEIL_AUDIT_TAXONOMY,
  auditVisionTokenizerSpecialTokens,
  buildInitialVisionSamples,
  buildVeilPrivacyAuditSeedSamples,
  buildVisionCurriculumPlan,
  calculateVisualPatchTokens,
  capExcessiveNavitPatches,
  cleanAndNormalizeVisionSample,
  computeVisionPipelineHardwarePlan,
  computeVisionTokenizerAndPatchDensity,
  generateVisionTeacherDistillationBatch,
  parseCustomVisionDatasetUpload,
  pruneVisionDatasetByAlignmentAndPhash,
  validateVisionDatasetCorpus,
  VEIL_SANDBOX_TEST_SCENES,
  computeBox2dIoU,
  computeVisionGrpoIoURewardPlan,
  evaluateVeilVisualSandboxScene,
} from './visionPipelineLogic.ts';

test('INITIAL_VISION_MODELS and CURATED_VISION_BLENDS include 5 multimodal models and 25 HuggingFace dataset blends including VEIL Privacy Hazard Auditing', () => {
  const modelIds = INITIAL_VISION_MODELS.map((m) => m.id);
  assert.ok(modelIds.includes('LiquidAI/LFM2.5-VL-450M-Instruct'));
  assert.ok(modelIds.includes('LiquidAI/LFM2.5-VL-1.6B-Instruct'));
  assert.ok(modelIds.includes('Qwen/Qwen3.5-VL-0.8B-Instruct'));
  assert.ok(modelIds.includes('Qwen/Qwen3.5-VL-2B-Instruct'));
  assert.ok(modelIds.includes('google/gemma-4-e2b-vision-it'));
  assert.equal(CURATED_VISION_BLENDS.length, 25);

  const veilBlends = CURATED_VISION_BLENDS.filter(
    (b) => b.taskCategory === 'VEIL_PRIVACY_AUDIT'
  );
  assert.equal(veilBlends.length, 6);

  // Every blend dataset weight sum must equal 100%
  for (const blend of CURATED_VISION_BLENDS) {
    const sumWeights = blend.datasets.reduce((acc, ds) => acc + ds.weightPct, 0);
    assert.equal(sumWeights, 100, `Expected blend ${blend.id} weights to sum to 100, got ${sumWeights}`);
  }
});

test('VEIL_AUDIT_TAXONOMY and buildVeilPrivacyAuditSeedSamples cover all 5 VEIL categories with 25/20/25/10/20 split and BLUR/MOSAIC/SAFE_PASS actions', () => {
  assert.equal(VEIL_AUDIT_TAXONOMY.length, 5);
  const totalSplit = VEIL_AUDIT_TAXONOMY.reduce((acc, c) => acc + c.splitAllocationPct, 0);
  assert.equal(totalSplit, 100);
  assert.equal(VEIL_AUDIT_TAXONOMY[3].defaultSuggestedAction, 'MOSAIC');
  assert.equal(VEIL_AUDIT_TAXONOMY[4].defaultSuggestedAction, 'SAFE_PASS');

  const veilSamples = buildVeilPrivacyAuditSeedSamples('QWEN_VL_CHATML');
  assert.equal(veilSamples.length, 5);
  assert.ok(veilSamples[3].assistantOutput.includes('"suggested_action": "MOSAIC"'));
  assert.ok(veilSamples[4].assistantOutput.includes('"suggested_action": "SAFE_PASS"'));
});

test('computeVisionTokenizerAndPatchDensity and capExcessiveNavitPatches quantify image-to-text token ratios and clamp excessive NaViT patches', () => {
  const initialSamples = buildInitialVisionSamples('QWEN_VL_CHATML');
  assert.equal(initialSamples.length, 6);

  const densityBefore = computeVisionTokenizerAndPatchDensity(initialSamples, 5.5, 896);
  assert.equal(densityBefore.totalSamples, 6);
  assert.equal(densityBefore.ratioDistributionBuckets.length, 4);
  assert.ok(densityBefore.excessiveNavitSampleCount >= 1);
  assert.ok(densityBefore.excessiveSampleIds.includes('vlm-sample-006'));
  assert.ok(densityBefore.maxImageToTextRatio > 6.5);

  const capped = capExcessiveNavitPatches(initialSamples, 896, 5.5, 'QWEN_VL_CHATML');
  assert.ok(capped.cappedCount >= 1);
  assert.ok(capped.tokensSavedPerEpoch > 100);

  const densityAfter = computeVisionTokenizerAndPatchDensity(capped.updatedSamples, 5.5, 896);
  assert.equal(densityAfter.excessiveNavitSampleCount, 0);
});

test('computeVisionPipelineHardwarePlan isolates VLM gradients, calculates frozen vs unfrozen Vision Tower VRAM, and outputs mmproj-f16.gguf', () => {
  const lfm450m = INITIAL_VISION_MODELS[0];
  const frozenPlan = computeVisionPipelineHardwarePlan(
    lfm450m,
    {
      freezeVisionTower: true,
      trainMlpProjector: true,
      applyLoraToLlmBackbone: true,
      minPixels: 256 * 28 * 28,
      maxPixels: 768 * 28 * 28,
      patchSizePx: 28,
      enable2dRopeScaling: true,
      maskImagePromptTokens: true,
      exportMmprojFp16: true,
    },
    32,
    8
  );
  const unfrozenPlan = computeVisionPipelineHardwarePlan(
    lfm450m,
    {
      freezeVisionTower: false,
      trainMlpProjector: true,
      applyLoraToLlmBackbone: true,
      minPixels: 256 * 28 * 28,
      maxPixels: 768 * 28 * 28,
      patchSizePx: 28,
      enable2dRopeScaling: true,
      maskImagePromptTokens: true,
      exportMmprojFp16: true,
    },
    32,
    8
  );
  assert.ok(frozenPlan.visionTowerVramGb < unfrozenPlan.visionTowerVramGb);
  assert.ok(frozenPlan.mmprojExportFilename.endsWith('-f16.gguf'));
  assert.ok(frozenPlan.modalityIsolationGuarantee.includes('ISOLATED_VLM_TRACK'));
});

test('Vision Dataset Studio functions: validateVisionDatasetCorpus, cleanAndNormalizeVisionSample, generateVisionTeacherDistillationBatch, pruneVisionDatasetByAlignmentAndPhash, buildVisionCurriculumPlan, auditVisionTokenizerSpecialTokens, and parseCustomVisionDatasetUpload work end-to-end', () => {
  const initialSamples = buildInitialVisionSamples('QWEN_VL_CHATML');
  assert.equal(initialSamples.length, 6);
  assert.ok(initialSamples[0].formattedPayload.includes('<|vision_start|><|image_pad|><|vision_end|>'));

  const reportBefore = validateVisionDatasetCorpus(initialSamples);
  assert.equal(reportBefore.totalSamples, 6);
  assert.equal(reportBefore.exifOrFacePiiCount, 1);

  const cleanedSamples = initialSamples.map((s) =>
    cleanAndNormalizeVisionSample(s, 'QWEN_VL_CHATML')
  );
  const reportAfter = validateVisionDatasetCorpus(cleanedSamples);
  assert.equal(reportAfter.exifOrFacePiiCount, 0);
  assert.equal(reportAfter.healthScore, 100);

  const distilled = generateVisionTeacherDistillationBatch({
    teacherModel: 'Qwen2.5-VL-72B-Instruct',
    taskCategory: 'UI_GROUNDING',
    seedTopic: 'Mobile checkout button grounding',
    targetFormat: 'QWEN_VL_CHATML',
  });
  assert.equal(distilled.samples.length, 2);
  assert.ok(distilled.avgJudgeScore >= 9.0);

  const pruned = pruneVisionDatasetByAlignmentAndPhash(cleanedSamples, 0.90);
  assert.ok(pruned.retainedSamples.length >= 5);
  assert.ok(pruned.meanAlignmentAfter >= pruned.meanAlignmentBefore);

  const curriculum = buildVisionCurriculumPlan(cleanedSamples, 600);
  assert.equal(curriculum.phases.length, 3);
  assert.equal(curriculum.orderedSamples.length, 6);

  const tokAudit = auditVisionTokenizerSpecialTokens('Qwen3.5');
  assert.equal(tokAudit.allSpecialTokensAtomic, true);
  assert.equal(tokAudit.tokens.length, 7);

  const customParsed = parseCustomVisionDatasetUpload(
    '{"image": "s3://vision/receipt_01.png", "width": 1280, "height": 960, "instruction": "Extract total", "output": "{\\"total\\": 42.50}", "box2d": [100, 200, 800, 900]}',
    'CustomVision/receipts.jsonl',
    'QWEN_VL_CHATML'
  );
  assert.equal(customParsed.detectedSchema, 'VISION_JSONL');
  assert.equal(customParsed.samples.length, 1);
  assert.equal(calculateVisualPatchTokens(1024, 768), 1003);
});

test('buildTextPythonEngineManifest and buildVisionPythonEngineManifest generate executable Python CLI manifests and commands', () => {
  const textBundle = buildTextPythonEngineManifest({
    modelId: 'Qwen/Qwen3.5-2B-Instruct',
    architecture: 'Qwen3.5',
    contextWindow: 8192,
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
    useDora: true,
    useRsLora: true,
    samplePackingEnabled: true,
    completionOnlyLossEnabled: true,
    stage2Enabled: true,
    alignmentMethod: 'ORPO',
    alignmentBeta: 0.1,
    preferenceDataset: 'argilla/ultrafeedback-binarized-preferences-cleaned',
    grpoEnabled: true,
    grpoRolloutSize: 8,
    grpoKlBeta: 0.04,
    grpoVerifiers: ['json_schema', 'python_pytest'],
  });

  assert.equal(textBundle.track, 'TEXT_LLM');
  assert.equal(textBundle.manifestFilename, 'training_manifest.json');
  assert.ok(textBundle.pythonTrainCommand.includes('python3 engine/train_text_pipeline.py --config'));
  assert.ok(textBundle.pythonDryRunCommand.includes('--dry-run'));
  assert.ok(textBundle.pythonExportCommand.includes('python3 engine/export_and_merge.py'));
  const parsedTextManifest = JSON.parse(textBundle.manifestJson);
  assert.equal(parsedTextManifest.hpoConfig.loraRank, 32);
  assert.equal(parsedTextManifest.preferenceConfig.method, 'ORPO');
  assert.equal(parsedTextManifest.grpoConfig.enabled, true);

  const visionBundle = buildVisionPythonEngineManifest({
    modelId: 'LiquidAI/LFM2.5-VL-450M-Instruct',
    architecture: 'LFM2.5',
    visionEncoder: 'SigLIP-NaViT-400M',
    projectorType: 'Spatial-2x2-MLP-Connector',
    freezeVisionTower: true,
    trainMlpProjector: true,
    loraRank: 32,
    batchSize: 8,
    minPixels: 256 * 28 * 28,
    maxPixels: 1024 * 28 * 28,
    patchSizePx: 28,
  });

  assert.equal(visionBundle.track, 'VISION_VLM');
  assert.equal(visionBundle.manifestFilename, 'vision_manifest.json');
  assert.ok(visionBundle.pythonTrainCommand.includes('python3 engine/train_vision_pipeline.py --config'));
  assert.ok(visionBundle.pythonExportCommand.includes('--mmproj'));
  const parsedVisionManifest = JSON.parse(visionBundle.manifestJson);
  assert.equal(parsedVisionManifest.pipelineTrack, 'VISION_VLM_ISOLATED');
  assert.equal(parsedVisionManifest.hardwareConfig.freezeVisionTower, true);
});

test('computeSpinAndBestOfNAlignmentPlan, computeWhiteBoxLogitDistillationPlan, computeLoftQAndQatEdgeHardeningPlan, computeContinualReplayAndLisaPlan, buildWebGpuAndTransformersJsExportBundle, runPipelineDoctorAudit, and parseEngineMetricLine work accurately', () => {
  const spinPlan = computeSpinAndBestOfNAlignmentPlan({
    modelId: 'Qwen/Qwen3.5-0.8B-Instruct',
    paramsBillions: 0.8,
    config: {
      enabled: true,
      candidatesPerPrompt: 8,
      spinIterations: 2,
      samplingTemperature: 0.7,
      judgeMarginThreshold: 0.25,
    },
  });
  assert.equal(spinPlan.totalGeneratedRolloutsPer1k, 16000);
  assert.ok(spinPlan.hallucinationLoopReductionPct > 50);
  assert.equal(spinPlan.iterationTrajectory.length, 2);
  assert.ok(spinPlan.spinPythonSnippet.includes('SamplingParams'));

  const kdPlan = computeWhiteBoxLogitDistillationPlan({
    studentModelId: 'LiquidAI/LFM2.5-350M-Instruct',
    studentParamsBillions: 0.35,
    config: {
      enabled: true,
      teacherModelId: 'Qwen/Qwen3.5-9B-Instruct',
      divergenceType: 'REVERSE_KL',
      temperature: 2.0,
      alphaKdWeight: 0.5,
      topKLogitsCached: 64,
    },
  });
  assert.ok(kdPlan.logitCacheSizeMbPer10kSamples > 1000);
  assert.ok(kdPlan.darkKnowledgeTransferEfficiencyPct >= 85);
  assert.ok(kdPlan.projectedGsm8kGainPct > 3.0);
  assert.ok(kdPlan.kdTrainerPythonSnippet.includes('Reverse-KL'));

  const loftqPlan = computeLoftQAndQatEdgeHardeningPlan({
    modelId: 'Qwen/Qwen3.5-0.8B-Instruct',
    baseFp16Perplexity: 5.12,
    loraRank: 32,
    config: {
      enableLoftQInit: true,
      loftqIterSteps: 4,
      enableFakeQuantQat: true,
      qatTailStepsPct: 15,
      targetEdgeBitwidth: 'GGUF_Q3_K_M',
    },
  });
  assert.ok(loftqPlan.loftqQatRecoveredPerplexity < loftqPlan.ptqBaselinePerplexity);
  assert.ok(loftqPlan.svdResidualErrorReductionPct > 60);
  assert.ok(loftqPlan.loftqPythonSnippet.includes('LoftQConfig'));

  const lisaPlan = computeContinualReplayAndLisaPlan({
    totalLayers: 24,
    config: {
      replayBufferRatioPct: 10,
      replayAnchorSource: 'HuggingFaceH4/ultrachat_200k',
      enableLisaLayerSampling: true,
      lisaActiveMiddleLayers: 4,
    },
  });
  assert.equal(lisaPlan.catastrophicForgettingRisk, 'MINIMAL');
  assert.ok(lisaPlan.generalCapabilityRetentionPct >= 95);
  assert.ok(lisaPlan.lisaOptimizerVramSavingsPct > 35);

  const webgpuBundle = buildWebGpuAndTransformersJsExportBundle({
    modelId: 'LiquidAI/LFM2.5-350M-Instruct',
    modelName: 'LFM2.5-350M-Instruct',
    paramsBillions: 0.35,
    contextWindow: 8192,
    dtype: 'q4f16',
  });
  assert.ok(webgpuBundle.estimatedBrowserVramMb < 1000);
  assert.ok(webgpuBundle.estimatedWebGpuTokPerSec > 80);
  assert.ok(webgpuBundle.transformersJsSnippet.includes('device: "webgpu"'));

  const doctorReport = runPipelineDoctorAudit({
    modelName: 'Qwen3.5-2B-Instruct',
    paramsBillions: 2.0,
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
    samplePackingEnabled: true,
    completionOnlyLossEnabled: true,
    stage2Enabled: true,
    alignmentMethod: 'ORPO',
    grpoEnabled: true,
    vramUsedGb: 11.4,
    vramTotalGb: 24,
    datasetHealthScore: 98,
    freezeVisionTower: true,
    excessiveNavitSampleCount: 0,
  });
  assert.equal(doctorReport.checks.length, 12);
  assert.equal(doctorReport.overallScore, 100);
  assert.equal(doctorReport.readinessVerdict, 'PRODUCTION_READY');

  const parsedMetric = parseEngineMetricLine(
    '[FORGEML_METRIC] {"step": 120, "epoch": 1.2, "trainLoss": 0.78, "valLoss": 0.84, "gradNorm": 0.89, "learningRate": 0.00018, "tokensPerSec": 14200}'
  );
  assert.ok(parsedMetric !== null);
  assert.equal(parsedMetric?.step, 120);
  assert.equal(parsedMetric?.trainLoss, 0.78);
});

test('computeBox2dIoU, computeVisionGrpoIoURewardPlan, and evaluateVeilVisualSandboxScene verify spatial grounding and VEIL redaction scenes', () => {
  const iouExact = computeBox2dIoU([200, 100, 600, 500], [200, 100, 600, 500]);
  assert.equal(iouExact, 1.0);
  const iouDrifted = computeBox2dIoU([220, 140, 560, 540], [180, 95, 620, 610]);
  assert.ok(iouDrifted > 0.5 && iouDrifted < 0.85);

  const visionGrpo = computeVisionGrpoIoURewardPlan({
    modelId: 'LiquidAI/LFM2.5-VL-450M-Instruct',
    totalParamsBillions: 0.45,
    config: {
      enabled: true,
      groupRolloutSize: 8,
      klBeta: 0.04,
      giouRewardWeight: 1.5,
      falsePositivePenaltyWeight: 1.0,
      activeVerifiers: [
        'giou_box_coordinate',
        'veil_false_positive_penalty',
        'ocr_exact_match',
        'json_schema_strict',
      ],
    },
  });
  assert.equal(visionGrpo.criticVramOverheadGb, 0);
  assert.ok(visionGrpo.projectedGroundingIoUGainPct > 8.0);
  assert.ok(visionGrpo.veilFalseAlarmReductionPct > 65.0);

  assert.equal(VEIL_SANDBOX_TEST_SCENES.length, 5);
  const scene1Eval = evaluateVeilVisualSandboxScene(VEIL_SANDBOX_TEST_SCENES[0]);
  assert.ok(scene1Eval.fineTunedMeanIoU > scene1Eval.baseMeanIoU);
  assert.equal(scene1Eval.recommendedRedactionMode, 'BLUR');

  const scene4Eval = evaluateVeilVisualSandboxScene(VEIL_SANDBOX_TEST_SCENES[3]);
  assert.equal(scene4Eval.recommendedRedactionMode, 'MOSAIC');

  const scene5Eval = evaluateVeilVisualSandboxScene(VEIL_SANDBOX_TEST_SCENES[4]);
  assert.equal(scene5Eval.falsePositiveEliminated, true);
  assert.equal(scene5Eval.recommendedRedactionMode, 'SAFE_PASS');
});

