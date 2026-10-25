import React, { useEffect, useState } from 'react';
import {
  Activity,
  Bookmark,
  Check,
  Clock,
  Copy,
  Cpu,
  Download,
  FileCode,
  FileText,
  GitCompare,
  GitMerge,
  HardDrive,
  Layers,
  PackageCheck,
  Play,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sliders,
  Sparkles,
  Terminal,
  Zap,
} from 'lucide-react';
import {
  AdapterMergeAlgorithm,
  EXPERIMENT_SNAPSHOTS_STORAGE_KEY,
  EdgeDeviceTargetId,
  ExperimentSnapshot,
  GpuAlertThresholds,
  GrpoVerifierId,
  HyperparameterConfig,
  ImatrixQuantType,
  LoggingFramework,
  LoraAllocationPolicy,
  PreferenceAlignmentMethod,
  SafetensorsDeploymentFormat,
  SafetensorsExportPlan,
  SafetensorsExportPrecision,
  SafetensorsRuntimeTarget,
  SpecialistAdapterInput,
  StandardizedModelCardResult,
  TrainingGpuTargetId,
  TrialOutcome,
  buildSafetensorsExportPlan,
  computeGrpoVerifiableRewardPlan,
  computeImatrixQuantizationCalibration,
  computeLayerWiseAdaptiveLoraPlan,
  computeMultiAdapterMergePlan,
  computeNextHyperparameters,
  computePackingAndMaskingMetrics,
  computePreferenceAlignmentPlan,
  computeTrainingProgressTelemetry,
  createSafetensorsBinaryArtifact,
  diffExperimentSnapshots,
  evaluateEarlyStoppingWatchdog,
  evaluateGpuThresholds,
  generateExecutableRunBundle,
  generateStandardizedModelCard,
  simulateHardwareAndEdgeTargets,
  buildTextPythonEngineManifest,
} from '../lib/pipelineLogic';
import {
  DatasetVersionCommit,
  LogEntry,
  RegisteredModel,
  TrainingMetricPoint,
} from '../data/initialPipelineData';
import { InfoTooltip } from './InfoTooltip';
import { NextGenQualityAndWebGpuPanel } from './NextGenQualityAndWebGpuPanel';
import { PythonEngineRunnerCard } from './PythonEngineRunnerCard';
import { ConstraintSolverAndScriptDiffPanel } from './ConstraintSolverAndScriptDiffPanel';

interface CommandCenterViewProps {
  models: RegisteredModel[];
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  trainingSeries: TrainingMetricPoint[];
  isTrainingActive: boolean;
  onRunTrainingStep: () => void;
  onResetTrainingSeries?: () => void;
  onRollbackTrainingToIndex?: (stepIndex: number) => void;
  onSimulateOverfitStep?: () => void;
  onApplyStage2AlignmentBoost?: (method: PreferenceAlignmentMethod, ifEvalGain: number, mtBenchGain: number) => void;
  onApplyStage3GrpoBoost?: (humanEvalGain: number, gsm8kGain: number, ifEvalGain: number) => void;
  onOpenCommandPalette?: () => void;
  onRunHpoSweep: () => void;
  activeHpoConfig: HyperparameterConfig;
  onUpdateHpoConfig: (cfg: HyperparameterConfig) => void;
  logs: LogEntry[];
  activeLogger: LoggingFramework;
  onSelectLogger: (fw: LoggingFramework) => void;
  onClearLogs: () => void;
  gpuStats: {
    vramUsedGb: number;
    vramTotalGb: number;
    smUtilPct: number;
    tensorCorePct: number;
    tempCelsius: number;
    powerWatts: number;
    nvlinkGbps: number;
  };
  vramHistoryGb: number[];
  activeDatasetCommit: DatasetVersionCommit;
  gpuThresholds: GpuAlertThresholds;
  onUpdateGpuThresholds: (thresholds: GpuAlertThresholds) => void;
  onSimulateGpuStressSpike: () => void;
  onMitigateGpuOverload: () => void;
  onRecordSafetensorsExport: (
    plan: SafetensorsExportPlan,
    model: RegisteredModel,
    modelCard?: StandardizedModelCardResult
  ) => void;
  onStreamEngineMetricPoint?: (pt: TrainingMetricPoint) => void;
}

export const CommandCenterView: React.FC<CommandCenterViewProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  trainingSeries,
  isTrainingActive,
  onRunTrainingStep,
  onResetTrainingSeries,
  onRollbackTrainingToIndex,
  onSimulateOverfitStep,
  onApplyStage2AlignmentBoost,
  onApplyStage3GrpoBoost,
  onOpenCommandPalette,
  onRunHpoSweep,
  activeHpoConfig,
  onUpdateHpoConfig,
  logs,
  activeLogger,
  onSelectLogger,
  onClearLogs,
  gpuStats,
  vramHistoryGb,
  activeDatasetCommit,
  gpuThresholds,
  onUpdateGpuThresholds,
  onSimulateGpuStressSpike,
  onMitigateGpuOverload,
  onRecordSafetensorsExport,
  onStreamEngineMetricPoint,
}) => {
  const gpuAlertEval = evaluateGpuThresholds(gpuStats, gpuThresholds);
  const [logLevelFilter, setLogLevelFilter] = useState<'ALL' | 'INFO' | 'METRIC' | 'WARN' | 'ERROR'>('ALL');
  const [logSearch, setLogSearch] = useState('');
  const [hoveredStep, setHoveredStep] = useState<TrainingMetricPoint | null>(null);
  const [targetEpochs, setTargetEpochs] = useState<number>(3);
  const [stepsPerEpoch, setStepsPerEpoch] = useState<number>(200);

  // Safetensors Merger & Exporter + Model Card Generator State (Optimized for <= 2B models)
  const [deploymentFormat, setDeploymentFormat] =
    useState<SafetensorsDeploymentFormat>('SAFETENSORS');
  const [exportPrecision, setExportPrecision] = useState<SafetensorsExportPrecision>('bfloat16');
  const [maxShardSizeGb, setMaxShardSizeGb] = useState<number>(5);
  const [runtimeTarget, setRuntimeTarget] = useState<SafetensorsRuntimeTarget>('ollama');
  const [includeTokenizer, setIncludeTokenizer] = useState<boolean>(true);
  const [autoIncludeModelCard, setAutoIncludeModelCard] = useState<boolean>(true);
  const [safetensorsInspectorTab, setSafetensorsInspectorTab] = useState<
    'modelcard' | 'header' | 'python' | 'deploy'
  >('modelcard');
  const [lastExportedArtifact, setLastExportedArtifact] = useState<{
    filename: string;
    modelCardFilename?: string;
    sizeGb: number;
    sha256: string;
    timestamp: string;
    runtime: SafetensorsRuntimeTarget;
  } | null>(null);
  const [copiedExportSnippet, setCopiedExportSnippet] = useState(false);

  const selectedModel = models.find((m) => m.id === selectedModelId) ?? models[0];
  const latestPoint = trainingSeries[trainingSeries.length - 1] ?? {
    step: 0,
    epoch: 0,
    trainLoss: 2.1,
    valLoss: 2.15,
    gradNorm: 1.2,
    learningRate: activeHpoConfig.learningRate,
    tokensPerSec: 9500,
  };

  const latestOutcome: TrialOutcome = {
    trialNumber: Math.max(1, Math.floor(trainingSeries.length / 2)),
    config: activeHpoConfig,
    trainLoss: latestPoint.trainLoss,
    valLoss: latestPoint.valLoss,
    gradNorm: latestPoint.gradNorm,
    evalScore: selectedModel.fineTunedBenchmarks?.mmluPro ?? selectedModel.baseBenchmarks.mmluPro,
  };

  const hpoProposal = computeNextHyperparameters(latestOutcome);

  const filteredLogs = logs.filter((entry) => {
    const matchesLevel = logLevelFilter === 'ALL' || entry.level === logLevelFilter;
    const matchesSearch =
      !logSearch.trim() ||
      entry.message.toLowerCase().includes(logSearch.toLowerCase()) ||
      entry.component.toLowerCase().includes(logSearch.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  const handleExportLogsJsonl = () => {
    const content = filteredLogs.map((l) => JSON.stringify(l)).join('\n');
    const blob = new Blob([content], { type: 'application/x-jsonlines' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forgeml-${activeLogger}-telemetry.jsonl`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // 1. Stage 2 Preference Alignment (ORPO / SimPO / DPO / KTO) & Sequence Packing State
  const [stage2Enabled, setStage2Enabled] = useState<boolean>(true);
  const [alignmentMethod, setAlignmentMethod] = useState<PreferenceAlignmentMethod>('ORPO');
  const [alignmentBeta, setAlignmentBeta] = useState<number>(0.1);
  const [preferenceDataset, setPreferenceDataset] = useState<string>(
    'argilla/ultrafeedback-binarized-preferences-cleaned'
  );
  const [alignmentExecutedMsg, setAlignmentExecutedMsg] = useState<string | null>(null);

  const [samplePackingEnabled, setSamplePackingEnabled] = useState<boolean>(true);
  const [completionOnlyLossEnabled, setCompletionOnlyLossEnabled] = useState<boolean>(true);

  // 2. Early-Stopping Watchdog & Checkpoint Rollback State
  const [watchdogPatience, setWatchdogPatience] = useState<number>(2);
  const [watchdogMinDelta, setWatchdogMinDelta] = useState<number>(0.005);
  const [watchdogRollbackBanner, setWatchdogRollbackBanner] = useState<string | null>(null);

  // 3. Multi-Adapter Merger Studio State (DARE-TIES / TIES / SLERP / LINEAR)
  const [mergeAlgorithm, setMergeAlgorithm] = useState<AdapterMergeAlgorithm>('DARE-TIES');
  const [mergeDensity, setMergeDensity] = useState<number>(0.8);
  const [specialistAdapters, setSpecialistAdapters] = useState<SpecialistAdapterInput[]>([
    {
      id: 'adapter-code-r32',
      name: 'Code & Refactoring Specialist (r=32)',
      domain: 'CODING',
      weight: 0.4,
      loraRank: 32,
      enabled: true,
    },
    {
      id: 'adapter-math-cot-r32',
      name: 'Math & Step-by-Step CoT Adapter (r=32)',
      domain: 'REASONING_MATH',
      weight: 0.3,
      loraRank: 32,
      enabled: true,
    },
    {
      id: 'adapter-json-tools-r16',
      name: 'Strict JSON & Tool-Calling Router (r=16)',
      domain: 'TOOL_CALLING',
      weight: 0.2,
      loraRank: 16,
      enabled: true,
    },
    {
      id: 'adapter-chat-ifeval-r16',
      name: 'IFEval Constraint & Multi-Turn Chat (r=16)',
      domain: 'INSTRUCTION_CHAT',
      weight: 0.1,
      loraRank: 16,
      enabled: true,
    },
  ]);
  const [customMergedCheckpointOverride, setCustomMergedCheckpointOverride] = useState<string | null>(
    null
  );
  const [adapterPreviewTab, setAdapterPreviewTab] = useState<'yaml' | 'python'>('yaml');

  // 4. 1-Click Executable Run Bundle Exporter State
  const [bundlePreviewTab, setBundlePreviewTab] = useState<
    'axolotl' | 'unsloth' | 'trl_orpo' | 'blend_json' | 'modelfile'
  >('unsloth');
  const [bundleDownloadedBanner, setBundleDownloadedBanner] = useState<string | null>(null);

  // 5. Saved Experiment Profiles & Run History Diffing (Persisted in localStorage)
  const defaultSnapshots: ExperimentSnapshot[] = [
    {
      id: 'snap-qwen2b-sft-base',
      name: 'Run #1: Qwen3.5-2B Standard SFT (r=16, Padded)',
      createdAt: '2026-10-19 14:20 UTC',
      modelId: 'Qwen/Qwen3.5-2B-Instruct',
      modelName: 'Qwen3.5-2B-Instruct',
      architecture: 'Qwen3.5',
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
        lrScheduler: 'cosine',
      },
      samplePackingEnabled: false,
      completionOnlyLossEnabled: false,
      stage2Method: 'NONE',
      finalStep: 300,
      finalTrainLoss: 0.94,
      finalValLoss: 1.06,
      mmluPro: 63.2,
      humanEval: 74.8,
      gsm8k: 79.5,
      ifEval: 76.4,
    },
    {
      id: 'snap-qwen2b-multipack-orpo',
      name: 'Run #2: Qwen3.5-2B Multipack + Stage 2 ORPO (r=32)',
      createdAt: '2026-10-20 09:45 UTC',
      modelId: 'Qwen/Qwen3.5-2B-Instruct',
      modelName: 'Qwen3.5-2B-Instruct',
      architecture: 'Qwen3.5',
      paramsBillions: 2.0,
      datasetVersionId: 'v2.4.0',
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
      stage2Method: 'ORPO',
      finalStep: 350,
      finalTrainLoss: 0.82,
      finalValLoss: 0.93,
      mmluPro: 65.8,
      humanEval: 79.1,
      gsm8k: 83.4,
      ifEval: 81.2,
    },
    {
      id: 'snap-lfm12b-edge',
      name: 'Run #3: LFM2.5-1.2B Liquid Edge + SimPO (r=16)',
      createdAt: '2026-10-20 18:10 UTC',
      modelId: 'LiquidAI/LFM2.5-1.2B-Instruct',
      modelName: 'LFM2.5-1.2B-Instruct (Liquid Edge)',
      architecture: 'LFM2.5',
      paramsBillions: 1.2,
      datasetVersionId: 'v2.4.0',
      hpoConfig: {
        learningRate: 0.00025,
        loraRank: 16,
        loraAlpha: 32,
        loraDropout: 0.05,
        warmupRatio: 0.04,
        weightDecay: 0.01,
        neftuneAlpha: 5,
        batchSize: 8,
        lrScheduler: 'wsd',
      },
      samplePackingEnabled: true,
      completionOnlyLossEnabled: true,
      stage2Method: 'SimPO',
      finalStep: 350,
      finalTrainLoss: 0.79,
      finalValLoss: 0.89,
      mmluPro: 61.9,
      humanEval: 73.5,
      gsm8k: 78.6,
      ifEval: 80.4,
    },
  ];

  const [experimentSnapshots, setExperimentSnapshots] = useState<ExperimentSnapshot[]>(() => {
    try {
      const raw = window.localStorage.getItem(EXPERIMENT_SNAPSHOTS_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length >= 2) {
          return parsed;
        }
      }
    } catch {
      // ignore storage errors
    }
    return defaultSnapshots;
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(
        EXPERIMENT_SNAPSHOTS_STORAGE_KEY,
        JSON.stringify(experimentSnapshots)
      );
    } catch {
      // ignore storage errors
    }
  }, [experimentSnapshots]);

  const [newSnapshotName, setNewSnapshotName] = useState('');
  const [diffSnapAId, setDiffSnapAId] = useState<string>(defaultSnapshots[0].id);
  const [diffSnapBId, setDiffSnapBId] = useState<string>(defaultSnapshots[1].id);
  const [snapshotStatusBanner, setSnapshotStatusBanner] = useState<string | null>(null);

  const alignmentPlan = computePreferenceAlignmentPlan({
    method: alignmentMethod,
    beta: alignmentBeta,
    preferenceDataset,
    paramsBillions: selectedModel.paramsBillions,
    baseIfEval: selectedModel.fineTunedBenchmarks?.ifEval ?? selectedModel.baseBenchmarks.ifEval,
    baseMtBench: selectedModel.fineTunedBenchmarks?.mtBench ?? selectedModel.baseBenchmarks.mtBench,
  });

  const packingMetrics = computePackingAndMaskingMetrics({
    samplePackingEnabled,
    completionOnlyLossEnabled,
    maxSeqLength: Math.min(8192, selectedModel.contextWindow),
    avgSampleTokens: 1480,
    baseTokensPerSec: latestPoint.tokensPerSec,
  });

  const multiAdapterPlan = computeMultiAdapterMergePlan({
    adapters: specialistAdapters,
    algorithm: mergeAlgorithm,
    density: mergeDensity,
    baseModelId: selectedModel.id,
    architecture: selectedModel.architecture,
  });

  const earlyStopReport = evaluateEarlyStoppingWatchdog({
    trainingSeries,
    patienceSteps: watchdogPatience,
    minDelta: watchdogMinDelta,
  });

  const executableRunBundle = generateExecutableRunBundle({
    modelId: selectedModel.id,
    modelName: selectedModel.name,
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    contextWindow: selectedModel.contextWindow,
    hpoConfig: activeHpoConfig,
    datasetSources: activeDatasetCommit.sourceDatasets,
    targetFormat: activeDatasetCommit.targetFormat,
    samplePackingEnabled,
    completionOnlyLossEnabled,
    alignmentStage: {
      enabled: stage2Enabled,
      method: alignmentMethod,
      beta: alignmentBeta,
      preferenceDataset,
    },
  });

  const snapA =
    experimentSnapshots.find((s) => s.id === diffSnapAId) ?? experimentSnapshots[0];
  const snapB =
    experimentSnapshots.find((s) => s.id === diffSnapBId) ??
    experimentSnapshots[Math.min(1, experimentSnapshots.length - 1)];
  const snapshotDiff = diffExperimentSnapshots(snapA, snapB);

  // 6. Activation-Aware Importance Matrix (imatrix) & Quantization Calibration State
  const [imatrixQuantType, setImatrixQuantType] = useState<ImatrixQuantType>('Q4_K_M');
  const [imatrixSamplesCount, setImatrixSamplesCount] = useState<number>(256);
  const [protectSalientLayers, setProtectSalientLayers] = useState<boolean>(true);
  const [copiedImatrixCmd, setCopiedImatrixCmd] = useState<boolean>(false);

  const imatrixPlan = computeImatrixQuantizationCalibration({
    modelId: selectedModel.id,
    modelName: selectedModel.name,
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    basePerplexity:
      selectedModel.fineTunedBenchmarks?.perplexity ?? selectedModel.baseBenchmarks.perplexity,
    quantType: imatrixQuantType,
    calibrationSamplesCount: imatrixSamplesCount,
    protectSalientLayers,
    datasetSourceLabel: activeDatasetCommit.sourceDatasets[0] ?? 'Active ChatML Blend',
  });

  // 7. Stage 3 GRPO Verifiable Reward Studio (RLVR) State
  const [grpoEnabled, setGrpoEnabled] = useState<boolean>(true);
  const [grpoRolloutSize, setGrpoRolloutSize] = useState<4 | 8 | 16>(8);
  const [grpoKlBeta, setGrpoKlBeta] = useState<number>(0.04);
  const [grpoVerifiers, setGrpoVerifiers] = useState<GrpoVerifierId[]>([
    'json_schema',
    'python_pytest',
    'math_symbolic',
    'think_format_length',
  ]);
  const [grpoExecutedBanner, setGrpoExecutedBanner] = useState<string | null>(null);

  const grpoPlan = computeGrpoVerifiableRewardPlan({
    enabled: grpoEnabled,
    modelId: selectedModel.id,
    modelName: selectedModel.name,
    paramsBillions: selectedModel.paramsBillions,
    groupRolloutSize: grpoRolloutSize,
    klBeta: grpoKlBeta,
    activeVerifiers: grpoVerifiers,
  });

  // 8. Layer-Wise Adaptive LoRA (DoRA / rsLoRA / SVD Rank Allocation) State
  const [useDora, setUseDora] = useState<boolean>(true);
  const [useRsLora, setUseRsLora] = useState<boolean>(true);
  const [loraAllocationPolicy, setLoraAllocationPolicy] =
    useState<LoraAllocationPolicy>('svd_salience');

  const adaptiveLoraPlan = computeLayerWiseAdaptiveLoraPlan({
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    baseRank: activeHpoConfig.loraRank,
    baseAlpha: activeHpoConfig.loraAlpha,
    useDora,
    useRsLora,
    allocationPolicy: loraAllocationPolicy,
  });

  const textEngineBundle = buildTextPythonEngineManifest({
    modelId: selectedModel.id,
    architecture: selectedModel.architecture,
    contextWindow: selectedModel.contextWindow,
    hpoConfig: activeHpoConfig,
    useDora,
    useRsLora,
    samplePackingEnabled,
    completionOnlyLossEnabled,
    stage2Enabled,
    alignmentMethod,
    alignmentBeta,
    preferenceDataset,
    grpoEnabled,
    grpoRolloutSize,
    grpoKlBeta,
    grpoVerifiers,
  });

  // 9. Hardware Profile & Edge Target Simulator State
  const [trainingGpuId, setTrainingGpuId] = useState<TrainingGpuTargetId>('rtx4090-24gb');
  const [edgeDeviceId, setEdgeDeviceId] = useState<EdgeDeviceTargetId>('macbook-air-m2-8gb');

  const hwSimResult = simulateHardwareAndEdgeTargets({
    paramsBillions: selectedModel.paramsBillions,
    architecture: selectedModel.architecture,
    quantization: selectedModel.quantizationDefault,
    contextLength: Math.min(8192, selectedModel.contextWindow),
    trainingGpuId,
    edgeDeviceId,
  });

  const checkpointIdToExport =
    customMergedCheckpointOverride ??
    selectedModel.fineTunedCheckpointId ??
    `ft-${selectedModel.name.toLowerCase().replace(/[^a-z0-9.]/g, '-')}-r${activeHpoConfig.loraRank}`;

  const progressTelemetry = computeTrainingProgressTelemetry({
    currentStep: latestPoint.step,
    currentEpoch: latestPoint.epoch,
    targetEpochs,
    stepsPerEpoch,
    tokensPerSec: latestPoint.tokensPerSec,
    avgTokensPerStep: Math.max(1024, activeHpoConfig.batchSize * 1024),
    vramUsedGb: gpuStats.vramUsedGb,
    vramTotalGb: gpuStats.vramTotalGb,
    recentSteps: trainingSeries,
  });

  const safetensorsPlan = buildSafetensorsExportPlan({
    modelId: selectedModel.id,
    modelName: selectedModel.name,
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    adapterType: selectedModel.quantizationDefault,
    loraRank: activeHpoConfig.loraRank,
    loraAlpha: activeHpoConfig.loraAlpha,
    exportPrecision,
    deploymentFormat,
    maxShardSizeGb,
    includeTokenizerAndConfig: includeTokenizer,
    runtimeTarget,
    checkpointId: checkpointIdToExport,
  });

  const generatedModelCard = generateStandardizedModelCard({
    modelId: selectedModel.id,
    modelName: selectedModel.name,
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    contextWindow: selectedModel.contextWindow,
    quantizationMode: selectedModel.quantizationDefault,
    checkpointId: checkpointIdToExport,
    datasetVersionId: activeDatasetCommit.versionId,
    datasetCommitHash: activeDatasetCommit.commitHash,
    sourceDatasets: activeDatasetCommit.sourceDatasets,
    targetFormat: activeDatasetCommit.targetFormat,
    sampleCount: activeDatasetCommit.sampleCount,
    datasetHealthScore: activeDatasetCommit.healthScore,
    hpoConfig: activeHpoConfig,
    baseBenchmarks: selectedModel.baseBenchmarks,
    fineTunedBenchmarks: selectedModel.fineTunedBenchmarks,
    safetensorsPlan,
  });

  const triggerMarkdownDownload = (card: StandardizedModelCardResult) => {
    const mdBlob = new Blob([card.markdown], { type: 'text/markdown;charset=utf-8' });
    const mdUrl = URL.createObjectURL(mdBlob);
    const aMd = document.createElement('a');
    aMd.href = mdUrl;
    aMd.download = card.filename;
    aMd.click();
    URL.revokeObjectURL(mdUrl);
  };

  const handleDownloadModelCardOnly = () => {
    triggerMarkdownDownload(generatedModelCard);
    const ts = new Date().toTimeString().split(' ')[0] + ' UTC';
    setLastExportedArtifact({
      filename: safetensorsPlan.outputFilename,
      modelCardFilename: generatedModelCard.filename,
      sizeGb: safetensorsPlan.mergedFileSizeGb,
      sha256: safetensorsPlan.sha256Preview,
      timestamp: ts,
      runtime: runtimeTarget,
    });
    onRecordSafetensorsExport(safetensorsPlan, selectedModel, generatedModelCard);
  };

  const handleMergeAndDownloadSafetensors = () => {
    const binaryBytes = createSafetensorsBinaryArtifact(safetensorsPlan.safetensorsHeaderJson);
    const blob = new Blob([binaryBytes], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = safetensorsPlan.outputFilename;
    a.click();
    URL.revokeObjectURL(url);

    if (autoIncludeModelCard) {
      triggerMarkdownDownload(generatedModelCard);
    }

    const ts = new Date().toTimeString().split(' ')[0] + ' UTC';
    setLastExportedArtifact({
      filename: safetensorsPlan.outputFilename,
      modelCardFilename: autoIncludeModelCard ? generatedModelCard.filename : undefined,
      sizeGb: safetensorsPlan.mergedFileSizeGb,
      sha256: safetensorsPlan.sha256Preview,
      timestamp: ts,
      runtime: runtimeTarget,
    });
    onRecordSafetensorsExport(
      safetensorsPlan,
      selectedModel,
      autoIncludeModelCard ? generatedModelCard : undefined
    );
  };

  const handleDownloadPythonMerger = () => {
    const blob = new Blob([safetensorsPlan.pythonMergeScript], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `merge_${selectedModel.name.toLowerCase().replace(/[^a-z0-9]/g, '_')}_safetensors.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const activeSafetensorsSnippet =
    safetensorsInspectorTab === 'modelcard'
      ? generatedModelCard.markdown
      : safetensorsInspectorTab === 'header'
      ? safetensorsPlan.safetensorsHeaderJson
      : safetensorsInspectorTab === 'python'
      ? safetensorsPlan.pythonMergeScript
      : safetensorsPlan.localDeployCommand;

  // SVG chart geometry
  const chartWidth = 680;
  const chartHeight = 210;
  const padX = 44;
  const padY = 24;
  const maxLoss = Math.max(2.2, ...trainingSeries.map((p) => Math.max(p.trainLoss, p.valLoss)));
  const minLoss = Math.min(0.4, ...trainingSeries.map((p) => Math.min(p.trainLoss, p.valLoss)));

  const toCoords = (idx: number, val: number) => {
    const count = Math.max(1, trainingSeries.length - 1);
    const x = padX + (idx / count) * (chartWidth - padX * 2);
    const y =
      chartHeight -
      padY -
      ((val - minLoss) / Math.max(0.1, maxLoss - minLoss)) * (chartHeight - padY * 2);
    return { x, y };
  };

  const trainPath = trainingSeries
    .map((pt, i) => {
      const { x, y } = toCoords(i, pt.trainLoss);
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  const valPath = trainingSeries
    .map((pt, i) => {
      const { x, y } = toCoords(i, pt.valLoss);
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  const inspectedPoint = hoveredStep ?? latestPoint;

  return (
    <div className="space-y-8">
      {/* Top Action & Model Selection Bar */}
      <section className="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4 pb-6 border-b border-slate-800/80">
        <div>
          <p className="text-xs text-slate-400">
            Active Target · {selectedModel.architecture} · {selectedModel.paramsBillions}B Params · {selectedModel.contextWindow.toLocaleString()} Context
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100 mt-1">
            Training &amp; Telemetry Control Plane
          </h1>
        </div>

        <div className="flex flex-wrap sm:flex-nowrap items-center gap-2.5 shrink-0">
          <label htmlFor="active-model-select" className="sr-only">
            Select Active Model
          </label>
          <select
            id="active-model-select"
            aria-label="Select Active Model"
            value={selectedModelId}
            onChange={(e) => onSelectModel(e.target.value)}
            className="bg-[#111827] border border-slate-700/80 rounded-lg px-3 py-2 text-xs font-medium text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.architecture})
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={onRunTrainingStep}
            disabled={isTrainingActive}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 text-emerald-400" />
            {isTrainingActive ? 'Running Step...' : 'Run Training Step (+50 Steps)'}
          </button>

          <button
            type="button"
            onClick={onRunHpoSweep}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Sliders className="w-3.5 h-3.5" />
            Run HPO + Eval Trial
          </button>

          {onOpenCommandPalette && (
            <button
              type="button"
              onClick={onOpenCommandPalette}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-mono text-slate-200 bg-[#111827] hover:bg-slate-800 border border-slate-700/80 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Terminal className="w-3.5 h-3.5 text-blue-400" />
              Cmd+K
            </button>
          )}
        </div>
      </section>

      {/* Primary KPI Strip */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-4">
          <div className="text-xs text-slate-400 flex items-center">
            <span>Training / Validation Loss</span>
            <InfoTooltip termKey="trainValLoss" align="left" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-semibold text-slate-100">
              {latestPoint.trainLoss.toFixed(3)}
            </span>
            <span className="text-xs text-slate-400">/ {latestPoint.valLoss.toFixed(3)} val</span>
          </div>
          <div className="mt-2 text-xs text-emerald-400 font-mono tabular-nums">
            Step {latestPoint.step} · Epoch {latestPoint.epoch.toFixed(2)} · Converging
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-4">
          <div className="text-xs text-slate-400 flex items-center">
            <span>Gradient L2 Norm & Schedule</span>
            <InfoTooltip termKey="gradNorm" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-semibold text-slate-100">
              {latestPoint.gradNorm.toFixed(3)}
            </span>
            <span className="text-xs text-slate-400">
              · lr {activeHpoConfig.learningRate.toExponential(2)}
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono tabular-nums">
            Scheduler: {activeHpoConfig.lrScheduler.toUpperCase()} · LoRA r={activeHpoConfig.loraRank} α={activeHpoConfig.loraAlpha}
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-4">
          <div className="text-xs text-slate-400 flex items-center">
            <span>Throughput & Kernel Path</span>
            <InfoTooltip termKey="lfmArchitecture" />
          </div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-semibold text-slate-100">
              {latestPoint.tokensPerSec.toLocaleString()}
            </span>
            <span className="text-xs text-slate-400">tok/sec</span>
          </div>
          <div className="mt-2 text-xs text-blue-400 font-mono tabular-nums">
            {selectedModel.architecture === 'LFM2.5'
              ? 'Liquid Hybrid SSM + Conv1D Fused'
              : selectedModel.architecture === 'Gemma4'
              ? 'FlashAttention-3 + Softcap Fused'
              : 'FlashAttention-3 + Unsloth RoPE'}
          </div>
        </div>

        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-4">
          <div className="text-xs text-slate-400">Base vs. Fine-Tuned Accuracy Gain</div>
          <div className="mt-2 flex items-baseline gap-2 font-mono tabular-nums">
            <span className="text-2xl font-semibold text-emerald-400">
              {selectedModel.fineTunedBenchmarks
                ? `+${(
                    (selectedModel.fineTunedBenchmarks.mmluPro -
                      selectedModel.baseBenchmarks.mmluPro +
                      (selectedModel.fineTunedBenchmarks.gsm8k -
                        selectedModel.baseBenchmarks.gsm8k)) /
                    2
                  ).toFixed(2)}%`
                : 'Baseline Stored'}
            </span>
            <span className="text-xs text-slate-400">mean delta</span>
          </div>
          <div className="mt-2 text-xs text-slate-400 font-mono tabular-nums">
            Base GSM8K {selectedModel.baseBenchmarks.gsm8k}% → FT{' '}
            {selectedModel.fineTunedBenchmarks?.gsm8k ?? selectedModel.baseBenchmarks.gsm8k}%
          </div>
        </div>
      </section>

      {/* Visual Progress Bars: Current Epoch Completion, ETA, and Memory Utilization Trends */}
      <section
        aria-label="Training Progress and Memory Utilization Trends"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5"
      >
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">
              Active Training Progress, ETA Countdown &amp; Memory Utilization Trends
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Real-time epoch progress bar, wall-clock ETA estimation, and step-by-step VRAM allocation trajectory for {selectedModel.name}
            </p>
          </div>

          <div className="flex flex-wrap xl:flex-nowrap items-center gap-2 text-xs font-mono shrink-0">
            <span
              role="status"
              aria-label="LocalStorage Training Progress Persistence Status"
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#0B0F17] border border-slate-800 text-emerald-400 whitespace-nowrap"
            >
              <HardDrive className="w-3.5 h-3.5 shrink-0" />
              Persisted (Step {latestPoint.step} · {trainingSeries.length} pts)
            </span>
            {onResetTrainingSeries && trainingSeries.length > 8 && (
              <button
                type="button"
                onClick={onResetTrainingSeries}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors whitespace-nowrap"
              >
                <RefreshCw className="w-3 h-3 shrink-0" />
                Reset Series
              </button>
            )}
            <label htmlFor="target-epochs-select" className="text-slate-400 whitespace-nowrap">
              Horizon:
            </label>
            <select
              id="target-epochs-select"
              aria-label="Target Training Epochs"
              value={targetEpochs}
              onChange={(e) => setTargetEpochs(Number(e.target.value))}
              className="bg-[#0B0F17] border border-slate-800 rounded px-2 py-1 text-xs font-mono text-slate-100"
            >
              <option value={2}>2 Epochs ({2 * stepsPerEpoch} steps)</option>
              <option value={3}>3 Epochs ({3 * stepsPerEpoch} steps)</option>
              <option value={5}>5 Epochs ({5 * stepsPerEpoch} steps)</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mt-4">
          {/* Bar 1: Current Epoch Completion & Overall Run Progress */}
          <div className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">
                  Current Epoch ({progressTelemetry.currentEpochIndex} of {progressTelemetry.totalTargetEpochs})
                </span>
                <span className="font-mono tabular-nums font-semibold text-emerald-400">
                  {progressTelemetry.currentEpochCompletionPct.toFixed(1)}%
                </span>
              </div>
              <div
                role="progressbar"
                aria-label="Current Epoch Completion Percentage"
                aria-valuenow={progressTelemetry.currentEpochCompletionPct}
                aria-valuemin={0}
                aria-valuemax={100}
                className="w-full h-2.5 bg-slate-800 rounded overflow-hidden mt-2"
              >
                <div
                  className="h-full bg-emerald-500 transition-all duration-200"
                  style={{ width: `${progressTelemetry.currentEpochCompletionPct}%` }}
                />
              </div>
              <div className="flex justify-between text-[11px] text-slate-400 font-mono tabular-nums mt-1.5">
                <span>
                  Epoch Step {progressTelemetry.stepWithinCurrentEpoch} / {progressTelemetry.stepsPerEpoch}
                </span>
                <span>Epoch {latestPoint.epoch.toFixed(2)}</span>
              </div>
            </div>

            <div className="pt-2.5 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Total Multi-Epoch Run Completion</span>
                <span className="font-mono tabular-nums text-blue-400 font-semibold">
                  {progressTelemetry.overallRunCompletionPct.toFixed(1)}%
                </span>
              </div>
              <div
                role="progressbar"
                aria-label="Overall Training Run Completion Percentage"
                aria-valuenow={progressTelemetry.overallRunCompletionPct}
                aria-valuemin={0}
                aria-valuemax={100}
                className="w-full h-2 bg-slate-800 rounded overflow-hidden mt-1.5"
              >
                <div
                  className="h-full bg-blue-600 transition-all duration-200"
                  style={{ width: `${progressTelemetry.overallRunCompletionPct}%` }}
                />
              </div>
              <div className="flex justify-between text-[11px] text-slate-400 font-mono tabular-nums mt-1">
                <span>
                  Global Step {latestPoint.step} / {progressTelemetry.totalTargetSteps}
                </span>
                <span>{progressTelemetry.remainingSteps} steps left</span>
              </div>
            </div>
          </div>

          {/* Bar 2: Estimated Time Remaining (ETA) & Throughput Budget */}
          <div className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium inline-flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-amber-400" />
                  Estimated Time Remaining (ETA)
                </span>
                <span className="font-mono tabular-nums font-semibold text-amber-400">
                  {progressTelemetry.etaFormatted}
                </span>
              </div>
              <div
                role="progressbar"
                aria-label="Estimated Time Remaining Progress"
                aria-valuenow={progressTelemetry.overallRunCompletionPct}
                aria-valuemin={0}
                aria-valuemax={100}
                className="w-full h-2.5 bg-slate-800 rounded overflow-hidden mt-2 flex"
              >
                <div
                  className="h-full bg-blue-600 transition-all duration-200"
                  title={`Elapsed: ${progressTelemetry.elapsedFormatted}`}
                  style={{ width: `${progressTelemetry.overallRunCompletionPct}%` }}
                />
                <div
                  className="h-full bg-amber-400/35 transition-all duration-200"
                  title={`Remaining ETA: ${progressTelemetry.etaFormatted}`}
                  style={{
                    width: `${Math.max(0, 100 - progressTelemetry.overallRunCompletionPct)}%`,
                  }}
                />
              </div>
              <div className="flex justify-between text-[11px] text-slate-400 font-mono tabular-nums mt-1.5">
                <span>Elapsed: {progressTelemetry.elapsedFormatted}</span>
                <span>Remaining: {progressTelemetry.etaFormatted}</span>
              </div>
            </div>

            <div className="pt-2.5 border-t border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono tabular-nums">
              <div>
                <div className="text-slate-400">Remaining Steps</div>
                <div className="text-slate-200 font-semibold mt-0.5">
                  {progressTelemetry.remainingSteps} steps
                </div>
              </div>
              <div>
                <div className="text-slate-400">Token Throughput</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  {latestPoint.tokensPerSec.toLocaleString()} tok/s
                </div>
              </div>
            </div>
          </div>

          {/* Bar 3: Memory Utilization Trends & Step Trajectory */}
          <div className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Memory Utilization Trend</span>
                <span
                  className={`font-mono tabular-nums font-semibold ${
                    progressTelemetry.vramUtilizationPct > gpuThresholds.vramPctThreshold
                      ? 'text-red-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {gpuStats.vramUsedGb.toFixed(1)} / {gpuStats.vramTotalGb.toFixed(0)} GB (
                  {progressTelemetry.vramUtilizationPct}%)
                </span>
              </div>
              <div
                role="progressbar"
                aria-label="Memory Utilization Percentage"
                aria-valuenow={progressTelemetry.vramUtilizationPct}
                aria-valuemin={0}
                aria-valuemax={100}
                className="relative w-full h-2.5 bg-slate-800 rounded overflow-hidden mt-2"
              >
                <div
                  className={`h-full transition-all duration-200 ${
                    progressTelemetry.vramUtilizationPct > gpuThresholds.vramPctThreshold
                      ? 'bg-red-500'
                      : 'bg-emerald-500'
                  }`}
                  style={{ width: `${Math.min(100, progressTelemetry.vramUtilizationPct)}%` }}
                />
              </div>
              <div className="flex justify-between text-[11px] text-slate-400 font-mono tabular-nums mt-1.5">
                <span>Peak Reserved: {progressTelemetry.peakReservedVramGb.toFixed(1)} GB</span>
                <span>
                  Delta:{' '}
                  {progressTelemetry.memoryTrendDeltaPct >= 0
                    ? `+${progressTelemetry.memoryTrendDeltaPct.toFixed(1)}`
                    : progressTelemetry.memoryTrendDeltaPct.toFixed(1)}
                  % ({progressTelemetry.memoryTrendStatus})
                </span>
              </div>
            </div>

            {/* Step-by-Step Memory Utilization Trend Mini-Bars */}
            <div className="pt-2.5 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono mb-1.5">
                <span>Recent Step VRAM Trend Bars</span>
                <span>Alert &gt; {gpuThresholds.vramPctThreshold}%</span>
              </div>
              <div
                aria-label="Memory Utilization Trend History Bars"
                className="grid grid-cols-8 gap-1.5 items-end pt-1"
              >
                {vramHistoryGb.slice(-8).map((gbVal, idx) => {
                  const pct = Number(
                    Math.min(100, (gbVal / Math.max(1, gpuStats.vramTotalGb)) * 100).toFixed(1)
                  );
                  const maxWindowGb = Math.max(6.0, ...vramHistoryGb.slice(-8), gpuStats.vramUsedGb);
                  const visualFillPct = Math.max(
                    28,
                    Math.min(100, Math.round((gbVal / maxWindowGb) * 100))
                  );
                  const isBreach = pct > gpuThresholds.vramPctThreshold;
                  return (
                    <div
                      key={idx}
                      className="flex flex-col items-center gap-1 justify-end"
                      title={`Sample ${idx + 1}: ${gbVal.toFixed(1)} GB (${pct}% of ${gpuStats.vramTotalGb} GB)`}
                    >
                      <div className="w-full bg-slate-800 rounded-sm h-7 flex items-end overflow-hidden">
                        <div
                          className={`w-full transition-all duration-200 ${
                            isBreach ? 'bg-red-500' : 'bg-blue-500'
                          }`}
                          style={{ height: `${visualFillPct}%` }}
                        />
                      </div>
                      <span className="text-[9px] font-mono tabular-nums text-slate-400">
                        {gbVal.toFixed(1)}G
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Main 2-Column Split: Loss Curve + GPU Hardware Telemetry */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 7 cols: Loss & Validation Trajectory */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-slate-800/80">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                01. Real-Time Training & Validation Loss Curve
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Cross-entropy loss across steps · Hover any checkpoint node to inspect exact metrics
              </p>
            </div>
            <div className="text-xs font-mono tabular-nums text-slate-300">
              Step {inspectedPoint.step} · Train {inspectedPoint.trainLoss.toFixed(3)} · Val{' '}
              {inspectedPoint.valLoss.toFixed(3)}
            </div>
          </div>

          <div className="my-4 overflow-x-auto">
            <svg
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              className="w-full h-52 select-none"
              role="img"
              aria-label="Training and validation loss trajectory chart"
            >
              {/* Horizontal grid lines */}
              {[0.6, 1.0, 1.4, 1.8, 2.2].map((val) => {
                const { y } = toCoords(0, val);
                return (
                  <g key={val}>
                    <line
                      x1={padX}
                      y1={y}
                      x2={chartWidth - padX}
                      y2={y}
                      stroke="#3F3B35"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padX - 8}
                      y={y + 4}
                      textAnchor="end"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      {val.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Validation Loss Path (Gruvbox Amber #FABD2F) */}
              <path
                d={valPath}
                fill="none"
                stroke="#FABD2F"
                strokeWidth="2.2"
                strokeLinecap="round"
              />

              {/* Training Loss Path (Gruvbox Aqua/Blue #83A598) */}
              <path
                d={trainPath}
                fill="none"
                stroke="#83A598"
                strokeWidth="2.5"
                strokeLinecap="round"
              />

              {/* Interactive Data Points */}
              {trainingSeries.map((pt, i) => {
                const tCoord = toCoords(i, pt.trainLoss);
                const vCoord = toCoords(i, pt.valLoss);
                return (
                  <g
                    key={pt.step}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredStep(pt)}
                    onMouseLeave={() => setHoveredStep(null)}
                  >
                    <circle
                      cx={vCoord.x}
                      cy={vCoord.y}
                      r={4}
                      fill="#272624"
                      stroke="#FABD2F"
                      strokeWidth="2"
                    />
                    <circle
                      cx={tCoord.x}
                      cy={tCoord.y}
                      r={4.5}
                      fill="#83A598"
                      stroke="#272624"
                      strokeWidth="1.5"
                    />
                    <text
                      x={tCoord.x}
                      y={chartHeight - 6}
                      textAnchor="middle"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      s{pt.step}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-400">
            <div className="flex items-center gap-4">
              <span>Teal Curve: Training Cross-Entropy Loss</span>
              <span aria-hidden="true">·</span>
              <span>Amber Curve: Held-Out Validation Loss</span>
            </div>
            <div className="font-mono tabular-nums text-slate-300">
              Generalization Gap: +{(latestPoint.valLoss - latestPoint.trainLoss).toFixed(3)}
            </div>
          </div>
        </div>

        {/* Right 5 cols: GPU & Hardware Resource Telemetry */}
        <div
          className={`lg:col-span-5 bg-[#111827] border rounded-lg p-5 flex flex-col justify-between transition-colors ${
            gpuAlertEval.hasAlerts ? 'border-red-500/80' : 'border-slate-800/90'
          }`}
        >
          <div className="pb-4 border-b border-slate-800/80 flex items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                02. GPU & Compute Resource Monitor
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Threshold Watchdog: VRAM &gt; {gpuThresholds.vramPctThreshold}% · Temp &gt;{' '}
                {gpuThresholds.tempCelsiusThreshold}°C · Power &gt; {gpuThresholds.powerWattsThreshold}W
              </p>
            </div>
            <Cpu
              className={`w-4 h-4 shrink-0 ${
                gpuAlertEval.hasAlerts ? 'text-red-400' : 'text-slate-400'
              }`}
            />
          </div>

          <div className="space-y-4 my-4">
            {/* VRAM Utilization */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300">
                  VRAM Allocation (Alert &gt; {gpuThresholds.vramPctThreshold}%)
                </span>
                <span
                  className={`font-mono tabular-nums ${
                    gpuAlertEval.vramUsagePct > gpuThresholds.vramPctThreshold
                      ? 'text-red-400 font-semibold'
                      : 'text-slate-100'
                  }`}
                >
                  {gpuStats.vramUsedGb.toFixed(1)} / {gpuStats.vramTotalGb.toFixed(0)} GB (
                  {gpuAlertEval.vramUsagePct}%)
                  {gpuAlertEval.vramUsagePct > gpuThresholds.vramPctThreshold
                    ? ' · BREACH'
                    : ''}
                </span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded overflow-hidden">
                <div
                  className={`h-full transition-transform duration-150 origin-left ${
                    gpuAlertEval.vramUsagePct > gpuThresholds.vramPctThreshold
                      ? 'bg-red-500'
                      : 'bg-blue-500'
                  }`}
                  style={{
                    transform: `scaleX(${Math.min(1, gpuStats.vramUsedGb / gpuStats.vramTotalGb)})`,
                  }}
                />
              </div>
            </div>

            {/* SM Compute Occupancy */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300 inline-flex items-center">
                  SM Compute Occupancy
                  <InfoTooltip termKey="smOccupancy" />
                </span>
                <span className="font-mono tabular-nums text-slate-100">{gpuStats.smUtilPct}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-transform duration-150 origin-left"
                  style={{ transform: `scaleX(${gpuStats.smUtilPct / 100})` }}
                />
              </div>
            </div>

            {/* Tensor Core BF16/FP8 Activity */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-slate-300">Tensor Core Matrix Pipe (BF16 / NF4)</span>
                <span className="font-mono tabular-nums text-slate-100">
                  {gpuStats.tensorCorePct}%
                </span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded overflow-hidden">
                <div
                  className="h-full bg-indigo-500 transition-transform duration-150 origin-left"
                  style={{ transform: `scaleX(${gpuStats.tensorCorePct / 100})` }}
                />
              </div>
            </div>

            {/* Hardware Vitals Grid */}
            <div className="grid grid-cols-3 gap-3 pt-2 border-t border-slate-800/80 text-xs">
              <div>
                <div className="text-slate-400">
                  Die Temp (&gt;{gpuThresholds.tempCelsiusThreshold}°C)
                </div>
                <div
                  className={`font-mono tabular-nums text-sm font-semibold mt-0.5 ${
                    gpuStats.tempCelsius > gpuThresholds.tempCelsiusThreshold
                      ? 'text-red-400'
                      : 'text-slate-100'
                  }`}
                >
                  {gpuStats.tempCelsius}°C
                  {gpuStats.tempCelsius > gpuThresholds.tempCelsiusThreshold ? ' (!)' : ''}
                </div>
              </div>
              <div>
                <div className="text-slate-400">
                  Power (&gt;{gpuThresholds.powerWattsThreshold}W)
                </div>
                <div
                  className={`font-mono tabular-nums text-sm font-semibold mt-0.5 ${
                    gpuStats.powerWatts > gpuThresholds.powerWattsThreshold
                      ? 'text-amber-400'
                      : 'text-slate-100'
                  }`}
                >
                  {gpuStats.powerWatts} W
                </div>
              </div>
              <div>
                <div className="text-slate-400">PCIe / NVLink</div>
                <div className="font-mono tabular-nums text-sm font-semibold text-slate-100 mt-0.5">
                  {gpuStats.nvlinkGbps} GB/s
                </div>
              </div>
            </div>

            {/* Configurable Alert Thresholds & Stress Test Trigger */}
            <div className="pt-3 border-t border-slate-800/80 space-y-2.5">
              <div className="grid grid-cols-2 gap-2.5 text-xs">
                <div>
                  <label htmlFor="vram-threshold-input" className="block text-slate-400 mb-1">
                    VRAM Alert Threshold (%)
                  </label>
                  <input
                    id="vram-threshold-input"
                    type="number"
                    min={50}
                    max={99}
                    value={gpuThresholds.vramPctThreshold}
                    onChange={(e) =>
                      onUpdateGpuThresholds({
                        ...gpuThresholds,
                        vramPctThreshold: Number(e.target.value) || 90,
                      })
                    }
                    className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1 text-xs font-mono tabular-nums text-slate-100"
                  />
                </div>
                <div>
                  <label htmlFor="temp-threshold-input" className="block text-slate-400 mb-1">
                    Temp Alert Threshold (°C)
                  </label>
                  <input
                    id="temp-threshold-input"
                    type="number"
                    min={60}
                    max={95}
                    value={gpuThresholds.tempCelsiusThreshold}
                    onChange={(e) =>
                      onUpdateGpuThresholds({
                        ...gpuThresholds,
                        tempCelsiusThreshold: Number(e.target.value) || 85,
                      })
                    }
                    className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1 text-xs font-mono tabular-nums text-slate-100"
                  />
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={onSimulateGpuStressSpike}
                  className="flex-1 px-3 py-1.5 text-xs font-medium text-amber-200 bg-amber-950/60 hover:bg-amber-900/60 border border-amber-700/60 rounded transition-colors whitespace-nowrap"
                >
                  Simulate GPU Overload (VRAM 95% · 89°C)
                </button>
                {gpuAlertEval.hasAlerts && (
                  <button
                    type="button"
                    onClick={onMitigateGpuOverload}
                    className="px-3 py-1.5 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded transition-colors whitespace-nowrap"
                  >
                    Mitigate &amp; Cool Down
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <span>OOM Guard: PagedAdamW8bit + Unsloth Checkpointing</span>
            <span
              className={`font-mono tabular-nums font-medium ${
                gpuAlertEval.hasAlerts ? 'text-red-400' : 'text-emerald-400'
              }`}
            >
              {gpuAlertEval.hasAlerts
                ? `ALERT (${gpuAlertEval.breaches.length} Breached)`
                : 'Nominal'}
            </span>
          </div>
        </div>
      </section>

      {/* Automated Hyperparameter Optimization (HPO) Tuner & Bayesian Proposal */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              03. Automated Hyperparameter Optimization (Optuna Bayesian TPE + Agentic Refinement)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Diagnosis: {hpoProposal.diagnosis} · {hpoProposal.rationale}
            </p>
          </div>
          <button
            type="button"
            onClick={() => onUpdateHpoConfig(hpoProposal.nextConfig)}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Zap className="w-3.5 h-3.5" />
            Apply Recommended HPO Config (+{hpoProposal.expectedImprovementPct}% est.)
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 mt-4">
          <div>
            <label htmlFor="hpo-lr" className="flex items-center text-xs text-slate-400 mb-1">
              <span>Learning Rate</span>
              <InfoTooltip termKey="learningRate" align="left" />
            </label>
            <input
              id="hpo-lr"
              type="number"
              step="0.00001"
              value={activeHpoConfig.learningRate}
              onChange={(e) =>
                onUpdateHpoConfig({
                  ...activeHpoConfig,
                  learningRate: Number(e.target.value) || 0.0001,
                })
              }
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
            />
          </div>

          <div>
            <label htmlFor="hpo-rank" className="flex items-center text-xs text-slate-400 mb-1">
              <span>LoRA Rank (r)</span>
              <InfoTooltip termKey="loraRank" />
            </label>
            <select
              id="hpo-rank"
              value={activeHpoConfig.loraRank}
              onChange={(e) => {
                const r = Number(e.target.value);
                onUpdateHpoConfig({
                  ...activeHpoConfig,
                  loraRank: r,
                  loraAlpha: r * 2,
                });
              }}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
            >
              {[8, 16, 32, 64, 128].map((r) => (
                <option key={r} value={r}>
                  r = {r} (α = {r * 2})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="hpo-scheduler" className="flex items-center text-xs text-slate-400 mb-1">
              <span>LR Scheduler</span>
              <InfoTooltip termKey="wsdScheduler" />
            </label>
            <select
              id="hpo-scheduler"
              value={activeHpoConfig.lrScheduler}
              onChange={(e) =>
                onUpdateHpoConfig({
                  ...activeHpoConfig,
                  lrScheduler: e.target.value as HyperparameterConfig['lrScheduler'],
                })
              }
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
            >
              <option value="wsd">Warmup-Stable-Decay (WSD)</option>
              <option value="cosine">Cosine Annealing</option>
              <option value="linear">Linear Decay</option>
              <option value="constant_with_warmup">Constant + Warmup</option>
            </select>
          </div>

          <div>
            <label htmlFor="hpo-neftune" className="flex items-center text-xs text-slate-400 mb-1">
              <span>NEFTune Noise (α)</span>
              <InfoTooltip termKey="neftune" />
            </label>
            <input
              id="hpo-neftune"
              type="number"
              step="1"
              min="0"
              max="15"
              value={activeHpoConfig.neftuneAlpha}
              onChange={(e) =>
                onUpdateHpoConfig({
                  ...activeHpoConfig,
                  neftuneAlpha: Number(e.target.value),
                })
              }
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
            />
          </div>

          <div>
            <label htmlFor="hpo-warmup" className="block text-xs text-slate-400 mb-1">
              Warmup Ratio
            </label>
            <input
              id="hpo-warmup"
              type="number"
              step="0.01"
              min="0"
              max="0.3"
              value={activeHpoConfig.warmupRatio}
              onChange={(e) =>
                onUpdateHpoConfig({
                  ...activeHpoConfig,
                  warmupRatio: Number(e.target.value),
                })
              }
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
            />
          </div>

          <div>
            <label htmlFor="hpo-batch" className="block text-xs text-slate-400 mb-1">
              Micro-Batch Size
            </label>
            <select
              id="hpo-batch"
              value={activeHpoConfig.batchSize}
              onChange={(e) =>
                onUpdateHpoConfig({
                  ...activeHpoConfig,
                  batchSize: Number(e.target.value),
                })
              }
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
            >
              {[1, 2, 4, 8, 16].map((b) => (
                <option key={b} value={b}>
                  {b} sequences/GPU
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      {/* Section 04: Direct Local Safetensors Checkpoint Merger & Automated Model Card Generator */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              04. Export Merged Checkpoint ({deploymentFormat}) &amp; Automated Standardized Model Card (README.md)
              <InfoTooltip termKey="safetensors" align="left" />
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Fuses{' '}
              <span className="font-mono text-slate-300">
                W_merged = W_base + ({activeHpoConfig.loraAlpha}/{activeHpoConfig.loraRank}) · (B @ A)
              </span>{' '}
              into <span className="font-mono text-slate-300">{safetensorsPlan.outputFilename}</span> ({safetensorsPlan.quantizationSummary}) and updates the accompanying Markdown Model Card
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={handleDownloadPythonMerger}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors whitespace-nowrap"
            >
              <Terminal className="w-3.5 h-3.5 text-blue-400" />
              Download PEFT Merger (.py)
            </button>

            <button
              type="button"
              onClick={handleDownloadModelCardOnly}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors whitespace-nowrap"
            >
              <FileText className="w-3.5 h-3.5 text-emerald-400" />
              Export Model Card (README.md)
            </button>

            <button
              type="button"
              onClick={handleMergeAndDownloadSafetensors}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap"
            >
              <PackageCheck className="w-3.5 h-3.5" />
              Merge &amp; Export {deploymentFormat} ({safetensorsPlan.mergedFileSizeGb} GB)
            </button>
          </div>
        </div>

        {/* Export Configuration + Live Plan Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-5 items-stretch">
          {/* Left 5 cols: Export Configuration & Extracted Model Card Metadata */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
            {/* Deployment Format Selector (SAFETENSORS / GGUF-Q4_K_M / GPTQ / ONNX) */}
            <div className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-3.5 space-y-2.5">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="st-deployment-format"
                  className="text-xs font-medium text-slate-200 flex items-center gap-1"
                >
                  <span>Target Deployment Format</span>
                  <InfoTooltip
                    termKey={
                      deploymentFormat === 'GGUF-Q4_K_M'
                        ? 'ggufQ4KM'
                        : deploymentFormat === 'GPTQ'
                        ? 'gptq'
                        : deploymentFormat === 'ONNX'
                        ? 'onnx'
                        : 'safetensors'
                    }
                    align="left"
                  />
                </label>
                <span className="text-[11px] font-mono text-emerald-400">
                  {safetensorsPlan.mergedFileSizeGb} GB Est.
                </span>
              </div>

              <div
                role="group"
                aria-label="Select Deployment Export Format"
                className="grid grid-cols-2 sm:grid-cols-4 gap-1.5"
              >
                {(
                  [
                    { id: 'SAFETENSORS', label: 'Safetensors', tip: 'safetensors' },
                    { id: 'GGUF-Q4_K_M', label: 'GGUF-Q4_K_M', tip: 'ggufQ4KM' },
                    { id: 'GPTQ', label: 'GPTQ (4-Bit)', tip: 'gptq' },
                    { id: 'ONNX', label: 'ONNX Graph', tip: 'onnx' },
                  ] as Array<{
                    id: SafetensorsDeploymentFormat;
                    label: string;
                    tip: string;
                  }>
                ).map((fmt) => (
                  <button
                    key={fmt.id}
                    type="button"
                    onClick={() => setDeploymentFormat(fmt.id)}
                    className={`px-2 py-1.5 text-xs font-mono font-medium rounded border transition-colors text-center ${
                      deploymentFormat === fmt.id
                        ? 'bg-blue-600 text-white border-blue-500'
                        : 'bg-[#111827] text-slate-300 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {fmt.label}
                  </button>
                ))}
              </div>

              <select
                id="st-deployment-format"
                aria-label="Target Deployment Format"
                value={deploymentFormat}
                onChange={(e) =>
                  setDeploymentFormat(e.target.value as SafetensorsDeploymentFormat)
                }
                className="w-full bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
              >
                <option value="SAFETENSORS">
                  SAFETENSORS — Contiguous Zero-Copy Weights ({exportPrecision.toUpperCase()})
                </option>
                <option value="GGUF-Q4_K_M">
                  GGUF-Q4_K_M — 4-Bit K-Quant Medium (llama.cpp / Ollama Edge)
                </option>
                <option value="GPTQ">
                  GPTQ — 4-Bit Hessian Calibrated (vLLM Marlin / TensorRT-LLM)
                </option>
                <option value="ONNX">
                  ONNX — O3 Fused Computation Graph (ONNX Runtime / WebGPU / DirectML)
                </option>
              </select>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label htmlFor="st-precision" className="block text-xs text-slate-400 mb-1">
                  Base Merge Precision
                </label>
                <select
                  id="st-precision"
                  value={exportPrecision}
                  onChange={(e) =>
                    setExportPrecision(e.target.value as SafetensorsExportPrecision)
                  }
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value="bfloat16">BF16 (bfloat16 · 2B/param)</option>
                  <option value="float16">FP16 (float16 · 2B/param)</option>
                  <option value="fp8_e4m3fn">FP8 (fp8_e4m3fn · 1B/param)</option>
                </select>
              </div>

              <div>
                <label htmlFor="st-shard" className="block text-xs text-slate-400 mb-1">
                  Max Shard Size
                </label>
                <select
                  id="st-shard"
                  value={maxShardSizeGb}
                  onChange={(e) => setMaxShardSizeGb(Number(e.target.value))}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
                >
                  <option value={5}>5 GB (Single-file for ≤2.5B)</option>
                  <option value={10}>10 GB (Single-file for ≤5B)</option>
                  <option value={2}>2 GB (Sharded chunks)</option>
                </select>
              </div>

              <div className="sm:col-span-2">
                <label htmlFor="st-runtime" className="block text-xs text-slate-400 mb-1">
                  Direct Local Deployment Target Runtime
                </label>
                <select
                  id="st-runtime"
                  value={runtimeTarget}
                  onChange={(e) =>
                    setRuntimeTarget(e.target.value as SafetensorsRuntimeTarget)
                  }
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value="ollama">Ollama Local (Native Safetensors / GGUF Modelfile)</option>
                  <option value="llama-cpp">llama.cpp / llama-server (Local Port 8080)</option>
                  <option value="vllm-local">vLLM Local OpenAI Server (Marlin / Safetensors)</option>
                  <option value="mlx-apple">Apple Silicon MLX (M1/M2/M3/M4 Local)</option>
                  <option value="transformers-local">PyTorch / HuggingFace / ONNX Runtime Local</option>
                </select>
              </div>
            </div>

            <div className="space-y-2">
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeTokenizer}
                  onChange={(e) => setIncludeTokenizer(e.target.checked)}
                  className="rounded border-slate-700 bg-[#0B0F17] text-blue-600"
                />
                <span>
                  Bundle <span className="font-mono">tokenizer.json</span>,{' '}
                  <span className="font-mono">config.json</span> &amp; chat template metadata
                </span>
              </label>

              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={autoIncludeModelCard}
                  onChange={(e) => setAutoIncludeModelCard(e.target.checked)}
                  className="rounded border-slate-700 bg-[#0B0F17] text-blue-600"
                />
                <span>
                  Auto-accompany <span className="font-mono">{deploymentFormat}</span> export with standardized{' '}
                  <span className="font-mono">{generatedModelCard.filename}</span> Model Card
                </span>
              </label>
            </div>

            {/* Extracted Model Card & Contiguous File Metadata Breakdown */}
            <div className="pt-3 border-t border-slate-800/80 space-y-2 text-xs font-mono tabular-nums">
              <div className="flex justify-between">
                <span className="text-slate-400">Deployment Format &amp; Quant:</span>
                <span className="text-amber-400 font-semibold">
                  {generatedModelCard.extractedMetadata.deploymentFormat} (
                  {generatedModelCard.extractedMetadata.mergedSizeGb})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Extracted Architecture &amp; Scale:</span>
                <span className="text-slate-200">
                  {generatedModelCard.extractedMetadata.architecture} ·{' '}
                  {generatedModelCard.extractedMetadata.parameterScale}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Training Dataset Version:</span>
                <span className="text-emerald-400 font-semibold">
                  {generatedModelCard.extractedMetadata.datasetVersion}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Base vs. FT MMLU-Pro:</span>
                <span className="text-slate-200">
                  {generatedModelCard.extractedMetadata.baseMmluPro} →{' '}
                  {generatedModelCard.extractedMetadata.fineTunedMmluPro} (
                  {generatedModelCard.extractedMetadata.meanAccuracyDelta})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Output Checkpoint &amp; Card:</span>
                <span className="text-blue-400 truncate max-w-[240px]">
                  {safetensorsPlan.outputFilename} + {generatedModelCard.filename}
                </span>
              </div>
            </div>

            {lastExportedArtifact && (
              <div
                role="status"
                className="pt-3 border-t border-slate-800/80 text-xs text-emerald-400 font-mono tabular-nums"
              >
                Exported {lastExportedArtifact.filename} ({deploymentFormat})
                {lastExportedArtifact.modelCardFilename
                  ? ` + ${lastExportedArtifact.modelCardFilename}`
                  : ''}{' '}
                ({lastExportedArtifact.sizeGb} GB · {lastExportedArtifact.sha256}) at{' '}
                {lastExportedArtifact.timestamp}
              </div>
            )}
          </div>

          {/* Right 7 cols: Standardized Model Card & Safetensors Header Inspector */}
          <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between">
            <div>
              <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800">
                <div className="flex flex-wrap items-center gap-1">
                  {(
                    [
                      { id: 'modelcard', label: 'Model Card (README.md)' },
                      { id: 'header', label: 'Safetensors Header' },
                      { id: 'python', label: 'PEFT Merger (.py)' },
                      { id: 'deploy', label: 'Local Run CLI' },
                    ] as const
                  ).map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setSafetensorsInspectorTab(tab.id)}
                      className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                        safetensorsInspectorTab === tab.id
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard?.writeText(activeSafetensorsSnippet);
                    setCopiedExportSnippet(true);
                    setTimeout(() => setCopiedExportSnippet(false), 1800);
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  {copiedExportSnippet ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      Copy
                    </>
                  )}
                </button>
              </div>

              <pre className="mt-3 font-mono text-[11px] text-slate-200 overflow-x-auto max-h-64 whitespace-pre leading-relaxed">
                {activeSafetensorsSnippet}
              </pre>
            </div>

            <div className="pt-3 mt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-400 font-mono">
              <span>
                Bundle files: {safetensorsPlan.companionFiles.join(', ')},{' '}
                {generatedModelCard.filename}
              </span>
              <span>Standardized HF YAML Frontmatter · Zero pickle execution</span>
            </div>
          </div>
        </div>
      </section>

      {/* Section 05: Post-SFT Stage 2 Preference Alignment (ORPO / SimPO / DPO / KTO), Sequence Packing & Early-Stopping Watchdog */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                Sub-2B Quality Optimizer · Stage 2 Alignment &amp; Packing
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                {alignmentPlan.requiresReferenceModel
                  ? 'Ref Model Required'
                  : `Reference-Free (${alignmentPlan.vramSavingsPct}% VRAM Saved)`}
              </span>
            </div>
            <h2 className="text-base font-semibold text-slate-100 mt-1.5 flex items-center">
              <span>
                05. Post-SFT Preference Alignment (ORPO / SimPO / DPO), Multipack &amp; Early-Stopping Watchdog
              </span>
              <InfoTooltip termKey="orpoAlignment" align="left" />
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Eliminate sub-2B repetition loops and boost IFEval/MT-Bench via reference-free preference optimization, completion-only loss masking, and validation patience rollback
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setStage2Enabled((v) => !v)}
              className={`px-3 py-1.5 text-xs font-mono font-medium rounded-lg border transition-colors ${
                stage2Enabled
                  ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-300'
                  : 'bg-[#0B0F17] border-slate-800 text-slate-400'
              }`}
            >
              {stage2Enabled ? 'Stage 1 SFT → Stage 2 Alignment: ON' : 'Stage 1 SFT Only'}
            </button>

            <button
              type="button"
              onClick={() => {
                onApplyStage2AlignmentBoost?.(
                  alignmentMethod,
                  alignmentPlan.projectedIfEvalGain,
                  alignmentPlan.projectedMtBenchGain
                );
                setAlignmentExecutedMsg(
                  `Executed Stage 2 ${alignmentMethod} pass (β=${alignmentBeta}) on ${selectedModel.name}: IFEval +${alignmentPlan.projectedIfEvalGain}%, MT-Bench +${alignmentPlan.projectedMtBenchGain}, Repetition -${alignmentPlan.repetitionReductionPct}%`
                );
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Run Stage 2 {alignmentMethod} Alignment Pass
            </button>
          </div>
        </div>

        {alignmentExecutedMsg && (
          <div
            role="status"
            className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300 flex items-center justify-between"
          >
            <span>{alignmentExecutedMsg}</span>
            <button
              type="button"
              onClick={() => setAlignmentExecutedMsg(null)}
              className="text-slate-400 hover:text-slate-200 text-[11px]"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 5 cols: Stage 2 Preference Alignment Controls */}
          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200 inline-flex items-center">
                Preference Optimization Method
                <InfoTooltip termKey="orpoAlignment" align="left" />
              </span>
              <span className="text-[11px] font-mono text-emerald-400">
                Est. VRAM: {alignmentPlan.estimatedVramGb} GB
              </span>
            </div>

            <div className="grid grid-cols-4 gap-1.5">
              {(['ORPO', 'SimPO', 'DPO', 'KTO'] as PreferenceAlignmentMethod[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setAlignmentMethod(m)}
                  className={`py-2 px-2 rounded text-xs font-mono font-semibold border transition-colors ${
                    alignmentMethod === m
                      ? 'bg-blue-600/20 border-blue-500 text-blue-300'
                      : 'bg-[#111827] border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div>{m}</div>
                  <div className="text-[9px] font-normal text-slate-400 mt-0.5">
                    {m === 'ORPO' || m === 'SimPO' ? 'Ref-Free' : 'Ref-Model'}
                  </div>
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label
                  htmlFor="alignment-beta-input"
                  className="block text-[11px] text-slate-400 mb-1"
                >
                  Odds/Margin Weight (β / γ):{' '}
                  <span className="font-mono text-slate-200">{alignmentBeta}</span>
                </label>
                <input
                  id="alignment-beta-input"
                  type="range"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  value={alignmentBeta}
                  onChange={(e) => setAlignmentBeta(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              <div>
                <label
                  htmlFor="pref-dataset-select"
                  className="block text-[11px] text-slate-400 mb-1"
                >
                  HF Preference Pair Corpus
                </label>
                <select
                  id="pref-dataset-select"
                  value={preferenceDataset}
                  onChange={(e) => setPreferenceDataset(e.target.value)}
                  className="w-full bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-200"
                >
                  <option value="argilla/ultrafeedback-binarized-preferences-cleaned">
                    argilla/ultrafeedback-binarized
                  </option>
                  <option value="HuggingFaceH4/orca_dpo_pairs">
                    HuggingFaceH4/orca_dpo_pairs
                  </option>
                  <option value="princeton-nlp/llama3-ultrafeedback">
                    princeton-nlp/llama3-ultrafeedback
                  </option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800/80 font-mono text-xs tabular-nums">
              <div className="bg-[#111827] border border-slate-800/80 rounded p-2.5">
                <div className="text-[10px] text-slate-400">IFEval Gain</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{alignmentPlan.projectedIfEvalGain}%
                </div>
              </div>
              <div className="bg-[#111827] border border-slate-800/80 rounded p-2.5">
                <div className="text-[10px] text-slate-400">MT-Bench Gain</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{alignmentPlan.projectedMtBenchGain}
                </div>
              </div>
              <div className="bg-[#111827] border border-slate-800/80 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Repetition Drop</div>
                <div className="text-blue-400 font-semibold mt-0.5">
                  -{alignmentPlan.repetitionReductionPct}%
                </div>
              </div>
            </div>

            <div className="bg-[#111827] border border-slate-800/80 rounded p-3 text-[11px] font-mono space-y-1.5">
              <div className="text-slate-400 flex justify-between">
                <span>Chosen vs. Rejected Margin Preview</span>
                <span className="text-emerald-400">
                  Log-Odds Δ = +{alignmentPlan.chosenRejectedPreview.marginScore}
                </span>
              </div>
              <div className="text-emerald-300 truncate">
                [+] Chosen: {alignmentPlan.chosenRejectedPreview.chosen}
              </div>
              <div className="text-red-400/90 truncate">
                [-] Rejected: {alignmentPlan.chosenRejectedPreview.rejected}
              </div>
            </div>
          </div>

          {/* Middle 4 cols: Sequence Packing (FlashAttention Multipack) & Completion-Only Loss Masking */}
          <div className="lg:col-span-4 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200 inline-flex items-center">
                  Multipack &amp; Loss Masking
                  <InfoTooltip termKey="sequencePacking" />
                </span>
                <span className="text-[11px] font-mono text-emerald-400">
                  {packingMetrics.speedupMultiplier}x Speedup
                </span>
              </div>

              <label className="flex items-start gap-2.5 p-2.5 rounded bg-[#111827] border border-slate-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={samplePackingEnabled}
                  onChange={(e) => setSamplePackingEnabled(e.target.checked)}
                  className="mt-0.5 rounded border-slate-700 bg-slate-900 text-blue-600"
                />
                <div className="text-xs">
                  <div className="font-medium text-slate-200">
                    FlashAttention-2/3 Sample Packing (Multipack)
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    Concatenates short examples with block-diagonal varlen masks; cuts &lt;pad&gt; waste to {packingMetrics.paddingWastePct}%
                  </div>
                </div>
              </label>

              <label className="flex items-start gap-2.5 p-2.5 rounded bg-[#111827] border border-slate-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={completionOnlyLossEnabled}
                  onChange={(e) => setCompletionOnlyLossEnabled(e.target.checked)}
                  className="mt-0.5 rounded border-slate-700 bg-slate-900 text-blue-600"
                />
                <div className="text-xs">
                  <div className="font-medium text-slate-200">
                    Completion-Only Loss Masking (Train on Responses Only)
                  </div>
                  <div className="text-[11px] text-slate-400 mt-0.5">
                    Sets label=-100 on &lt;|im_start|&gt;user tokens ({packingMetrics.maskedPromptTokenRatioPct}% masked) so sub-2B capacity focuses 100% on completions
                  </div>
                </div>
              </label>
            </div>

            <div className="pt-3 border-t border-slate-800/80 space-y-2 font-mono text-xs tabular-nums">
              <div className="flex justify-between">
                <span className="text-slate-400">Padding Token Waste:</span>
                <span
                  className={
                    samplePackingEnabled ? 'text-emerald-400 font-semibold' : 'text-amber-400'
                  }
                >
                  {packingMetrics.paddingWastePct}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Effective Packed Throughput:</span>
                <span className="text-slate-100 font-semibold">
                  {packingMetrics.effectiveThroughputTokSec.toLocaleString()} tok/s
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Est. Sub-2B Accuracy Lift:</span>
                <span className="text-emerald-400 font-semibold">
                  +{packingMetrics.estimatedSub2bAccuracyGainPct}% downstream
                </span>
              </div>
            </div>
          </div>

          {/* Right 3 cols: Automated Early-Stopping & Checkpoint Rollback Watchdog */}
          <div className="lg:col-span-3 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200 inline-flex items-center">
                  Early-Stop Watchdog
                  <InfoTooltip termKey="earlyStopping" align="right" />
                </span>
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-semibold ${
                    earlyStopReport.shouldTriggerEarlyStop
                      ? 'bg-red-500/20 text-red-400'
                      : 'bg-emerald-500/15 text-emerald-400'
                  }`}
                >
                  {earlyStopReport.statusLabel}
                </span>
              </div>

              <div className="bg-[#111827] border border-slate-800 rounded p-2.5 font-mono text-xs tabular-nums space-y-1">
                <div className="text-[10px] text-slate-400 uppercase">
                  Best Val Loss Checkpoint Snapshot
                </div>
                <div className="text-sm font-semibold text-emerald-400">
                  Step {earlyStopReport.bestCheckpoint.step} · Val{' '}
                  {earlyStopReport.bestCheckpoint.valLoss.toFixed(3)}
                </div>
                <div className="text-[11px] text-slate-400">
                  Epoch {earlyStopReport.bestCheckpoint.epoch} · Train{' '}
                  {earlyStopReport.bestCheckpoint.trainLoss.toFixed(3)} · Gap{' '}
                  {earlyStopReport.generalizationGap}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <label
                    htmlFor="watchdog-patience-select"
                    className="block text-[10px] text-slate-400 mb-1"
                  >
                    Patience (Evals)
                  </label>
                  <select
                    id="watchdog-patience-select"
                    value={watchdogPatience}
                    onChange={(e) => setWatchdogPatience(Number(e.target.value))}
                    className="w-full bg-[#111827] border border-slate-800 rounded px-2 py-1 text-xs font-mono text-slate-200"
                  >
                    <option value={1}>1 eval step</option>
                    <option value={2}>2 eval steps</option>
                    <option value={3}>3 eval steps</option>
                    <option value={4}>4 eval steps</option>
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="watchdog-delta-select"
                    className="block text-[10px] text-slate-400 mb-1"
                  >
                    Min Val Δ
                  </label>
                  <select
                    id="watchdog-delta-select"
                    value={watchdogMinDelta}
                    onChange={(e) => setWatchdogMinDelta(Number(e.target.value))}
                    className="w-full bg-[#111827] border border-slate-800 rounded px-2 py-1 text-xs font-mono text-slate-200"
                  >
                    <option value={0.002}>0.002</option>
                    <option value={0.005}>0.005</option>
                    <option value={0.01}>0.010</option>
                  </select>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed">
                {earlyStopReport.recommendation}
              </p>
            </div>

            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => {
                    onSimulateOverfitStep?.();
                    setWatchdogRollbackBanner(null);
                  }}
                  className="flex-1 px-2.5 py-1.5 text-[11px] font-mono text-amber-300 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 rounded transition-colors"
                >
                  Simulate Overfit Spike
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onRollbackTrainingToIndex?.(earlyStopReport.bestIndex);
                    setWatchdogRollbackBanner(
                      `Rolled back training state to Best Checkpoint Step ${earlyStopReport.bestCheckpoint.step} (val_loss=${earlyStopReport.bestCheckpoint.valLoss.toFixed(3)})`
                    );
                  }}
                  className="flex-1 inline-flex items-center justify-center gap-1 px-2.5 py-1.5 text-[11px] font-mono font-semibold text-emerald-300 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 rounded transition-colors"
                >
                  <RotateCcw className="w-3 h-3" />
                  Rollback to Best
                </button>
              </div>
              {watchdogRollbackBanner && (
                <div className="text-[11px] font-mono text-emerald-400">
                  {watchdogRollbackBanner}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Section 06: Multi-Adapter Model Merging (DARE-TIES / SLERP) & 1-Click Executable Run Bundle Exporter */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 6 cols: Multi-Adapter Model Merging & Task Arithmetic Studio */}
        <section className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100 flex items-center">
                  <span>06. Multi-Adapter Task Arithmetic Merger (DARE-TIES / SLERP)</span>
                  <InfoTooltip termKey="adapterMerging" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Combine specialized sub-2B LoRA adapters without multi-task gradient interference
                </p>
              </div>

              <div className="flex items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800">
                {(['DARE-TIES', 'TIES', 'SLERP', 'LINEAR'] as AdapterMergeAlgorithm[]).map(
                  (alg) => (
                    <button
                      key={alg}
                      type="button"
                      onClick={() => setMergeAlgorithm(alg)}
                      className={`px-2 py-1 text-[11px] font-mono rounded transition-colors ${
                        mergeAlgorithm === alg
                          ? 'bg-blue-600 text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {alg}
                    </button>
                  )
                )}
              </div>
            </div>

            {/* Specialist Adapter Weight Sliders */}
            <div className="space-y-2.5">
              {specialistAdapters.map((adapter, idx) => (
                <div
                  key={adapter.id}
                  className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                >
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={adapter.enabled}
                      onChange={(e) => {
                        const next = [...specialistAdapters];
                        next[idx] = { ...adapter, enabled: e.target.checked };
                        setSpecialistAdapters(next);
                      }}
                      className="rounded border-slate-700 bg-slate-900 text-blue-600"
                    />
                    <div>
                      <div className="font-medium text-slate-200">{adapter.name}</div>
                      <div className="text-[10px] font-mono text-slate-400">
                        {adapter.id} · Domain: {adapter.domain}
                      </div>
                    </div>
                  </label>

                  <div className="flex items-center gap-2.5 sm:w-48">
                    <input
                      type="range"
                      aria-label={`Weight for ${adapter.name}`}
                      min={0.05}
                      max={1.0}
                      step={0.05}
                      disabled={!adapter.enabled}
                      value={adapter.weight}
                      onChange={(e) => {
                        const next = [...specialistAdapters];
                        next[idx] = { ...adapter, weight: Number(e.target.value) };
                        setSpecialistAdapters(next);
                      }}
                      className="w-full accent-blue-500"
                    />
                    <span className="text-xs font-mono tabular-nums text-slate-200 w-10 text-right">
                      {adapter.enabled ? adapter.weight.toFixed(2) : 'OFF'}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Predicted Composite Gains & Sign Conflict Resolution */}
            <div className="grid grid-cols-4 gap-2 font-mono text-xs tabular-nums">
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">MMLU-Pro Δ</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{multiAdapterPlan.predictedCompositeGains.mmluProDelta}%
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">HumanEval+ Δ</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{multiAdapterPlan.predictedCompositeGains.humanEvalDelta}%
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">GSM8K Δ</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{multiAdapterPlan.predictedCompositeGains.gsm8kDelta}%
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Sign Resolved</div>
                <div className="text-blue-400 font-semibold mt-0.5">
                  {multiAdapterPlan.signConflictsResolvedPct}%
                </div>
              </div>
            </div>

            {/* Mergekit YAML / PEFT Python Script Preview */}
            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={() => setAdapterPreviewTab('yaml')}
                    className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                      adapterPreviewTab === 'yaml'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    mergekit.yaml
                  </button>
                  <button
                    type="button"
                    onClick={() => setAdapterPreviewTab('python')}
                    className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                      adapterPreviewTab === 'python'
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    peft_weighted_merge.py
                  </button>
                </div>
                <span className="text-[11px] font-mono text-slate-400">
                  Density: {mergeDensity} · {multiAdapterPlan.activeAdapterCount} adapters
                </span>
              </div>
              <pre className="mt-2 font-mono text-[11px] text-slate-300 overflow-x-auto max-h-36 whitespace-pre leading-relaxed">
                {adapterPreviewTab === 'yaml'
                  ? multiAdapterPlan.mergekitYaml
                  : multiAdapterPlan.peftMergeScript}
              </pre>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-slate-800/80">
            <span className="text-xs font-mono text-slate-400">
              Target ID: <strong className="text-slate-200">{multiAdapterPlan.mergedAdapterCheckpointId}</strong>
            </span>
            <button
              type="button"
              onClick={() =>
                setCustomMergedCheckpointOverride(multiAdapterPlan.mergedAdapterCheckpointId)
              }
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
            >
              <GitMerge className="w-3.5 h-3.5" />
              Bind Merged Adapter to Section 04 Exporter
            </button>
          </div>
        </section>

        {/* Right 6 cols: 1-Click Executable Training Script & Config Bundle Exporter */}
        <section className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  07. 1-Click Executable Training Script &amp; Config Bundle Exporter
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Ready-to-run Axolotl YAML, Unsloth + TRL Python scripts, Dataset Blend JSON, and Ollama Modelfile
                </p>
              </div>

              <button
                type="button"
                onClick={() => {
                  const files: Array<{ name: string; content: string }> = [
                    { name: 'axolotl_config.yaml', content: executableRunBundle.axolotlConfigYaml },
                    { name: 'unsloth_train.py', content: executableRunBundle.unslothTrainPy },
                    { name: 'trl_stage2_orpo.py', content: executableRunBundle.trlSftOrpoPy },
                    { name: 'dataset_blend.json', content: executableRunBundle.datasetBlendJson },
                    { name: 'Modelfile', content: executableRunBundle.ollamaModelfile },
                  ];
                  const combinedBundle = files
                    .map((f) => `# ==========================================\n# FILE: ${f.name}\n# ==========================================\n${f.content}`)
                    .join('\n\n');
                  const blob = new Blob([combinedBundle], { type: 'text/plain;charset=utf-8' });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `forgeml-${selectedModel.name.toLowerCase().replace(/[^a-z0-9.-]+/g, '-')}-run-bundle.txt`;
                  a.click();
                  URL.revokeObjectURL(url);
                  setBundleDownloadedBanner(
                    `Exported 5-file executable training bundle for ${selectedModel.name} (Axolotl + Unsloth + TRL ${alignmentMethod} + Blend + Modelfile)`
                  );
                }}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                Download Complete Run Bundle (5 Files)
              </button>
            </div>

            {bundleDownloadedBanner && (
              <div
                role="status"
                className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-3.5 py-2 text-xs font-mono text-emerald-300"
              >
                {bundleDownloadedBanner}
              </div>
            )}

            <div className="flex flex-wrap items-center gap-1 bg-[#0B0F17] p-1.5 rounded-lg border border-slate-800">
              {(
                [
                  { id: 'unsloth', label: 'unsloth_train.py' },
                  { id: 'axolotl', label: 'axolotl_config.yaml' },
                  { id: 'trl_orpo', label: `trl_${alignmentMethod.toLowerCase()}.py` },
                  { id: 'blend_json', label: 'dataset_blend.json' },
                  { id: 'modelfile', label: 'Modelfile' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setBundlePreviewTab(tab.id)}
                  className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                    bundlePreviewTab === tab.id
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5">
              <pre className="font-mono text-[11px] text-slate-200 overflow-x-auto max-h-72 whitespace-pre leading-relaxed">
                {bundlePreviewTab === 'unsloth'
                  ? executableRunBundle.unslothTrainPy
                  : bundlePreviewTab === 'axolotl'
                  ? executableRunBundle.axolotlConfigYaml
                  : bundlePreviewTab === 'trl_orpo'
                  ? executableRunBundle.trlSftOrpoPy
                  : bundlePreviewTab === 'blend_json'
                  ? executableRunBundle.datasetBlendJson
                  : executableRunBundle.ollamaModelfile}
              </pre>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
            <span>
              Synced with: {selectedModel.name} · LoRA r={activeHpoConfig.loraRank} α={activeHpoConfig.loraAlpha}
            </span>
            <span>
              Multipack: {samplePackingEnabled ? 'ON' : 'OFF'} · Masking: {completionOnlyLossEnabled ? '-100 Prompt' : 'OFF'}
            </span>
          </div>
        </section>
      </div>

      {/* Section 07B: Interactive VRAM & Training Time Budget Constraint Solver + Side-by-Side Script Diff Viewer */}
      <ConstraintSolverAndScriptDiffPanel
        modelId={selectedModel.id}
        modelName={selectedModel.name}
        architecture={selectedModel.architecture}
        paramsBillions={selectedModel.paramsBillions}
        contextWindow={selectedModel.contextWindow}
        activeHpoConfig={activeHpoConfig}
        samplePackingEnabled={samplePackingEnabled}
        completionOnlyLossEnabled={completionOnlyLossEnabled}
        stage2Enabled={stage2Enabled}
        alignmentMethod={alignmentMethod}
        alignmentBeta={alignmentBeta}
        preferenceDataset={preferenceDataset}
        datasetSources={activeDatasetCommit.sourceDatasets}
        targetFormat={activeDatasetCommit.targetFormat}
        currentBundle={executableRunBundle}
        onApplySolvedConstraints={(solution) => {
          onUpdateHpoConfig(solution.hpoConfig);
          setSamplePackingEnabled(solution.samplePackingEnabled);
          setCompletionOnlyLossEnabled(solution.completionOnlyLossEnabled);
        }}
      />

      {/* Section 08: Saved Experiment Profiles & Run History Diffing (Run Ledger) */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-blue-400" />
              <span>08. Saved Experiment Profiles &amp; Side-by-Side Run History Diffing</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Persisted to LocalStorage ({experimentSnapshots.length} snapshots) · Snapshot current HPO + Dataset + Stage 2 state, compare runs, or restore configurations in 1 click
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <input
              type="text"
              aria-label="New experiment snapshot name"
              placeholder={`e.g. ${selectedModel.name} r=${activeHpoConfig.loraRank} + ${alignmentMethod}`}
              value={newSnapshotName}
              onChange={(e) => setNewSnapshotName(e.target.value)}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 w-64"
            />
            <button
              type="button"
              onClick={() => {
                const label =
                  newSnapshotName.trim() ||
                  `Run #${experimentSnapshots.length + 1}: ${selectedModel.name} (r=${activeHpoConfig.loraRank}, ${stage2Enabled ? alignmentMethod : 'SFT'})`;
                const created: ExperimentSnapshot = {
                  id: `snap-${Date.now()}`,
                  name: label,
                  createdAt: new Date().toTimeString().split(' ')[0] + ' UTC',
                  modelId: selectedModel.id,
                  modelName: selectedModel.name,
                  architecture: selectedModel.architecture,
                  paramsBillions: selectedModel.paramsBillions,
                  datasetVersionId: activeDatasetCommit.versionId,
                  hpoConfig: { ...activeHpoConfig },
                  samplePackingEnabled,
                  completionOnlyLossEnabled,
                  stage2Method: stage2Enabled ? alignmentMethod : 'NONE',
                  finalStep: latestPoint.step,
                  finalTrainLoss: latestPoint.trainLoss,
                  finalValLoss: latestPoint.valLoss,
                  mmluPro:
                    selectedModel.fineTunedBenchmarks?.mmluPro ??
                    selectedModel.baseBenchmarks.mmluPro,
                  humanEval:
                    selectedModel.fineTunedBenchmarks?.humanEval ??
                    selectedModel.baseBenchmarks.humanEval,
                  gsm8k:
                    selectedModel.fineTunedBenchmarks?.gsm8k ??
                    selectedModel.baseBenchmarks.gsm8k,
                  ifEval:
                    selectedModel.fineTunedBenchmarks?.ifEval ??
                    selectedModel.baseBenchmarks.ifEval,
                };
                setExperimentSnapshots((prev) => [created, ...prev]);
                setDiffSnapBId(created.id);
                setNewSnapshotName('');
                setSnapshotStatusBanner(`Saved snapshot "${created.name}" to LocalStorage.`);
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
            >
              <Bookmark className="w-3.5 h-3.5" />
              Save Current Run Snapshot
            </button>
          </div>
        </div>

        {snapshotStatusBanner && (
          <div
            role="status"
            className="bg-blue-950/30 border border-blue-500/40 rounded-lg px-3.5 py-2 text-xs font-mono text-blue-300"
          >
            {snapshotStatusBanner}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 6 cols: Saved Experiment Snapshot Ledger */}
          <div className="lg:col-span-6 space-y-2.5">
            <div className="text-xs font-semibold text-slate-300">
              Saved Experiment Snapshots (Click Restore to Load Config)
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
              {experimentSnapshots.map((snap) => (
                <div
                  key={snap.id}
                  className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                >
                  <div>
                    <div className="text-xs font-semibold text-slate-100">{snap.name}</div>
                    <div className="text-[11px] font-mono text-slate-400 mt-0.5">
                      {snap.architecture} · r={snap.hpoConfig.loraRank} · lr={snap.hpoConfig.learningRate} · Stage 2: {snap.stage2Method} · Val Loss: {snap.finalValLoss.toFixed(2)}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      onSelectModel(snap.modelId);
                      onUpdateHpoConfig(snap.hpoConfig);
                      setSamplePackingEnabled(snap.samplePackingEnabled);
                      setCompletionOnlyLossEnabled(snap.completionOnlyLossEnabled);
                      if (snap.stage2Method !== 'NONE') {
                        setStage2Enabled(true);
                        setAlignmentMethod(snap.stage2Method);
                      } else {
                        setStage2Enabled(false);
                      }
                      setSnapshotStatusBanner(
                        `Restored configuration from "${snap.name}" (LoRA r=${snap.hpoConfig.loraRank}, lr=${snap.hpoConfig.learningRate}).`
                      );
                    }}
                    className="px-2.5 py-1 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded transition-colors shrink-0"
                  >
                    Restore Config
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Right 6 cols: Side-by-Side Snapshot Diff Viewer */}
          <div className="lg:col-span-6 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <div>
                <label htmlFor="diff-snap-a" className="block text-[10px] text-slate-400 mb-1">
                  Baseline Snapshot (A)
                </label>
                <select
                  id="diff-snap-a"
                  value={diffSnapAId}
                  onChange={(e) => setDiffSnapAId(e.target.value)}
                  className="w-full bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-200"
                >
                  {experimentSnapshots.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="diff-snap-b" className="block text-[10px] text-slate-400 mb-1">
                  Comparison Snapshot (B)
                </label>
                <select
                  id="diff-snap-b"
                  value={diffSnapBId}
                  onChange={(e) => setDiffSnapBId(e.target.value)}
                  className="w-full bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-200"
                >
                  {experimentSnapshots.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2 font-mono text-xs tabular-nums pt-1">
              <div className="bg-[#111827] border border-slate-800 rounded p-2">
                <div className="text-[10px] text-slate-400">Val Loss Δ</div>
                <div
                  className={`font-semibold mt-0.5 ${
                    snapshotDiff.valLossDelta <= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {snapshotDiff.valLossDelta > 0 ? '+' : ''}
                  {snapshotDiff.valLossDelta}
                </div>
              </div>
              <div className="bg-[#111827] border border-slate-800 rounded p-2">
                <div className="text-[10px] text-slate-400">MMLU-Pro Δ</div>
                <div
                  className={`font-semibold mt-0.5 ${
                    snapshotDiff.mmluProDelta >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {snapshotDiff.mmluProDelta >= 0 ? '+' : ''}
                  {snapshotDiff.mmluProDelta}%
                </div>
              </div>
              <div className="bg-[#111827] border border-slate-800 rounded p-2">
                <div className="text-[10px] text-slate-400">HumanEval Δ</div>
                <div
                  className={`font-semibold mt-0.5 ${
                    snapshotDiff.humanEvalDelta >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {snapshotDiff.humanEvalDelta >= 0 ? '+' : ''}
                  {snapshotDiff.humanEvalDelta}%
                </div>
              </div>
              <div className="bg-[#111827] border border-slate-800 rounded p-2">
                <div className="text-[10px] text-slate-400">IFEval Δ</div>
                <div
                  className={`font-semibold mt-0.5 ${
                    snapshotDiff.ifEvalDelta >= 0 ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {snapshotDiff.ifEvalDelta >= 0 ? '+' : ''}
                  {snapshotDiff.ifEvalDelta}%
                </div>
              </div>
            </div>

            {snapshotDiff.changedHyperparameters.length > 0 && (
              <div className="space-y-1 text-[11px] font-mono">
                <div className="text-slate-400">Parameter Deltas (A → B):</div>
                {snapshotDiff.changedHyperparameters.map((ch) => (
                  <div key={ch.param} className="flex justify-between text-slate-300">
                    <span>{ch.param}:</span>
                    <span>
                      <span className="text-slate-400">{ch.valueA}</span> →{' '}
                      <span className="text-emerald-400 font-semibold">{ch.valueB}</span>
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className="pt-2 border-t border-slate-800/80 text-xs text-emerald-400 font-mono">
              {snapshotDiff.winnerSummary}
            </div>
          </div>
        </div>
      </section>

      {/* Section 09: Activation-Aware imatrix Quantization Calibration & Stage 3 GRPO Verifiable Reward Studio */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 6 cols: Activation-Aware Importance Matrix (imatrix) & Quantization Calibration Studio */}
        <section className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100 flex items-center">
                  <span>09. Activation-Aware imatrix &amp; Quantization Calibration Studio</span>
                  <InfoTooltip termKey="imatrixCalibration" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Protect salient outlier channels in {selectedModel.name} before GGUF quantization
                </p>
              </div>

              <div className="flex items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800">
                {(['Q4_K_M', 'Q4_K_S', 'IQ4_XS', 'Q5_K_M', 'Q8_0'] as ImatrixQuantType[]).map(
                  (qt) => (
                    <button
                      key={qt}
                      type="button"
                      onClick={() => setImatrixQuantType(qt)}
                      className={`px-2 py-1 text-[11px] font-mono rounded transition-colors ${
                        imatrixQuantType === qt
                          ? 'bg-blue-600 text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {qt}
                    </button>
                  )
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label htmlFor="imatrix-samples-select" className="block text-xs text-slate-400 mb-1">
                  Calibration Corpus Chunks
                </label>
                <select
                  id="imatrix-samples-select"
                  value={imatrixSamplesCount}
                  onChange={(e) => setImatrixSamplesCount(Number(e.target.value))}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value={128}>128 Chunks (Fast Edge Calibration)</option>
                  <option value={256}>256 Chunks (Recommended Sub-2B)</option>
                  <option value={512}>512 Chunks (Maximum Outlier Protection)</option>
                </select>
              </div>

              <div className="flex items-end pb-1">
                <label className="inline-flex items-center gap-2 text-xs text-slate-200 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={protectSalientLayers}
                    onChange={(e) => setProtectSalientLayers(e.target.checked)}
                    className="rounded border-slate-700 bg-[#0B0F17] text-blue-600"
                  />
                  <span>Keep embed_tokens &amp; lm_head at Q8_0</span>
                </label>
              </div>
            </div>

            {/* Perplexity Comparison Cards */}
            <div className="grid grid-cols-4 gap-2 font-mono text-xs tabular-nums">
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">FP16 Baseline PPL</div>
                <div className="text-slate-100 font-semibold mt-0.5">
                  {imatrixPlan.fp16Perplexity}
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Naive {imatrixQuantType}</div>
                <div className="text-amber-400 font-semibold mt-0.5">
                  {imatrixPlan.naiveQuantPerplexity}
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">imatrix {imatrixQuantType}</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  {imatrixPlan.imatrixQuantPerplexity}
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">PPL Recovery</div>
                <div className="text-blue-400 font-semibold mt-0.5">
                  {imatrixPlan.perplexityRecoveryPct}%
                </div>
              </div>
            </div>

            {/* Protected Salient Tensors */}
            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3 space-y-1.5 text-xs font-mono">
              <div className="text-[11px] text-slate-400 flex justify-between">
                <span>Salient Tensor Outlier Protection ({selectedModel.architecture})</span>
                <span>Artifact: {imatrixPlan.quantizedSizeGb} GB</span>
              </div>
              {imatrixPlan.salientTensorsProtected.map((t) => (
                <div
                  key={t.tensorName}
                  className="flex items-center justify-between text-[11px] py-1 border-t border-slate-800/70"
                >
                  <span className="text-slate-300 truncate">{t.tensorName}</span>
                  <span className="text-emerald-400 shrink-0">
                    Salience {t.activationSalienceScore} · {t.fallbackPrecision}
                  </span>
                </div>
              ))}
            </div>

            <pre className="p-3 bg-[#0B0F17] border border-slate-800 rounded-lg font-mono text-[11px] text-slate-300 overflow-x-auto max-h-36 whitespace-pre leading-relaxed">
              {imatrixPlan.calibrationCliPipeline}
            </pre>
          </div>

          <div className="flex items-center justify-between gap-2 pt-3 border-t border-slate-800/80">
            <span className="text-xs font-mono text-slate-400">
              imatrix file: {imatrixPlan.imatrixArtifactSizeMb} MB
            </span>
            <button
              type="button"
              onClick={() => {
                navigator.clipboard?.writeText(imatrixPlan.calibrationCliPipeline);
                setCopiedImatrixCmd(true);
                setTimeout(() => setCopiedImatrixCmd(false), 1800);
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
            >
              {copiedImatrixCmd ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  Copied imatrix Pipeline
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy imatrix CLI Pipeline
                </>
              )}
            </button>
          </div>
        </section>

        {/* Right 6 cols: Stage 3 GRPO Verifiable Reward Studio (RLVR) */}
        <section className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100 flex items-center">
                  <span>Stage 3 · GRPO Verifiable Reward Studio (RLVR)</span>
                  <InfoTooltip termKey="grpoRlvr" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Critic-free group relative policy optimization (0 GB Critic VRAM overhead)
                </p>
              </div>

              <div className="flex items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800">
                {([4, 8, 16] as const).map((g) => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => setGrpoRolloutSize(g)}
                    className={`px-2.5 py-1 text-[11px] font-mono rounded transition-colors ${
                      grpoRolloutSize === g
                        ? 'bg-blue-600 text-white font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    G={g} Rollouts
                  </button>
                ))}
              </div>
            </div>

            {/* Deterministic Verifier Checkboxes */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {(
                [
                  { id: 'json_schema', label: 'Strict JSON Schema Verifier (+1.5)' },
                  { id: 'python_pytest', label: 'Python AST & Pytest Sandbox (+1.5)' },
                  { id: 'math_symbolic', label: 'Symbolic Math Answer Matcher (+1.5)' },
                  { id: 'think_format_length', label: '<think> XML Envelope Verifier (+1.0)' },
                ] as Array<{ id: GrpoVerifierId; label: string }>
              ).map((v) => {
                const checked = grpoVerifiers.includes(v.id);
                return (
                  <label
                    key={v.id}
                    className="flex items-center gap-2 p-2.5 rounded bg-[#0B0F17] border border-slate-800 text-xs text-slate-200 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setGrpoVerifiers([...grpoVerifiers, v.id]);
                        } else if (grpoVerifiers.length > 1) {
                          setGrpoVerifiers(grpoVerifiers.filter((x) => x !== v.id));
                        }
                      }}
                      className="rounded border-slate-700 bg-slate-900 text-blue-600"
                    />
                    <span className="truncate">{v.label}</span>
                  </label>
                );
              })}
            </div>

            {/* GRPO Metrics */}
            <div className="grid grid-cols-4 gap-2 font-mono text-xs tabular-nums">
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Critic VRAM</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  {grpoPlan.criticVramOverheadGb} GB (Free)
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">HumanEval+ Δ</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{grpoPlan.projectedHumanEvalGain}%
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">GSM8K Δ</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{grpoPlan.projectedGsm8kGain}%
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Rollout VRAM</div>
                <div className="text-blue-400 font-semibold mt-0.5">
                  {grpoPlan.totalGrpoVramGb} GB
                </div>
              </div>
            </div>

            <pre className="p-3 bg-[#0B0F17] border border-slate-800 rounded-lg font-mono text-[11px] text-slate-300 overflow-x-auto max-h-36 whitespace-pre leading-relaxed">
              {grpoPlan.trlGrpoScriptPy}
            </pre>

            {grpoExecutedBanner && (
              <div className="text-xs font-mono text-emerald-400">{grpoExecutedBanner}</div>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-slate-800/80">
            <span className="text-xs font-mono text-slate-400">
              Final Pass Rate:{' '}
              <strong className="text-emerald-400">
                {grpoPlan.rewardTrajectory[grpoPlan.rewardTrajectory.length - 1].verifierPassRatePct}%
              </strong>
            </span>
            <button
              type="button"
              onClick={() => {
                onApplyStage3GrpoBoost?.(
                  grpoPlan.projectedHumanEvalGain,
                  grpoPlan.projectedGsm8kGain,
                  grpoPlan.projectedIfEvalGain
                );
                setGrpoExecutedBanner(
                  `Executed Stage 3 GRPO RLVR pass (G=${grpoRolloutSize}): HumanEval+ +${grpoPlan.projectedHumanEvalGain}%, GSM8K +${grpoPlan.projectedGsm8kGain}%.`
                );
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors"
            >
              <Zap className="w-3.5 h-3.5" />
              Run Stage 3 GRPO Verifier Pass
            </button>
          </div>
        </section>
      </div>

      {/* Section 10: Layer-Wise Adaptive LoRA (DoRA / rsLoRA) & Sub-2B Hardware / Edge Target Simulator */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 6 cols: Layer-Wise Adaptive LoRA (DoRA / rsLoRA / SVD Rank Allocation) */}
        <section className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100 flex items-center">
                  <span>10. Layer-Wise Adaptive LoRA (DoRA / rsLoRA / SVD Rank)</span>
                  <InfoTooltip termKey="doraRankAllocation" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Concentrate trainable rank in upper reasoning layers without bloating adapter size
                </p>
              </div>

              <select
                aria-label="LoRA Layer Allocation Policy"
                value={loraAllocationPolicy}
                onChange={(e) =>
                  setLoraAllocationPolicy(e.target.value as LoraAllocationPolicy)
                }
                className="bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
              >
                <option value="svd_salience">SVD Salience Profile (Recommended)</option>
                <option value="top_heavy_reasoning">Top-Heavy Reasoning Quartiles</option>
                <option value="attention_mlp_split">Attention vs. MLP Split</option>
                <option value="uniform">Uniform Rank Across All Layers</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              <label className="flex items-center gap-2 p-2.5 rounded bg-[#0B0F17] border border-slate-800 text-xs text-slate-200 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useDora}
                  onChange={(e) => setUseDora(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-900 text-blue-600"
                />
                <span>Enable DoRA (Weight Decomposition)</span>
              </label>
              <label className="flex items-center gap-2 p-2.5 rounded bg-[#0B0F17] border border-slate-800 text-xs text-slate-200 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useRsLora}
                  onChange={(e) => setUseRsLora(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-900 text-blue-600"
                />
                <span>Enable rsLoRA (α / √r Scaling)</span>
              </label>
            </div>

            {/* 4-Quartile Layer Allocation Table */}
            <div className="space-y-1.5 font-mono text-xs">
              {adaptiveLoraPlan.layerQuartiles.map((q) => (
                <div
                  key={q.quartileLabel}
                  className="p-2.5 rounded bg-[#0B0F17] border border-slate-800/90 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                >
                  <div>
                    <div className="text-slate-200 font-semibold">
                      {q.quartileLabel} ({q.layerRange})
                    </div>
                    <div className="text-[10px] text-slate-400">
                      Modules: {q.primaryModules} · Sensitivity: {q.gradientSensitivity}
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="px-2 py-0.5 rounded bg-blue-600/20 text-blue-300 font-semibold">
                      r={q.allocatedRank} · α={q.effectiveAlpha}
                    </span>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      Scale: {q.scalingFactor}x
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <span className="text-slate-400">
              Trainable: <strong className="text-slate-200">{adaptiveLoraPlan.trainableParamsMillions}M params</strong>
            </span>
            <span className="text-emerald-400">
              Projected Lift: +{adaptiveLoraPlan.projectedMmluGainPct}% MMLU-Pro · +
              {adaptiveLoraPlan.projectedHumanEvalGainPct}% HumanEval+
            </span>
          </div>
        </section>

        {/* Right 6 cols: Hardware Profile & Edge Target Simulator */}
        <section className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="pb-3 border-b border-slate-800/80">
              <h2 className="text-base font-semibold text-slate-100">
                Sub-2B Training GPU &amp; Edge Inference Target Simulator
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Simulate training VRAM headroom and local edge device throughput for {selectedModel.name}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label htmlFor="sim-train-gpu" className="block text-xs text-slate-400 mb-1">
                  Training GPU Hardware Profile
                </label>
                <select
                  id="sim-train-gpu"
                  value={trainingGpuId}
                  onChange={(e) => setTrainingGpuId(e.target.value as TrainingGpuTargetId)}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value="rtx3060-12gb">NVIDIA RTX 3060 (12 GB)</option>
                  <option value="rtx4090-24gb">NVIDIA RTX 4090 (24 GB)</option>
                  <option value="mac-m3-max-64gb">Apple M3 Max Unified (64 GB)</option>
                  <option value="dual-t4-32gb">2x Cloud Tesla T4 (32 GB)</option>
                  <option value="a100-80gb">NVIDIA A100 SXM4 (80 GB)</option>
                </select>
              </div>

              <div>
                <label htmlFor="sim-edge-device" className="block text-xs text-slate-400 mb-1">
                  Local Edge Inference Target
                </label>
                <select
                  id="sim-edge-device"
                  value={edgeDeviceId}
                  onChange={(e) => setEdgeDeviceId(e.target.value as EdgeDeviceTargetId)}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value="macbook-air-m2-8gb">MacBook Air M2 (8 GB Unified)</option>
                  <option value="rpi5-8gb">Raspberry Pi 5 (8 GB LPDDR4X)</option>
                  <option value="iphone-16-pro-ane">iPhone 16 Pro (A18 Pro 8 GB)</option>
                  <option value="steam-deck-oled-16gb">Steam Deck OLED APU (16 GB)</option>
                  <option value="rtx4060-laptop-8gb">RTX 4060 Laptop (8 GB GDDR6)</option>
                </select>
              </div>
            </div>

            {/* Training GPU Simulation Card */}
            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5 space-y-2 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-semibold">
                  Training Target: {hwSimResult.trainingProfile.gpuName}
                </span>
                <span
                  className={
                    hwSimResult.trainingProfile.fitsInVram
                      ? 'text-emerald-400 font-semibold'
                      : 'text-red-400 font-semibold'
                  }
                >
                  {hwSimResult.trainingProfile.fitsInVram ? '[FITS IN VRAM]' : '[OOM WARNING]'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] tabular-nums">
                <div>
                  <div className="text-slate-400">Required / Total</div>
                  <div className="text-slate-100 font-semibold">
                    {hwSimResult.trainingProfile.requiredVramGb} /{' '}
                    {hwSimResult.trainingProfile.vramTotalGb} GB
                  </div>
                </div>
                <div>
                  <div className="text-slate-400">Max Safe Batch</div>
                  <div className="text-blue-400 font-semibold">
                    micro_batch={hwSimResult.trainingProfile.maxSafeBatchSize}
                  </div>
                </div>
                <div>
                  <div className="text-slate-400">Est. Epoch Time</div>
                  <div className="text-emerald-400 font-semibold">
                    {hwSimResult.trainingProfile.estimatedEpochMinutes} min/epoch
                  </div>
                </div>
              </div>
            </div>

            {/* Edge Device Inference Simulation Card */}
            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5 space-y-2 font-mono text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-300 font-semibold">
                  Edge Target: {hwSimResult.edgeProfile.deviceName}
                </span>
                <span className="text-blue-400 text-[11px]">
                  TTFT: {hwSimResult.edgeProfile.firstTokenLatencyMs} ms
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 pt-1 text-[11px] tabular-nums">
                <div className="p-2 rounded bg-[#111827] border border-slate-800">
                  <div className="text-slate-400">FP16 ({hwSimResult.edgeProfile.footprintFp16Gb} GB)</div>
                  <div className="text-slate-200 font-semibold mt-0.5">
                    {hwSimResult.edgeProfile.tokPerSecFp16} tok/s
                  </div>
                </div>
                <div className="p-2 rounded bg-[#111827] border border-slate-800">
                  <div className="text-slate-400">Q8_0 ({hwSimResult.edgeProfile.footprintQ8Gb} GB)</div>
                  <div className="text-emerald-400 font-semibold mt-0.5">
                    {hwSimResult.edgeProfile.tokPerSecQ8} tok/s
                  </div>
                </div>
                <div className="p-2 rounded bg-[#111827] border border-slate-800">
                  <div className="text-slate-400">Q4_K_M ({hwSimResult.edgeProfile.footprintQ4KmGb} GB)</div>
                  <div className="text-emerald-400 font-semibold mt-0.5">
                    {hwSimResult.edgeProfile.tokPerSecQ4Km} tok/s
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Recommended Edge Deployment Format:</span>
            <strong className="text-emerald-400">
              {hwSimResult.edgeProfile.recommendedEdgeFormat}
            </strong>
          </div>
        </section>
      </div>

      {/* Centralized Logging & Telemetry Framework Stream */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              11. Centralized Logging & Error-Resilient Telemetry Stream
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Active sink: {activeLogger.toUpperCase()} · Built-in NaN gradient recovery, OOM micro-batch backoff, and checkpoint verification
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Interactive Logging Framework Selector */}
            <div className="flex items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
              {(['wandb', 'mlflow', 'tensorboard', 'otel'] as LoggingFramework[]).map((fw) => (
                <button
                  key={fw}
                  type="button"
                  onClick={() => onSelectLogger(fw)}
                  className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    activeLogger === fw
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {fw === 'wandb'
                    ? 'W&B'
                    : fw === 'mlflow'
                    ? 'MLflow'
                    : fw === 'tensorboard'
                    ? 'TensorBoard'
                    : 'OpenTelemetry'}
                </button>
              ))}
            </div>

            {/* Level Filter */}
            <div className="flex items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
              {(['ALL', 'INFO', 'METRIC', 'WARN', 'ERROR'] as const).map((lvl) => (
                <button
                  key={lvl}
                  type="button"
                  onClick={() => setLogLevelFilter(lvl)}
                  className={`px-2 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                    logLevelFilter === lvl
                      ? 'bg-slate-800 text-slate-100'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {lvl}
                </button>
              ))}
            </div>

            <input
              type="search"
              aria-label="Filter log messages"
              placeholder="Search logs..."
              value={logSearch}
              onChange={(e) => setLogSearch(e.target.value)}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 w-40"
            />

            <button
              type="button"
              onClick={handleExportLogsJsonl}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              Export JSONL
            </button>
          </div>
        </div>

        <div className="mt-4 divide-y divide-slate-800/70 font-mono text-xs max-h-72 overflow-y-auto">
          {filteredLogs.length === 0 ? (
            <div className="py-8 text-center text-slate-400">
              No structured telemetry events match the current filter.
            </div>
          ) : (
            filteredLogs.map((entry) => (
              <div
                key={entry.id}
                className="py-2.5 px-2 flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:bg-slate-800/30"
              >
                <div className="flex items-start sm:items-center gap-3 min-w-0">
                  <span className="text-slate-500 tabular-nums shrink-0">{entry.timestamp}</span>
                  <span
                    className={`font-semibold shrink-0 w-14 ${
                      entry.level === 'ERROR'
                        ? 'text-red-400'
                        : entry.level === 'WARN'
                        ? 'text-amber-400'
                        : entry.level === 'METRIC'
                        ? 'text-emerald-400'
                        : 'text-blue-400'
                    }`}
                  >
                    {entry.level}
                  </span>
                  <span className="text-slate-400 shrink-0">[{entry.component}]</span>
                  <span className="text-slate-200 break-words">{entry.message}</span>
                </div>
                {entry.metadata && (
                  <span className="text-slate-400 text-[11px] shrink-0 tabular-nums">
                    {entry.metadata}
                  </span>
                )}
              </div>
            ))
          )}
        </div>
      </section>

      {/* Next-Gen Sub-2B Quality Accelerators (SPIN, White-Box Logit KD, LoftQ/QAT, Replay/LISA, WebGPU) */}
      <NextGenQualityAndWebGpuPanel
        modelId={selectedModel.id}
        modelName={selectedModel.name}
        paramsBillions={selectedModel.paramsBillions}
        contextWindow={selectedModel.contextWindow}
        loraRank={activeHpoConfig.loraRank}
        basePerplexity={
          selectedModel.fineTunedBenchmarks?.perplexity ?? selectedModel.baseBenchmarks.perplexity
        }
        onApplyNextGenQualityBoost={(_label, mmluGain, gsm8kGain, ifEvalGain) => {
          onApplyStage3GrpoBoost?.(mmluGain, gsm8kGain, ifEvalGain);
        }}
      />

      {/* Real Python Engine Runner (Unsloth + Transformers + TRL + PEFT + SSE Metric Stream) */}
      <PythonEngineRunnerCard
        bundle={textEngineBundle}
        onStreamMetricPoint={(m) => {
          onStreamEngineMetricPoint?.(m);
        }}
      />
    </div>
  );
};
