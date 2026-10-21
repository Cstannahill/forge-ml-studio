import React, { useState } from 'react';
import {
  Activity,
  Check,
  Clock,
  Copy,
  Cpu,
  Download,
  FileText,
  HardDrive,
  PackageCheck,
  Play,
  RefreshCw,
  Sliders,
  Terminal,
  Zap,
} from 'lucide-react';
import {
  GpuAlertThresholds,
  HyperparameterConfig,
  LoggingFramework,
  SafetensorsDeploymentFormat,
  SafetensorsExportPlan,
  SafetensorsExportPrecision,
  SafetensorsRuntimeTarget,
  StandardizedModelCardResult,
  TrialOutcome,
  buildSafetensorsExportPlan,
  computeNextHyperparameters,
  computeTrainingProgressTelemetry,
  createSafetensorsBinaryArtifact,
  evaluateGpuThresholds,
  generateStandardizedModelCard,
} from '../lib/pipelineLogic';
import {
  DatasetVersionCommit,
  LogEntry,
  RegisteredModel,
  TrainingMetricPoint,
} from '../data/initialPipelineData';
import { InfoTooltip } from './InfoTooltip';

interface CommandCenterViewProps {
  models: RegisteredModel[];
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  trainingSeries: TrainingMetricPoint[];
  isTrainingActive: boolean;
  onRunTrainingStep: () => void;
  onResetTrainingSeries?: () => void;
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
}

export const CommandCenterView: React.FC<CommandCenterViewProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  trainingSeries,
  isTrainingActive,
  onRunTrainingStep,
  onResetTrainingSeries,
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

  const checkpointIdToExport =
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

      {/* Centralized Logging & Telemetry Framework Stream */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              05. Centralized Logging & Error-Resilient Telemetry Stream
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
    </div>
  );
};
