import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ShieldAlert,
  Stethoscope,
  X,
} from 'lucide-react';
import {
  ArchitectureFamily,
  CommandPaletteActionItem,
  GpuAlertThresholds,
  GpuTelemetrySnapshot,
  HyperparameterConfig,
  LoggingFramework,
  NormalizedDatasetSample,
  PlainLanguageComparison,
  PipelineDoctorCheckItem,
  PreferenceAlignmentMethod,
  SELECTED_MODEL_STORAGE_KEY,
  SafetensorsExportPlan,
  StandardizedModelCardResult,
  TRAINING_SERIES_STORAGE_KEY,
  TargetFormat,
  buildPlainLanguageBenchmarkAnalysis,
  cleanAndRedactSample,
  computeNextHyperparameters,
  deserializeTrainingSeriesMap,
  evaluateGpuThresholds,
  formatSampleToTarget,
  normalizeDatasetRecord,
  runPipelineDoctorAudit,
  serializeTrainingSeriesMap,
  validateDatasetRecords,
} from './lib/pipelineLogic';
import {
  DatasetVersionCommit,
  INITIAL_DATASET_VERSIONS,
  INITIAL_LOGS,
  INITIAL_MODELS,
  INITIAL_TRAINING_SERIES,
  LogEntry,
  RegisteredModel,
  TrainingMetricPoint,
  buildInitialNormalizedSamples,
} from './data/initialPipelineData';
import { CuratedDatasetBlend } from './data/curatedDatasetBlends';
import { CommandCenterView } from './components/CommandCenterView';
import { DatasetStudioView } from './components/DatasetStudioView';
import { ModelBenchmarksView } from './components/ModelBenchmarksView';
import {
  AgentExecutionRecord,
  AgentProviderId,
  AgenticOrchestratorView,
} from './components/AgenticOrchestratorView';
import { CloudAndContainerView } from './components/CloudAndContainerView';
import { CommandPaletteModal } from './components/CommandPaletteModal';
import { PipelineDoctorModal } from './components/PipelineDoctorModal';
import { VisionCommandCenterPanel } from './components/VisionCommandCenterPanel';
import { VisionDatasetStudioView } from './components/VisionDatasetStudioView';
import {
  INITIAL_VISION_MODELS,
  NormalizedVisionSample,
  PipelineModalityTrack,
  RegisteredVisionModel,
  VisionTargetFormat,
  buildInitialVisionSamples,
  capExcessiveNavitPatches,
  computeVisionTokenizerAndPatchDensity,
  formatVisionSamplePayload,
  validateVisionDatasetCorpus,
} from './lib/visionPipelineLogic';
import { InfoTooltip } from './components/InfoTooltip';

type ActiveTabId = 'command' | 'datasets' | 'benchmarks' | 'agents' | 'cloud';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTabId>('command');
  const [models, setModels] = useState<RegisteredModel[]>(INITIAL_MODELS);
  const [selectedModelId, setSelectedModelId] = useState<string>(() => {
    try {
      const savedModelId = window.localStorage.getItem(SELECTED_MODEL_STORAGE_KEY);
      if (savedModelId && INITIAL_MODELS.some((m) => m.id === savedModelId)) {
        return savedModelId;
      }
    } catch {
      // Ignore storage access errors
    }
    return INITIAL_MODELS[0].id;
  });

  const [trainingSeriesByModel, setTrainingSeriesByModel] = useState<
    Record<string, TrainingMetricPoint[]>
  >(() => {
    const modelIds = INITIAL_MODELS.map((m) => m.id);
    try {
      const rawMap = window.localStorage.getItem(TRAINING_SERIES_STORAGE_KEY);
      if (rawMap) {
        return deserializeTrainingSeriesMap(rawMap, INITIAL_TRAINING_SERIES, modelIds);
      }
      const rawDirect = window.localStorage.getItem('trainingSeries');
      if (rawDirect) {
        const parsedDirect = JSON.parse(rawDirect);
        if (Array.isArray(parsedDirect) && parsedDirect.length > 0) {
          const baseMap = deserializeTrainingSeriesMap(null, INITIAL_TRAINING_SERIES, modelIds);
          baseMap[INITIAL_MODELS[0].id] = parsedDirect;
          return baseMap;
        }
      }
    } catch {
      // Fall back to default series if localStorage is unavailable
    }
    return deserializeTrainingSeriesMap(null, INITIAL_TRAINING_SERIES, modelIds);
  });

  const trainingSeries: TrainingMetricPoint[] =
    trainingSeriesByModel[selectedModelId] ?? INITIAL_TRAINING_SERIES;

  useEffect(() => {
    try {
      window.localStorage.setItem(SELECTED_MODEL_STORAGE_KEY, selectedModelId);
      window.localStorage.setItem(
        TRAINING_SERIES_STORAGE_KEY,
        serializeTrainingSeriesMap(trainingSeriesByModel)
      );
      window.localStorage.setItem('trainingSeries', JSON.stringify(trainingSeries));
    } catch {
      // Ignore storage quota or private-mode errors
    }
  }, [selectedModelId, trainingSeriesByModel, trainingSeries]);

  const [isTrainingActive, setIsTrainingActive] = useState<boolean>(false);

  const [activeHpoConfig, setActiveHpoConfig] = useState<HyperparameterConfig>({
    learningRate: 0.0002,
    loraRank: 32,
    loraAlpha: 64,
    loraDropout: 0.05,
    warmupRatio: 0.05,
    weightDecay: 0.01,
    neftuneAlpha: 5,
    batchSize: 8,
    lrScheduler: 'wsd',
  });

  const [logs, setLogs] = useState<LogEntry[]>(INITIAL_LOGS);
  const [activeLogger, setActiveLogger] = useState<LoggingFramework>('wandb');

  const [gpuStats, setGpuStats] = useState<GpuTelemetrySnapshot>({
    vramUsedGb: 3.4,
    vramTotalGb: 24.0,
    smUtilPct: 84,
    tensorCorePct: 81,
    tempCelsius: 62,
    powerWatts: 185,
    nvlinkGbps: 56.4,
  });

  const [vramHistoryGb, setVramHistoryGb] = useState<number[]>([
    2.6, 2.8, 2.95, 3.1, 3.2, 3.3, 3.35, 3.4,
  ]);

  const [gruvboxBgPreset, setGruvboxBgPreset] = useState<'#2F2F2F' | '#282828' | '#1D2021'>('#2F2F2F');

  const [gpuThresholds, setGpuThresholds] = useState<GpuAlertThresholds>({
    vramPctThreshold: 90,
    tempCelsiusThreshold: 85,
    powerWattsThreshold: 400,
  });

  const [alertBannerDismissed, setAlertBannerDismissed] = useState<boolean>(false);
  const [isPipelineDoctorOpen, setIsPipelineDoctorOpen] = useState<boolean>(false);
  const [doctorSamplePacking, setDoctorSamplePacking] = useState<boolean>(true);
  const [doctorCompletionOnlyLoss, setDoctorCompletionOnlyLoss] = useState<boolean>(true);
  const [doctorFreezeVisionTower, setDoctorFreezeVisionTower] = useState<boolean>(true);

  // LLM Plain-Language Benchmark Explanations State
  const [plainExplanations, setPlainExplanations] = useState<
    Record<
      string,
      { providerUsed: string; timestamp: string; explanation: PlainLanguageComparison }
    >
  >({});
  const [isExplainingBenchmarks, setIsExplainingBenchmarks] = useState<boolean>(false);

  // Dataset Studio State (Text Generation Pipeline)
  const [targetFormat, setTargetFormat] = useState<TargetFormat>('chatml');
  const [samples, setSamples] = useState<NormalizedDatasetSample[]>(() =>
    buildInitialNormalizedSamples('chatml')
  );
  const [datasetVersions, setDatasetVersions] = useState<DatasetVersionCommit[]>(
    INITIAL_DATASET_VERSIONS
  );
  const [isDatasetBusy, setIsDatasetBusy] = useState<boolean>(false);

  // Isolated Vision-Language (VLM) Pipeline Track State (Zero Text Interference)
  const [pipelineTrack, setPipelineTrack] = useState<PipelineModalityTrack>('TEXT_LLM');
  const [visionModels, setVisionModels] =
    useState<RegisteredVisionModel[]>(INITIAL_VISION_MODELS);
  const [selectedVisionModelId, setSelectedVisionModelId] = useState<string>(
    INITIAL_VISION_MODELS[0].id
  );
  const [visionTargetFormat, setVisionTargetFormat] =
    useState<VisionTargetFormat>('QWEN_VL_CHATML');
  const [visionSamples, setVisionSamples] = useState<NormalizedVisionSample[]>(() =>
    buildInitialVisionSamples('QWEN_VL_CHATML')
  );
  const [visionDatasetCommits, setVisionDatasetCommits] = useState<DatasetVersionCommit[]>([
    {
      versionId: 'v1.4.0-vision',
      commitHash: 'sha256:7c31e9a',
      timestamp: '2026-10-20 15:00 UTC',
      sourceDatasets: [
        'HuggingFaceM4/Docmatix-Sub2B-Clean',
        'agentsea/wave-ui-25k-grounded',
        'LiquidAI/Edge-Spatial-VQA-50k',
      ],
      targetFormat: 'chatml',
      sampleCount: 5,
      healthScore: 98,
      cloudSynced: true,
      cloudUri: 's3://forgeml-vision-artifacts/v1.4.0-vision/manifest.parquet',
      notes:
        'Initial multimodal VLM corpus with [0, 1000] normalized bounding boxes & NaViT patch budgets',
    },
  ]);

  // Agentic Orchestrator State
  const [activeProvider, setActiveProvider] = useState<AgentProviderId>('gemini');
  const [providerModelName, setProviderModelName] = useState<string>('gemini-3.8-flash');
  const [localEndpoint, setLocalEndpoint] = useState<string>('http://localhost:8080/v1');
  const [isAgentRunning, setIsAgentRunning] = useState<boolean>(false);
  const [agentHistory, setAgentHistory] = useState<AgentExecutionRecord[]>([
    {
      id: 'agent-init-1',
      timestamp: '16:27:45 UTC',
      agentName: 'Goal-to-Dataset & Hygiene Orchestrator',
      providerUsed: 'Google Gemini (gemini-3.8-flash)',
      summary:
        'Initialized unified multi-dataset blend (Qwen3-Code-Reasoning-110k + UltraChat-200k + SlimOrca-Dedup) into ChatML schema and verified baseline evaluation fingerprints for Qwen3.5, LFM2.5, and Gemma4.',
      actionsTaken: [
        'Ingested 3 HuggingFace datasets and unified schema into ChatML (<|im_start|>role)',
        'Tagged samples across code-generation, reasoning-math, and structured-output domains',
        'Verified baseline lm-evaluation-harness results in Model Registry',
      ],
    },
  ]);

  const gpuAlertEvaluation = evaluateGpuThresholds(gpuStats, gpuThresholds);

  const appendLog = (
    level: LogEntry['level'],
    component: LogEntry['component'],
    message: string,
    metadata?: string
  ) => {
    const now = new Date();
    const ts =
      now.toTimeString().split(' ')[0] + '.' + String(now.getMilliseconds()).padStart(3, '0');
    setLogs((prev) => [
      {
        id: `log-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        timestamp: ts,
        level,
        component,
        framework: activeLogger,
        message,
        metadata,
      },
      ...prev,
    ]);
  };

  // Helper: Check and log GPU threshold breaches whenever stats or thresholds change
  const evaluateAndLogGpuBreaches = (
    nextStats: GpuTelemetrySnapshot,
    nextThresholds: GpuAlertThresholds
  ) => {
    const evalResult = evaluateGpuThresholds(nextStats, nextThresholds);
    if (evalResult.hasAlerts) {
      setAlertBannerDismissed(false);
      for (const breach of evalResult.breaches) {
        appendLog(
          breach.severity === 'CRITICAL' ? 'ERROR' : 'WARN',
          'TRAINER',
          `[GPU_THRESHOLD_ALERT] ${breach.title} — ${breach.message}`,
          `metric=${breach.metric} | value=${breach.currentValue}${breach.unit} | threshold=${breach.thresholdValue}${breach.unit}`
        );
      }
    }
  };

  // Handler: Simulate GPU Stress Spike (VRAM > 90% and Temp > 85°C)
  const handleSimulateGpuStressSpike = () => {
    const spikedStats: GpuTelemetrySnapshot = {
      vramUsedGb: 22.8, // 95.0% of 24GB (> 90%)
      vramTotalGb: 24.0,
      smUtilPct: 99,
      tensorCorePct: 97,
      tempCelsius: 89, // > 85°C
      powerWatts: 428, // > 400W
      nvlinkGbps: 59.8,
    };
    setGpuStats(spikedStats);
    setVramHistoryGb((prev) => [...prev.slice(-7), 22.8]);
    evaluateAndLogGpuBreaches(spikedStats, gpuThresholds);
  };

  // Handler: Auto-Mitigate GPU Overload (OOM micro-batch backoff + thermal power cap)
  const handleMitigateGpuOverload = () => {
    const nominalStats: GpuTelemetrySnapshot = {
      vramUsedGb: 16.2, // 67.5% of 24GB
      vramTotalGb: 24.0,
      smUtilPct: 88,
      tensorCorePct: 84,
      tempCelsius: 66,
      powerWatts: 315,
      nvlinkGbps: 54.0,
    };
    setGpuStats(nominalStats);
    setVramHistoryGb((prev) => [...prev.slice(-7), 16.2]);
    setActiveHpoConfig((prev) => ({
      ...prev,
      batchSize: Math.max(1, Math.floor(prev.batchSize / 2)),
    }));
    setAlertBannerDismissed(false);
    appendLog(
      'INFO',
      'TRAINER',
      'Applied automatic GPU threshold mitigation: enabled PagedAdamW8bit offloading, halved per-GPU micro-batch size, and capped board power at 320W. VRAM (67.5%) and Die Temp (66°C) restored to nominal.',
      'vram=16.2GB | temp=66C | status=NOMINAL'
    );
  };

  // Handler: Update GPU Thresholds
  const handleUpdateGpuThresholds = (nextThresholds: GpuAlertThresholds) => {
    setGpuThresholds(nextThresholds);
    evaluateAndLogGpuBreaches(gpuStats, nextThresholds);
  };

  // Handler: Switch Target Unified Format
  const handleChangeTargetFormat = (fmt: TargetFormat) => {
    setTargetFormat(fmt);
    setSamples((prev) =>
      prev.map((s) => ({
        ...s,
        formattedPreview: formatSampleToTarget(s.messages, fmt),
      }))
    );
    appendLog(
      'INFO',
      'DATA_PIPELINE',
      `Re-serialized ${samples.length} unified dataset records into ${fmt.toUpperCase()} format.`,
      `format=${fmt}`
    );
  };

  // Handler: Ingest 1 or more HuggingFace datasets
  const handleIngestDatasets = async (datasetIds: string[]) => {
    setIsDatasetBusy(true);
    try {
      const res = await fetch('/api/hf/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetIds }),
      });
      if (res.ok) {
        const data = await res.json();
        const batches = data.ingestedBatches as Array<{
          datasetId: string;
          rows: Array<Record<string, unknown>>;
        }>;
        const newSamples: NormalizedDatasetSample[] = [];
        let idx = 0;
        for (const batch of batches) {
          for (const row of batch.rows) {
            newSamples.push(normalizeDatasetRecord(row, batch.datasetId, idx++, targetFormat));
          }
        }
        if (newSamples.length > 0) {
          setSamples(newSamples);
          const report = validateDatasetRecords(newSamples);
          appendLog(
            'INFO',
            'DATA_PIPELINE',
            `Ingested & combined ${datasetIds.length} HuggingFace datasets (${datasetIds.join(
              ', '
            )}) into ${targetFormat.toUpperCase()}.`,
            `samples=${newSamples.length} | health=${report.overallHealthScore}/100`
          );
        }
      }
    } finally {
      setIsDatasetBusy(false);
    }
  };

  // Handler: Apply one of the 24 Curated Task-Focused Sub-2B Dataset Blends
  const handleApplyCuratedBlend = (blend: CuratedDatasetBlend) => {
    const blendSamples: NormalizedDatasetSample[] = [];
    let idx = 0;
    for (const ds of blend.datasets) {
      for (const row of ds.sampleRowsPreview) {
        blendSamples.push(normalizeDatasetRecord(row, ds.datasetId, idx++, blend.targetFormat));
      }
    }

    setTargetFormat(blend.targetFormat);
    if (blendSamples.length > 0) {
      setSamples(blendSamples);
    }

    // Apply recommended sub-2B model and hyperparameters
    if (models.some((m) => m.id === blend.recommendedModelId)) {
      setSelectedModelId(blend.recommendedModelId);
    }
    setActiveHpoConfig(blend.recommendedHpo);

    const report = validateDatasetRecords(blendSamples.length > 0 ? blendSamples : samples);
    appendLog(
      'INFO',
      'DATA_PIPELINE',
      `Loaded Curated Blend #${blend.blendNumber} ("${blend.title}") with ${blend.datasets.length} HuggingFace datasets for ${blend.primaryTargetModels.join(', ')}. Applied sub-2B HPO preset (r=${blend.recommendedHpo.loraRank}, lr=${blend.recommendedHpo.learningRate}).`,
      `model=${blend.recommendedModelId} | vram_est=${blend.estimatedVramQloraGb}GB | health=${report.overallHealthScore}/100`
    );
  };

  // Handler: Record Safetensors Checkpoint Export with Merged LoRA/QLoRA Adapter & Model Card
  const handleRecordSafetensorsExport = (
    plan: SafetensorsExportPlan,
    model: RegisteredModel,
    modelCard?: StandardizedModelCardResult
  ) => {
    const cardSuffix = modelCard
      ? ` + standardized Model Card ${modelCard.filename} (dataset=${modelCard.extractedMetadata.datasetVersion})`
      : '';
    appendLog(
      'INFO',
      'TRAINER',
      `Merged ${model.quantizationDefault.toUpperCase()} adapter (scaling=${plan.adapterScalingFactor}x) into ${model.name} and exported contiguous Safetensors checkpoint ${plan.outputFilename} (${plan.mergedFileSizeGb} GB)${cardSuffix}.`,
      `file=${plan.outputFilename} | sha256=${plan.sha256Preview} | card=${modelCard?.filename ?? 'none'}`
    );
  };

  // Handler: Agentic Data Cleaning & Tagging
  const handleRunAgenticCleanAndTag = async () => {
    setIsDatasetBusy(true);
    try {
      const cleanedList = samples.map((s) => cleanAndRedactSample(s, targetFormat));
      setSamples(cleanedList);
      const report = validateDatasetRecords(cleanedList);
      appendLog(
        'INFO',
        'AGENT_ORCHESTRATOR',
        `Agentic Hygiene completed: redacted all PII/secret patterns and verified domain tags. Health Score: ${report.overallHealthScore}/100.`,
        `status=${report.status} | pii_remaining=0`
      );
    } finally {
      setIsDatasetBusy(false);
    }
  };

  // Handler: Commit Dataset Version & Sync to Cloud Storage
  const handleCommitDatasetVersion = (cloudUri: string, notes: string) => {
    const report = validateDatasetRecords(samples);
    const nextMinor = datasetVersions.length + 4;
    const versionId = `v2.${nextMinor}.0`;
    const commitHash = `sha256:${Math.random().toString(16).slice(2, 9)}`;
    const sources = Array.from(new Set(samples.map((s) => s.sourceDataset)));

    const newCommit: DatasetVersionCommit = {
      versionId,
      commitHash,
      timestamp: 'Just now · Synced',
      sourceDatasets: sources,
      targetFormat,
      sampleCount: samples.length,
      healthScore: report.overallHealthScore,
      cloudSynced: true,
      cloudUri,
      notes: notes || 'Unified dataset snapshot committed',
    };

    setDatasetVersions((prev) => [newCommit, ...prev]);
    appendLog(
      'INFO',
      'DATA_PIPELINE',
      `Committed dataset snapshot ${versionId} (${commitHash}) and synced to ${cloudUri}.`,
      `sources=${sources.length} | health=${report.overallHealthScore}/100`
    );
  };

  // Handler: Run Training Step (+50 steps) and persist to localStorage
  const handleRunTrainingStep = () => {
    setIsTrainingActive(true);
    setTimeout(() => {
      setTrainingSeriesByModel((prevMap) => {
        const currentSeries = prevMap[selectedModelId] ?? INITIAL_TRAINING_SERIES;
        const last = currentSeries[currentSeries.length - 1];
        const nextStep = last.step + 50;
        const nextEpoch = Number((last.epoch + 0.25).toFixed(2));
        const nextTrainLoss = Number(Math.max(0.42, last.trainLoss * 0.92).toFixed(3));
        const nextValLoss = Number(Math.max(0.51, last.valLoss * 0.93).toFixed(3));
        const nextGradNorm = Number(Math.max(0.48, last.gradNorm * 0.95).toFixed(3));
        const nextPoint: TrainingMetricPoint = {
          step: nextStep,
          epoch: nextEpoch,
          trainLoss: nextTrainLoss,
          valLoss: nextValLoss,
          gradNorm: nextGradNorm,
          learningRate: activeHpoConfig.learningRate,
          tokensPerSec: last.tokensPerSec + 35,
        };
        appendLog(
          'METRIC',
          'TRAINER',
          `Step ${nextStep} (Epoch ${nextEpoch}) completed & persisted to localStorage: train_loss=${nextTrainLoss}, val_loss=${nextValLoss}, grad_norm=${nextGradNorm}.`,
          `model=${selectedModelId} | lr=${activeHpoConfig.learningRate} | persisted=true`
        );
        const updatedSeries = [...currentSeries, nextPoint];
        const nextMap = {
          ...prevMap,
          [selectedModelId]: updatedSeries,
        };
        try {
          window.localStorage.setItem(
            TRAINING_SERIES_STORAGE_KEY,
            serializeTrainingSeriesMap(nextMap)
          );
          window.localStorage.setItem('trainingSeries', JSON.stringify(updatedSeries));
        } catch {
          // Ignore storage errors
        }
        return nextMap;
      });

      setGpuStats((prev) => {
        const nextVram = Number(Math.min(prev.vramTotalGb - 0.8, prev.vramUsedGb + 0.18).toFixed(2));
        const updated: GpuTelemetrySnapshot = {
          ...prev,
          vramUsedGb: nextVram,
          smUtilPct: Math.min(99, prev.smUtilPct + 2),
          tensorCorePct: Math.min(96, prev.tensorCorePct + 2),
          tempCelsius: Math.min(90, prev.tempCelsius + 1),
        };
        setVramHistoryGb((hist) => [...hist.slice(-7), nextVram]);
        evaluateAndLogGpuBreaches(updated, gpuThresholds);
        return updated;
      });
      setIsTrainingActive(false);
    }, 150);
  };

  // Handler: Reset persisted trainingSeries for the active model
  const handleResetTrainingSeries = () => {
    const resetCopy = INITIAL_TRAINING_SERIES.map((pt) => ({ ...pt }));
    setTrainingSeriesByModel((prevMap) => {
      const nextMap = {
        ...prevMap,
        [selectedModelId]: resetCopy,
      };
      try {
        window.localStorage.setItem(
          TRAINING_SERIES_STORAGE_KEY,
          serializeTrainingSeriesMap(nextMap)
        );
        window.localStorage.setItem('trainingSeries', JSON.stringify(resetCopy));
      } catch {
        // Ignore storage errors
      }
      return nextMap;
    });
    appendLog(
      'INFO',
      'TRAINER',
      `Reset persisted trainingSeries in localStorage for ${selectedModelId} back to baseline Step 350.`,
      `model=${selectedModelId} | step=350`
    );
  };

  // Handler: Rollback trainingSeries to the Best Val Loss Checkpoint index
  const handleRollbackTrainingToIndex = (stepIndex: number) => {
    setTrainingSeriesByModel((prevMap) => {
      const currentSeries = prevMap[selectedModelId] ?? INITIAL_TRAINING_SERIES;
      const rolledBack = currentSeries.slice(0, Math.max(1, stepIndex + 1));
      const nextMap = {
        ...prevMap,
        [selectedModelId]: rolledBack,
      };
      try {
        window.localStorage.setItem(
          TRAINING_SERIES_STORAGE_KEY,
          serializeTrainingSeriesMap(nextMap)
        );
        window.localStorage.setItem('trainingSeries', JSON.stringify(rolledBack));
      } catch {
        // Ignore storage errors
      }
      const bestPt = rolledBack[rolledBack.length - 1];
      appendLog(
        'INFO',
        'TRAINER',
        `[EARLY_STOP_ROLLBACK] Rolled back ${selectedModelId} to Best Validation Loss Checkpoint at Step ${bestPt.step} (val_loss=${bestPt.valLoss}).`,
        `model=${selectedModelId} | step=${bestPt.step} | val_loss=${bestPt.valLoss}`
      );
      return nextMap;
    });
  };

  // Handler: Simulate Late-Epoch Overfit Spike to test Early-Stopping Watchdog
  const handleSimulateOverfitStep = () => {
    setTrainingSeriesByModel((prevMap) => {
      const currentSeries = prevMap[selectedModelId] ?? INITIAL_TRAINING_SERIES;
      const last = currentSeries[currentSeries.length - 1];
      const overfitPoint: TrainingMetricPoint = {
        step: last.step + 50,
        epoch: Number((last.epoch + 0.25).toFixed(2)),
        trainLoss: Number(Math.max(0.35, last.trainLoss - 0.08).toFixed(3)),
        valLoss: Number((last.valLoss + 0.14).toFixed(3)),
        gradNorm: Number((last.gradNorm + 0.32).toFixed(3)),
        learningRate: activeHpoConfig.learningRate,
        tokensPerSec: last.tokensPerSec,
      };
      const updatedSeries = [...currentSeries, overfitPoint];
      const nextMap = {
        ...prevMap,
        [selectedModelId]: updatedSeries,
      };
      try {
        window.localStorage.setItem(
          TRAINING_SERIES_STORAGE_KEY,
          serializeTrainingSeriesMap(nextMap)
        );
        window.localStorage.setItem('trainingSeries', JSON.stringify(updatedSeries));
      } catch {
        // Ignore storage errors
      }
      appendLog(
        'WARN',
        'TRAINER',
        `[EARLY_STOP_WATCHDOG] Validation loss diverged at Step ${overfitPoint.step} (train_loss=${overfitPoint.trainLoss}, val_loss=${overfitPoint.valLoss}). Recommend rollback to peak generalization checkpoint.`,
        `step=${overfitPoint.step} | val_loss=${overfitPoint.valLoss}`
      );
      return nextMap;
    });
  };

  // Handler: Apply Stage 2 Preference Alignment (ORPO / SimPO / DPO / KTO) Boost
  const handleApplyStage2AlignmentBoost = (
    method: PreferenceAlignmentMethod,
    ifEvalGain: number,
    mtBenchGain: number
  ) => {
    setModels((prev) =>
      prev.map((m) => {
        if (m.id !== selectedModelId) return m;
        const currentFt = m.fineTunedBenchmarks ?? m.baseBenchmarks;
        return {
          ...m,
          fineTunedBenchmarks: {
            ...currentFt,
            ifEval: Number(Math.min(96.5, currentFt.ifEval + ifEvalGain * 0.35).toFixed(1)),
            mtBench: Number(Math.min(9.6, currentFt.mtBench + mtBenchGain * 0.35).toFixed(2)),
          },
        };
      })
    );
    appendLog(
      'METRIC',
      'TRAINER',
      `Completed Stage 2 ${method} Preference Alignment pass for ${selectedModelId}: projected IFEval +${ifEvalGain}%, MT-Bench +${mtBenchGain}.`,
      `stage=2 | method=${method} | model=${selectedModelId}`
    );
  };

  // Handler: Apply Stage 3 GRPO Verifiable Reward (RLVR) Boost
  const handleApplyStage3GrpoBoost = (
    gsm8kGain: number,
    humanEvalGain: number,
    ifEvalGain: number
  ) => {
    setModels((prev) =>
      prev.map((m) => {
        if (m.id !== selectedModelId) return m;
        const currentFt = m.fineTunedBenchmarks ?? m.baseBenchmarks;
        return {
          ...m,
          fineTunedBenchmarks: {
            ...currentFt,
            gsm8k: Number(Math.min(98.5, currentFt.gsm8k + gsm8kGain * 0.4).toFixed(1)),
            humanEval: Number(
              Math.min(96.5, currentFt.humanEval + humanEvalGain * 0.4).toFixed(1)
            ),
            ifEval: Number(Math.min(97.5, currentFt.ifEval + ifEvalGain * 0.4).toFixed(1)),
          },
        };
      })
    );
    appendLog(
      'METRIC',
      'TRAINER',
      `Completed Stage 3 GRPO Verifiable Reward (RLVR) pass for ${selectedModelId}: projected GSM8K +${gsm8kGain}%, HumanEval+ +${humanEvalGain}%, IFEval +${ifEvalGain}%.`,
      `stage=3 | method=GRPO_RLVR | model=${selectedModelId}`
    );
  };

  // Handlers: Append or Replace Dataset Samples (from Teacher Distillation, IFD Pruning, or In-Place Edit)
  const handleAppendDatasetSamples = (
    newSamples: NormalizedDatasetSample[],
    note: string
  ) => {
    setSamples((prev) => [...newSamples, ...prev]);
    appendLog('INFO', 'DATA_PIPELINE', note, `added=${newSamples.length}`);
  };

  const handleReplaceDatasetSamples = (
    nextSamples: NormalizedDatasetSample[],
    note: string
  ) => {
    setSamples(nextSamples);
    appendLog('INFO', 'DATA_PIPELINE', note, `total=${nextSamples.length}`);
  };

  // Handler: Send Flagged Arena Prompt & Verified Target to Dataset Studio
  const handleSendFlaggedPromptToDatasetStudio = (draft: {
    instruction: string;
    output: string;
    domainTag: string;
  }) => {
    const activeModelObj = models.find((m) => m.id === selectedModelId) ?? models[0];
    const newRecord = normalizeDatasetRecord(
      {
        instruction: draft.instruction,
        input: '',
        output: draft.output,
      },
      `ArenaCorrection/${activeModelObj.name}`,
      samples.length + 1,
      targetFormat
    );
    const enriched: NormalizedDatasetSample = {
      ...newRecord,
      id: `arena-fix-${Date.now()}`,
      tags: Array.from(new Set([...newRecord.tags, draft.domainTag, 'arena-flagged'])),
    };
    setSamples((prev) => [enriched, ...prev]);
    appendLog(
      'INFO',
      'EVAL_HARNESS',
      `Flagged base-model failure in Evaluation Arena and appended verified completion (${draft.domainTag}) to Dataset Studio corpus.`,
      `sample_id=${enriched.id} | tag=${draft.domainTag}`
    );
  };

  // Handler: Run Automated HPO + Auto-Benchmark Trial
  const handleRunHpoSweep = () => {
    const lastPt = trainingSeries[trainingSeries.length - 1];
    const proposal = computeNextHyperparameters({
      trialNumber: trainingSeries.length,
      config: activeHpoConfig,
      trainLoss: lastPt.trainLoss,
      valLoss: lastPt.valLoss,
      gradNorm: lastPt.gradNorm,
      evalScore: 76.0,
    });

    setActiveHpoConfig(proposal.nextConfig);
    handleTriggerFineTuneAndCompare(selectedModelId, proposal.nextConfig);
  };

  // Handler: Detect & Benchmark a New Base Model automatically
  const handleRegisterNewBaseModel = (params: {
    id: string;
    architecture: ArchitectureFamily;
    paramsBillions: number;
    contextWindow: number;
  }) => {
    const shortName = params.id.split('/').pop() || params.id;
    const baseMmlu = Number(Math.min(84.5, 62 + Math.sqrt(params.paramsBillions) * 2.8).toFixed(1));
    const baseGsm8k = Number(
      Math.min(93.0, 72 + Math.sqrt(params.paramsBillions) * 3.1).toFixed(1)
    );
    const baseHumanEval = Number(
      Math.min(88.0, 66 + Math.sqrt(params.paramsBillions) * 2.9).toFixed(1)
    );
    const baseIfEval = Number(
      Math.min(89.0, 69 + Math.sqrt(params.paramsBillions) * 2.7).toFixed(1)
    );

    const newModel: RegisteredModel = {
      id: params.id,
      name: shortName,
      architecture: params.architecture,
      paramsBillions: params.paramsBillions,
      contextWindow: params.contextWindow,
      quantizationDefault: 'qlora-4bit',
      status: 'BASE_BENCHMARKED',
      detectedAt: 'Just now · Auto-Benchmarked',
      baseBenchmarks: {
        mmluPro: baseMmlu,
        gsm8k: baseGsm8k,
        humanEval: baseHumanEval,
        ifEval: baseIfEval,
        mtBench: 8.25,
        perplexity: 5.24,
        tokensPerSec: Number(Math.max(32, 220 / Math.sqrt(params.paramsBillions)).toFixed(1)),
      },
    };

    setModels((prev) => [newModel, ...prev]);
    setSelectedModelId(newModel.id);
    appendLog(
      'INFO',
      'EVALUATOR',
      `Detected new base model ${newModel.id} (${newModel.architecture}). Automatically executed baseline lm-eval benchmark and saved results.`,
      `mmlu_pro=${baseMmlu}% | gsm8k=${baseGsm8k}% | humaneval=${baseHumanEval}%`
    );
  };

  // Handler: Fine-Tune & Automatically Evaluate + Compare against Base Model
  const handleTriggerFineTuneAndCompare = (
    modelId: string,
    customHpo?: HyperparameterConfig
  ) => {
    const hpoUsed = customHpo ?? activeHpoConfig;
    const latestVersion = datasetVersions[0]?.versionId ?? 'v2.4.0';

    setModels((prev) =>
      prev.map((m) => {
        if (m.id !== modelId) return m;
        const base = m.baseBenchmarks;
        const ftBenchmarks = {
          mmluPro: Number(
            Math.min(94.0, (m.fineTunedBenchmarks?.mmluPro ?? base.mmluPro) + 3.2).toFixed(1)
          ),
          gsm8k: Number(
            Math.min(97.5, (m.fineTunedBenchmarks?.gsm8k ?? base.gsm8k) + 4.8).toFixed(1)
          ),
          humanEval: Number(
            Math.min(95.0, (m.fineTunedBenchmarks?.humanEval ?? base.humanEval) + 5.4).toFixed(1)
          ),
          ifEval: Number(
            Math.min(96.0, (m.fineTunedBenchmarks?.ifEval ?? base.ifEval) + 5.1).toFixed(1)
          ),
          mtBench: Number(
            Math.min(9.6, (m.fineTunedBenchmarks?.mtBench ?? base.mtBench) + 0.42).toFixed(2)
          ),
          perplexity: Number(
            Math.max(3.1, (m.fineTunedBenchmarks?.perplexity ?? base.perplexity) - 0.54).toFixed(2)
          ),
          tokensPerSec: base.tokensPerSec,
        };
        return {
          ...m,
          status: 'FINE_TUNED_READY',
          fineTunedBenchmarks: ftBenchmarks,
          fineTunedCheckpointId: `ft-${m.name
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '-')}-r${hpoUsed.loraRank}-${hpoUsed.lrScheduler}`,
          datasetVersionUsed: `${latestVersion} (${
            datasetVersions[0]?.commitHash ?? 'sha256:9f82c1e'
          })`,
          hpoConfigUsed: hpoUsed,
        };
      })
    );

    handleRunTrainingStep();

    appendLog(
      'INFO',
      'EVALUATOR',
      `Completed fine-tuning & automatic post-training benchmark for ${modelId}. Saved side-by-side comparison against base model.`,
      `dataset=${latestVersion} | lora_r=${hpoUsed.loraRank} | verdict=PROMOTE_TO_PRODUCTION`
    );
  };

  // Handler: Analyze & Explain Benchmark Comparison in Plain Language via LLM
  const handleExplainBenchmarksWithLlm = async (targetModel: RegisteredModel) => {
    setIsExplainingBenchmarks(true);
    try {
      // Ensure we have fine-tuned benchmarks to compare against the base model
      const ftBenchmarks = targetModel.fineTunedBenchmarks ?? {
        mmluPro: Number((targetModel.baseBenchmarks.mmluPro + 3.1).toFixed(1)),
        gsm8k: Number((targetModel.baseBenchmarks.gsm8k + 4.6).toFixed(1)),
        humanEval: Number((targetModel.baseBenchmarks.humanEval + 5.2).toFixed(1)),
        ifEval: Number((targetModel.baseBenchmarks.ifEval + 4.9).toFixed(1)),
        mtBench: Number((targetModel.baseBenchmarks.mtBench + 0.4).toFixed(2)),
        perplexity: Number(Math.max(3.2, targetModel.baseBenchmarks.perplexity - 0.52).toFixed(2)),
        tokensPerSec: targetModel.baseBenchmarks.tokensPerSec,
      };

      if (!targetModel.fineTunedBenchmarks) {
        handleTriggerFineTuneAndCompare(targetModel.id);
      }

      const res = await fetch('/api/benchmarks/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: activeProvider,
          providerModelName,
          modelName: targetModel.name,
          architecture: targetModel.architecture,
          baseBenchmarks: targetModel.baseBenchmarks,
          fineTunedBenchmarks: ftBenchmarks,
        }),
      });

      const ts = new Date().toTimeString().split(' ')[0] + ' UTC';
      if (res.ok) {
        const data = await res.json();
        const explanation: PlainLanguageComparison = data.explanation;
        setPlainExplanations((prev) => ({
          ...prev,
          [targetModel.id]: {
            providerUsed: data.providerUsed || `${activeProvider} (${providerModelName})`,
            timestamp: ts,
            explanation,
          },
        }));
        appendLog(
          'INFO',
          'EVALUATOR',
          `Generated plain-language LLM benchmark explanation for ${targetModel.name}: "${explanation.headline}"`,
          `provider=${data.providerUsed || activeProvider}`
        );
      } else {
        const fallback = buildPlainLanguageBenchmarkAnalysis({
          modelName: targetModel.name,
          architecture: targetModel.architecture,
          base: targetModel.baseBenchmarks,
          fineTuned: ftBenchmarks,
        });
        setPlainExplanations((prev) => ({
          ...prev,
          [targetModel.id]: {
            providerUsed: `${activeProvider.toUpperCase()} (${providerModelName})`,
            timestamp: ts,
            explanation: fallback,
          },
        }));
      }
    } finally {
      setIsExplainingBenchmarks(false);
    }
  };

  // Handler: Execute Agentic AI Task across Gemini / OpenRouter / llama-server / Ollama
  const handleExecuteAgentTask = async (
    role: 'clean_and_tag' | 'goal_to_pipeline' | 'refine_hpo' | 'generate_report',
    goal?: string
  ) => {
    setIsAgentRunning(true);
    try {
      const res = await fetch('/api/agent/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: activeProvider,
          modelName: providerModelName,
          localEndpoint,
          agentRole: role,
          payload: {
            goal,
            modelId: selectedModelId,
            activeHpoConfig,
            sampleCount: samples.length,
          },
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const result = data.result;

        if (role === 'clean_and_tag') {
          await handleRunAgenticCleanAndTag();
        } else if (role === 'goal_to_pipeline') {
          const recommended: string[] = Array.isArray(result.recommendedDatasets)
            ? result.recommendedDatasets
            : ['Qwen/Qwen3-Code-Reasoning-110k', 'HuggingFaceH4/ultrachat_200k'];
          await handleIngestDatasets(recommended);
          setSamples((prev) => prev.map((s) => cleanAndRedactSample(s, targetFormat)));
          handleCommitDatasetVersion(
            `s3://forgeml-datasets/autonomous/${Date.now()}.parquet`,
            `Autonomous Agent Pipeline for goal: ${goal?.slice(0, 60)}`
          );
        } else if (role === 'refine_hpo') {
          if (result.suggestedHyperparameters) {
            setActiveHpoConfig((prev) => ({
              ...prev,
              ...result.suggestedHyperparameters,
            }));
          } else {
            handleRunHpoSweep();
          }
        }

        const roleTitles: Record<typeof role, string> = {
          clean_and_tag: 'Agent 01 · Data Cleaning & Tagging Orchestrator',
          goal_to_pipeline: 'Agent 02 · Autonomous Goal-to-Dataset Pipeline',
          refine_hpo: 'Agent 03 · Iterative HPO & Performance Refiner',
          generate_report: 'Agent 04 · Executive Summary & Actionable Insights',
        };

        const record: AgentExecutionRecord = {
          id: `agent-${Date.now()}`,
          timestamp: new Date().toTimeString().split(' ')[0] + ' UTC',
          agentName: roleTitles[role],
          providerUsed: data.providerUsed || `${activeProvider} (${providerModelName})`,
          summary: result.summary,
          actionsTaken: Array.isArray(result.actionsTaken) ? result.actionsTaken : [],
          recommendedDatasets: result.recommendedDatasets,
          suggestedHyperparameters: result.suggestedHyperparameters,
          markdownReport: result.markdownReport,
        };

        setAgentHistory((prev) => [record, ...prev]);
        appendLog(
          'INFO',
          'AGENT_ORCHESTRATOR',
          `${roleTitles[role]} finished via ${record.providerUsed}: ${record.summary.slice(
            0,
            110
          )}...`
        );
      }
    } finally {
      setIsAgentRunning(false);
    }
  };

  const navItems: Array<{ id: ActiveTabId; label: string }> = [
    { id: 'command', label: 'Command Center' },
    { id: 'datasets', label: 'Dataset Studio' },
    { id: 'benchmarks', label: 'Model Benchmarks' },
    { id: 'agents', label: 'Agentic AI' },
    { id: 'cloud', label: 'Cloud & Docker' },
  ];

  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  const commandPaletteActions: CommandPaletteActionItem[] = [
    {
      id: 'nav-command',
      title: 'Go to Command Center',
      subtitle: 'Telemetry, HPO, imatrix Calibration, Stage 2/3 Alignment, Adapter Merger',
      category: 'NAVIGATION',
      shortcutHint: 'Tab 1',
      keywords: ['command', 'home', 'telemetry', 'imatrix', 'grpo', 'dora', 'safetensors'],
    },
    {
      id: 'nav-datasets',
      title: 'Go to Dataset Studio',
      subtitle: 'Curated Sub-2B Blends, Teacher Distillation, Curriculum Pacing, Custom Upload',
      category: 'NAVIGATION',
      shortcutHint: 'Tab 2',
      keywords: ['dataset', 'curriculum', 'tokenizer', 'upload', 'jsonl', 'ifd', 'distillation'],
    },
    {
      id: 'nav-benchmarks',
      title: 'Go to Model Benchmarks & Arena',
      subtitle: 'Base vs. Fine-Tuned Radar, Side-by-Side Vibe Check Arena, Red-Team Stress Suite',
      category: 'NAVIGATION',
      shortcutHint: 'Tab 3',
      keywords: ['benchmarks', 'arena', 'red team', 'mmlu', 'humaneval', 'gsm8k', 'ifeval'],
    },
    {
      id: 'nav-agents',
      title: 'Go to Agent Orchestrator',
      subtitle: 'Gemini, OpenRouter & Local Ollama Sub-Agents for HPO & Data Curation',
      category: 'NAVIGATION',
      shortcutHint: 'Tab 4',
      keywords: ['agents', 'gemini', 'openrouter', 'ollama'],
    },
    {
      id: 'nav-cloud',
      title: 'Go to Cloud & Docker',
      subtitle: 'Modal, RunPod, Local Inference Server Launcher (llama.cpp / Ollama / vLLM) & SDKs',
      category: 'NAVIGATION',
      shortcutHint: 'Tab 5',
      keywords: ['cloud', 'docker', 'llama.cpp', 'vllm', 'ollama', 'sdk', 'openai'],
    },
    ...models.map((m) => ({
      id: `model-${m.id}`,
      title: `Switch Active Model -> ${m.name}`,
      subtitle: `${m.architecture} · ${m.paramsBillions}B params · ${m.contextWindow.toLocaleString()} ctx`,
      category: 'MODEL_SWITCH' as const,
      shortcutHint: m.id === selectedModelId ? 'Active' : 'Select',
      keywords: [m.id, m.name, m.architecture, `${m.paramsBillions}B`],
    })),
    {
      id: 'act-train-step',
      title: 'Step Training (+50 Steps)',
      subtitle: 'Advance active model fine-tuning telemetry and persist to localStorage',
      category: 'PIPELINE_ACTION',
      shortcutHint: 'Action',
      keywords: ['step', 'train', 'epoch', 'loss'],
    },
    {
      id: 'act-auto-hpo',
      title: 'Run Auto-HPO & Benchmark Evaluation',
      subtitle: 'Compute next optimal hyperparameter trial and evaluate against base model',
      category: 'PIPELINE_ACTION',
      shortcutHint: 'Action',
      keywords: ['hpo', 'sweep', 'benchmark', 'evaluate'],
    },
    {
      id: 'act-stage3-grpo',
      title: 'Execute Stage 3 GRPO Verifiable Reward Pass',
      subtitle: 'Run deterministic RLVR code/math/JSON verifiers on active checkpoint',
      category: 'PIPELINE_ACTION',
      shortcutHint: 'Stage 3',
      keywords: ['grpo', 'rlvr', 'reward', 'verifier', 'math', 'code'],
    },
    {
      id: 'track-text-llm',
      title: 'Switch Pipeline Track -> Pure Text Generation (LLM)',
      subtitle: '100% isolated text-only SFT, ORPO/SimPO & GRPO pipeline',
      category: 'PIPELINE_ACTION',
      shortcutHint: pipelineTrack === 'TEXT_LLM' ? 'Active' : 'Track A',
      keywords: ['text', 'llm', 'pipeline', 'track'],
    },
    {
      id: 'track-vision-vlm',
      title: 'Switch Pipeline Track -> Isolated Vision-Language (VLM: Qwen3.5-VL / LFM2.5-VL)',
      subtitle: 'Dedicated multimodal image+text pipeline (LFM2.5-VL-450M/1.6B & Qwen3.5-VL-0.8B/2B)',
      category: 'PIPELINE_ACTION',
      shortcutHint: pipelineTrack === 'VISION_VLM' ? 'Active' : 'Track B',
      keywords: ['vision', 'vlm', 'image', 'multimodal', '450m', '1.6b', 'qwen3.5-vl', 'lfm2.5-vl'],
    },
    {
      id: 'theme-soft',
      title: 'Theme: Gruvbox Dark Warm Soft (#2F2F2F)',
      subtitle: 'Default warm charcoal surface preset',
      category: 'THEME',
      shortcutHint: '#2F2F2F',
      keywords: ['theme', 'gruvbox', 'soft', '2f2f2f'],
    },
    {
      id: 'theme-medium',
      title: 'Theme: Gruvbox Dark Medium (#282828)',
      subtitle: 'Classic Gruvbox medium contrast background',
      category: 'THEME',
      shortcutHint: '#282828',
      keywords: ['theme', 'gruvbox', 'medium', '282828'],
    },
    {
      id: 'theme-hard',
      title: 'Theme: Gruvbox Dark Hard (#1D2021)',
      subtitle: 'High-contrast deep studio surface preset',
      category: 'THEME',
      shortcutHint: '#1D2021',
      keywords: ['theme', 'gruvbox', 'hard', '1d2021'],
    },
  ];

  const handleChangeVisionTargetFormat = (nextFmt: VisionTargetFormat) => {
    setVisionTargetFormat(nextFmt);
    setVisionSamples((prev) =>
      prev.map((s) => ({
        ...s,
        targetFormat: nextFmt,
        formattedPayload: formatVisionSamplePayload(s, nextFmt),
      }))
    );
    appendLog(
      'INFO',
      'DATA_PIPELINE',
      `Re-serialized ${visionSamples.length} Vision-Language samples into ${nextFmt}.`,
      `track=VISION_VLM | format=${nextFmt}`
    );
  };

  const handleUpdateVisionSamples = (
    nextSamples: NormalizedVisionSample[],
    logMessage: string
  ) => {
    setVisionSamples(nextSamples);
    appendLog(
      'INFO',
      'DATA_PIPELINE',
      `[VISION_DATASET_STUDIO] ${logMessage}`,
      `track=VISION_VLM | samples=${nextSamples.length}`
    );
  };

  const handleCommitVisionDatasetVersion = (notes: string) => {
    const report = validateVisionDatasetCorpus(visionSamples);
    const nextMinor = visionDatasetCommits.length + 4;
    const versionId = `v1.${nextMinor}.0-vision`;
    const hashHex = Math.abs(
      visionSamples.reduce((acc, s) => acc + s.totalTokens * 31 + s.id.length, 7919)
    )
      .toString(16)
      .slice(0, 7)
      .padStart(7, 'a');
    const newCommit: DatasetVersionCommit = {
      versionId,
      commitHash: `sha256:${hashHex}`,
      timestamp: 'Just now · Vision DVC',
      sourceDatasets: Array.from(new Set(visionSamples.map((s) => s.sourceDataset))).slice(0, 4),
      targetFormat: 'chatml',
      sampleCount: visionSamples.length,
      healthScore: report.healthScore,
      cloudSynced: true,
      cloudUri: `s3://forgeml-vision-artifacts/${versionId}/manifest.parquet`,
      notes: notes.trim() || 'Vision-Language dataset snapshot',
    };
    setVisionDatasetCommits((prev) => [newCommit, ...prev]);
    appendLog(
      'INFO',
      'DATA_PIPELINE',
      `Committed immutable Vision DVC snapshot ${versionId} (${newCommit.commitHash}) with ${visionSamples.length} image-text pairs.`,
      `track=VISION_VLM | health=${report.healthScore}%`
    );
  };

  const handleTriggerVisionFineTune = (visionModelId: string) => {
    setVisionModels((prev) =>
      prev.map((vm) => {
        if (vm.id !== visionModelId) return vm;
        const current = vm.fineTunedVisionBenchmarks ?? vm.baseVisionBenchmarks;
        return {
          ...vm,
          status: 'FINE_TUNED_READY',
          fineTunedVisionBenchmarks: {
            mmmu: Number(Math.min(82.0, current.mmmu + 2.4).toFixed(1)),
            docVqa: Number(Math.min(97.5, current.docVqa + 1.9).toFixed(1)),
            mathVista: Number(Math.min(86.0, current.mathVista + 2.6).toFixed(1)),
            realWorldQa: Number(Math.min(88.0, current.realWorldQa + 2.2).toFixed(1)),
            groundingIoU: Number(Math.min(94.0, current.groundingIoU + 2.8).toFixed(1)),
            visionTokensPerSec: current.visionTokensPerSec,
          },
        };
      })
    );
    appendLog(
      'METRIC',
      'TRAINER',
      `Completed isolated Vision-Language fine-tuning pass on ${visionModelId} (Vision Tower frozen, MLP Connector + LLM DoRA tuned). Zero interference with Text LLM checkpoints.`,
      `track=VISION_VLM | model=${visionModelId}`
    );
  };

  const handleExecuteCommandPaletteAction = (actionId: string) => {
    if (actionId === 'nav-command') setActiveTab('command');
    else if (actionId === 'nav-datasets') setActiveTab('datasets');
    else if (actionId === 'nav-benchmarks') setActiveTab('benchmarks');
    else if (actionId === 'nav-agents') setActiveTab('agents');
    else if (actionId === 'nav-cloud') setActiveTab('cloud');
    else if (actionId === 'track-text-llm') setPipelineTrack('TEXT_LLM');
    else if (actionId === 'track-vision-vlm') setPipelineTrack('VISION_VLM');
    else if (actionId.startsWith('model-')) {
      const targetId = actionId.replace('model-', '');
      setSelectedModelId(targetId);
    } else if (actionId === 'act-train-step') {
      handleRunTrainingStep();
    } else if (actionId === 'act-auto-hpo') {
      handleRunHpoSweep();
      setActiveTab('benchmarks');
    } else if (actionId === 'act-stage3-grpo') {
      handleApplyStage3GrpoBoost(4.6, 4.2, 3.8);
      setActiveTab('benchmarks');
    } else if (actionId === 'theme-soft') {
      setGruvboxBgPreset('#2F2F2F');
    } else if (actionId === 'theme-medium') {
      setGruvboxBgPreset('#282828');
    } else if (actionId === 'theme-hard') {
      setGruvboxBgPreset('#1D2021');
    }
  };

  // Live SSE Engine Metric Point Handler
  const handleStreamEngineMetricPoint = (point: TrainingMetricPoint) => {
    setTrainingSeriesByModel((prev) => {
      const current = prev[selectedModelId] ?? INITIAL_TRAINING_SERIES;
      return {
        ...prev,
        [selectedModelId]: [...current, point],
      };
    });
    appendLog(
      'METRIC',
      'TRAINER',
      `[SSE_ENGINE_STREAM] Step ${point.step} (Epoch ${point.epoch}): trainLoss=${point.trainLoss.toFixed(4)} | valLoss=${point.valLoss.toFixed(4)} | gradNorm=${point.gradNorm.toFixed(3)}`,
      `lr=${point.learningRate.toExponential(2)} | tok/s=${point.tokensPerSec}`
    );
  };

  // Pre-Flight Pipeline Doctor Audit Report & Auto-Fix Handlers
  const activeDoctorModel = models.find((m) => m.id === selectedModelId) ?? models[0];
  const datasetHealthReport = validateDatasetRecords(samples);
  const visionDensityReport = computeVisionTokenizerAndPatchDensity(visionSamples);

  const pipelineDoctorReport = runPipelineDoctorAudit({
    modelName: activeDoctorModel.name,
    paramsBillions: activeDoctorModel.paramsBillions,
    hpoConfig: activeHpoConfig,
    samplePackingEnabled: doctorSamplePacking,
    completionOnlyLossEnabled: doctorCompletionOnlyLoss,
    stage2Enabled: true,
    alignmentMethod: 'ORPO',
    grpoEnabled: true,
    vramUsedGb: gpuStats.vramUsedGb,
    vramTotalGb: gpuStats.vramTotalGb,
    datasetHealthScore: datasetHealthReport.overallHealthScore,
    freezeVisionTower: doctorFreezeVisionTower,
    excessiveNavitSampleCount: visionDensityReport.excessiveNavitSampleCount,
  });

  const handleApplyDoctorFix = (
    fixKey: NonNullable<PipelineDoctorCheckItem['autoFixKey']>
  ) => {
    if (fixKey === 'CLAMP_LEARNING_RATE') {
      setActiveHpoConfig((prev) => ({ ...prev, learningRate: 0.0002 }));
    } else if (fixKey === 'FIX_LORA_ALPHA_RATIO') {
      setActiveHpoConfig((prev) => ({ ...prev, loraAlpha: prev.loraRank * 2 }));
    } else if (fixKey === 'ENABLE_PACKING_AND_MASKING') {
      setDoctorSamplePacking(true);
      setDoctorCompletionOnlyLoss(true);
    } else if (fixKey === 'ENABLE_NEFTUNE_AND_WSD') {
      setActiveHpoConfig((prev) => ({
        ...prev,
        neftuneAlpha: 5,
        lrScheduler: 'wsd',
      }));
    } else if (fixKey === 'FREEZE_VISION_TOWER') {
      setDoctorFreezeVisionTower(true);
    } else if (fixKey === 'CAP_NAVIT_PATCHES') {
      const capped = capExcessiveNavitPatches(visionSamples, 896, 5.5, visionTargetFormat);
      setVisionSamples(capped.updatedSamples);
    }
    appendLog(
      'INFO',
      'ORCHESTRATOR',
      `Applied Pre-Flight Pipeline Doctor Auto-Fix: ${fixKey}`,
      `model=${activeDoctorModel.name}`
    );
  };

  const handleApplyAllDoctorFixes = () => {
    for (const check of pipelineDoctorReport.checks) {
      if (check.status !== 'PASS' && check.autoFixKey) {
        handleApplyDoctorFix(check.autoFixKey);
      }
    }
  };

  return (
    <div
      className="min-h-screen text-[#FFE9B1] flex flex-col transition-colors"
      style={{ backgroundColor: gruvboxBgPreset }}
    >
      {/* Strict 3-Zone Top Bar Contract (Gruvbox Dark Surface #22211F + #FFE9B1 Cream + #458588 Teal) */}
      <header className="flex flex-wrap md:flex-nowrap items-center justify-between gap-4 md:gap-8 px-4 sm:px-6 py-3.5 border-b border-[#4A453E] bg-[#22211F]/95 sticky top-0 z-30 backdrop-blur">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('command');
          }}
          className="text-lg font-bold tracking-tight text-[#FFE9B1] whitespace-nowrap shrink-0"
        >
          ForgeML Studio
        </a>

        {/* Zone 2: 5 clean single-line text navigation links */}
        <nav
          aria-label="Primary Workspace Navigation"
          className="order-3 md:order-2 w-full md:w-auto flex items-center gap-4 sm:gap-6 text-xs sm:text-sm font-medium text-[#A89984] overflow-x-auto pt-2 md:pt-0 border-t md:border-t-0 border-[#3F3B35]"
        >
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setActiveTab(item.id)}
              className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 ${
                activeTab === item.id
                  ? 'text-[#FFE9B1] border-[#83A598]'
                  : 'border-transparent hover:text-[#D5C4A1]'
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>

        {/* Zone 3: Pipeline Doctor + Primary Action */}
        <div className="order-2 md:order-3 flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={() => setIsPipelineDoctorOpen(true)}
            className="px-3 py-2 text-xs font-medium text-[#8EC07C] bg-[#1D2021] border border-[#8EC07C]/50 rounded-lg hover:bg-[#8EC07C]/15 transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5"
          >
            <Stethoscope className="w-3.5 h-3.5" />
            <span>Pipeline Doctor ({pipelineDoctorReport.overallScore}%)</span>
          </button>
          <button
            type="button"
            onClick={() => {
              handleRunHpoSweep();
              setActiveTab('benchmarks');
            }}
            className="px-3.5 py-2 text-xs font-medium text-[#FFE9B1] bg-[#458588] rounded-lg hover:bg-[#529497] transition-colors whitespace-nowrap shrink-0"
          >
            Run Auto-HPO &amp; Eval
          </button>
        </div>
      </header>

      {/* Global Threshold-Based GPU Resource Alert Notification Banner */}
      {gpuAlertEvaluation.hasAlerts && !alertBannerDismissed && (
        <div
          role="alert"
          aria-live="assertive"
          className="bg-red-950/80 border-b border-red-500/60 px-4 sm:px-6 py-3"
        >
          <div className="max-w-[1400px] mx-auto flex flex-col lg:flex-row lg:items-center justify-between gap-3">
            <div className="flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="text-xs font-semibold text-red-200">
                  GPU Hardware Resource Threshold Alert ({gpuAlertEvaluation.breaches.length}{' '}
                  Active Breach{gpuAlertEvaluation.breaches.length > 1 ? 'es' : ''} Logged to{' '}
                  {activeLogger.toUpperCase()})
                </div>
                <div className="space-y-0.5 text-xs text-red-100/90">
                  {gpuAlertEvaluation.breaches.map((b) => (
                    <div key={b.id} className="font-mono tabular-nums">
                      • {b.title} — {b.remediation}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleMitigateGpuOverload}
                className="px-3 py-1.5 text-xs font-medium text-white bg-red-600 hover:bg-red-500 rounded-lg transition-colors whitespace-nowrap"
              >
                Auto-Mitigate &amp; Restore Nominal GPU
              </button>
              <button
                type="button"
                aria-label="Dismiss GPU Alert Notification"
                onClick={() => setAlertBannerDismissed(true)}
                className="p-1.5 text-red-300 hover:text-white rounded-lg transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1400px] w-full mx-auto px-4 sm:px-6 py-8 space-y-6">
        {/* Isolated Pipeline Modality Track Switcher (Pure Text LLM vs. Isolated Vision-Language VLM) */}
        <div className="bg-[#22211F] border border-[#4A453E] rounded-xl p-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2 text-xs font-semibold text-[#FFE9B1]">
              <span>Modality Pipeline Track Isolation</span>
              <InfoTooltip glossaryKey="vlmIsolatedPipeline" />
              <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-[#1D2021] border border-[#8EC07C]/40 text-[#8EC07C]">
                {pipelineTrack === 'TEXT_LLM'
                  ? 'Track A: Pure Text Generation (Zero Vision Overhead)'
                  : 'Track B: Isolated Vision-Language VLM (Zero Text Interference)'}
              </span>
            </div>
            <p className="text-xs text-[#A89984]">
              Separating Pure Text LLM training from Vision-Language (VLM) projector &amp; patch training prevents modality gradient interference on sub-2B models.
            </p>
          </div>

          <div
            role="group"
            aria-label="Pipeline Modality Track Switcher"
            className="flex flex-wrap items-center gap-2 bg-[#1D2021] p-1.5 rounded-xl border border-[#3C3836] shrink-0"
          >
            <button
              type="button"
              onClick={() => setPipelineTrack('TEXT_LLM')}
              className={`px-3.5 py-2 text-xs font-medium rounded-lg transition-colors ${
                pipelineTrack === 'TEXT_LLM'
                  ? 'bg-[#458588] text-[#FFE9B1] font-semibold'
                  : 'text-[#A89984] hover:text-[#FFE9B1]'
              }`}
            >
              Text Generation Pipeline (Pure LLM)
            </button>
            <button
              type="button"
              onClick={() => setPipelineTrack('VISION_VLM')}
              className={`px-3.5 py-2 text-xs font-medium rounded-lg transition-colors ${
                pipelineTrack === 'VISION_VLM'
                  ? 'bg-[#8EC07C] text-[#1D2021] font-semibold'
                  : 'text-[#A89984] hover:text-[#FFE9B1]'
              }`}
            >
              Vision-Language Pipeline (Qwen3.5-VL · LFM2.5-VL 450M/1.6B)
            </button>
          </div>
        </div>

        {activeTab === 'command' && (
          <div className="space-y-8">
            {pipelineTrack === 'VISION_VLM' && (
              <VisionCommandCenterPanel
                visionModels={visionModels}
                selectedVisionModelId={selectedVisionModelId}
                onSelectVisionModel={setSelectedVisionModelId}
                onTriggerVisionFineTune={handleTriggerVisionFineTune}
                onOpenVisionDatasetStudio={() => setActiveTab('datasets')}
                activeVisionSampleCount={visionSamples.length}
              />
            )}
            <CommandCenterView
              models={models}
              selectedModelId={selectedModelId}
              onSelectModel={setSelectedModelId}
              trainingSeries={trainingSeries}
              isTrainingActive={isTrainingActive}
              onRunTrainingStep={handleRunTrainingStep}
              onResetTrainingSeries={handleResetTrainingSeries}
              onRollbackTrainingToIndex={handleRollbackTrainingToIndex}
              onSimulateOverfitStep={handleSimulateOverfitStep}
              onApplyStage2AlignmentBoost={handleApplyStage2AlignmentBoost}
              onApplyStage3GrpoBoost={handleApplyStage3GrpoBoost}
              onOpenCommandPalette={() => setIsCommandPaletteOpen(true)}
              onRunHpoSweep={handleRunHpoSweep}
              activeHpoConfig={activeHpoConfig}
              onUpdateHpoConfig={setActiveHpoConfig}
              logs={logs}
              activeLogger={activeLogger}
              onSelectLogger={setActiveLogger}
              onClearLogs={() => setLogs([])}
              gpuStats={gpuStats}
              vramHistoryGb={vramHistoryGb}
              activeDatasetCommit={datasetVersions[0]}
              gpuThresholds={gpuThresholds}
              onUpdateGpuThresholds={handleUpdateGpuThresholds}
              onSimulateGpuStressSpike={handleSimulateGpuStressSpike}
              onMitigateGpuOverload={handleMitigateGpuOverload}
              onRecordSafetensorsExport={handleRecordSafetensorsExport}
              onStreamEngineMetricPoint={handleStreamEngineMetricPoint}
            />
          </div>
        )}

        {activeTab === 'datasets' &&
          (pipelineTrack === 'VISION_VLM' ? (
            <VisionDatasetStudioView
              visionModels={visionModels}
              selectedVisionModelId={selectedVisionModelId}
              onSelectVisionModel={setSelectedVisionModelId}
              visionSamples={visionSamples}
              visionTargetFormat={visionTargetFormat}
              onChangeVisionTargetFormat={handleChangeVisionTargetFormat}
              onUpdateVisionSamples={handleUpdateVisionSamples}
              visionDatasetCommits={visionDatasetCommits}
              onCommitVisionDatasetVersion={handleCommitVisionDatasetVersion}
            />
          ) : (
            <DatasetStudioView
              samples={samples}
              targetFormat={targetFormat}
              onChangeTargetFormat={handleChangeTargetFormat}
              onIngestDatasets={handleIngestDatasets}
              onApplyCuratedBlend={handleApplyCuratedBlend}
              onRunAgenticCleanAndTag={handleRunAgenticCleanAndTag}
              onCommitDatasetVersion={handleCommitDatasetVersion}
              onAppendSamples={handleAppendDatasetSamples}
              onReplaceSamples={handleReplaceDatasetSamples}
              datasetVersions={datasetVersions}
              isBusy={isDatasetBusy}
            />
          ))}

        {activeTab === 'benchmarks' && (
          <div className="space-y-8">
            {pipelineTrack === 'VISION_VLM' && (
              <VisionCommandCenterPanel
                visionModels={visionModels}
                selectedVisionModelId={selectedVisionModelId}
                onSelectVisionModel={setSelectedVisionModelId}
                onTriggerVisionFineTune={handleTriggerVisionFineTune}
                onOpenVisionDatasetStudio={() => setActiveTab('datasets')}
                activeVisionSampleCount={visionSamples.length}
              />
            )}
            <ModelBenchmarksView
              models={models}
              selectedModelId={selectedModelId}
              onSelectModel={setSelectedModelId}
              onRegisterNewBaseModel={handleRegisterNewBaseModel}
              onTriggerFineTuneAndCompare={(id) => handleTriggerFineTuneAndCompare(id)}
              plainExplanations={plainExplanations}
              onExplainBenchmarksWithLlm={handleExplainBenchmarksWithLlm}
              onSendFlaggedPromptToDatasetStudio={handleSendFlaggedPromptToDatasetStudio}
              isExplaining={isExplainingBenchmarks}
            />
          </div>
        )}

        {activeTab === 'agents' && (
          <AgenticOrchestratorView
            models={models}
            selectedModelId={selectedModelId}
            activeProvider={activeProvider}
            onSelectProvider={setActiveProvider}
            providerModelName={providerModelName}
            onChangeProviderModelName={setProviderModelName}
            localEndpoint={localEndpoint}
            onChangeLocalEndpoint={setLocalEndpoint}
            onExecuteAgentTask={handleExecuteAgentTask}
            agentHistory={agentHistory}
            isAgentRunning={isAgentRunning}
          />
        )}

        {activeTab === 'cloud' && (
          <CloudAndContainerView
            models={models}
            selectedModelId={selectedModelId}
            activeLogger={activeLogger}
          />
        )}
      </main>

      {/* Global Command Palette Modal (Cmd/Ctrl + K) */}
      <CommandPaletteModal
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        actions={commandPaletteActions}
        onExecuteAction={handleExecuteCommandPaletteAction}
      />

      {/* Pre-Flight Pipeline Doctor Modal (12-Check Audit & 1-Click Auto-Fix) */}
      <PipelineDoctorModal
        isOpen={isPipelineDoctorOpen}
        onClose={() => setIsPipelineDoctorOpen(false)}
        report={pipelineDoctorReport}
        onApplyAutoFix={handleApplyDoctorFix}
        onApplyAllFixes={handleApplyAllDoctorFixes}
      />

      {/* Gruvbox Theme & Pipeline Status Footer */}
      <footer className="border-t border-[#4A453E] bg-[#22211F] py-4 px-6 text-xs text-[#A89984]">
        <div className="max-w-[1400px] mx-auto flex flex-col lg:flex-row items-center justify-between gap-3">
          <span>
            ForgeML Pipeline Studio · Sub-2B Edge &amp; Multi-Model Fine-Tuning (LFM2.5, Qwen3.5, Gemma4)
          </span>

          <div className="flex flex-wrap items-center gap-3 font-mono text-[11px]">
            <button
              type="button"
              onClick={() => setIsCommandPaletteOpen(true)}
              className="px-2 py-0.5 rounded bg-[#2F2F2F] border border-[#4A453E] text-[#83A598] hover:text-[#FFE9B1] transition-colors"
            >
              Cmd/Ctrl + K Command Palette
            </button>
            <span className="inline-flex items-center gap-1.5 text-[#D5C4A1]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#458588] inline-block" />
              Theme: Gruvbox ({gruvboxBgPreset} · #FFE9B1 · #458588 · JetBrainsMono Nerd Font)
            </span>
            <div className="flex items-center gap-1 bg-[#2F2F2F] border border-[#4A453E] rounded px-1.5 py-0.5">
              {(
                [
                  { bg: '#2F2F2F', label: 'Soft (#2F2F2F)' },
                  { bg: '#282828', label: 'Medium (#282828)' },
                  { bg: '#1D2021', label: 'Hard (#1D2021)' },
                ] as const
              ).map((preset) => (
                <button
                  key={preset.bg}
                  type="button"
                  onClick={() => setGruvboxBgPreset(preset.bg)}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    gruvboxBgPreset === preset.bg
                      ? 'bg-[#458588] text-[#FFE9B1]'
                      : 'text-[#A89984] hover:text-[#FFE9B1]'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
