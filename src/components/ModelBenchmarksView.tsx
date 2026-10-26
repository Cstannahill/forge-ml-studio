import React, { useState } from 'react';
import {
  Award,
  BarChart3,
  Bot,
  CheckCircle2,
  Cpu,
  Eye,
  Flag,
  Plus,
  Play,
  Send,
  Sparkles,
  Terminal,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { VeilVisionRedactionCanvas } from './VeilVisionRedactionCanvas';
import { PopeAndHardNegativeMinerPanel } from './PopeAndHardNegativeMinerPanel';
import { MultiRunAblationMatrixPanel } from './MultiRunAblationMatrixPanel';
import {
  ArchitectureFamily,
  PlainLanguageComparison,
  buildBenchmarkSummaryTableRows,
  buildPlainLanguageBenchmarkAnalysis,
  buildRadarChartComparisonData,
  compareBaseAndFineTuned,
  generateArenaComparisonResponse,
  runAutomatedRedTeamStressSuite,
} from '../lib/pipelineLogic';
import {
  INITIAL_VISION_MODELS,
  PipelineModalityTrack,
  RegisteredVisionModel,
  buildVisionVlmBenchmarkDashboard,
} from '../lib/visionPipelineLogic';
import { RegisteredModel } from '../data/initialPipelineData';
import { InfoTooltip } from './InfoTooltip';

interface ModelBenchmarksViewProps {
  models: RegisteredModel[];
  selectedModelId: string;
  onSelectModel: (id: string) => void;
  onRegisterNewBaseModel: (params: {
    id: string;
    architecture: ArchitectureFamily;
    paramsBillions: number;
    contextWindow: number;
  }) => void;
  onTriggerFineTuneAndCompare: (modelId: string) => void;
  plainExplanations: Record<
    string,
    { providerUsed: string; timestamp: string; explanation: PlainLanguageComparison }
  >;
  onExplainBenchmarksWithLlm: (model: RegisteredModel) => Promise<void>;
  onSendFlaggedPromptToDatasetStudio?: (draft: {
    instruction: string;
    output: string;
    domainTag: string;
  }) => void;
  isExplaining: boolean;
  visionModels?: RegisteredVisionModel[];
  selectedVisionModelId?: string;
  onSelectVisionModel?: (id: string) => void;
  onTriggerVisionFineTune?: (modelId: string) => void;
  pipelineTrack?: PipelineModalityTrack;
}

const ARENA_PRESETS: Array<{
  id: 'JSON_SCHEMA' | 'PYTHON_REFACTOR' | 'MATH_COT' | 'TOOL_CALLING';
  label: string;
  defaultPrompt: string;
}> = [
  {
    id: 'JSON_SCHEMA',
    label: 'Strict JSON Schema Extraction',
    defaultPrompt:
      'Extract incident_id, severity, root_cause, and mitigation_steps from the log into strict JSON with zero markdown fences.',
  },
  {
    id: 'PYTHON_REFACTOR',
    label: 'Python Async Concurrency Refactor',
    defaultPrompt:
      'Refactor the sequential task runner into an asyncio.Semaphore bounded worker pool with full type hints.',
  },
  {
    id: 'MATH_COT',
    label: 'Multi-Step Optimizer Math CoT',
    defaultPrompt:
      'Calculate total optimizer steps for 3 epochs over 12.8M tokens with micro_batch=8, grad_accum=4, and seq_len=2048.',
  },
  {
    id: 'TOOL_CALLING',
    label: 'Edge GPU Tool-Call Router',
    defaultPrompt:
      'Check if GPU die temperature exceeds 85C and invoke gpu_telemetry_action with NF4 quantization fallback.',
  },
];

export const ModelBenchmarksView: React.FC<ModelBenchmarksViewProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  onRegisterNewBaseModel,
  onTriggerFineTuneAndCompare,
  plainExplanations,
  onExplainBenchmarksWithLlm,
  onSendFlaggedPromptToDatasetStudio,
  isExplaining,
  visionModels = INITIAL_VISION_MODELS,
  selectedVisionModelId,
  onSelectVisionModel,
  onTriggerVisionFineTune,
  pipelineTrack = 'TEXT_LLM',
}) => {
  const [newModelId, setNewModelId] = useState('');
  const [newArch, setNewArch] = useState<ArchitectureFamily>('Qwen3.5');
  const [newParamsB, setNewParamsB] = useState(2.0);
  const [newContext, setNewContext] = useState(32768);
  const [formError, setFormError] = useState('');

  // Local fallback selection for VISION_VLM benchmark dashboard if not controlled by parent
  const [localVisionModelId, setLocalVisionModelId] = useState<string>(
    visionModels[0]?.id ?? 'google/gemma-4-e2b-vision-it'
  );
  const effectiveVisionModelId = selectedVisionModelId ?? localVisionModelId;
  const handleSelectVisionModel = (id: string) => {
    setLocalVisionModelId(id);
    if (onSelectVisionModel) {
      onSelectVisionModel(id);
    }
  };

  const vlmDashboard = buildVisionVlmBenchmarkDashboard(
    visionModels,
    effectiveVisionModelId
  );

  // Side-by-Side Interactive Evaluation Arena State
  const [arenaPreset, setArenaPreset] = useState<
    'JSON_SCHEMA' | 'PYTHON_REFACTOR' | 'MATH_COT' | 'TOOL_CALLING'
  >('JSON_SCHEMA');
  const [arenaCustomPrompt, setArenaCustomPrompt] = useState<string>(
    ARENA_PRESETS[0].defaultPrompt
  );
  const [flaggedNotice, setFlaggedNotice] = useState<string | null>(null);

  const selectedModel = models.find((m) => m.id === selectedModelId) ?? models[0];

  const comparison = selectedModel.fineTunedBenchmarks
    ? compareBaseAndFineTuned(selectedModel.baseBenchmarks, selectedModel.fineTunedBenchmarks)
    : null;

  const radarData = buildRadarChartComparisonData(
    selectedModel.baseBenchmarks,
    selectedModel.fineTunedBenchmarks
  );

  const summaryTableRows = buildBenchmarkSummaryTableRows(models);
  const selectedSummaryRow =
    summaryTableRows.find((r) => r.modelId === selectedModel.id) ?? summaryTableRows[0];

  const arenaResponse = generateArenaComparisonResponse({
    prompt: arenaCustomPrompt,
    presetCategory: arenaPreset,
    modelName: selectedModel.name,
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    stage2Aligned: true,
  });

  const [redTeamNotice, setRedTeamNotice] = useState<string | null>(null);
  const redTeamReport = runAutomatedRedTeamStressSuite({
    modelName: selectedModel.name,
    architecture: selectedModel.architecture,
    paramsBillions: selectedModel.paramsBillions,
    stage2Aligned: true,
  });

  const storedExplanationEntry = plainExplanations[selectedModel.id];
  const activeExplanation: PlainLanguageComparison | null =
    storedExplanationEntry?.explanation ??
    (selectedModel.fineTunedBenchmarks
      ? buildPlainLanguageBenchmarkAnalysis({
          modelName: selectedModel.name,
          architecture: selectedModel.architecture,
          base: selectedModel.baseBenchmarks,
          fineTuned: selectedModel.fineTunedBenchmarks,
        })
      : null);

  const handleDetectAndBenchmarkBase = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = newModelId.trim();
    if (!trimmed) {
      setFormError(
        'Please enter a HuggingFace model identifier or local path (e.g. Qwen/Qwen3.5-32B-Instruct).'
      );
      return;
    }
    setFormError('');
    onRegisterNewBaseModel({
      id: trimmed,
      architecture: newArch,
      paramsBillions: newParamsB,
      contextWindow: newContext,
    });
    setNewModelId('');
  };

  const benchmarkRows: Array<{
    key: keyof RegisteredModel['baseBenchmarks'];
    label: string;
    unit: string;
    lowerIsBetter?: boolean;
    description: string;
    glossaryKey?: string;
  }> = [
    {
      key: 'mmluPro',
      label: 'MMLU-Pro (10-Choice STEM & Reasoning)',
      unit: '%',
      description: 'Graduate-level multi-task reasoning and domain knowledge',
      glossaryKey: 'mmluPro',
    },
    {
      key: 'gsm8k',
      label: 'GSM8K (8-Shot Chain-of-Thought Math)',
      unit: '%',
      description: 'Multi-step mathematical word problem derivation accuracy',
      glossaryKey: 'gsm8k',
    },
    {
      key: 'humanEval',
      label: 'HumanEval+ (Pass@1 Code Synthesis)',
      unit: '%',
      description: 'Executable Python & polyglot unit-test synthesis pass rate',
      glossaryKey: 'humanEval',
    },
    {
      key: 'ifEval',
      label: 'IFEval (Strict Instruction Following)',
      unit: '%',
      description: 'Verifiable formatting, length, and schema constraint compliance',
      glossaryKey: 'ifEval',
    },
    {
      key: 'mtBench',
      label: 'MT-Bench (Multi-Turn Judge Score)',
      unit: '/10',
      description: 'Multi-turn conversational coherence and instruction retention',
      glossaryKey: 'mtBench',
    },
    {
      key: 'perplexity',
      label: 'Held-Out Validation Perplexity (PPL)',
      unit: 'ppl',
      lowerIsBetter: true,
      description:
        'Token-level cross-entropy perplexity on held-out evaluation split (lower is better)',
      glossaryKey: 'ppl',
    },
    {
      key: 'tokensPerSec',
      label: 'Inference Decode Throughput (vLLM / FP16)',
      unit: 'tok/s',
      description: 'Single-stream token generation velocity at batch=1, ctx=4096',
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-6 border-b border-slate-800/80">
        <div>
          <p className="text-xs text-slate-400">
            Automated Base Model Detection · Post-Fine-Tuning Evaluation · Plain-Language LLM Comparison
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100 mt-1">
            Model Registry & Automated Benchmarking
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={isExplaining}
            onClick={() => onExplainBenchmarksWithLlm(selectedModel)}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            {isExplaining
              ? 'Analyzing Benchmarks with LLM...'
              : 'Explain Benchmark Results in Plain Terms'}
          </button>

          <button
            type="button"
            onClick={() => onTriggerFineTuneAndCompare(selectedModel.id)}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            {selectedModel.fineTunedBenchmarks
              ? `Re-Run Fine-Tune & Auto-Evaluate (${selectedModel.name})`
              : `Fine-Tune & Auto-Benchmark Against Base (${selectedModel.name})`}
          </button>
        </div>
      </section>

      {/* VISION_VLM Visual Benchmark Dashboard (M-MMLU · MMMU · DocVQA) */}
      <section
        aria-label="VISION_VLM Visual Benchmark Dashboard"
        className={`bg-[#252423] border-2 rounded-xl p-6 space-y-6 ${
          pipelineTrack === 'VISION_VLM'
            ? 'border-[#8EC07C] ring-1 ring-[#8EC07C]/30'
            : 'border-[#8EC07C]/50'
        }`}
      >
        <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#8EC07C]">
              <Eye className="w-4 h-4" />
              <span>VISION_VLM Track Visual Benchmark Dashboard · M-MMLU · MMMU · DocVQA</span>
              <span className="px-2 py-0.5 text-[10px] rounded bg-[#1D2021] border border-[#8EC07C]/40 text-[#FFE9B1]">
                {vlmDashboard.registryTableRows.length} Multimodal Checkpoints (LFM2.5-VL · Qwen3.5-VL · Gemma4-VL)
              </span>
            </div>
            <h2 className="text-lg font-bold text-[#FFE9B1]">
              Multimodal VLM Evaluation Suite: M-MMLU, MMMU &amp; DocVQA Visual Analytics
            </h2>
            <p className="text-xs text-[#A89984] max-w-4xl">
              Tracks cross-lingual &amp; cultural multimodal reasoning (<strong className="text-[#8EC07C]">M-MMLU</strong>),
              college-level multi-discipline STEM diagrams (<strong className="text-[#83A598]">MMMU</strong>),
              and high-resolution Pan-&amp;-Scan / NaViT document extraction (<strong className="text-[#B8BB26]">DocVQA</strong>)
              across all models in the <code className="font-mono text-[#8EC07C]">VISION_VLM</code> track.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <div className="text-xs font-mono text-[#D5C4A1] bg-[#1D2021] border border-[#3C3836] px-3 py-1.5 rounded-lg">
              Active VLM:{' '}
              <strong className="text-[#FFE9B1]">{vlmDashboard.selectedModel.name}</strong>
            </div>
            {onTriggerVisionFineTune && (
              <button
                type="button"
                onClick={() => onTriggerVisionFineTune(vlmDashboard.selectedModel.id)}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors whitespace-nowrap"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Re-Run VLM Fine-Tune &amp; Eval ({vlmDashboard.selectedModel.architecture}-VL)
              </button>
            )}
          </div>
        </div>

        {/* VLM Model Selector Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-2.5">
          {visionModels.map((vm) => {
            const isSelected = vm.id === vlmDashboard.selectedModel.id;
            const ft = vm.fineTunedVisionBenchmarks ?? vm.baseVisionBenchmarks;
            return (
              <button
                key={vm.id}
                type="button"
                onClick={() => handleSelectVisionModel(vm.id)}
                className={`text-left p-3 rounded-xl border transition-all flex flex-col justify-between gap-2 ${
                  isSelected
                    ? 'bg-[#1D2021] border-[#8EC07C] ring-1 ring-[#8EC07C]/40'
                    : 'bg-[#1D2021]/60 border-[#3C3836] hover:border-[#504945]'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#252423] text-[#83A598]">
                      {vm.architecture}-VL
                    </span>
                    <span className="text-[11px] font-mono font-bold text-[#FFE9B1] tabular-nums">
                      {vm.totalParamsBillions}B
                    </span>
                  </div>
                  <div className="text-xs font-semibold text-[#FFE9B1] mt-1 leading-snug">
                    {vm.name}
                  </div>
                </div>
                <div className="pt-1.5 border-t border-[#3C3836] flex items-center justify-between text-[10px] font-mono tabular-nums">
                  <span className="text-[#8EC07C]">M-MMLU {ft.mMmlu}%</span>
                  <span className="text-[#83A598]">MMMU {ft.mmmu}%</span>
                  <span className="text-[#B8BB26]">Doc {ft.docVqa}%</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Core Triad Spotlight Cards: M-MMLU, MMMU, DocVQA */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {vlmDashboard.coreTriadMetrics.map((card) => (
            <div
              key={card.key}
              className="bg-[#1D2021] border border-[#4A453E] rounded-xl p-4 space-y-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="text-xs font-mono uppercase tracking-wider text-[#A89984]">
                    {card.shortLabel} Primary VLM Benchmark
                  </div>
                  <h3 className="text-sm font-bold text-[#FFE9B1] mt-0.5">{card.label}</h3>
                </div>
                <span
                  className="px-2 py-0.5 text-xs font-mono font-bold rounded bg-[#252423] border border-[#4A453E] tabular-nums"
                  style={{ color: card.accentColor }}
                >
                  +{card.delta}% ({card.relativeGainPct >= 0 ? '+' : ''}
                  {card.relativeGainPct}% rel)
                </span>
              </div>

              <p className="text-[11px] text-[#A89984] leading-relaxed">{card.subtitle}</p>

              <div className="pt-2 border-t border-[#3C3836] flex items-baseline justify-between font-mono tabular-nums">
                <div className="text-xs text-[#D5C4A1]">
                  Base Checkpoint: <span className="text-[#A89984]">{card.baseScore}%</span>
                </div>
                <div className="text-base font-bold text-[#FFE9B1]">
                  Fine-Tuned:{' '}
                  <span style={{ color: card.accentColor }}>{card.fineTunedScore}%</span>
                </div>
              </div>

              {/* Dual Progress Bar */}
              <div className="space-y-1">
                <div className="w-full h-2 rounded-full bg-[#252423] overflow-hidden flex">
                  <div
                    className="h-full bg-[#665C54]"
                    style={{ width: `${Math.min(100, card.baseScore)}%` }}
                    title={`Base ${card.shortLabel}: ${card.baseScore}%`}
                  />
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.max(0, Math.min(100 - card.baseScore, card.delta))}%`,
                      backgroundColor: card.accentColor,
                    }}
                    title={`Fine-Tuned Gain: +${card.delta}%`}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Visual Charts Row: Selected Model Base vs Fine-Tuned + Cross-Model M-MMLU / MMMU / DocVQA Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 6 Cols: Selected VLM Base vs Fine-Tuned Across All 6 Multimodal Benchmarks */}
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4 text-[#8EC07C]" />
                  <span>
                    Base vs. Fine-Tuned VLM Profile ({vlmDashboard.selectedModel.name})
                  </span>
                </h3>
                <p className="text-[11px] text-[#A89984]">
                  Direct comparison across M-MMLU, MMMU, DocVQA, MathVista, RealWorldQA &amp; BBox IoU
                </p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#252423] text-[#8EC07C] border border-[#3C3836]">
                {vlmDashboard.selectedModel.visionEncoder}
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={vlmDashboard.selectedModelBarChartData}
                  margin={{ top: 10, right: 12, left: -12, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#3C3836" />
                  <XAxis
                    dataKey="metric"
                    tick={{ fill: '#D5C4A1', fontSize: 11, fontFamily: 'JetBrainsMono Nerd Font, monospace' }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: '#A89984', fontSize: 11, fontFamily: 'JetBrainsMono Nerd Font, monospace' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1D2021',
                      borderColor: '#4A453E',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: '#FFE9B1',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="Base" name="Base VLM (%)" fill="#665C54" radius={[4, 4, 0, 0]} />
                  <Bar
                    dataKey="FineTuned"
                    name="Fine-Tuned VLM (%)"
                    fill="#8EC07C"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Right 6 Cols: Cross-Model VISION_VLM Leaderboard on M-MMLU, MMMU, and DocVQA */}
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                  <Award className="w-4 h-4 text-[#FABD2F]" />
                  <span>
                    Cross-Model VISION_VLM Triad Comparison (M-MMLU · MMMU · DocVQA)
                  </span>
                </h3>
                <p className="text-[11px] text-[#A89984]">
                  Side-by-side Fine-Tuned scores across LFM2.5-VL (450M/1.6B), Qwen3.5-VL (0.8B/2B) &amp; Gemma4-VL (E2B/E4B)
                </p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#252423] text-[#FABD2F] border border-[#3C3836]">
                Triad Leaderboard
              </span>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={vlmDashboard.crossModelTriadChartData}
                  margin={{ top: 10, right: 12, left: -12, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#3C3836" />
                  <XAxis
                    dataKey="modelShortName"
                    tick={{ fill: '#D5C4A1', fontSize: 10, fontFamily: 'JetBrainsMono Nerd Font, monospace' }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: '#A89984', fontSize: 11, fontFamily: 'JetBrainsMono Nerd Font, monospace' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1D2021',
                      borderColor: '#4A453E',
                      borderRadius: '8px',
                      fontSize: '12px',
                      color: '#FFE9B1',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="M-MMLU" fill="#8EC07C" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="MMMU" fill="#83A598" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="DocVQA" fill="#B8BB26" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* VISION_VLM Registry Table: Base vs Fine-Tuned M-MMLU, MMMU, DocVQA & Spatial Metrics */}
        <div className="overflow-x-auto rounded-xl border border-[#3C3836] bg-[#1D2021]">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[#3C3836] text-[11px] font-mono uppercase text-[#A89984] bg-[#252423]">
                <th className="py-2.5 px-3">VISION_VLM Model</th>
                <th className="py-2.5 px-3">Vision Tower</th>
                <th className="py-2.5 px-3 text-right">M-MMLU (Base → FT)</th>
                <th className="py-2.5 px-3 text-right">MMMU (Base → FT)</th>
                <th className="py-2.5 px-3 text-right">DocVQA (Base → FT)</th>
                <th className="py-2.5 px-3 text-right">MathVista</th>
                <th className="py-2.5 px-3 text-right">BBox IoU</th>
                <th className="py-2.5 px-3 text-right">Avg Triad Δ</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#3C3836] text-xs font-mono tabular-nums">
              {vlmDashboard.registryTableRows.map((row) => {
                const isSelected = row.modelId === vlmDashboard.selectedModel.id;
                return (
                  <tr
                    key={row.modelId}
                    onClick={() => handleSelectVisionModel(row.modelId)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-[#458588]/25' : 'hover:bg-[#252423]/70'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans">
                      <div className="font-semibold text-[#FFE9B1]">{row.modelName}</div>
                      <div className="text-[10px] font-mono text-[#A89984]">
                        {row.architecture}-VL · {row.totalParamsBillions}B · {row.visionTokensPerSec}{' '}
                        tok/s
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-[11px] text-[#D5C4A1] max-w-[200px] truncate">
                      {row.visionEncoder}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className="text-[#A89984]">{row.mMmluBase}%</span> →{' '}
                      <strong className="text-[#FFE9B1]">{row.mMmluFineTuned}%</strong>{' '}
                      <span className="text-[#8EC07C]">(+{row.mMmluDelta}%)</span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className="text-[#A89984]">{row.mmmuBase}%</span> →{' '}
                      <strong className="text-[#FFE9B1]">{row.mmmuFineTuned}%</strong>{' '}
                      <span className="text-[#83A598]">(+{row.mmmuDelta}%)</span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <span className="text-[#A89984]">{row.docVqaBase}%</span> →{' '}
                      <strong className="text-[#FFE9B1]">{row.docVqaFineTuned}%</strong>{' '}
                      <span className="text-[#B8BB26]">(+{row.docVqaDelta}%)</span>
                    </td>
                    <td className="py-2.5 px-3 text-right text-[#D5C4A1]">
                      {row.mathVistaFineTuned}%
                    </td>
                    <td className="py-2.5 px-3 text-right text-[#FABD2F]">
                      {row.groundingIoUFineTuned}%
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-[#8EC07C]">
                      +{row.averageCoreTriadDelta}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Detect & Auto-Benchmark New Base Model Form */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="pb-4 border-b border-slate-800/80">
          <h2 className="text-base font-semibold text-slate-100">
            01. Detect & Benchmark New Base Model
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Point to any base checkpoint (Qwen3.5, LFM2.5, Gemma4, Llama-3.3, Mistral, DeepSeek, or custom) to automatically run and store baseline evaluations
          </p>
        </div>

        <form
          onSubmit={handleDetectAndBenchmarkBase}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-12 gap-3 mt-4 items-end"
        >
          <div className="lg:col-span-5">
            <label htmlFor="new-base-model-id" className="block text-xs text-slate-400 mb-1">
              HuggingFace Model ID or Local Checkpoint Path
            </label>
            <input
              id="new-base-model-id"
              type="text"
              placeholder="e.g. Qwen/Qwen3.5-32B-Instruct or LiquidAI/LFM2.5-40B-MoE"
              value={newModelId}
              onChange={(e) => setNewModelId(e.target.value)}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 placeholder:text-slate-500"
            />
          </div>

          <div className="lg:col-span-2">
            <label htmlFor="new-base-arch" className="flex items-center text-xs text-slate-400 mb-1">
              <span>Architecture Family</span>
              <InfoTooltip termKey="lfmArchitecture" />
            </label>
            <select
              id="new-base-arch"
              value={newArch}
              onChange={(e) => setNewArch(e.target.value as ArchitectureFamily)}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100"
            >
              <option value="Qwen3.5">Qwen3.5 (GQA + Dynamic NTK)</option>
              <option value="LFM2.5">LFM2.5 (Liquid Hybrid SSM)</option>
              <option value="Gemma4">Gemma4 (Sliding-Window + Softcap)</option>
              <option value="Llama3.3">Llama 3.3</option>
              <option value="Mistral3">Mistral Small 3</option>
              <option value="DeepSeekV3">DeepSeek MLA/MoE</option>
              <option value="Custom">Custom Transformer</option>
            </select>
          </div>

          <div className="lg:col-span-2">
            <label htmlFor="new-base-params" className="block text-xs text-slate-400 mb-1">
              Parameters (Billions)
            </label>
            <input
              id="new-base-params"
              type="number"
              min="0.5"
              max="405"
              step="0.5"
              value={newParamsB}
              onChange={(e) => setNewParamsB(Number(e.target.value) || 7)}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono tabular-nums text-slate-100"
            />
          </div>

          <div className="lg:col-span-3">
            <button
              type="submit"
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors whitespace-nowrap"
            >
              <Plus className="w-3.5 h-3.5" />
              Detect & Run Baseline Benchmark
            </button>
          </div>
        </form>

        {formError && (
          <p className="text-xs text-amber-400 mt-2" role="alert">
            {formError}
          </p>
        )}
      </section>

      {/* Model Architecture Cards Grid */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {models.map((m) => {
          const isSelected = m.id === selectedModel.id;
          const comp = m.fineTunedBenchmarks
            ? compareBaseAndFineTuned(m.baseBenchmarks, m.fineTunedBenchmarks)
            : null;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onSelectModel(m.id)}
              className={`text-left p-4 rounded-lg border transition-colors ${
                isSelected
                  ? 'bg-[#111827] border-blue-500/80'
                  : 'bg-[#111827]/70 border-slate-800/80 hover:border-slate-700'
              }`}
            >
              <div className="text-xs text-slate-400 font-mono">
                {m.architecture} · {m.paramsBillions}B · {(m.contextWindow / 1024).toFixed(0)}k ctx
              </div>
              <div className="text-sm font-semibold text-slate-100 mt-1 truncate">{m.name}</div>
              <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono tabular-nums">
                <span className="text-slate-400">Base MMLU: {m.baseBenchmarks.mmluPro}%</span>
                {comp ? (
                  <span className="text-emerald-400">+{comp.averageAccuracyGain}% FT Gain</span>
                ) : (
                  <span className="text-amber-400">Base Stored</span>
                )}
              </div>
            </button>
          );
        })}
      </section>

      {/* Section 02: Base vs. Fine-Tuned Performance Summary Table (Highlighting MMLU-Pro & HumanEval Deltas) */}
      <section
        aria-label="Base vs Fine-Tuned Performance Summary"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              02. Base vs. Fine-Tuned Performance Summary Table (MMLU-Pro &amp; HumanEval+ Delta Spotlight)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Side-by-side baseline vs. fine-tuned scores across the model registry, specifically highlighting net accuracy deltas on MMLU-Pro and HumanEval+
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs font-mono tabular-nums">
            <span className="text-slate-400">Active Focus:</span>
            <span className="px-2.5 py-1 rounded bg-[#0B0F17] border border-slate-800 text-slate-100 font-semibold">
              {selectedModel.name}
            </span>
          </div>
        </div>

        {/* Highlighted Delta Spotlight Cards for Active Model (MMLU-Pro & HumanEval+ Emphasized) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#0B0F17] border border-emerald-500/40 rounded-lg p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-semibold inline-flex items-center">
                MMLU-Pro Delta
                <InfoTooltip termKey="mmluPro" align="left" />
              </span>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                Highlighted
              </span>
            </div>
            <div className="mt-2 flex items-baseline justify-between font-mono tabular-nums">
              <span className="text-xs text-slate-400">
                Base {selectedSummaryRow.mmluProBase}% → FT{' '}
                <strong className="text-slate-100">
                  {selectedSummaryRow.mmluProFineTuned !== null
                    ? `${selectedSummaryRow.mmluProFineTuned}%`
                    : 'Pending'}
                </strong>
              </span>
              <span className="text-lg font-semibold text-emerald-400">
                {selectedSummaryRow.mmluProDelta !== null
                  ? `${selectedSummaryRow.mmluProDelta >= 0 ? '+' : ''}${selectedSummaryRow.mmluProDelta}%`
                  : '—'}
              </span>
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-emerald-500/40 rounded-lg p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-semibold inline-flex items-center">
                HumanEval+ Delta
                <InfoTooltip termKey="humanEval" align="left" />
              </span>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                Highlighted
              </span>
            </div>
            <div className="mt-2 flex items-baseline justify-between font-mono tabular-nums">
              <span className="text-xs text-slate-400">
                Base {selectedSummaryRow.humanEvalBase}% → FT{' '}
                <strong className="text-slate-100">
                  {selectedSummaryRow.humanEvalFineTuned !== null
                    ? `${selectedSummaryRow.humanEvalFineTuned}%`
                    : 'Pending'}
                </strong>
              </span>
              <span className="text-lg font-semibold text-emerald-400">
                {selectedSummaryRow.humanEvalDelta !== null
                  ? `${selectedSummaryRow.humanEvalDelta >= 0 ? '+' : ''}${selectedSummaryRow.humanEvalDelta}%`
                  : '—'}
              </span>
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-medium inline-flex items-center">
                GSM8K Math Delta
                <InfoTooltip termKey="gsm8k" />
              </span>
              <span className="text-[10px] font-mono text-slate-400">CoT Logic</span>
            </div>
            <div className="mt-2 flex items-baseline justify-between font-mono tabular-nums">
              <span className="text-xs text-slate-400">
                Base {selectedSummaryRow.gsm8kBase}% → FT{' '}
                <strong className="text-slate-100">
                  {selectedSummaryRow.gsm8kFineTuned !== null
                    ? `${selectedSummaryRow.gsm8kFineTuned}%`
                    : 'Pending'}
                </strong>
              </span>
              <span className="text-lg font-semibold text-emerald-400">
                {selectedSummaryRow.gsm8kDelta !== null
                  ? `${selectedSummaryRow.gsm8kDelta >= 0 ? '+' : ''}${selectedSummaryRow.gsm8kDelta}%`
                  : '—'}
              </span>
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-medium inline-flex items-center">
                IFEval &amp; PPL Delta
                <InfoTooltip termKey="ppl" align="right" />
              </span>
              <span className="text-[10px] font-mono text-slate-400">Constraints</span>
            </div>
            <div className="mt-2 flex items-baseline justify-between font-mono tabular-nums">
              <span className="text-xs text-slate-400">
                IFEval {selectedSummaryRow.ifEvalBase}% →{' '}
                <strong className="text-slate-100">
                  {selectedSummaryRow.ifEvalFineTuned !== null
                    ? `${selectedSummaryRow.ifEvalFineTuned}%`
                    : '—'}
                </strong>
              </span>
              <span className="text-lg font-semibold text-emerald-400">
                {selectedSummaryRow.ifEvalDelta !== null
                  ? `${selectedSummaryRow.ifEvalDelta >= 0 ? '+' : ''}${selectedSummaryRow.ifEvalDelta}%`
                  : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* Fleet-Wide Base vs. Fine-Tuned Summary Comparison Table */}
        <div className="overflow-x-auto">
          <table
            aria-label="Base vs Fine-Tuned Performance Summary Table"
            className="w-full text-left border-collapse text-xs whitespace-nowrap"
          >
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 pr-2 font-medium">Model &amp; Architecture</th>
                <th className="py-2.5 px-2 font-semibold text-right text-slate-200 bg-slate-800/25">
                  MMLU-Pro (Base → FT · Δ)
                </th>
                <th className="py-2.5 px-2 font-semibold text-right text-slate-200 bg-slate-800/25">
                  HumanEval+ (Base → FT · Δ)
                </th>
                <th className="py-2.5 px-2 font-medium text-right">GSM8K (Base → FT · Δ)</th>
                <th className="py-2.5 px-2 font-medium text-right">IFEval (Base → FT · Δ)</th>
                <th className="py-2.5 px-2 font-medium text-right">Val PPL (Base → FT)</th>
                <th className="py-2.5 pl-2 font-medium text-right">Mean Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono tabular-nums">
              {summaryTableRows.map((row) => {
                const isCurrent = row.modelId === selectedModel.id;
                return (
                  <tr
                    key={row.modelId}
                    onClick={() => onSelectModel(row.modelId)}
                    className={`cursor-pointer transition-colors ${
                      isCurrent ? 'bg-blue-600/15' : 'hover:bg-slate-800/30'
                    }`}
                  >
                    <td className="py-2.5 pr-2 font-sans">
                      <div className="font-semibold text-slate-100 flex items-center gap-1.5">
                        <span>{row.modelName}</span>
                        {isCurrent && (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-blue-600 text-white">
                            Active
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono">
                        {row.architecture} · {row.paramsBillions}B
                      </div>
                    </td>

                    {/* Highlighted MMLU-Pro Base -> FT + Delta */}
                    <td className="py-2.5 px-2 text-right bg-slate-800/15">
                      {row.hasFineTuned ? (
                        <div className="inline-flex items-center justify-end gap-1">
                          <span className="text-slate-400">{row.mmluProBase}%</span>
                          <span className="text-slate-500">→</span>
                          <span className="text-slate-100 font-semibold">
                            {row.mmluProFineTuned}%
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-semibold">
                            {row.mmluProDelta! >= 0 ? `+${row.mmluProDelta}%` : `${row.mmluProDelta}%`}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400">{row.mmluProBase}% (Base)</span>
                      )}
                    </td>

                    {/* Highlighted HumanEval+ Base -> FT + Delta */}
                    <td className="py-2.5 px-2 text-right bg-slate-800/15">
                      {row.hasFineTuned ? (
                        <div className="inline-flex items-center justify-end gap-1">
                          <span className="text-slate-400">{row.humanEvalBase}%</span>
                          <span className="text-slate-500">→</span>
                          <span className="text-slate-100 font-semibold">
                            {row.humanEvalFineTuned}%
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-semibold">
                            {row.humanEvalDelta! >= 0
                              ? `+${row.humanEvalDelta}%`
                              : `${row.humanEvalDelta}%`}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-400">{row.humanEvalBase}% (Base)</span>
                      )}
                    </td>

                    {/* GSM8K */}
                    <td className="py-2.5 px-2 text-right">
                      {row.hasFineTuned ? (
                        <span>
                          <span className="text-slate-400">{row.gsm8kBase}%</span>→
                          <span className="text-slate-200">{row.gsm8kFineTuned}%</span>{' '}
                          <span className="text-emerald-400">
                            ({row.gsm8kDelta! >= 0 ? `+${row.gsm8kDelta}%` : `${row.gsm8kDelta}%`})
                          </span>
                        </span>
                      ) : (
                        <span className="text-slate-400">{row.gsm8kBase}%</span>
                      )}
                    </td>

                    {/* IFEval */}
                    <td className="py-2.5 px-2 text-right">
                      {row.hasFineTuned ? (
                        <span>
                          <span className="text-slate-400">{row.ifEvalBase}%</span>→
                          <span className="text-slate-200">{row.ifEvalFineTuned}%</span>{' '}
                          <span className="text-emerald-400">
                            ({row.ifEvalDelta! >= 0 ? `+${row.ifEvalDelta}%` : `${row.ifEvalDelta}%`})
                          </span>
                        </span>
                      ) : (
                        <span className="text-slate-400">{row.ifEvalBase}%</span>
                      )}
                    </td>

                    {/* Perplexity */}
                    <td className="py-2.5 px-2 text-right">
                      {row.hasFineTuned ? (
                        <span>
                          <span className="text-slate-400">{row.perplexityBase}</span>→
                          <span className="text-slate-200">{row.perplexityFineTuned}</span>{' '}
                          <span className="text-emerald-400">({row.perplexityDelta})</span>
                        </span>
                      ) : (
                        <span className="text-slate-400">{row.perplexityBase} ppl</span>
                      )}
                    </td>

                    {/* Mean Accuracy Gain */}
                    <td className="py-2.5 pl-2 text-right">
                      {row.hasFineTuned ? (
                        <span className="text-emerald-400 font-semibold">
                          +{row.meanAccuracyGain}% · Ready
                        </span>
                      ) : (
                        <span className="text-amber-400">Base Stored</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Side-by-Side Base vs. Fine-Tuned Radar Chart & Evaluation Matrix */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              03. Side-by-Side Base vs. Fine-Tuned Radar &amp; Detailed Metric Matrix — {selectedModel.name}
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Base Detected: {selectedModel.detectedAt}
              {selectedModel.fineTunedCheckpointId
                ? ` · Fine-Tuned Checkpoint: ${selectedModel.fineTunedCheckpointId} · Dataset: ${selectedModel.datasetVersionUsed}`
                : ' · No fine-tuned checkpoint evaluated yet (click Fine-Tune & Auto-Benchmark above)'}
            </p>
          </div>

          {comparison && (
            <div className="text-xs font-mono tabular-nums">
              <span className="text-slate-400">Verdict: </span>
              <span
                className={
                  comparison.verdict === 'PROMOTE_TO_PRODUCTION'
                    ? 'text-emerald-400 font-semibold'
                    : comparison.verdict === 'REGRESSION_ALERT'
                    ? 'text-red-400 font-semibold'
                    : 'text-amber-400 font-semibold'
                }
              >
                {comparison.verdict} (Mean Accuracy +{comparison.averageAccuracyGain}%)
              </span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-5 items-stretch">
          {/* Left 5 cols: Recharts Multi-Axis Radar Chart (MMLU-Pro, GSM8K, HumanEval, IFEval, MT-Bench) */}
          <div
            aria-label="Base vs Fine-Tuned Performance Radar Chart"
            className="lg:col-span-5 bg-[#0B0F17] border border-slate-800/90 rounded-lg p-4 flex flex-col justify-between"
          >
            <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
              <div>
                <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-1">
                  <span>Multi-Axis Capability Radar (Recharts)</span>
                  <InfoTooltip termKey="mmluPro" align="left" />
                </h3>
                <p className="text-[11px] text-slate-400">
                  MMLU-Pro · GSM8K · HumanEval · IFEval · MT-Bench
                </p>
              </div>
              <span className="text-[11px] font-mono text-emerald-400">
                {selectedModel.fineTunedBenchmarks ? 'Base vs. FT Overlay' : 'Baseline Polygon'}
              </span>
            </div>

            <div className="w-full h-72 my-2">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="72%" data={radarData}>
                  <PolarGrid stroke="#4A453E" strokeDasharray="3 3" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fill: '#FFE9B1', fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={{ fill: '#A89984', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
                  />
                  <Radar
                    name={`Base (${selectedModel.name})`}
                    dataKey="base"
                    stroke="#FABD2F"
                    fill="#FABD2F"
                    fillOpacity={0.22}
                    strokeWidth={2}
                  />
                  <Radar
                    name={
                      selectedModel.fineTunedCheckpointId
                        ? `Fine-Tuned (${selectedModel.fineTunedCheckpointId})`
                        : 'Fine-Tuned (Pending)'
                    }
                    dataKey="fineTuned"
                    stroke="#8EC07C"
                    fill="#8EC07C"
                    fillOpacity={0.35}
                    strokeWidth={2.5}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#22211F',
                      borderColor: '#4A453E',
                      borderRadius: '8px',
                      color: '#FFE9B1',
                      fontSize: '11px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Legend
                    wrapperStyle={{
                      fontSize: '11px',
                      fontFamily: 'JetBrains Mono, monospace',
                      color: '#D5C4A1',
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="pt-2 border-t border-slate-800/80 grid grid-cols-4 gap-2 text-center font-mono text-[11px] tabular-nums">
              {radarData.slice(0, 4).map((pt) => (
                <div key={pt.metric} className="bg-[#111827] rounded p-1.5 border border-slate-800">
                  <div className="text-slate-400 text-[10px] truncate">{pt.metric}</div>
                  <div className="text-emerald-400 font-semibold">
                    {pt.delta >= 0 ? `+${pt.delta}%` : `${pt.delta}%`}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right 7 cols: Evaluation Matrix Table with Instructive Tooltips */}
          <div className="lg:col-span-7 overflow-x-auto lg:overflow-visible flex flex-col justify-between">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-xs text-slate-400">
                  <th className="py-2.5 pr-4 font-medium">Evaluation Benchmark</th>
                  <th className="py-2.5 px-4 font-medium text-right">Base Model Score</th>
                  <th className="py-2.5 px-4 font-medium text-right">Fine-Tuned Score</th>
                  <th className="py-2.5 pl-4 font-medium text-right">Net Delta</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 text-xs">
                {benchmarkRows.map((row) => {
                  const baseVal = selectedModel.baseBenchmarks[row.key];
                  const ftVal = selectedModel.fineTunedBenchmarks?.[row.key];
                  const delta =
                    ftVal !== undefined ? Number((ftVal - baseVal).toFixed(2)) : undefined;

                  const isPositiveImprovement =
                    delta !== undefined && (row.lowerIsBetter ? delta <= 0 : delta >= 0);

                  return (
                    <tr key={row.key} className="hover:bg-slate-800/30">
                      <td className="py-3 pr-4">
                        <div className="font-medium text-slate-200 flex items-center">
                          <span>{row.label}</span>
                          {row.glossaryKey && (
                            <InfoTooltip termKey={row.glossaryKey} align="left" />
                          )}
                        </div>
                        <div className="text-[11px] text-slate-500">{row.description}</div>
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-300">
                        {baseVal} {row.unit}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-100 font-medium">
                        {ftVal !== undefined ? `${ftVal} ${row.unit}` : 'Pending FT Run'}
                      </td>
                      <td className="py-3 pl-4 text-right font-mono tabular-nums">
                        {delta !== undefined ? (
                          <span
                            className={
                              isPositiveImprovement ? 'text-emerald-400' : 'text-amber-400'
                            }
                          >
                            {delta > 0 ? `+${delta}` : delta} {row.unit}
                          </span>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Section 3: LLM Plain-Language Benchmark Explainer */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              03. Plain-Language LLM Benchmark Interpretation — {selectedModel.name}
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Translates technical evaluation numbers into clear, everyday explanations of what improved, what traded off, and how behavior changed
              {storedExplanationEntry
                ? ` · Generated via ${storedExplanationEntry.providerUsed} at ${storedExplanationEntry.timestamp}`
                : ''}
            </p>
          </div>

          <button
            type="button"
            disabled={isExplaining}
            onClick={() => onExplainBenchmarksWithLlm(selectedModel)}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
          >
            <Bot className="w-3.5 h-3.5" />
            {isExplaining
              ? 'Generating Plain-English Analysis...'
              : 'Run LLM Plain-Language Analysis'}
          </button>
        </div>

        {activeExplanation ? (
          <div className="mt-5 space-y-6">
            {/* Executive Plain-Language Summary */}
            <div className="pb-5 border-b border-slate-800/80">
              <div className="text-xs text-blue-400 font-medium">
                Plain-English Executive Takeaway
              </div>
              <h3 className="text-base font-semibold text-slate-100 mt-1">
                {activeExplanation.headline}
              </h3>
              <p className="text-xs text-slate-300 mt-2 leading-relaxed max-w-4xl">
                {activeExplanation.plainSummary}
              </p>
            </div>

            {/* 2-Column Split: What It's Better At vs. What It's Worse At / Trade-Offs */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Better At */}
              <div className="space-y-3">
                <div className="text-xs font-semibold text-emerald-400">
                  What the Fine-Tuned Model Is Better At (Strengths &amp; Gains)
                </div>
                <div className="divide-y divide-slate-800/80">
                  {activeExplanation.betterAt.map((item, idx) => (
                    <div key={idx} className="py-3 first:pt-0 space-y-1">
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        <span className="font-semibold text-slate-100">{item.area}</span>
                        <span className="font-mono tabular-nums text-emerald-400">
                          {item.metricEvidence}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {item.plainExplanation}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Worse At / Trade-offs */}
              <div className="space-y-3 lg:border-l lg:border-slate-800/80 lg:pl-6">
                <div className="text-xs font-semibold text-amber-400">
                  What the Model Is Worse At / Trade-Offs to Watch
                </div>
                <div className="divide-y divide-slate-800/80">
                  {activeExplanation.worseAt.map((item, idx) => (
                    <div key={idx} className="py-3 first:pt-0 space-y-1">
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        <span className="font-semibold text-slate-100">{item.area}</span>
                        <span className="font-mono tabular-nums text-amber-400">
                          {item.metricEvidence}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {item.plainExplanation}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Other Behavioral & Operational Differences */}
            <div className="pt-5 border-t border-slate-800/80">
              <div className="text-xs font-semibold text-slate-200 mb-3">
                Other Behavioral &amp; Operational Differences
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {activeExplanation.otherDifferences.map((diff, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="text-xs font-medium text-slate-100">{diff.aspect}</div>
                    <p className="text-xs text-slate-400 leading-relaxed">{diff.detail}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Actionable Next Step */}
            <div className="pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <span className="text-slate-200 font-medium">
                {activeExplanation.recommendation}
              </span>
              <span className="font-mono text-slate-400 shrink-0">
                Base vs. Fine-Tuned Verified
              </span>
            </div>
          </div>
        ) : (
          <div className="py-8 text-center space-y-3">
            <p className="text-xs text-slate-400">
              {selectedModel.name} currently has only baseline base-model scores stored. Click{' '}
              <strong className="text-slate-200">
                Run LLM Plain-Language Analysis
              </strong>{' '}
              or{' '}
              <strong className="text-slate-200">
                Fine-Tune &amp; Auto-Benchmark Against Base
              </strong>{' '}
              to compare and explain the outcome in simple terms.
            </p>
          </div>
        )}
      </section>

      {/* Section 05: Side-by-Side Interactive Evaluation Arena ("Vibe Check" Sandbox: Base vs. Fine-Tuned) */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                Interactive Vibe-Check Sandbox
              </span>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                {selectedModel.name} ({selectedModel.paramsBillions}B)
              </span>
            </div>
            <h2 className="text-base font-semibold text-slate-100 mt-1.5">
              05. Side-by-Side Interactive Evaluation Arena (Base Model vs. Fine-Tuned Checkpoint)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Test prompts side-by-side to inspect reasoning traces, strict schema compliance, token throughput, and send flagged responses directly to Dataset Studio
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {ARENA_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => {
                  setArenaPreset(preset.id);
                  setArenaCustomPrompt(preset.defaultPrompt);
                  setFlaggedNotice(null);
                }}
                className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors ${
                  arenaPreset === preset.id
                    ? 'bg-blue-600 text-white border-blue-500 font-semibold'
                    : 'bg-[#0B0F17] border-slate-800 text-slate-300 hover:text-slate-100'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Prompt Input Bar */}
        <div className="flex flex-col sm:flex-row gap-2.5">
          <input
            type="text"
            aria-label="Evaluation Arena Prompt Input"
            value={arenaCustomPrompt}
            onChange={(e) => setArenaCustomPrompt(e.target.value)}
            className="flex-1 bg-[#0B0F17] border border-slate-800 rounded-lg px-3.5 py-2 text-xs font-mono text-slate-100"
          />
          <button
            type="button"
            onClick={() => {
              onSendFlaggedPromptToDatasetStudio?.(arenaResponse.correctedSampleDraft);
              setFlaggedNotice(
                `Sent flagged prompt & verified fine-tuned completion (${arenaResponse.correctedSampleDraft.domainTag}) to Dataset Studio corpus.`
              );
            }}
            className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-amber-300 bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 rounded-lg transition-colors whitespace-nowrap shrink-0"
          >
            <Flag className="w-3.5 h-3.5" />
            Flag Base Failure → Send to Dataset Studio
          </button>
        </div>

        {flaggedNotice && (
          <div
            role="status"
            className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-3.5 py-2 text-xs font-mono text-emerald-300"
          >
            {flaggedNotice}
          </div>
        )}

        {/* Split-Screen Base vs. Fine-Tuned Output Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Base Model Column */}
          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-3">
              <div className="flex items-center justify-between pb-2.5 border-b border-slate-800">
                <div>
                  <span className="text-xs font-semibold text-slate-300">
                    Base Model ({selectedModel.name} Untuned)
                  </span>
                  <div className="text-[11px] font-mono text-slate-400">
                    {arenaResponse.baseTokPerSec} tok/s · First-token latency: {arenaResponse.baseLatencyMs} ms
                  </div>
                </div>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-amber-500/15 text-amber-400">
                  Constraint Drift / Prose Wrapper
                </span>
              </div>

              <pre className="font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed">
                {arenaResponse.baseOutput}
              </pre>
            </div>

            <div className="pt-2.5 border-t border-slate-800/80 text-[11px] font-mono text-amber-400">
              [WARN] Failed strict IFEval / formatting constraint check
            </div>
          </div>

          {/* Fine-Tuned Checkpoint Column */}
          <div className="bg-[#0B0F17] border border-emerald-500/40 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-3">
              <div className="flex items-center justify-between pb-2.5 border-b border-slate-800">
                <div>
                  <span className="text-xs font-semibold text-emerald-400">
                    Fine-Tuned Checkpoint ({selectedModel.fineTunedCheckpointId ?? 'ft-merged-orpo'})
                  </span>
                  <div className="text-[11px] font-mono text-slate-400">
                    {arenaResponse.ftTokPerSec} tok/s · First-token latency: {arenaResponse.ftLatencyMs} ms
                  </div>
                </div>
                <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                  [PASS] 100% Schema &amp; CoT Verified
                </span>
              </div>

              <pre className="font-mono text-xs text-slate-100 whitespace-pre-wrap leading-relaxed">
                {arenaResponse.fineTunedOutput}
              </pre>
            </div>

            <div className="pt-2.5 border-t border-slate-800/80 space-y-1">
              {arenaResponse.keyImprovements.map((imp, idx) => (
                <div key={idx} className="text-[11px] font-mono text-emerald-400">
                  [+] {imp}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Section 06: Automated Red-Teaming, Hallucination & Edge-Case Stress-Test Matrix */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center">
              <span>06. Automated Red-Teaming, Hallucination &amp; Edge-Case Stress-Test Matrix</span>
              <InfoTooltip termKey="redTeamSuite" align="left" />
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Adversarial prompt injection, truncated JSON recovery, fake-API hallucination traps, and n-gram repetition stress probes
            </p>
          </div>

          <div className="flex items-center gap-3 font-mono text-xs tabular-nums">
            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5">
              <span className="text-slate-400">Base Resilience: </span>
              <strong className="text-amber-400">{redTeamReport.baseResilienceScorePct}%</strong>
            </div>
            <div className="bg-[#0B0F17] border border-emerald-500/40 rounded-lg px-3 py-1.5">
              <span className="text-slate-400">Fine-Tuned Resilience: </span>
              <strong className="text-emerald-400">
                {redTeamReport.fineTunedResilienceScorePct}% ({redTeamReport.passedProbeCount}/
                {redTeamReport.totalProbeCount} Passed)
              </strong>
            </div>
          </div>
        </div>

        {redTeamNotice && (
          <div
            role="status"
            className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-3.5 py-2 text-xs font-mono text-emerald-300"
          >
            {redTeamNotice}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {redTeamReport.probes.map((probe) => (
            <div
              key={probe.id}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3"
            >
              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold text-slate-100">{probe.title}</div>
                    <div className="text-[10px] font-mono text-slate-400">
                      Category: {probe.category}
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 font-mono text-[10px]">
                    <span className="px-2 py-0.5 rounded bg-amber-500/15 text-amber-400">
                      Base: {probe.baseStatus}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                      FT: {probe.fineTunedStatus}
                    </span>
                  </div>
                </div>

                <div className="p-2.5 rounded bg-[#111827] border border-slate-800/90 text-[11px] font-mono text-slate-300">
                  Prompt: &ldquo;{probe.adversarialPrompt}&rdquo;
                </div>

                <div className="space-y-1 text-[11px] font-mono">
                  <div className="text-amber-400">[-] Base: {probe.baseFailureSummary}</div>
                  <div className="text-emerald-400">[+] Fine-Tuned: {probe.fineTunedDefenseSummary}</div>
                </div>
              </div>

              <div className="pt-2.5 border-t border-slate-800/80 flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    onSendFlaggedPromptToDatasetStudio?.(probe.correctiveSampleDraft);
                    setRedTeamNotice(
                      `Appended corrective hardening sample for "${probe.title}" (${probe.correctiveSampleDraft.domainTag}) to Dataset Studio corpus.`
                    );
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded transition-colors"
                >
                  <Plus className="w-3.5 h-3.5 text-emerald-400" />
                  Add Hardening Sample to Dataset Studio
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Section 06B: POPE Adversarial Visual Hallucination Suite & Closed-Loop Hard-Negative DPO/ORPO Preference Pair Miner */}
      <PopeAndHardNegativeMinerPanel
        modality={pipelineTrack}
        activeModelId={
          pipelineTrack === 'VISION_VLM'
            ? vlmDashboard.selectedModel.id
            : selectedModel.id
        }
        activeModelName={
          pipelineTrack === 'VISION_VLM'
            ? vlmDashboard.selectedModel.name
            : selectedModel.name
        }
        activeVisionModel={vlmDashboard.selectedModel}
        onSendMinedPairToDatasetStudio={(draft) => {
          onSendFlaggedPromptToDatasetStudio?.(draft);
        }}
      />

      {/* Section 06C: Pinned Multi-Run Ablation Matrix & Pareto Experiment Comparison Table */}
      <MultiRunAblationMatrixPanel
        activeModality={pipelineTrack}
        activeModelId={
          pipelineTrack === 'VISION_VLM'
            ? vlmDashboard.selectedModel.id
            : selectedModel.id
        }
        activeModelName={
          pipelineTrack === 'VISION_VLM'
            ? vlmDashboard.selectedModel.name
            : selectedModel.name
        }
        activeParamsBillions={
          pipelineTrack === 'VISION_VLM'
            ? vlmDashboard.selectedModel.totalParamsBillions
            : selectedModel.paramsBillions
        }
        onRestoreAblationRunModel={(restoredId) => {
          if (pipelineTrack === 'VISION_VLM') {
            onSelectVisionModel?.(restoredId);
          } else {
            onSelectModel(restoredId);
          }
        }}
      />

      {/* Section 07: Interactive Vision Grounding & VEIL Privacy Auto-Redaction Canvas */}
      <VeilVisionRedactionCanvas />
    </div>
  );
};

