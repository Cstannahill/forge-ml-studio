import React, { useMemo, useState } from 'react';
import {
  Cpu,
  Layers,
  Zap,
  Terminal,
  Copy,
  Check,
  Sparkles,
  Sliders,
  Database,
  ArrowRight,
  GitBranch,
} from 'lucide-react';
import {
  HyperparameterConfig,
  Lfm3LayerType,
  Lfm3SamplingModeId,
  computeLfm3TriHybridArchitecturePlan,
} from '../lib/pipelineLogic';
import {
  CURATED_DATASET_BLENDS,
  CuratedDatasetBlend,
} from '../data/curatedDatasetBlends';

interface Lfm3TriHybridStudioPanelProps {
  selectedModelId: string;
  activeHpoConfig: HyperparameterConfig;
  onSelectModel: (modelId: string) => void;
  onApplyHpoConfig: (nextConfig: HyperparameterConfig) => void;
  onApplyCuratedBlend?: (blend: CuratedDatasetBlend) => void;
}

export const Lfm3TriHybridStudioPanel: React.FC<Lfm3TriHybridStudioPanelProps> = ({
  selectedModelId,
  activeHpoConfig,
  onSelectModel,
  onApplyHpoConfig,
  onApplyCuratedBlend,
}) => {
  const [samplingMode, setSamplingMode] =
    useState<Lfm3SamplingModeId>('GENERAL_AGENT_TOOL');
  const [layerFilter, setLayerFilter] = useState<'ALL' | Lfm3LayerType>('ALL');
  const [selectedCheckpointTag, setSelectedCheckpointTag] = useState<string>('GGUF');
  const [selectedLfmBlendId, setSelectedLfmBlendId] = useState<string>(
    'blend-33-lfm3-agentic-harness-mcp-openclaw'
  );
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [applyBanner, setApplyBanner] = useState<string | null>(null);

  const plan = useMemo(
    () =>
      computeLfm3TriHybridArchitecturePlan({
        loraRank: activeHpoConfig.loraRank,
        loraAlpha: activeHpoConfig.loraAlpha,
        useDora: true,
        samplingMode,
      }),
    [activeHpoConfig.loraRank, activeHpoConfig.loraAlpha, samplingMode]
  );

  const lfm3CompatibleBlends = useMemo(
    () =>
      CURATED_DATASET_BLENDS.filter((b) =>
        b.primaryTargetModels.includes('LFM3-1.7B')
      ),
    []
  );

  const activeBlend = useMemo(
    () =>
      lfm3CompatibleBlends.find((b) => b.id === selectedLfmBlendId) ??
      lfm3CompatibleBlends[0],
    [lfm3CompatibleBlends, selectedLfmBlendId]
  );

  const filteredLayers = useMemo(
    () =>
      layerFilter === 'ALL'
        ? plan.layerStack
        : plan.layerStack.filter((l) => l.layerType === layerFilter),
    [plan.layerStack, layerFilter]
  );

  const activeCheckpoint = useMemo(
    () =>
      plan.officialCheckpoints.find((c) => c.formatTag === selectedCheckpointTag) ??
      plan.officialCheckpoints[1],
    [plan.officialCheckpoints, selectedCheckpointTag]
  );

  const isLfm3Selected = selectedModelId === 'LiquidAI/LFM3-1.7B';

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleActivateLfm3Profile = () => {
    onSelectModel('LiquidAI/LFM3-1.7B');
    if (activeBlend) {
      onApplyHpoConfig(activeBlend.recommendedHpo);
      if (onApplyCuratedBlend) {
        onApplyCuratedBlend(activeBlend);
      }
      setApplyBanner(
        `Activated LiquidAI/LFM3-1.7B (24L Tri-Hybrid · 256K Context) + "${activeBlend.title}" (DoRA r=${activeBlend.recommendedHpo.loraRank}, α=${activeBlend.recommendedHpo.loraAlpha}, lr=${activeBlend.recommendedHpo.learningRate})`
      );
    } else {
      setApplyBanner(
        'Activated LiquidAI/LFM3-1.7B (24L Tri-Hybrid · 256K Context) with w1/w2/w3 + in_proj/out_proj DoRA target modules.'
      );
    }
    setTimeout(() => setApplyBanner(null), 5000);
  };

  return (
    <div
      data-testid="lfm3-tri-hybrid-studio-panel"
      className="bg-[#111827] border border-cyan-500/30 rounded-lg p-5 space-y-5"
    >
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 flex items-center gap-1">
              <Cpu className="w-3.5 h-3.5" />
              NEW ARCHITECTURE · Lfm3ForCausalLM
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
              13 ShortConv + 5 SWA-256 + 6 Global Attn
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-blue-500/15 text-blue-300 border border-blue-500/30">
              256K Context · ~1.0 GB 4-Bit · {lfm3CompatibleBlends.length}/34 Blends Ready
            </span>
          </div>
          <h3 className="text-base font-semibold text-slate-100 mt-1.5 flex items-center gap-2">
            <span>
              LiquidAI/LFM3-1.7B Tri-Hybrid Architecture, PEFT Target Map &amp; 6-Teacher MOPD Studio
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            First-class support for <code className="text-cyan-300 font-mono">LiquidAI/LFM3-1.7B</code> (1.73B params,{' '}
            <code className="text-slate-300 font-mono">max_position_embeddings=262,144</code>,{' '}
            <code className="text-slate-300 font-mono">use_sandwich_norm=true</code>): prefills{' '}
            <span className="text-emerald-300 font-medium">2.5x faster</span> and decodes{' '}
            <span className="text-emerald-300 font-medium">1.8x faster</span> at 32K context than 2B–4B dense transformers.
          </p>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            data-testid="activate-lfm3-model-btn"
            onClick={handleActivateLfm3Profile}
            className={`px-3.5 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-colors cursor-pointer ${
              isLfm3Selected
                ? 'bg-emerald-600/20 text-emerald-300 border border-emerald-500/40 hover:bg-emerald-600/30'
                : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-sm'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>
              {isLfm3Selected
                ? 'LFM3-1.7B Active (Apply Selected Blend & HPO)'
                : 'Activate LiquidAI/LFM3-1.7B + Apply Recipe'}
            </span>
          </button>
        </div>
      </div>

      {applyBanner && (
        <div className="p-3 rounded-lg bg-emerald-950/50 border border-emerald-500/40 text-xs text-emerald-200 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Check className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{applyBanner}</span>
          </div>
        </div>
      )}

      {/* Top KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
        <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
          <div className="text-[11px] text-slate-400">24-Layer Tri-Hybrid</div>
          <div className="text-sm font-mono font-semibold text-cyan-300 mt-0.5">
            13 Conv · 5 SWA · 6 Full
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            conv_L_cache=7 · SWA=256
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
          <div className="text-[11px] text-slate-400">Context Window</div>
          <div className="text-sm font-mono font-semibold text-emerald-300 mt-0.5">
            262,144 (256K)
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            81.2% KV savings @ 32K
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
          <div className="text-[11px] text-slate-400">SwiGLU + ShortConv</div>
          <div className="text-sm font-mono font-semibold text-slate-100 mt-0.5">
            2048 &rarr; 7680
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            w1, w2, w3 + in/out_proj
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
          <div className="text-[11px] text-slate-400">Attention GQA + Gate</div>
          <div className="text-sm font-mono font-semibold text-blue-300 mt-0.5">
            16Q / 4KV + gate_proj
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            Head-wise sigmoid gate + Q/K Norm
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
          <div className="text-[11px] text-slate-400">Vocab &amp; Sandwich Norm</div>
          <div className="text-sm font-mono font-semibold text-amber-300 mt-0.5">
            128,000 Tied
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            BOS=124894 · EOS=124900
          </div>
        </div>
        <div className="p-3 rounded-lg bg-[#0B0F17] border border-slate-800">
          <div className="text-[11px] text-slate-400">4-Bit Edge Footprint</div>
          <div className="text-sm font-mono font-semibold text-purple-300 mt-0.5">
            ~1.00 GB RAM
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            GGUF / NVFP4 / MLX-4bit
          </div>
        </div>
      </div>

      {/* Prebuilt Dataset & Fine-Tuning Selector for ALL 34 Applicable Blends */}
      <div className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span>
                Pre-Built LFM3-1.7B Dataset &amp; Fine-Tuning Configurations ({lfm3CompatibleBlends.length} All-Purpose Blends)
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Select any of the {lfm3CompatibleBlends.length} task-specific dataset blends (Agentic MCP/OpenClaw, 6-Teacher MOPD RAG, Prompt Enhancement, Code/SQL, JSON Schema, Math, Security) to configure <code className="text-cyan-300 font-mono">LiquidAI/LFM3-1.7B</code> in one click.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              aria-label="Select LFM3-1.7B Pre-Built Dataset Blend"
              data-testid="lfm3-blend-selector"
              value={selectedLfmBlendId}
              onChange={(e) => setSelectedLfmBlendId(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-500 max-w-md"
            >
              {lfm3CompatibleBlends.map((blend) => (
                <option key={blend.id} value={blend.id}>
                  #{blend.blendNumber} · [{blend.taskCategory}] {blend.title}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={handleActivateLfm3Profile}
              className="px-3 py-1.5 rounded bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 border border-cyan-500/40 text-xs font-mono font-semibold flex items-center gap-1 shrink-0 cursor-pointer"
            >
              <span>Load Config</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {activeBlend && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 pt-2 border-t border-slate-800/80">
            <div className="lg:col-span-2 space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  Blend #{activeBlend.blendNumber} · {activeBlend.taskCategory}
                </span>
                <span className="text-xs font-semibold text-slate-100">
                  {activeBlend.title}
                </span>
              </div>
              <p className="text-xs text-slate-400">{activeBlend.specificTaskFocus}</p>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {activeBlend.datasets.map((ds) => (
                  <span
                    key={ds.datasetId}
                    className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300"
                  >
                    {ds.datasetId} ({ds.weightPct}%)
                  </span>
                ))}
              </div>
            </div>
            <div className="p-2.5 rounded bg-slate-900/70 border border-slate-800 flex flex-col justify-between text-[11px] font-mono">
              <div className="text-slate-400">LFM3-1.7B Tuned HPO Preset:</div>
              <div className="grid grid-cols-2 gap-1 text-slate-200 mt-1">
                <span>DoRA Rank: r={activeBlend.recommendedHpo.loraRank}</span>
                <span>Alpha: α={activeBlend.recommendedHpo.loraAlpha}</span>
                <span>LR: {activeBlend.recommendedHpo.learningRate}</span>
                <span>Sched: {activeBlend.recommendedHpo.lrScheduler.toUpperCase()}</span>
                <span>Ctx: {activeBlend.recommendedContextLen.toLocaleString()}</span>
                <span>QLoRA VRAM: ~2.3 GB</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Middle Grid: 24-Layer Tri-Hybrid Stack + 6-Teacher MOPD & Agentic Harnesses */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: 24-Layer Tri-Hybrid Stack Inspector */}
        <div className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                <span>24-Layer Tri-Hybrid Stack (`config.json` `layer_types`)</span>
              </h4>
              <p className="text-[11px] text-slate-400">
                10 PEFT linear targets:{' '}
                <code className="text-cyan-300 font-mono">
                  {plan.peftTargetModules.join(', ')}
                </code>
              </p>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 p-1 rounded border border-slate-800">
              {(
                [
                  { id: 'ALL', label: 'All (24)' },
                  { id: 'conv', label: 'ShortConv (13)' },
                  { id: 'sliding_attention', label: 'SWA-256 (5)' },
                  { id: 'full_attention', label: 'Full-256K (6)' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setLayerFilter(tab.id)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors cursor-pointer ${
                    layerFilter === tab.id
                      ? 'bg-cyan-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* 24-Layer Visual Strip */}
          <div className="grid grid-cols-12 gap-1 pt-1">
            {plan.layerStack.map((layer) => {
              const badgeColor =
                layer.layerType === 'conv'
                  ? 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300'
                  : layer.layerType === 'sliding_attention'
                  ? 'bg-amber-500/20 border-amber-500/40 text-amber-300'
                  : 'bg-cyan-500/25 border-cyan-400/50 text-cyan-200';
              const shortTag =
                layer.layerType === 'conv'
                  ? 'CONV'
                  : layer.layerType === 'sliding_attention'
                  ? 'SWA'
                  : 'FULL';
              return (
                <div
                  key={layer.layerIndex}
                  title={`Layer ${layer.layerIndex}: ${layer.blockName} (${layer.ropeOrCacheSpec})`}
                  className={`p-1 rounded border text-center font-mono text-[9px] leading-tight ${badgeColor}`}
                >
                  <div className="opacity-75">L{layer.layerIndex}</div>
                  <div className="font-bold">{shortTag}</div>
                </div>
              );
            })}
          </div>

          {/* Filtered Layer Table */}
          <div className="max-h-48 overflow-y-auto border border-slate-800/80 rounded divide-y divide-slate-800/60">
            {filteredLayers.map((l) => (
              <div
                key={l.layerIndex}
                className="px-2.5 py-1.5 flex items-center justify-between gap-2 text-[11px] bg-slate-900/40"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-mono font-semibold text-slate-300 w-7">
                    L{l.layerIndex}
                  </span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                      l.layerType === 'conv'
                        ? 'bg-emerald-500/15 text-emerald-300'
                        : l.layerType === 'sliding_attention'
                        ? 'bg-amber-500/15 text-amber-300'
                        : 'bg-cyan-500/20 text-cyan-300'
                    }`}
                  >
                    {l.layerType}
                  </span>
                  <span className="text-slate-300 truncate">{l.ropeOrCacheSpec}</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400 shrink-0">
                  {l.kvStateComplexity}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: 4-Stage Post-Training (SFT -> Mixed RL -> 6-Teacher MOPD -> Agentic RL) */}
        <div className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5 text-emerald-400" />
                <span>
                  LFM3 4-Stage Post-Training: 6-Teacher MOPD &amp; Agentic RL Harnesses
                </span>
              </h4>
              <p className="text-[11px] text-slate-400">
                1.75T SFT &rarr; Mixed RL (Length Penalty) &rarr; 6-Teacher MOPD &rarr; Agentic RL (Hermes / Pi / OpenClaw)
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {plan.mopdTeachers.map((t) => (
              <div
                key={t.teacherId}
                className="p-2 rounded bg-slate-900/70 border border-slate-800 space-y-0.5"
              >
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-semibold text-slate-200">{t.domainName}</span>
                  <span className="font-mono text-cyan-300">{t.weightPct}%</span>
                </div>
                <div className="text-[10px] font-mono text-emerald-400">
                  {t.targetBenchmark}
                </div>
                <div className="text-[10px] text-slate-400 leading-snug">
                  {t.distillationObjective}
                </div>
              </div>
            ))}
          </div>

          {/* Agentic Harnesses Row */}
          <div className="grid grid-cols-3 gap-2 pt-1">
            {plan.agenticHarnessesSupported.map((h) => (
              <div
                key={h.harnessName}
                className="p-2 rounded bg-slate-900/90 border border-cyan-500/20 text-[11px]"
              >
                <div className="font-mono font-semibold text-cyan-300">
                  {h.harnessName}
                </div>
                <div className="text-[10px] text-slate-400">
                  Horizon: {h.multiTurnToolHorizon} turns · +{h.passRateGainPct}%
                </div>
                <div className="text-[10px] font-mono text-slate-500 truncate mt-0.5">
                  {h.targetBenchmarks.join(' · ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Row: Official Checkpoints + Sampling Preset + PEFT/MOPD Code */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Official 6 Checkpoints & Sampling Selector */}
        <div className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-blue-400" />
              <span>Official Checkpoints &amp; Recommended Sampling Parameters</span>
            </div>
            <div className="flex items-center gap-1 bg-slate-900 p-1 rounded border border-slate-800">
              <button
                type="button"
                onClick={() => setSamplingMode('GENERAL_AGENT_TOOL')}
                className={`px-2 py-0.5 rounded text-[10px] font-mono cursor-pointer ${
                  samplingMode === 'GENERAL_AGENT_TOOL'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                General / Agents (T=0.1)
              </button>
              <button
                type="button"
                onClick={() => setSamplingMode('MATH_STEM_REASONING')}
                className={`px-2 py-0.5 rounded text-[10px] font-mono cursor-pointer ${
                  samplingMode === 'MATH_STEM_REASONING'
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Math &amp; STEM (T=1.0)
              </button>
            </div>
          </div>

          <div className="p-2.5 rounded bg-slate-900/80 border border-slate-800 text-xs space-y-1">
            <div className="flex flex-wrap items-center gap-3 font-mono text-[11px] text-cyan-300">
              <span>temperature={plan.samplingPreset.temperature}</span>
              <span>top_k={plan.samplingPreset.topK}</span>
              <span>top_p={plan.samplingPreset.topP}</span>
              <span>repetition_penalty={plan.samplingPreset.repetitionPenalty}</span>
            </div>
            <p className="text-[11px] text-slate-400">{plan.samplingPreset.rationale}</p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {plan.officialCheckpoints.map((cp) => (
              <button
                key={cp.formatTag}
                type="button"
                onClick={() => setSelectedCheckpointTag(cp.formatTag)}
                className={`px-2.5 py-1 rounded text-[11px] font-mono border transition-colors cursor-pointer ${
                  selectedCheckpointTag === cp.formatTag
                    ? 'bg-cyan-600/25 border-cyan-400 text-cyan-200 font-semibold'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {cp.formatTag} ({cp.footprintGb} GB)
              </button>
            ))}
          </div>

          <div className="p-2.5 rounded bg-slate-950 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-emerald-300">{activeCheckpoint.repoId}</span>
              <span className="text-slate-400">{activeCheckpoint.targetRuntimes}</span>
            </div>
            <div className="flex items-center justify-between gap-2 bg-slate-900 px-2.5 py-1.5 rounded border border-slate-800 font-mono text-[11px] text-slate-200">
              <code className="truncate">{activeCheckpoint.launchCli}</code>
              <button
                type="button"
                onClick={() => handleCopy('cli', activeCheckpoint.launchCli)}
                className="text-slate-400 hover:text-slate-200 shrink-0 cursor-pointer"
                title="Copy launch command"
              >
                {copiedId === 'cli' ? (
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Generated LFM3 PEFT DoRA Python Script */}
        <div className="p-4 rounded-lg bg-[#0B0F17] border border-slate-800 space-y-2.5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-emerald-400" />
              <span>
                LFM3-1.7B Tri-Hybrid PEFT / DoRA Target Script (`w1,w2,w3` + `in/out_proj` + `gate_proj`)
              </span>
            </div>
            <button
              type="button"
              onClick={() => handleCopy('script', plan.peftAndMopdTrainingScriptPy)}
              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-mono text-slate-200 flex items-center gap-1 cursor-pointer"
            >
              {copiedId === 'script' ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Script</span>
                </>
              )}
            </button>
          </div>
          <pre className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-56 leading-relaxed">
            {plan.peftAndMopdTrainingScriptPy}
          </pre>
        </div>
      </div>
    </div>
  );
};
