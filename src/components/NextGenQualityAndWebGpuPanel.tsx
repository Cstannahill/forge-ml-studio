import React, { useEffect, useState } from 'react';
import {
  Check,
  Copy,
  Cpu,
  Globe,
  Layers,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  ContinualReplayAndLisaConfig,
  LoftQAndQatConfig,
  LogitDistillationDivergence,
  QatTargetBitwidth,
  SpinBestOfNConfig,
  WebGpuDtype,
  WhiteBoxLogitDistillationConfig,
  buildWebGpuAndTransformersJsExportBundle,
  computeContinualReplayAndLisaPlan,
  computeLoftQAndQatEdgeHardeningPlan,
  computeSpinAndBestOfNAlignmentPlan,
  computeWhiteBoxLogitDistillationPlan,
} from '../lib/pipelineLogic';
import { InfoTooltip } from './InfoTooltip';

interface NextGenQualityAndWebGpuPanelProps {
  modelId: string;
  modelName: string;
  paramsBillions: number;
  contextWindow: number;
  loraRank: number;
  basePerplexity: number;
  onApplyNextGenQualityBoost?: (
    stageLabel: string,
    mmluGain: number,
    gsm8kGain: number,
    ifEvalGain: number
  ) => void;
}

export const NextGenQualityAndWebGpuPanel: React.FC<NextGenQualityAndWebGpuPanelProps> = ({
  modelId,
  modelName,
  paramsBillions,
  contextWindow,
  loraRank,
  basePerplexity,
  onApplyNextGenQualityBoost,
}) => {
  const [activeTab, setActiveTab] = useState<
    'spin' | 'whitebox_kd' | 'loftq_qat' | 'replay_lisa' | 'webgpu_export'
  >('spin');
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [appliedBanner, setAppliedBanner] = useState<string | null>(null);
  const [webGpuSupported, setWebGpuSupported] = useState<boolean>(false);

  useEffect(() => {
    setWebGpuSupported(typeof navigator !== 'undefined' && 'gpu' in navigator);
  }, []);

  // 1. SPIN / Best-of-N Self-Play Config
  const [spinConfig, setSpinConfig] = useState<SpinBestOfNConfig>({
    enabled: true,
    candidatesPerPrompt: 8,
    spinIterations: 2,
    samplingTemperature: 0.7,
    judgeMarginThreshold: 0.25,
  });

  // 2. White-Box Logit KD Config
  const [kdConfig, setKdConfig] = useState<WhiteBoxLogitDistillationConfig>({
    enabled: true,
    teacherModelId: 'Qwen/Qwen3.5-9B-Instruct',
    divergenceType: 'REVERSE_KL',
    temperature: 2.0,
    alphaKdWeight: 0.5,
    topKLogitsCached: 64,
  });

  // 3. LoftQ + QAT Config
  const [loftqConfig, setLoftqConfig] = useState<LoftQAndQatConfig>({
    enableLoftQInit: true,
    loftqIterSteps: 4,
    enableFakeQuantQat: true,
    qatTailStepsPct: 15,
    targetEdgeBitwidth: 'GGUF_Q3_K_M',
  });

  // 4. Continual Learning Replay Buffer + LISA Config
  const [lisaConfig, setLisaConfig] = useState<ContinualReplayAndLisaConfig>({
    replayBufferRatioPct: 10,
    replayAnchorSource: 'HuggingFaceH4/ultrachat_200k',
    enableLisaLayerSampling: true,
    lisaActiveMiddleLayers: 4,
  });

  // 5. WebGPU / Transformers.js Export Config
  const [webGpuDtype, setWebGpuDtype] = useState<WebGpuDtype>('q4f16');

  const spinPlan = computeSpinAndBestOfNAlignmentPlan({
    modelId,
    paramsBillions,
    config: spinConfig,
  });

  const kdPlan = computeWhiteBoxLogitDistillationPlan({
    studentModelId: modelId,
    studentParamsBillions: paramsBillions,
    config: kdConfig,
  });

  const loftqPlan = computeLoftQAndQatEdgeHardeningPlan({
    modelId,
    baseFp16Perplexity: basePerplexity,
    loraRank,
    config: loftqConfig,
  });

  const lisaPlan = computeContinualReplayAndLisaPlan({
    totalLayers: paramsBillions <= 0.6 ? 16 : paramsBillions <= 1.5 ? 24 : 28,
    config: lisaConfig,
  });

  const webGpuBundle = buildWebGpuAndTransformersJsExportBundle({
    modelId,
    modelName,
    paramsBillions,
    contextWindow,
    dtype: webGpuDtype,
  });

  const handleCopySnippet = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 1800);
  };

  const handleApplyBoost = (
    label: string,
    mmluGain: number,
    gsm8kGain: number,
    ifEvalGain: number
  ) => {
    if (onApplyNextGenQualityBoost) {
      onApplyNextGenQualityBoost(label, mmluGain, gsm8kGain, ifEvalGain);
    }
    setAppliedBanner(
      `Applied ${label} to ${modelName}: +${mmluGain}% MMLU-Pro, +${gsm8kGain}% GSM8K, +${ifEvalGain}% IFEval`
    );
  };

  return (
    <section className="bg-[#252423] border border-[#83A598]/50 rounded-xl p-6 space-y-5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#83A598]">
            <Sparkles className="w-4 h-4 text-[#8EC07C]" />
            Next-Gen Sub-2B Quality &amp; Edge Hardening Suite (SPIN · Reverse-KL Logit KD · LoftQ/QAT · LISA · WebGPU)
          </div>
          <h3 className="text-base font-semibold text-[#FFE9B1]">
            Frontier-to-Edge Model Quality Accelerators &amp; Zero-Server Browser WebGPU Exporter
          </h3>
          <p className="text-xs text-[#A89984] max-w-4xl">
            Combines On-Policy Self-Play (<code className="font-mono text-[#8EC07C]">SPIN / Best-of-N</code>),
            Mode-Seeking Top-K Logit Distillation (<code className="font-mono text-[#83A598]">Reverse-KL</code>),
            SVD Residual Initialization (<code className="font-mono text-[#FABD2F]">LoftQ + QAT</code>),
            and <code className="font-mono text-[#B8BB26]">LISA</code> Continual Replay.
          </p>
        </div>

        {/* 5-Tab Switcher */}
        <div className="flex flex-wrap items-center gap-1.5 bg-[#1D2021] p-1.5 rounded-xl border border-[#3C3836]">
          {[
            { id: 'spin', label: '1. SPIN / Best-of-N' },
            { id: 'whitebox_kd', label: '2. White-Box Logit KD' },
            { id: 'loftq_qat', label: '3. LoftQ + QAT Edge' },
            { id: 'replay_lisa', label: '4. Replay + LISA' },
            { id: 'webgpu_export', label: '5. WebGPU / ONNX.js' },
          ].map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setActiveTab(t.id as typeof activeTab)}
              className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors ${
                activeTab === t.id
                  ? 'bg-[#458588] text-[#FFE9B1] font-semibold'
                  : 'text-[#A89984] hover:text-[#FFE9B1]'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {appliedBanner && (
        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#B8BB26]/50 text-xs font-mono text-[#B8BB26] flex items-center justify-between">
          <span>{appliedBanner}</span>
          <button
            type="button"
            onClick={() => setAppliedBanner(null)}
            className="text-[11px] underline hover:text-[#FFE9B1]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* TAB 1: SPIN / BEST-OF-N SELF-PLAY */}
      {activeTab === 'spin' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                <RefreshCw className="w-4 h-4 text-[#8EC07C]" />
                On-Policy Rejection Sampling &amp; SPIN Self-Play
                <InfoTooltip glossaryKey="spinBestOfN" />
              </h4>
              <span className="text-xs font-mono text-[#B8BB26]">
                -{spinPlan.hallucinationLoopReductionPct}% Hallucination Loops
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">
                  Rollouts per Prompt (N)
                </label>
                <select
                  value={spinConfig.candidatesPerPrompt}
                  onChange={(e) =>
                    setSpinConfig((p) => ({
                      ...p,
                      candidatesPerPrompt: Number(e.target.value) as 4 | 8 | 16,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={4}>Best-of-4</option>
                  <option value={8}>Best-of-8 (Recommended)</option>
                  <option value={16}>Best-of-16 (Dense RL)</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">SPIN Iterations</label>
                <select
                  value={spinConfig.spinIterations}
                  onChange={(e) =>
                    setSpinConfig((p) => ({
                      ...p,
                      spinIterations: Number(e.target.value) as 1 | 2 | 3,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={1}>1 Iteration</option>
                  <option value={2}>2 Iterations (Optimal)</option>
                  <option value={3}>3 Iterations</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">Min Reward Margin</label>
                <select
                  value={spinConfig.judgeMarginThreshold}
                  onChange={(e) =>
                    setSpinConfig((p) => ({
                      ...p,
                      judgeMarginThreshold: Number(e.target.value),
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={0.15}>0.15 (Relaxed)</option>
                  <option value={0.25}>0.25 (Strict)</option>
                  <option value={0.4}>0.40 (Ultra-High Margin)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 text-center font-mono text-xs">
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Rollouts / 1K Prompts</div>
                <div className="text-[#FFE9B1] font-bold mt-0.5">
                  {spinPlan.totalGeneratedRolloutsPer1k.toLocaleString()}
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Projected IFEval Lift</div>
                <div className="text-[#8EC07C] font-bold mt-0.5">
                  +{spinPlan.projectedIfEvalGainPct}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Projected MT-Bench</div>
                <div className="text-[#B8BB26] font-bold mt-0.5">
                  +{spinPlan.projectedMtBenchGain}
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() =>
                handleApplyBoost(
                  `SPIN Best-of-${spinConfig.candidatesPerPrompt} Self-Play`,
                  1.4,
                  1.8,
                  spinPlan.projectedIfEvalGainPct
                )
              }
              className="w-full py-2.5 px-4 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Run On-Policy SPIN Self-Play Pass (+{spinPlan.projectedIfEvalGainPct}% IFEval)
            </button>
          </div>

          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#A89984]">vLLM Best-of-N Rollout + SPIN Pair Builder:</span>
              <button
                type="button"
                onClick={() => handleCopySnippet(spinPlan.spinPythonSnippet)}
                className="text-[#83A598] hover:text-[#FFE9B1] inline-flex items-center gap-1"
              >
                {copiedCode ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedCode ? 'Copied' : 'Copy Python'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto max-h-52">
              {spinPlan.spinPythonSnippet}
            </pre>
          </div>
        </div>
      )}

      {/* TAB 2: WHITE-BOX LOGIT DISTILLATION */}
      {activeTab === 'whitebox_kd' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-[#83A598]" />
                White-Box Top-K Logit Distillation (Reverse-KL)
                <InfoTooltip glossaryKey="whiteBoxLogitKd" />
              </h4>
              <span className="text-xs font-mono text-[#8EC07C]">
                {kdPlan.darkKnowledgeTransferEfficiencyPct}% Dark-Knowledge Transfer
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">Teacher Logit Source</label>
                <select
                  value={kdConfig.teacherModelId}
                  onChange={(e) =>
                    setKdConfig((p) => ({
                      ...p,
                      teacherModelId: e.target.value as WhiteBoxLogitDistillationConfig['teacherModelId'],
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value="Qwen/Qwen3.5-9B-Instruct">Qwen3.5-9B-Instruct</option>
                  <option value="Qwen/Qwen3.5-27B-Instruct">Qwen3.5-27B-Instruct</option>
                  <option value="LiquidAI/LFM2.5-1.6B-Instruct">LFM2.5-1.6B-Instruct</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">KL Objective</label>
                <select
                  value={kdConfig.divergenceType}
                  onChange={(e) =>
                    setKdConfig((p) => ({
                      ...p,
                      divergenceType: e.target.value as LogitDistillationDivergence,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value="REVERSE_KL">Reverse-KL (Mode-Seeking)</option>
                  <option value="JSD_SYMMETRIC">JSD Symmetric (Beta=0.5)</option>
                  <option value="FORWARD_KL">Forward-KL (Mean-Seeking)</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">Cached Top-K Logits</label>
                <select
                  value={kdConfig.topKLogitsCached}
                  onChange={(e) =>
                    setKdConfig((p) => ({
                      ...p,
                      topKLogitsCached: Number(e.target.value) as 32 | 64 | 128,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={32}>Top-32 Logits</option>
                  <option value={64}>Top-64 Logits (Recommended)</option>
                  <option value={128}>Top-128 Logits</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 text-center font-mono text-xs">
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Logit Cache / 10K Rows</div>
                <div className="text-[#FFE9B1] font-bold mt-0.5">
                  {kdPlan.logitCacheSizeMbPer10kSamples} MB
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">MMLU-Pro Gain</div>
                <div className="text-[#8EC07C] font-bold mt-0.5">
                  +{kdPlan.projectedMmluProGainPct}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">GSM8K Math Gain</div>
                <div className="text-[#B8BB26] font-bold mt-0.5">
                  +{kdPlan.projectedGsm8kGainPct}%
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() =>
                handleApplyBoost(
                  `White-Box ${kdConfig.divergenceType} Logit KD`,
                  kdPlan.projectedMmluProGainPct,
                  kdPlan.projectedGsm8kGainPct,
                  1.9
                )
              }
              className="w-full py-2.5 px-4 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Zap className="w-3.5 h-3.5" />
              Apply White-Box Teacher Logit Distillation (+{kdPlan.projectedGsm8kGainPct}% GSM8K)
            </button>
          </div>

          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#A89984]">PyTorch Top-K Logit Gather &amp; Reverse-KL Loss:</span>
              <button
                type="button"
                onClick={() => handleCopySnippet(kdPlan.kdTrainerPythonSnippet)}
                className="text-[#83A598] hover:text-[#FFE9B1] inline-flex items-center gap-1"
              >
                {copiedCode ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedCode ? 'Copied' : 'Copy Python'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto max-h-52">
              {kdPlan.kdTrainerPythonSnippet}
            </pre>
          </div>
        </div>
      )}

      {/* TAB 3: LOFTQ + QAT EDGE HARDENING */}
      {activeTab === 'loftq_qat' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-[#FABD2F]" />
                LoftQ SVD Residual Init &amp; Fake-Quantization QAT
                <InfoTooltip glossaryKey="loftqQat" />
              </h4>
              <span className="text-xs font-mono text-[#B8BB26]">
                -{loftqPlan.svdResidualErrorReductionPct}% Quant Error
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">
                  Target Ultra-Low-Bit Edge Format
                </label>
                <select
                  value={loftqConfig.targetEdgeBitwidth}
                  onChange={(e) =>
                    setLoftqConfig((p) => ({
                      ...p,
                      targetEdgeBitwidth: e.target.value as QatTargetBitwidth,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value="INT4_WEBGPU">INT4 WebGPU (q4f16)</option>
                  <option value="GGUF_Q3_K_M">GGUF Q3_K_M (3.4-bit Edge)</option>
                  <option value="GGUF_IQ3_XXS">GGUF IQ3_XXS (3.06-bit Mobile)</option>
                  <option value="GGUF_Q2_K">GGUF Q2_K (2.6-bit Extreme)</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">
                  LoftQ Alternating SVD Iterations
                </label>
                <select
                  value={loftqConfig.loftqIterSteps}
                  onChange={(e) =>
                    setLoftqConfig((p) => ({
                      ...p,
                      loftqIterSteps: Number(e.target.value) as 1 | 2 | 4,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={1}>1 SVD Pass</option>
                  <option value={2}>2 SVD Passes</option>
                  <option value={4}>4 SVD Passes (Recommended)</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 text-center font-mono text-xs">
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Naive PTQ PPL</div>
                <div className="text-[#FB4934] font-bold mt-0.5">
                  {loftqPlan.ptqBaselinePerplexity}
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">LoftQ + QAT PPL</div>
                <div className="text-[#8EC07C] font-bold mt-0.5">
                  {loftqPlan.loftqQatRecoveredPerplexity}
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Edge Retention</div>
                <div className="text-[#B8BB26] font-bold mt-0.5">
                  {loftqPlan.edgeAccuracyRetentionPct}%
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => handleApplyBoost('LoftQ SVD + QAT Edge Hardening', 1.6, 2.1, 1.5)}
              className="w-full py-2.5 px-4 text-xs font-semibold text-[#1D2021] bg-[#FABD2F] hover:bg-[#fcc93d] rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <ShieldCheck className="w-4 h-4" />
              Run LoftQ SVD Init + QAT Tail Pass ({loftqPlan.edgeAccuracyRetentionPct}% Retention)
            </button>
          </div>

          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#A89984]">PEFT LoftQConfig Initialization Snippet:</span>
              <button
                type="button"
                onClick={() => handleCopySnippet(loftqPlan.loftqPythonSnippet)}
                className="text-[#83A598] hover:text-[#FFE9B1] inline-flex items-center gap-1"
              >
                {copiedCode ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedCode ? 'Copied' : 'Copy Python'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto max-h-52">
              {loftqPlan.loftqPythonSnippet}
            </pre>
          </div>
        </div>
      )}

      {/* TAB 4: CONTINUAL REPLAY BUFFER + LISA */}
      {activeTab === 'replay_lisa' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-[#B8BB26]" />
                Anti-Forgetting Replay Buffer &amp; LISA Layer Sampling
                <InfoTooltip glossaryKey="continualReplayLisa" />
              </h4>
              <span className="text-xs font-mono text-[#8EC07C]">
                Forgetting Risk: {lisaPlan.catastrophicForgettingRisk}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">
                  General Instruction Replay Mix (%)
                </label>
                <select
                  value={lisaConfig.replayBufferRatioPct}
                  onChange={(e) =>
                    setLisaConfig((p) => ({
                      ...p,
                      replayBufferRatioPct: Number(e.target.value),
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={0}>0% (No Replay Anchor)</option>
                  <option value={5}>5% Anchor Mix</option>
                  <option value={10}>10% Anchor Mix (Recommended)</option>
                  <option value={15}>15% Anchor Mix</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">
                  LISA Sampled Active Middle Layers (k)
                </label>
                <select
                  value={lisaConfig.lisaActiveMiddleLayers}
                  onChange={(e) =>
                    setLisaConfig((p) => ({
                      ...p,
                      lisaActiveMiddleLayers: Number(e.target.value) as 2 | 4 | 6,
                    }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={2}>k = 2 Layers (Max VRAM Save)</option>
                  <option value={4}>k = 4 Layers (Balanced)</option>
                  <option value={6}>k = 6 Layers</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2.5 text-center font-mono text-xs">
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">General Retention</div>
                <div className="text-[#8EC07C] font-bold mt-0.5">
                  {lisaPlan.generalCapabilityRetentionPct}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Optimizer VRAM Saved</div>
                <div className="text-[#83A598] font-bold mt-0.5">
                  -{lisaPlan.lisaOptimizerVramSavingsPct}%
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Domain Plasticity</div>
                <div className="text-[#B8BB26] font-bold mt-0.5">
                  {lisaPlan.domainPlasticityScorePct}%
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={() => handleApplyBoost('Continual Replay + LISA Optimizer', 1.9, 1.5, 2.2)}
              className="w-full py-2.5 px-4 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              Activate {lisaConfig.replayBufferRatioPct}% Replay Buffer + LISA Layer Sampler
            </button>
          </div>

          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#A89984]">LISA Dynamic Layer Sampling Callback:</span>
              <button
                type="button"
                onClick={() => handleCopySnippet(lisaPlan.lisaPythonSnippet)}
                className="text-[#83A598] hover:text-[#FFE9B1] inline-flex items-center gap-1"
              >
                {copiedCode ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedCode ? 'Copied' : 'Copy Python'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto max-h-52">
              {lisaPlan.lisaPythonSnippet}
            </pre>
          </div>
        </div>
      )}

      {/* TAB 5: BROWSER-NATIVE WEBGPU / ONNX / TRANSFORMERS.JS EXPORTER */}
      {activeTab === 'webgpu_export' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                <Globe className="w-4 h-4 text-[#83A598]" />
                Browser-Native WebGPU / ONNX &amp; Transformers.js Exporter
                <InfoTooltip glossaryKey="webGpuExport" />
              </h4>
              <span className="text-xs font-mono text-[#8EC07C]">
                {webGpuSupported ? 'navigator.gpu Detected' : 'WebGPU Ready Target'}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2.5">
              {(['q4f16', 'uint8', 'fp16'] as const).map((dt) => (
                <button
                  key={dt}
                  type="button"
                  onClick={() => setWebGpuDtype(dt)}
                  className={`py-2 px-3 rounded-lg border text-xs font-mono transition-colors ${
                    webGpuDtype === dt
                      ? 'bg-[#458588] border-[#83A598] text-[#FFE9B1] font-bold'
                      : 'bg-[#252423] border-[#3C3836] text-[#A89984] hover:text-[#FFE9B1]'
                  }`}
                >
                  dtype: {dt}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3 text-center font-mono text-xs">
              <div className="p-3 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Est. Browser Tab VRAM</div>
                <div className="text-sm font-bold text-[#FFE9B1] mt-0.5">
                  {webGpuBundle.estimatedBrowserVramMb} MB
                </div>
              </div>
              <div className="p-3 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Est. WebGPU Speed</div>
                <div className="text-sm font-bold text-[#8EC07C] mt-0.5">
                  {webGpuBundle.estimatedWebGpuTokPerSec} tok/s
                </div>
              </div>
            </div>

            <pre className="p-2.5 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#8EC07C] overflow-x-auto">
              {webGpuBundle.optimumOnnxCliCommand}
            </pre>
          </div>

          <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-[#A89984]">Transformers.js v3 In-Browser WebGPU Pipeline:</span>
              <button
                type="button"
                onClick={() => handleCopySnippet(webGpuBundle.transformersJsSnippet)}
                className="text-[#83A598] hover:text-[#FFE9B1] inline-flex items-center gap-1"
              >
                {copiedCode ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copiedCode ? 'Copied' : 'Copy JS'}
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto max-h-52">
              {webGpuBundle.transformersJsSnippet}
            </pre>
          </div>
        </div>
      )}
    </section>
  );
};
