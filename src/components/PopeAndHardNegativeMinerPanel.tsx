import React, { useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Copy,
  Database,
  Eye,
  GitCompare,
  Plus,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import {
  ModalityTrack,
  mineHardNegativePreferencePairs,
} from '../lib/pipelineLogic';
import {
  INITIAL_VISION_MODELS,
  RegisteredVisionModel,
  evaluatePopeVisualHallucinationSuite,
} from '../lib/visionPipelineLogic';

interface PopeAndHardNegativeMinerPanelProps {
  modality: ModalityTrack;
  activeModelId: string;
  activeModelName: string;
  activeVisionModel?: RegisteredVisionModel;
  onSendMinedPairToDatasetStudio?: (draft: {
    instruction: string;
    output: string;
    domainTag: string;
  }) => void;
}

export const PopeAndHardNegativeMinerPanel: React.FC<
  PopeAndHardNegativeMinerPanelProps
> = ({
  modality,
  activeModelId,
  activeModelName,
  activeVisionModel,
  onSendMinedPairToDatasetStudio,
}) => {
  const visionModel =
    activeVisionModel ??
    INITIAL_VISION_MODELS.find((m) => m.id === activeModelId) ??
    INITIAL_VISION_MODELS[4]; // Default to Gemma-4-E2B-Vision-IT

  const popeReport = evaluatePopeVisualHallucinationSuite(visionModel);

  const [alignmentMethod, setAlignmentMethod] = useState<'ORPO' | 'DPO' | 'SimPO' | 'KTO'>(
    'ORPO'
  );
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [customChosen, setCustomChosen] = useState<string>('');
  const [customRejected, setCustomRejected] = useState<string>('');
  const [submittedCustomPair, setSubmittedCustomPair] = useState<{
    prompt: string;
    chosen: string;
    rejected: string;
  } | null>(null);
  const [minerNotice, setMinerNotice] = useState<string | null>(null);
  const [copiedJsonl, setCopiedJsonl] = useState<boolean>(false);

  const miningReport = mineHardNegativePreferencePairs({
    modelId: activeModelId,
    modelName: activeModelName,
    modality,
    alignmentMethod,
    customPrompt: submittedCustomPair?.prompt,
    customChosen: submittedCustomPair?.chosen,
    customRejected: submittedCustomPair?.rejected,
  });

  return (
    <div className="space-y-6">
      {/* PART 1: POPE (Polling-Based Object Probing Evaluation) Visual Hallucination Suite */}
      <section
        aria-label="POPE Adversarial Visual Hallucination Benchmark Suite"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
      >
        <div className="flex flex-col gap-3 pb-4 border-b border-slate-800/80">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2.5 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30">
                POPE Object &amp; OCR Hallucination Benchmark
              </span>
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 font-semibold">
                -{popeReport.hallucinationReductionPct}% Object Hallucination Rate
              </span>
            </div>

            <div className="text-xs font-mono text-slate-400">
              Evaluated VLM: <strong className="text-slate-100">{visionModel.name}</strong>
            </div>
          </div>

          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Eye className="w-4 h-4 text-purple-400 shrink-0" />
              <span>
                POPE (Polling-Based Object Probing) Adversarial Visual Hallucination Benchmark Suite
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Measures whether the VLM hallucinates nonexistent objects or unreadable fine-print PII when asked leading adversarial questions (<code className="font-mono text-slate-300">Random</code>, <code className="font-mono text-slate-300">Popular Co-Occurrence</code>, and <code className="font-mono text-slate-300">Adversarial Trap</code> splits).
            </p>
          </div>
        </div>

        {/* 4-Card POPE Telemetry Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs tabular-nums">
          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">POPE F1 Score (Base → FT)</div>
            <div className="text-base font-bold text-emerald-400 mt-1">
              {popeReport.baseF1ScorePct}% → {popeReport.fineTunedF1ScorePct}%
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              +{(popeReport.fineTunedF1ScorePct - popeReport.baseF1ScorePct).toFixed(1)}% F1 Lift
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">Abstention Precision</div>
            <div className="text-base font-bold text-blue-300 mt-1">
              {popeReport.basePrecisionPct}% → {popeReport.fineTunedPrecisionPct}%
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Zero phantom Rx/barcode claims
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">&ldquo;Yes&rdquo;-Bias Calibration</div>
            <div className="text-base font-bold text-amber-300 mt-1">
              {popeReport.baseYesBiasRatioPct}% → {popeReport.fineTunedYesBiasRatioPct}%
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Ideal calibrated balance: 50.0%
            </div>
          </div>

          <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3">
            <div className="text-[11px] text-slate-400">Hallucination Suppression</div>
            <div className="text-base font-bold text-purple-300 mt-1">
              -{popeReport.hallucinationReductionPct}% False Alarms
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              Verified on VEIL Task 5 Negatives
            </div>
          </div>
        </div>

        {/* POPE Probes Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {popeReport.probes.map((probe) => (
            <div
              key={probe.id}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3"
            >
              <div className="space-y-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-100">
                    {probe.sceneTitle}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-500/15 text-purple-300">
                    {probe.split}
                  </span>
                </div>

                <div className="text-[11px] font-mono text-slate-400">
                  Probed Absent Artifact:{' '}
                  <strong className="text-amber-300">{probe.probedNonExistentObject}</strong> ·
                  Co-occurring priors: {probe.coOccurringContextObjects.join(', ')}
                </div>

                <div className="p-2.5 rounded bg-[#111827] border border-slate-800 text-xs font-mono text-slate-200">
                  Q: &ldquo;{probe.question}&rdquo;
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono pt-1">
                  <div className="p-2 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300">
                    Base VLM: <strong>{probe.baseModelAnswer}</strong>
                  </div>
                  <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
                    Fine-Tuned VLM: <strong>{probe.fineTunedModelAnswer}</strong>
                  </div>
                </div>
              </div>

              <div className="pt-2.5 border-t border-slate-800 flex items-center justify-between gap-2">
                <span className="text-[10px] font-mono text-slate-400">
                  Ground Truth: {probe.groundTruthAnswer} (SAFE_PASS)
                </span>
                <button
                  type="button"
                  onClick={() => {
                    onSendMinedPairToDatasetStudio?.({
                      instruction: probe.negativeTrainingPairDraft.instruction,
                      output: `<think>\n${probe.negativeTrainingPairDraft.assistantThinking}\n</think>\n${probe.negativeTrainingPairDraft.assistantOutput}`,
                      domainTag: probe.negativeTrainingPairDraft.domainTag,
                    });
                    setMinerNotice(
                      `Injected POPE negative abstention sample "${probe.sceneTitle}" into Dataset Studio.`
                    );
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono text-emerald-300 bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/30 rounded transition-colors"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Mine Abstention Pair → Dataset Studio</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* PART 2: Closed-Loop Hard-Negative DPO/ORPO Preference Pair Miner */}
      <section
        aria-label="Closed-Loop Hard-Negative DPO and ORPO Preference Pair Miner"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
      >
        <div className="flex flex-col gap-3 pb-4 border-b border-slate-800/80">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-mono uppercase tracking-wider px-2.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                Closed-Loop Evaluation → Alignment Flywheel
              </span>
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded bg-slate-800 text-amber-300">
                {miningReport.minedPairs.length} Contrastive Pairs Ready · +{miningReport.projectedIfEvalLiftPct}% IFEval Lift
              </span>
            </div>

            <div className="flex items-center gap-1.5 bg-[#0B0F17] p-1 rounded-lg border border-slate-800">
              {(['ORPO', 'DPO', 'SimPO', 'KTO'] as const).map((method) => (
                <button
                  key={method}
                  type="button"
                  onClick={() => setAlignmentMethod(method)}
                  className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                    alignmentMethod === method
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {method}
                </button>
              ))}
            </div>
          </div>

          <div>
            <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                Closed-Loop Hard-Negative {alignmentMethod} Preference Pair Miner ({activeModelName})
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Automatically harvests failed Base-Model outputs from the Evaluation Arena, Red-Team Jailbreak Probes, and POPE Hallucination Traps into structured <code className="font-mono text-slate-300">&lt;prompt, chosen, rejected&gt;</code> preference triples for Stage 2 {alignmentMethod} alignment.
            </p>
          </div>
        </div>

        {minerNotice && (
          <div
            role="status"
            className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300 flex items-center justify-between"
          >
            <span>{minerNotice}</span>
            <button
              type="button"
              onClick={() => setMinerNotice(null)}
              className="text-slate-400 hover:text-slate-200 text-[11px]"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Custom Contrastive Pair Builder */}
        <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-3">
          <div className="text-xs font-semibold text-slate-200">
            Mine Custom Contrastive Pair (Prompt + Chosen Verification + Rejected Failure)
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <div>
              <label className="block text-[11px] font-mono text-slate-400 mb-1">
                1. Evaluation Prompt
              </label>
              <input
                type="text"
                aria-label="Custom Preference Prompt"
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="e.g., Extract invoice total into strict JSON..."
                className="w-full bg-[#111827] border border-slate-800 rounded px-3 py-1.5 text-xs font-mono text-slate-100"
              />
            </div>
            <div>
              <label className="block text-[11px] font-mono text-emerald-400 mb-1">
                2. Chosen Target Response (y_w)
              </label>
              <input
                type="text"
                aria-label="Custom Chosen Response"
                value={customChosen}
                onChange={(e) => setCustomChosen(e.target.value)}
                placeholder='e.g., <think>Verify JSON</think>{"total": 42.50}'
                className="w-full bg-[#111827] border border-emerald-500/40 rounded px-3 py-1.5 text-xs font-mono text-emerald-200"
              />
            </div>
            <div>
              <label className="block text-[11px] font-mono text-amber-400 mb-1">
                3. Rejected Hard-Negative (y_l)
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  aria-label="Custom Rejected Response"
                  value={customRejected}
                  onChange={(e) => setCustomRejected(e.target.value)}
                  placeholder="e.g., Sure! The total is $42.50."
                  className="flex-1 bg-[#111827] border border-amber-500/40 rounded px-3 py-1.5 text-xs font-mono text-amber-200"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (!customPrompt.trim() || !customChosen.trim() || !customRejected.trim()) {
                      return;
                    }
                    setSubmittedCustomPair({
                      prompt: customPrompt,
                      chosen: customChosen,
                      rejected: customRejected,
                    });
                    setMinerNotice(
                      `Mined new custom ${alignmentMethod} preference pair into Stage 2 JSONL buffer.`
                    );
                  }}
                  className="px-3 py-1.5 text-xs font-mono font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded shrink-0"
                >
                  + Mine Pair
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Mined Preference Pairs List */}
        <div className="space-y-3">
          {miningReport.minedPairs.map((pair) => (
            <div
              key={pair.id}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-2.5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 font-mono text-xs">
                  <span className="px-2 py-0.5 rounded bg-blue-500/15 text-blue-300">
                    {pair.sourceType}
                  </span>
                  <span className="text-slate-300 font-semibold">#{pair.domainTag}</span>
                  <span className="text-slate-400">
                    ΔLogProb Margin: <strong className="text-emerald-400">+{pair.estimatedLogProbMargin}</strong> · β={pair.recommendedBeta}
                  </span>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    onSendMinedPairToDatasetStudio?.({
                      instruction: pair.prompt,
                      output: pair.chosenResponse,
                      domainTag: pair.domainTag,
                    });
                    setMinerNotice(
                      `Sent mined ${alignmentMethod} pair (${pair.domainTag}) to Dataset Studio & Stage 2 Alignment Queue.`
                    );
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  <Database className="w-3 h-3 text-emerald-400" />
                  <span>Push to Dataset &amp; Alignment Studio</span>
                </button>
              </div>

              <div className="text-xs font-mono text-slate-200 bg-[#111827] border border-slate-800/80 rounded p-2.5">
                <span className="text-slate-400">Prompt: </span>
                {pair.prompt}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 text-[11px] font-mono">
                <div className="p-2.5 rounded bg-emerald-950/20 border border-emerald-500/30 text-emerald-200 whitespace-pre-wrap">
                  <div className="text-emerald-400 font-semibold mb-1">
                    [+] CHOSEN TARGET (y_w):
                  </div>
                  {pair.chosenResponse}
                </div>
                <div className="p-2.5 rounded bg-red-950/20 border border-red-500/30 text-red-200 whitespace-pre-wrap">
                  <div className="text-red-400 font-semibold mb-1">
                    [-] REJECTED HARD-NEGATIVE (y_l):
                  </div>
                  {pair.rejectedResponse}
                  <div className="mt-2 pt-1.5 border-t border-red-500/20 text-amber-300">
                    Diagnosis: {pair.failureDiagnosis}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Combined TRL JSONL Export Box */}
        <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-semibold text-slate-300">
              TRL {alignmentMethod}Trainer Ready Preference Dataset (preference_mined.jsonl)
            </span>
            <button
              type="button"
              onClick={() => {
                navigator.clipboard?.writeText(miningReport.combinedJsonlPayload);
                setCopiedJsonl(true);
                setTimeout(() => setCopiedJsonl(false), 1500);
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded"
            >
              {copiedJsonl ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Copied JSONL</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy {miningReport.minedPairs.length} JSONL Records</span>
                </>
              )}
            </button>
          </div>
          <pre className="text-[11px] font-mono text-slate-400 overflow-x-auto max-h-28 overflow-y-auto leading-relaxed">
            {miningReport.combinedJsonlPayload}
          </pre>
        </div>
      </section>
    </div>
  );
};
