import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Database,
  Scale,
  ShieldCheck,
  Sliders,
  Zap,
} from 'lucide-react';
import {
  DECISION_EVALUATOR_MODELS,
  DecisionEvaluatorModelId,
  ModalityTrack,
  evaluateWithDecisionModelCascade,
} from '../lib/pipelineLogic';

interface DecisionModelEvaluatorTriagePanelProps {
  modality: ModalityTrack;
  targetModelId: string;
  targetModelName: string;
  onSendCaseToDatasetStudio?: (draft: {
    instruction: string;
    output: string;
    domainTag: string;
  }) => void;
}

export const DecisionModelEvaluatorTriagePanel: React.FC<
  DecisionModelEvaluatorTriagePanelProps
> = ({
  modality,
  targetModelId,
  targetModelName,
  onSendCaseToDatasetStudio,
}) => {
  const [evaluatorId, setEvaluatorId] = useState<DecisionEvaluatorModelId>(
    modality === 'VISION_VLM'
      ? 'google/gemma-4-e2b-decision-judge'
      : 'LiquidAI/LFM2.5-Decision-350M-Rubric'
  );
  const [enableStyleDebiasing, setEnableStyleDebiasing] = useState<boolean>(true);
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [customReference, setCustomReference] = useState<string>('');
  const [customResponse, setCustomResponse] = useState<string>('');
  const [submittedCustomCase, setSubmittedCustomCase] = useState<{
    prompt: string;
    reference: string;
    response: string;
  } | null>(null);
  const [triageNotice, setTriageNotice] = useState<string | null>(null);

  const cascadeReport = evaluateWithDecisionModelCascade({
    evaluatorModelId: evaluatorId,
    targetModelId,
    targetModelName,
    modality,
    enableStyleAndLengthDebiasing: enableStyleDebiasing,
    customTestCase: submittedCustomCase,
  });

  const axisItems = [
    {
      label: '1. Instruction Adherence',
      base: cascadeReport.baseSixAxes.instructionAdherence,
      ft: cascadeReport.fineTunedSixAxes.instructionAdherence,
      lowerIsBetter: false,
    },
    {
      label: '2. Factual Correctness (vs. Reference)',
      base: cascadeReport.baseSixAxes.factualCorrectness,
      ft: cascadeReport.fineTunedSixAxes.factualCorrectness,
      lowerIsBetter: false,
    },
    {
      label: '3. Completeness',
      base: cascadeReport.baseSixAxes.completeness,
      ft: cascadeReport.fineTunedSixAxes.completeness,
      lowerIsBetter: false,
    },
    {
      label: '4. Relevance',
      base: cascadeReport.baseSixAxes.relevance,
      ft: cascadeReport.fineTunedSixAxes.relevance,
      lowerIsBetter: false,
    },
    {
      label: '5. Format Compliance (Code + Schema)',
      base: cascadeReport.baseSixAxes.formatCompliance,
      ft: cascadeReport.fineTunedSixAxes.formatCompliance,
      lowerIsBetter: false,
    },
    {
      label: '6. Presence of Unsupported Statements (Risk ↓)',
      base: cascadeReport.baseSixAxes.unsupportedStatementRisk,
      ft: cascadeReport.fineTunedSixAxes.unsupportedStatementRisk,
      lowerIsBetter: true,
    },
  ];

  return (
    <section
      aria-label="3-Tier Decision Model Evaluator and 6-Axis Rubric Triage Matrix"
      className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
    >
      <div className="flex flex-col gap-3.5 pb-4 border-b border-slate-800/80">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
            <span className="text-emerald-400 font-semibold">
              {cascadeReport.speedupVsGenerativeJudgeMultiplier}x Faster than Generative LLM-as-a-Judge
            </span>
            <span aria-hidden="true">·</span>
            <span>
              Latency:{' '}
              <strong className="text-blue-300">
                {cascadeReport.evaluatorModel.latencyMsPerSample} ms/sample
              </strong>{' '}
              ({cascadeReport.evaluatorModel.throughputSamplesPerSec} samples/sec)
            </span>
            <span aria-hidden="true">·</span>
            <span>
              Judge Token Savings:{' '}
              <strong className="text-amber-300">-{cascadeReport.tokenCostSavedPct}%</strong>
            </span>
          </div>

          <label className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#0B0F17] border border-slate-800 text-xs font-mono text-slate-200 cursor-pointer">
            <input
              type="checkbox"
              checked={enableStyleDebiasing}
              onChange={(e) => setEnableStyleDebiasing(e.target.checked)}
              className="rounded border-slate-700 bg-slate-900 text-emerald-500"
            />
            <span>Length &amp; Style-Bias Debiasing (Active)</span>
          </label>
        </div>

        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Scale className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              3-Tier Decision Model Evaluator &amp; 6-Axis Rubric Triage Matrix ({targetModelName})
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Combines <strong>Tier 1 Deterministic Code Tests</strong> (JSON/AST/GIoU), <strong>Tier 2 Single-Pass Decision Model Scoring</strong> across 6 calibrated rubric axes (0 generative tokens), and <strong>Tier 3 Selective Heavy-Judge Escalation</strong> for ambiguous cases (<code className="font-mono text-slate-300">p &isin; [0.35, 0.68]</code>).
          </p>
        </div>

        {/* Decision Evaluator Model Selector */}
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          {DECISION_EVALUATOR_MODELS.map((dm) => (
            <button
              key={dm.id}
              type="button"
              onClick={() => setEvaluatorId(dm.id)}
              className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors whitespace-nowrap ${
                evaluatorId === dm.id
                  ? 'bg-emerald-600 text-white border-emerald-500 font-semibold'
                  : 'bg-[#0B0F17] border-slate-800 text-slate-300 hover:text-slate-100'
              }`}
            >
              {dm.name}
            </button>
          ))}
        </div>
      </div>

      {triageNotice && (
        <div
          role="status"
          className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300"
        >
          {triageNotice}
        </div>
      )}

      {/* 3-Tier Triage Funnel Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs tabular-nums">
        <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5">
          <div className="text-[11px] text-slate-400">
            Tier 1: Deterministic Code &amp; Schema Tests
          </div>
          <div className="text-base font-bold text-blue-300 mt-1">
            {cascadeReport.tier1CodeHandledPct}% Resolved Instantly (0 ms)
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            JSON AST, SQL parser, pytest &amp; [0, 1000] BBox GIoU
          </div>
        </div>

        <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5">
          <div className="text-[11px] text-slate-400">
            Tier 2: Decision Model 6-Axis Auto-Triage
          </div>
          <div className="text-base font-bold text-emerald-400 mt-1">
            {cascadeReport.tier2DecisionAutoTriagedPct}% Auto-Triaged ({cascadeReport.evaluatorModel.latencyMsPerSample} ms)
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            High-confidence Pass (&ge;0.68) or Unsupported-Claim Reject (&lt;0.35)
          </div>
        </div>

        <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5">
          <div className="text-[11px] text-slate-400">
            Tier 3: Ambiguous Escalation to Heavy Judge
          </div>
          <div className="text-base font-bold text-amber-300 mt-1">
            Only {cascadeReport.tier3AmbiguousEscalatedPct}% Escalated
          </div>
          <div className="text-[10px] text-slate-500 mt-0.5">
            Saves {cascadeReport.tokenCostSavedPct}% of 70B LLM-as-a-Judge compute
          </div>
        </div>
      </div>

      {/* 6-Axis Rubric Comparison + Interactive Test Case Probe */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 6 Cols: 6-Axis Structured Rubric Scores (Base vs Fine-Tuned) */}
        <div className="lg:col-span-6 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-200">
              6-Axis Decision Model Rubric (Base vs. Fine-Tuned {targetModelName})
            </span>
            <span className="font-mono text-[11px] text-emerald-400">
              {enableStyleDebiasing ? 'Debiased for Length/Markdown' : 'Raw Probabilities'}
            </span>
          </div>

          <div className="space-y-2.5">
            {axisItems.map((ax) => {
              const delta = Number((ax.ft - ax.base).toFixed(1));
              return (
                <div
                  key={ax.label}
                  className="p-2.5 rounded bg-[#111827] border border-slate-800/90 space-y-1.5 font-mono text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-slate-200 font-semibold">{ax.label}</span>
                    <div className="tabular-nums">
                      <span className="text-slate-400">Base {ax.base}% &rarr; </span>
                      <strong className="text-emerald-400">FT {ax.ft}%</strong>
                      <span className="text-blue-300 ml-1.5">
                        ({delta > 0 ? `+${delta}%` : `${delta}%`})
                      </span>
                    </div>
                  </div>
                  <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        ax.lowerIsBetter ? 'bg-amber-400' : 'bg-emerald-400'
                      }`}
                      style={{ width: `${Math.min(100, ax.ft)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right 6 Cols: Interactive Custom Decision Model Probe + Evaluated Cases */}
        <div className="lg:col-span-6 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-4">
          <div className="space-y-2.5">
            <div className="text-xs font-semibold text-slate-200">
              Run Single-Pass Decision Evaluation (Test Case + Response + Reference)
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <input
                type="text"
                aria-label="Decision Test Case Prompt"
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="1. Test Prompt..."
                className="bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
              />
              <input
                type="text"
                aria-label="Decision Reference Answer"
                value={customReference}
                onChange={(e) => setCustomReference(e.target.value)}
                placeholder="2. Reference Ground Truth..."
                className="bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
              />
              <div className="flex gap-1.5">
                <input
                  type="text"
                  aria-label="Decision Candidate Response"
                  value={customResponse}
                  onChange={(e) => setCustomResponse(e.target.value)}
                  placeholder='3. Model Response...'
                  className="flex-1 bg-[#111827] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100 min-w-0"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (!customPrompt.trim() || !customResponse.trim()) return;
                    setSubmittedCustomCase({
                      prompt: customPrompt,
                      reference: customReference,
                      response: customResponse,
                    });
                    setTriageNotice(
                      `Scored custom test case across all 6 Decision Rubric axes in ${cascadeReport.evaluatorModel.latencyMsPerSample} ms.`
                    );
                  }}
                  className="px-3 py-1.5 text-xs font-mono font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded shrink-0"
                >
                  Score
                </button>
              </div>
            </div>
          </div>

          {/* Triage Cases Queue */}
          <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
            {cascadeReport.testCases.map((tc) => (
              <div
                key={tc.id}
                className="p-3 rounded bg-[#111827] border border-slate-800/90 space-y-2 text-xs font-mono"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-semibold text-slate-100">{tc.title}</span>
                  <span
                    className={
                      tc.triageVerdict === 'AUTO_PASS_CODE_AND_DECISION'
                        ? 'text-emerald-400 font-semibold'
                        : tc.triageVerdict === 'AMBIGUOUS_ESCALATE_HEAVY_JUDGE'
                        ? 'text-amber-300 font-semibold'
                        : 'text-red-400 font-semibold'
                    }
                  >
                    {tc.triageVerdict} (p={tc.compositeConfidenceProb.toFixed(2)})
                  </span>
                </div>

                <div className="text-[11px] text-blue-300">{tc.tier1CodeCheckNote}</div>

                <div className="grid grid-cols-3 sm:grid-cols-6 gap-1.5 text-[10px] tabular-nums bg-[#0B0F17] p-2 rounded border border-slate-800">
                  <div>
                    <span className="text-slate-400">Adhere:</span>{' '}
                    <strong className="text-slate-100">{tc.axes.instructionAdherence}%</strong>
                  </div>
                  <div>
                    <span className="text-slate-400">Factual:</span>{' '}
                    <strong className="text-slate-100">{tc.axes.factualCorrectness}%</strong>
                  </div>
                  <div>
                    <span className="text-slate-400">Compl:</span>{' '}
                    <strong className="text-slate-100">{tc.axes.completeness}%</strong>
                  </div>
                  <div>
                    <span className="text-slate-400">Relev:</span>{' '}
                    <strong className="text-slate-100">{tc.axes.relevance}%</strong>
                  </div>
                  <div>
                    <span className="text-slate-400">Format:</span>{' '}
                    <strong className="text-slate-100">{tc.axes.formatCompliance}%</strong>
                  </div>
                  <div>
                    <span className="text-slate-400">Unsupp:</span>{' '}
                    <strong
                      className={
                        tc.axes.unsupportedStatementRisk > 25
                          ? 'text-red-400'
                          : 'text-emerald-400'
                      }
                    >
                      {tc.axes.unsupportedStatementRisk}%
                    </strong>
                  </div>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                  <span className="text-[11px] text-slate-400">{tc.recommendedAction}</span>
                  {tc.triageVerdict !== 'AUTO_PASS_CODE_AND_DECISION' && (
                    <button
                      type="button"
                      onClick={() => {
                        onSendCaseToDatasetStudio?.({
                          instruction: tc.testPrompt,
                          output: tc.referenceAnswer,
                          domainTag: 'decision-triage-remediation',
                        });
                        setTriageNotice(
                          `Sent "${tc.title}" with reference ground truth to Dataset Studio for remediation.`
                        );
                      }}
                      className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] text-slate-200 bg-slate-800 hover:bg-slate-700 rounded"
                    >
                      <Database className="w-3 h-3 text-emerald-400" />
                      <span>Send to Dataset Studio</span>
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
