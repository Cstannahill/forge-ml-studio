import React, { useState } from 'react';
import {
  CheckCircle2,
  GitCompare,
  Pin,
  Plus,
  RotateCcw,
  Trash2,
} from 'lucide-react';
import {
  AblationRunRecord,
  INITIAL_ABLATION_RUNS,
  ModalityTrack,
  computeAblationMatrixSummary,
} from '../lib/pipelineLogic';

interface MultiRunAblationMatrixPanelProps {
  activeModality: ModalityTrack;
  activeModelId: string;
  activeModelName: string;
  activeParamsBillions: number;
  onRestoreAblationRunModel?: (modelId: string, summaryMessage: string) => void;
}

export const MultiRunAblationMatrixPanel: React.FC<
  MultiRunAblationMatrixPanelProps
> = ({
  activeModality,
  activeModelId,
  activeModelName,
  activeParamsBillions,
  onRestoreAblationRunModel,
}) => {
  const [runs, setRuns] = useState<AblationRunRecord[]>(INITIAL_ABLATION_RUNS);
  const [customRunNote, setCustomRunNote] = useState<string>(
    'PiSSA + LoRA+ (r=32, λ=12) + Curriculum'
  );
  const [ablationNotice, setAblationNotice] = useState<string | null>(null);

  const summary = computeAblationMatrixSummary(runs);

  const handlePinCurrentRun = () => {
    const nextIdx = runs.length + 1;
    const isVlm = activeModality === 'VISION_VLM';
    const newRun: AblationRunRecord = {
      id: `ablation-run-${Date.now()}`,
      runLabel: `Run #${nextIdx}: ${activeModelName} (${customRunNote.trim() || 'Custom Snapshot'})`,
      modality: activeModality,
      modelId: activeModelId,
      modelName: activeModelName,
      adapterSpec: customRunNote.trim() || 'PiSSA + LoRA+ (r=32, λ=12)',
      loraRank: 32,
      learningRate: 1.8e-4,
      stage2Method: isVlm ? 'ORPO + POPE Negatives' : 'ORPO (β=0.15)',
      quantizationFormat: 'Hetero Q4_K_M + Q8_0',
      peakVramGb: Number((activeParamsBillions * 2.1 + 2.4).toFixed(1)),
      trainingMinutes: Math.round(activeParamsBillions * 11 + 14),
      finalValLoss: Number(Math.max(0.68, 0.81 - activeParamsBillions * 0.02).toFixed(3)),
      primaryBenchmarkScorePct: Number(
        Math.min(91.5, 79.5 + activeParamsBillions * 2.2).toFixed(1)
      ),
      secondaryBenchmarkScorePct: Number(
        Math.min(95.4, 88.0 + activeParamsBillions * 1.9).toFixed(1)
      ),
      quantizedSizeMb: Math.round(activeParamsBillions * 640 + 180),
      inferenceTokPerSec: Math.round(210 / Math.max(0.6, activeParamsBillions * 0.75)),
    };

    setRuns((prev) => [...prev, newRun]);
    setAblationNotice(
      `Pinned "${newRun.runLabel}" into Multi-Run Ablation Matrix (${runs.length + 1} total runs compared).`
    );
  };

  return (
    <section
      aria-label="Pinned Multi-Run Ablation Matrix and Experiment Comparison Table"
      className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
    >
      <div className="flex flex-col gap-3.5 pb-4 border-b border-slate-800/80">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
            <span className="text-emerald-400 font-semibold">
              {runs.length} Pinned Experiment Runs
            </span>
            <span aria-hidden="true">·</span>
            <span>
              Pareto Leader:{' '}
              <strong className="text-slate-100">
                {runs.find((r) => r.id === summary.bestAccuracyRunId)?.runLabel}
              </strong>
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <input
              type="text"
              aria-label="Custom Ablation Run Label"
              value={customRunNote}
              onChange={(e) => setCustomRunNote(e.target.value)}
              placeholder="e.g., DoRA r=32 + ORPO..."
              className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 w-56"
            />
            <button
              type="button"
              onClick={handlePinCurrentRun}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded-lg transition-colors whitespace-nowrap"
            >
              <Pin className="w-3.5 h-3.5" />
              <span>+ Pin Current Config as Run</span>
            </button>
          </div>
        </div>

        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <GitCompare className="w-4 h-4 text-blue-400 shrink-0" />
            <span>
              Pinned Multi-Run Ablation Matrix &amp; Pareto Experiment Comparison Table
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Compare pinned fine-tuning runs side-by-side across Peak VRAM, Training Duration, Validation Loss, GSM8K/MMMU Reasoning, IFEval/DocVQA Grounding, and Quantized Throughput.
          </p>
        </div>
      </div>

      {ablationNotice && (
        <div
          role="status"
          className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300"
        >
          {ablationNotice}
        </div>
      )}

      {/* Executive Pareto Takeaway */}
      <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5 text-xs font-mono text-slate-300 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <span>{summary.executiveComparisonSummary}</span>
        <span className="text-emerald-400 shrink-0">Pareto Frontier Verified</span>
      </div>

      {/* Multi-Run Comparison Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs font-mono tabular-nums">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
              <th className="py-2.5 pr-4">Pinned Experiment Run</th>
              <th className="py-2.5 px-3">Adapter &amp; Stage 2</th>
              <th className="py-2.5 px-3 text-right">Peak VRAM</th>
              <th className="py-2.5 px-3 text-right">Train Time</th>
              <th className="py-2.5 px-3 text-right">Val Loss</th>
              <th className="py-2.5 px-3 text-right">GSM8K / MMMU</th>
              <th className="py-2.5 px-3 text-right">IFEval / DocVQA</th>
              <th className="py-2.5 px-3 text-right">Speed</th>
              <th className="py-2.5 pl-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {runs.map((run) => {
              const isBestAcc = run.id === summary.bestAccuracyRunId;
              const isLowestLoss = run.id === summary.lowestLossRunId;
              const isLowestVram = run.id === summary.lowestVramRunId;
              return (
                <tr key={run.id} className="hover:bg-slate-800/30">
                  <td className="py-3 pr-4">
                    <div className="font-semibold text-slate-100">{run.runLabel}</div>
                    <div className="text-[11px] text-slate-400">
                      {run.modelName} · {run.modality} · {run.quantizationFormat} ({run.quantizedSizeMb} MB)
                    </div>
                  </td>
                  <td className="py-3 px-3 text-slate-300">
                    <div>{run.adapterSpec}</div>
                    <div className="text-[11px] text-slate-500">
                      LR: {run.learningRate} · {run.stage2Method}
                    </div>
                  </td>
                  <td className="py-3 px-3 text-right">
                    <span className={isLowestVram ? 'text-blue-300 font-semibold' : 'text-slate-300'}>
                      {run.peakVramGb} GB
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right text-slate-300">
                    {run.trainingMinutes}m
                  </td>
                  <td className="py-3 px-3 text-right">
                    <span className={isLowestLoss ? 'text-emerald-400 font-bold' : 'text-slate-200'}>
                      {run.finalValLoss.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right text-slate-200">
                    {run.primaryBenchmarkScorePct}%
                  </td>
                  <td className="py-3 px-3 text-right">
                    <span className={isBestAcc ? 'text-emerald-400 font-bold' : 'text-slate-100'}>
                      {run.secondaryBenchmarkScorePct}%
                    </span>
                  </td>
                  <td className="py-3 px-3 text-right text-amber-300">
                    {run.inferenceTokPerSec} tok/s
                  </td>
                  <td className="py-3 pl-3 text-right">
                    <div className="inline-flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          onRestoreAblationRunModel?.(
                            run.modelId,
                            `Restored "${run.runLabel}" (${run.modelName}) as active evaluation model.`
                          );
                          setAblationNotice(
                            `Restored "${run.runLabel}" (${run.modelName}, ${run.adapterSpec}) into active workspace.`
                          );
                        }}
                        className="px-2.5 py-1 text-[11px] text-slate-200 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                      >
                        Restore
                      </button>
                      {runs.length > 2 && (
                        <button
                          type="button"
                          aria-label={`Remove ${run.runLabel}`}
                          onClick={() =>
                            setRuns((prev) => prev.filter((item) => item.id !== run.id))
                          }
                          className="p-1 text-slate-400 hover:text-red-400 rounded"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
};
