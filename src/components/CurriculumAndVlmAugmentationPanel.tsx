import React, { useState } from 'react';
import {
  Check,
  Copy,
  Layers,
  RotateCw,
  Sliders,
  Sparkles,
  Wand2,
} from 'lucide-react';
import {
  DifficultyPacingScheduleMode,
  ModalityTrack,
  computeCurriculumDifficultySchedule,
} from '../lib/pipelineLogic';
import {
  NormalizedVisionSample,
  VlmSyntheticCorruptionPreset,
  computeSyntheticVlmAugmentationPreview,
} from '../lib/visionPipelineLogic';

interface CurriculumAndVlmAugmentationPanelProps {
  modality: ModalityTrack;
  samples: Array<{
    id: string;
    instruction: string;
    output: string;
    domainTag: string;
    tokenEstimate: number;
    boundingBoxCount?: number;
  }>;
  totalSteps?: number;
  onApplyCurriculumOrder?: (orderedIds: string[], summaryMessage: string) => void;
  activeVisionSample?: NormalizedVisionSample;
  onCommitTransformedVisionBboxes?: (
    updatedSample: NormalizedVisionSample,
    summaryMessage: string
  ) => void;
}

export const CurriculumAndVlmAugmentationPanel: React.FC<
  CurriculumAndVlmAugmentationPanelProps
> = ({
  modality,
  samples,
  totalSteps = 600,
  onApplyCurriculumOrder,
  activeVisionSample,
  onCommitTransformedVisionBboxes,
}) => {
  const [pacingStrategy, setPacingStrategy] =
    useState<DifficultyPacingScheduleMode>('ROOT_COMPETENCE_PACING');
  const [copiedCurriculumPy, setCopiedCurriculumPy] = useState<boolean>(false);
  const [curriculumNotice, setCurriculumNotice] = useState<string | null>(null);

  const curriculumReport = computeCurriculumDifficultySchedule({
    samples,
    totalSteps,
    strategy: pacingStrategy,
  });

  // Synthetic VLM Augmentation & Coordinate Transformer state
  const [corruptionPreset, setCorruptionPreset] =
    useState<VlmSyntheticCorruptionPreset>('PERSPECTIVE_TILT_12DEG');
  const [rotationDeg, setRotationDeg] = useState<number>(8);
  const [scaleCropFactor, setScaleCropFactor] = useState<number>(0.94);
  const [horizontalFlip, setHorizontalFlip] = useState<boolean>(false);
  const [copiedAlbumentations, setCopiedAlbumentations] = useState<boolean>(false);
  const [augmentationNotice, setAugmentationNotice] = useState<string | null>(null);

  const augmentationReport = activeVisionSample
    ? computeSyntheticVlmAugmentationPreview({
        sample: activeVisionSample,
        corruptionPreset,
        rotationDeg,
        scaleCropFactor,
        horizontalFlip,
      })
    : null;

  return (
    <div className="space-y-6">
      {/* SECTION 1: Multi-Stage Curriculum Learning & Difficulty Pacing Scheduler */}
      <section
        aria-label="Multi-Stage Curriculum Learning and Difficulty Pacing Scheduler"
        className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
      >
        <div className="flex flex-col gap-3.5 pb-4 border-b border-slate-800/80">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
              <span className="text-emerald-400 font-semibold">
                +{curriculumReport.projectedAccuracyLiftPct}% Benchmark Lift
              </span>
              <span aria-hidden="true">·</span>
              <span>
                Early Gradient Spike Reduction:{' '}
                <strong className="text-blue-300">
                  -{curriculumReport.earlyGradientSpikeReductionPct}%
                </strong>
              </span>
              <span aria-hidden="true">·</span>
              <span>
                Scored Corpus: <strong className="text-slate-200">{samples.length} samples</strong>
              </span>
            </div>

            <button
              type="button"
              onClick={() => {
                const orderedIds = curriculumReport.sortedSamples.map((s) => s.id);
                onApplyCurriculumOrder?.(
                  orderedIds,
                  `Sorted ${orderedIds.length} samples by ${pacingStrategy} (Easy -> Medium -> Hard difficulty pacing).`
                );
                setCurriculumNotice(
                  `Applied ${pacingStrategy} Easy -> Hard ordering across ${orderedIds.length} active samples (+${curriculumReport.projectedAccuracyLiftPct}% projected accuracy gain).`
                );
              }}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded-lg transition-colors whitespace-nowrap shrink-0"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Apply Easy &rarr; Hard Curriculum Pacing</span>
            </button>
          </div>

          <div>
            <h2 className="text-base font-semibold text-slate-100">
              Multi-Stage Curriculum Learning &amp; Difficulty Pacing Scheduler (Easy &rarr; Medium &rarr; Hard)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Scores every training sample&apos;s composite difficulty (<code className="font-mono text-slate-300">D &isin; [0, 100]</code>) from token length, <code className="font-mono text-slate-300">&lt;think&gt;</code> step depth, and <code className="font-mono text-slate-300">&lt;box&gt;</code> grounding density to prevent Epoch 1 gradient shock on sub-2B/4B checkpoints.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800 w-fit">
            {(
              [
                {
                  id: 'ROOT_COMPETENCE_PACING',
                  label: '1. Root-Competence Pacing (+3.4% Lift)',
                },
                {
                  id: 'LINEAR_3PHASE_RAMP',
                  label: '2. Linear 3-Phase Ramp (+2.8% Lift)',
                },
                {
                  id: 'STEPWISE_STAGE_GATED',
                  label: '3. Stepwise Stage-Gated (+3.1% Lift)',
                },
                {
                  id: 'UNIFORM_SHUFFLE',
                  label: '4. Standard Uniform Shuffle (Baseline)',
                },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setPacingStrategy(opt.id)}
                className={`px-3 py-1.5 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                  pacingStrategy === opt.id
                    ? 'bg-blue-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {curriculumNotice && (
          <div
            role="status"
            className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300"
          >
            {curriculumNotice}
          </div>
        )}

        {/* 3-Phase Curriculum Progression Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {curriculumReport.phases.map((phase, idx) => (
            <div
              key={phase.phaseId}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-2"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-blue-300 font-semibold">{phase.stepRangeLabel}</span>
                <span className="text-amber-300">
                  Mean Difficulty: D={phase.meanDifficultyScore}/100
                </span>
              </div>
              <div className="text-xs font-semibold text-slate-100">{phase.title}</div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                {phase.curriculumFocus}
              </p>
              <div className="pt-1 text-[11px] font-mono text-emerald-400">
                Assigned Corpus Slice: {phase.sampleCount} sample(s) (Stage {idx + 1}/3)
              </div>
            </div>
          ))}
        </div>

        {/* Scored Samples Table + PyTorch Sampler Snippet */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-2.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-200">
                Curriculum-Sorted Sample Queue (Easy &rarr; Hard Pacing Order)
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                Showing top {Math.min(6, curriculumReport.sortedSamples.length)} of{' '}
                {curriculumReport.sortedSamples.length} samples
              </span>
            </div>

            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {curriculumReport.sortedSamples.slice(0, 6).map((item, idx) => (
                <div
                  key={item.id}
                  className="p-2.5 rounded bg-[#111827] border border-slate-800/90 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs font-mono"
                >
                  <div className="space-y-0.5 min-w-0">
                    <div className="text-slate-200 truncate">
                      #{idx + 1} · {item.instructionPreview}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Domain: {item.domainTag} · {item.tokenEstimate} tok ·{' '}
                      {item.reasoningStepsCount} CoT steps
                      {item.groundingBoxCount > 0
                        ? ` · ${item.groundingBoxCount} <box> tags`
                        : ''}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 tabular-nums">
                    <span className="text-[11px] text-slate-400">{item.assignedPhase.replace('PHASE_', 'P')}</span>
                    <span
                      className={`font-semibold ${
                        item.difficultyScore < 45
                          ? 'text-emerald-400'
                          : item.difficultyScore < 72
                          ? 'text-amber-300'
                          : 'text-purple-300'
                      }`}
                    >
                      D={item.difficultyScore}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-semibold text-slate-200">
                  PyTorch Curriculum Difficulty Sorter
                </span>
                <button
                  type="button"
                  onClick={() => {
                    navigator.clipboard?.writeText(
                      curriculumReport.pytorchCurriculumSamplerSnippet
                    );
                    setCopiedCurriculumPy(true);
                    setTimeout(() => setCopiedCurriculumPy(false), 1500);
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded"
                >
                  {copiedCurriculumPy ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy PyTorch Code</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="p-3 rounded bg-[#111827] border border-slate-800 text-[11px] font-mono text-emerald-300 overflow-x-auto leading-relaxed whitespace-pre">
                {curriculumReport.pytorchCurriculumSamplerSnippet}
              </pre>
            </div>

            <div className="p-3 rounded bg-[#111827] border border-slate-800 text-[11px] font-mono text-slate-300">
              Curriculum pacing prevents early attention head entropy collapse by teaching atomic delimiter syntax (<code className="text-emerald-300">&lt;think&gt;</code>, <code className="text-emerald-300">&lt;box&gt;</code>) before dense adversarial scenes.
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: Synthetic VLM Visual Corruption & Coordinate-Preserving BBox Augmentation Lab (VISION_VLM Only) */}
      {modality === 'VISION_VLM' && activeVisionSample && augmentationReport && (
        <section
          aria-label="Synthetic VLM Visual Corruption and Coordinate-Preserving BBox Augmentation Lab"
          className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5"
        >
          <div className="flex flex-col gap-3.5 pb-4 border-b border-slate-800/80">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
                <span className="text-emerald-400 font-semibold">
                  +{augmentationReport.expectedRealWorldOcrRecallLiftPct}% Real-World Mobile OCR &amp; VEIL Recall
                </span>
                <span aria-hidden="true">·</span>
                <span>
                  Active Scene: <strong className="text-slate-200">{activeVisionSample.imageTitle}</strong>
                </span>
              </div>

              <button
                type="button"
                onClick={() => {
                  if (!onCommitTransformedVisionBboxes) return;
                  const updatedSample: NormalizedVisionSample = {
                    ...activeVisionSample,
                    boundingBoxes: activeVisionSample.boundingBoxes.map((b, i) => ({
                      ...b,
                      box2d:
                        augmentationReport.transformedBoxes[i]?.transformedBox2d ?? b.box2d,
                    })),
                  };
                  onCommitTransformedVisionBboxes(
                    updatedSample,
                    `Committed coordinate-transformed [0, 1000] bounding boxes (${corruptionPreset}, rot=${rotationDeg}deg, scale=${scaleCropFactor}x) to "${activeVisionSample.imageTitle}".`
                  );
                  setAugmentationNotice(
                    `Committed ${augmentationReport.transformedBoxes.length} coordinate-transformed [0, 1000] bounding box(es) to "${activeVisionSample.imageTitle}".`
                  );
                }}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-400 hover:bg-amber-300 rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                <Wand2 className="w-3.5 h-3.5" />
                <span>Commit Transformed [0, 1000] BBoxes to Sample</span>
              </button>
            </div>

            <div>
              <h2 className="text-base font-semibold text-slate-100">
                Synthetic VLM Corruption &amp; Coordinate-Preserving Bounding-Box Transformer
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Simulates real-world camera captures (perspective skew, specular glare on ID/Rx labels, motion blur, and JPEG compression) while mathematically transforming <code className="font-mono text-slate-300">&lt;box&gt;[y1, x1, y2, x2]&lt;/box&gt;</code> coordinates so spatial supervision never drifts.
              </p>
            </div>

            {/* Corruption Preset Selector */}
            <div className="flex flex-wrap items-center gap-1.5 pt-1">
              {(
                [
                  { id: 'PERSPECTIVE_TILT_12DEG', label: '1. Perspective Skew & Tilt (+9.4% Recall)' },
                  { id: 'SPECULAR_GLARE_ID_RX', label: '2. Specular Flash Glare (+11.8% Recall)' },
                  { id: 'MOTION_BLUR_HANDHELD', label: '3. Handheld Motion Blur (+10.6% Recall)' },
                  { id: 'JPEG_COMPRESSION_Q35', label: '4. Low-Bitrate JPEG Q35 (+8.2% Recall)' },
                  { id: 'CLEAN_ORIGINAL', label: '5. Clean Studio Baseline' },
                ] as const
              ).map((preset) => (
                <button
                  key={preset.id}
                  type="button"
                  onClick={() => setCorruptionPreset(preset.id)}
                  className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors whitespace-nowrap ${
                    corruptionPreset === preset.id
                      ? 'bg-amber-500 text-slate-950 border-amber-400 font-semibold'
                      : 'bg-[#0B0F17] border-slate-800 text-slate-300 hover:text-slate-100'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {augmentationNotice && (
            <div
              role="status"
              className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-4 py-2.5 text-xs font-mono text-emerald-300"
            >
              {augmentationNotice}
            </div>
          )}

          {/* Geometric Sliders + Live Coordinate Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
            {/* Left 5 Cols: Live Augmented SVG Canvas Preview + Geometric Sliders */}
            <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 space-y-4">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-200 font-semibold">
                  Live Geometric &amp; Photometric Preview
                </span>
                <span className="text-amber-300">
                  rot={rotationDeg}&deg; · scale={scaleCropFactor}x
                </span>
              </div>

              {/* SVG Preview Box showing Original (dashed slate) vs Transformed (emerald solid) BBoxes */}
              <div className="relative w-full h-48 rounded-lg bg-[#111827] border border-slate-800 overflow-hidden flex items-center justify-center">
                <svg
                  viewBox="0 0 1000 1000"
                  className="w-full h-full"
                  style={{
                    filter: augmentationReport.cssFilterString,
                  }}
                >
                  <rect x="0" y="0" width="1000" height="1000" fill="#0B0F17" />
                  {/* Grid lines */}
                  <line x1="500" y1="0" x2="500" y2="1000" stroke="#1E293B" strokeWidth="4" />
                  <line x1="0" y1="500" x2="1000" y2="500" stroke="#1E293B" strokeWidth="4" />

                  {corruptionPreset === 'SPECULAR_GLARE_ID_RX' && (
                    <circle cx="620" cy="420" r="210" fill="rgba(255,255,255,0.16)" />
                  )}

                  {augmentationReport.transformedBoxes.map((tb, idx) => {
                    const [oy1, ox1, oy2, ox2] = tb.originalBox2d;
                    const [ny1, nx1, ny2, nx2] = tb.transformedBox2d;
                    return (
                      <g key={`${tb.label}-${idx}`}>
                        {/* Original Unadjusted Box (Dashed Red/Amber) */}
                        <rect
                          x={ox1}
                          y={oy1}
                          width={Math.max(20, ox2 - ox1)}
                          height={Math.max(20, oy2 - oy1)}
                          fill="none"
                          stroke="#F59E0B"
                          strokeWidth="8"
                          strokeDasharray="18 12"
                        />
                        {/* Coordinate-Transformed Box (Solid Emerald) */}
                        <rect
                          x={nx1}
                          y={ny1}
                          width={Math.max(20, nx2 - nx1)}
                          height={Math.max(20, ny2 - ny1)}
                          fill="rgba(16, 185, 129, 0.18)"
                          stroke="#10B981"
                          strokeWidth="10"
                        />
                      </g>
                    );
                  })}
                </svg>
              </div>

              <div className="space-y-3 text-xs font-mono">
                <div>
                  <div className="flex justify-between mb-1">
                    <label htmlFor="vlm-rot-slider" className="text-slate-300">
                      Affine Camera Rotation (&deg;):
                    </label>
                    <span className="text-amber-300">{rotationDeg}&deg;</span>
                  </div>
                  <input
                    id="vlm-rot-slider"
                    type="range"
                    min={-15}
                    max={15}
                    step={1}
                    value={rotationDeg}
                    onChange={(e) => setRotationDeg(Number(e.target.value))}
                    className="w-full accent-amber-400"
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <label htmlFor="vlm-scale-slider" className="text-slate-300">
                      Scale / Zoom Jitter:
                    </label>
                    <span className="text-blue-300">{scaleCropFactor.toFixed(2)}x</span>
                  </div>
                  <input
                    id="vlm-scale-slider"
                    type="range"
                    min={0.85}
                    max={1.15}
                    step={0.02}
                    value={scaleCropFactor}
                    onChange={(e) => setScaleCropFactor(Number(e.target.value))}
                    className="w-full accent-blue-400"
                  />
                </div>

                <label className="flex items-center gap-2 text-slate-300 cursor-pointer pt-1">
                  <input
                    type="checkbox"
                    checked={horizontalFlip}
                    onChange={(e) => setHorizontalFlip(e.target.checked)}
                    className="rounded border-slate-700 bg-slate-900 text-amber-500"
                  />
                  <span>Horizontal Mirror Flip (x&apos; = 1000 - x)</span>
                </label>
              </div>
            </div>

            {/* Right 7 Cols: Transformed [0, 1000] Coordinates & Albumentations Pipeline */}
            <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-4">
              <div className="space-y-2.5">
                <div className="text-xs font-semibold text-slate-200">
                  Coordinate-Preserving [0, 1000] Bounding-Box Transformation Matrix
                </div>

                {augmentationReport.transformedBoxes.length > 0 ? (
                  <div className="space-y-2">
                    {augmentationReport.transformedBoxes.map((tb, idx) => (
                      <div
                        key={`${tb.label}-${idx}`}
                        className="p-2.5 rounded bg-[#111827] border border-slate-800 text-xs font-mono flex flex-wrap items-center justify-between gap-2"
                      >
                        <div>
                          <span className="text-slate-100 font-semibold">{tb.label}</span>
                          <div className="text-[11px] text-slate-400 mt-0.5">
                            Original: [{tb.originalBox2d.join(', ')}] &rarr;{' '}
                            <strong className="text-emerald-400">
                              Transformed: [{tb.transformedBox2d.join(', ')}]
                            </strong>
                          </div>
                        </div>
                        <span className="text-[11px] text-amber-300">
                          -{tb.unadjustedIoULossPct}% IoU if unadjusted
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-3 rounded bg-[#111827] border border-slate-800 text-xs font-mono text-slate-400">
                    Active sample is a clean negative baseline (0 bounding boxes). Select a scene with hazards or draw a box above to inspect coordinate transforms.
                  </div>
                )}
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-semibold text-slate-300">
                    Albumentations BBox-Aware Augmentation Pipeline
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard?.writeText(
                        augmentationReport.albumentationsCodeSnippet
                      );
                      setCopiedAlbumentations(true);
                      setTimeout(() => setCopiedAlbumentations(false), 1500);
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded"
                  >
                    {copiedAlbumentations ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-400" />
                        <span>Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        <span>Copy Albumentations Code</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="p-2.5 rounded bg-[#111827] border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto leading-relaxed">
                  {augmentationReport.albumentationsCodeSnippet}
                </pre>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
