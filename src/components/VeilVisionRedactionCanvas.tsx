import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  EyeOff,
  Layers,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import {
  VEIL_SANDBOX_TEST_SCENES,
  VeilSandboxScene,
  computeBox2dIoU,
  evaluateVeilVisualSandboxScene,
} from '../lib/visionPipelineLogic';
import { InfoTooltip } from './InfoTooltip';

export const VeilVisionRedactionCanvas: React.FC = () => {
  const [selectedSceneId, setSelectedSceneId] = useState<string>(VEIL_SANDBOX_TEST_SCENES[0].id);
  const [showBaseBoxes, setShowBaseBoxes] = useState<boolean>(true);
  const [showFineTunedBoxes, setShowFineTunedBoxes] = useState<boolean>(true);
  const [redactionMode, setRedactionMode] = useState<'AUTO' | 'BLUR' | 'MOSAIC' | 'OUTLINE_ONLY'>(
    'AUTO'
  );

  const activeScene: VeilSandboxScene = useMemo(
    () =>
      VEIL_SANDBOX_TEST_SCENES.find((s) => s.id === selectedSceneId) ?? VEIL_SANDBOX_TEST_SCENES[0],
    [selectedSceneId]
  );

  const evalSummary = useMemo(
    () => evaluateVeilVisualSandboxScene(activeScene),
    [activeScene]
  );

  const effectiveMaskType =
    redactionMode === 'AUTO' ? activeScene.defaultAction : redactionMode;

  return (
    <section className="bg-[#252423] border-2 border-[#8EC07C]/55 rounded-xl p-6 space-y-5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#8EC07C]">
            <Eye className="w-4 h-4" />
            Interactive Vision Grounding &amp; VEIL Privacy Auto-Redaction Canvas (&quot;Vision Vibe Check&quot;)
            <InfoTooltip glossaryKey="bboxGrounding" />
          </div>
          <h3 className="text-base font-semibold text-[#FFE9B1]">
            Base VLM vs. Fine-Tuned VLM Spatial `[ymin, xmin, ymax, xmax]` Localization &amp; Live `BLUR` / `MOSAIC` Masking
          </h3>
          <p className="text-xs text-[#A89984] max-w-4xl">
            Inspects normalized <code className="font-mono text-[#83A598]">[0, 1000]</code> bounding-box accuracy across all 5 VEIL Audit Taxonomy categories and previews real-time Gaussian Blur (<code className="font-mono text-[#FABD2F]">BLUR</code>) or 16px Pixelation (<code className="font-mono text-[#8EC07C]">MOSAIC</code>) privacy redaction.
          </p>
        </div>

        {/* Redaction Mask Mode Selector */}
        <div className="flex flex-wrap items-center gap-1.5 bg-[#1D2021] p-1.5 rounded-xl border border-[#3C3836]">
          {[
            { id: 'AUTO', label: `Auto Policy (${activeScene.defaultAction})` },
            { id: 'BLUR', label: 'Gaussian BLUR' },
            { id: 'MOSAIC', label: 'Pixel MOSAIC' },
            { id: 'OUTLINE_ONLY', label: 'Raw BBox Only' },
          ].map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setRedactionMode(m.id as typeof redactionMode)}
              className={`px-3 py-1.5 text-xs font-mono rounded-lg transition-colors ${
                redactionMode === m.id
                  ? 'bg-[#8EC07C] text-[#1D2021] font-bold'
                  : 'text-[#A89984] hover:text-[#FFE9B1]'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* 5 VEIL Scene Selector Buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5">
        {VEIL_SANDBOX_TEST_SCENES.map((sc) => {
          const isSelected = sc.id === activeScene.id;
          return (
            <button
              key={sc.id}
              type="button"
              onClick={() => setSelectedSceneId(sc.id)}
              className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between gap-2 ${
                isSelected
                  ? 'bg-[#1D2021] border-[#8EC07C] ring-1 ring-[#8EC07C]/40'
                  : 'bg-[#1D2021]/60 border-[#3C3836] hover:border-[#504945]'
              }`}
            >
              <div>
                <div className="text-[10px] font-mono uppercase text-[#83A598]">
                  Category {sc.veilCategoryIndex} · {sc.defaultAction}
                </div>
                <div className="text-xs font-semibold text-[#FFE9B1] mt-1 line-clamp-2">
                  {sc.title}
                </div>
              </div>
              <div className="text-[10px] font-mono text-[#B8BB26]">
                {sc.detections.length > 0
                  ? `${sc.detections.length} PII Hazard Region(s)`
                  : '0 Hazards (SAFE_PASS Control)'}
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Split View: Interactive Visual Canvas (Left 7 cols) + IoU & JSON Telemetry (Right 5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 bg-[#141617] border border-[#3C3836] rounded-xl p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <span className="text-[#FFE9B1] font-semibold">{activeScene.veilCategoryName}</span>
            <div className="flex items-center gap-3">
              <label className="inline-flex items-center gap-1.5 cursor-pointer text-[#FB4934]">
                <input
                  type="checkbox"
                  checked={showBaseBoxes}
                  onChange={(e) => setShowBaseBoxes(e.target.checked)}
                  className="accent-[#FB4934]"
                />
                <span>Base VLM Drift/Hallucination (Dashed Red)</span>
              </label>
              <label className="inline-flex items-center gap-1.5 cursor-pointer text-[#8EC07C]">
                <input
                  type="checkbox"
                  checked={showFineTunedBoxes}
                  onChange={(e) => setShowFineTunedBoxes(e.target.checked)}
                  className="accent-[#8EC07C]"
                />
                <span>Fine-Tuned VLM + Redaction (Solid Green)</span>
              </label>
            </div>
          </div>

          {/* Normalized [0, 1000] Coordinate Viewport */}
          <div
            data-testid="veil-vision-sandbox-canvas"
            className="relative w-full h-80 rounded-xl overflow-hidden border border-[#4A453E] bg-gradient-to-br from-[#1D2021] via-[#282828] to-[#1D2021] select-none"
          >
            {/* Subtle Coordinate Grid Overlay */}
            <div className="absolute inset-0 opacity-15 pointer-events-none grid grid-cols-10 grid-rows-10">
              {Array.from({ length: 100 }).map((_, idx) => (
                <div key={idx} className="border border-[#A89984]/30" />
              ))}
            </div>

            {/* Scene Context Watermark */}
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-[#141617]/85 border border-[#3C3836] text-[10px] font-mono text-[#D5C4A1]">
              Canvas Coordinate Space: [0, 1000] × [0, 1000] · Theme: {activeScene.backgroundTheme}
            </div>

            {/* Simulated Underlying Visual Artifacts + Redaction Overlays */}
            {activeScene.detections.map((det) => {
              const [y1, x1, y2, x2] = det.fineTunedBox2d;
              const topPct = y1 / 10;
              const leftPct = x1 / 10;
              const heightPct = (y2 - y1) / 10;
              const widthPct = (x2 - x1) / 10;

              const baseBox = det.baseModelBox2d;

              return (
                <React.Fragment key={det.id}>
                  {/* Underlying Artifact Card */}
                  <div
                    style={{
                      top: `${topPct}%`,
                      left: `${leftPct}%`,
                      width: `${widthPct}%`,
                      height: `${heightPct}%`,
                    }}
                    className="absolute rounded-lg bg-[#32302F] border border-[#504945] p-3 flex flex-col justify-between overflow-hidden shadow-lg"
                  >
                    <div className="text-[10px] font-mono uppercase text-[#FABD2F] font-bold">
                      RAW ARTIFACT: {det.label}
                    </div>
                    <div
                      className={`text-xs font-mono text-[#FFE9B1] bg-[#1D2021] p-2 rounded border border-[#3C3836] transition-all ${
                        showFineTunedBoxes && effectiveMaskType === 'BLUR'
                          ? 'blur-xs select-none'
                          : ''
                      }`}
                    >
                      {det.extractedOcrSnippet}
                    </div>

                    {/* Pixelation MOSAIC Grid Overlay when MOSAIC is active */}
                    {showFineTunedBoxes && effectiveMaskType === 'MOSAIC' && (
                      <div
                        aria-label={`Mosaic redaction mask on ${det.label}`}
                        className="absolute inset-0 grid grid-cols-8 grid-rows-6 opacity-90 pointer-events-none"
                      >
                        {Array.from({ length: 48 }).map((_, cellIdx) => (
                          <div
                            key={cellIdx}
                            className={
                              cellIdx % 3 === 0
                                ? 'bg-[#3C3836]'
                                : cellIdx % 2 === 0
                                  ? 'bg-[#504945]'
                                  : 'bg-[#282828]'
                            }
                          />
                        ))}
                        <div className="col-span-8 flex items-center justify-center text-[10px] font-mono font-bold text-[#8EC07C] bg-black/65">
                          [MOSAIC 16px REDACTED · {det.label}]
                        </div>
                      </div>
                    )}

                    {/* Gaussian BLUR Badge when BLUR is active */}
                    {showFineTunedBoxes && effectiveMaskType === 'BLUR' && (
                      <div className="text-[10px] font-mono font-bold text-[#FABD2F] flex items-center gap-1">
                        <EyeOff className="w-3 h-3" />
                        GAUSSIAN BLUR ACTIVE ([{y1}, {x1}, {y2}, {x2}])
                      </div>
                    )}
                  </div>

                  {/* Base Model Drifted Bounding Box (Dashed Red) */}
                  {showBaseBoxes && baseBox && (
                    <div
                      style={{
                        top: `${baseBox[0] / 10}%`,
                        left: `${baseBox[1] / 10}%`,
                        width: `${(baseBox[3] - baseBox[1]) / 10}%`,
                        height: `${(baseBox[2] - baseBox[0]) / 10}%`,
                      }}
                      className="absolute border-2 border-dashed border-[#FB4934] rounded-lg pointer-events-none"
                    >
                      <span className="absolute -top-5 left-0 px-1.5 py-0.5 rounded bg-[#FB4934] text-[#1D2021] text-[9px] font-mono font-bold">
                        Base VLM: [{baseBox.join(', ')}] (IoU{' '}
                        {computeBox2dIoU(det.groundTruthBox2d, baseBox)})
                      </span>
                    </div>
                  )}

                  {/* Fine-Tuned Model Grounded Bounding Box (Solid Green) */}
                  {showFineTunedBoxes && (
                    <div
                      style={{
                        top: `${topPct}%`,
                        left: `${leftPct}%`,
                        width: `${widthPct}%`,
                        height: `${heightPct}%`,
                      }}
                      className="absolute border-2 border-[#8EC07C] rounded-lg pointer-events-none ring-2 ring-[#8EC07C]/30"
                    >
                      <span className="absolute -bottom-5 right-0 px-1.5 py-0.5 rounded bg-[#8EC07C] text-[#1D2021] text-[9px] font-mono font-bold">
                        Fine-Tuned VLM: [{det.fineTunedBox2d.join(', ')}] (IoU{' '}
                        {computeBox2dIoU(det.groundTruthBox2d, det.fineTunedBox2d)})
                      </span>
                    </div>
                  )}
                </React.Fragment>
              );
            })}

            {/* Category 5 Clean Studio Baseline: Base Model False Positive Hallucination vs Fine-Tuned SAFE_PASS */}
            {activeScene.detections.length === 0 && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
                <div className="p-4 rounded-xl bg-[#1D2021]/90 border border-[#B8BB26]/60 max-w-md space-y-2">
                  <div className="inline-flex items-center gap-1.5 text-xs font-mono font-bold text-[#B8BB26]">
                    <ShieldCheck className="w-4 h-4" />
                    FINE-TUNED VLM VERDICT: SAFE_PASS (0 PII Hazards)
                  </div>
                  <p className="text-xs text-[#D5C4A1]">{activeScene.sceneSummary}</p>
                </div>

                {showBaseBoxes && activeScene.baseHallucinatedBox && (
                  <div
                    style={{
                      top: `${activeScene.baseHallucinatedBox.box2d[0] / 10}%`,
                      left: `${activeScene.baseHallucinatedBox.box2d[1] / 10}%`,
                      width: `${
                        (activeScene.baseHallucinatedBox.box2d[3] -
                          activeScene.baseHallucinatedBox.box2d[1]) /
                        10
                      }%`,
                      height: `${
                        (activeScene.baseHallucinatedBox.box2d[2] -
                          activeScene.baseHallucinatedBox.box2d[0]) /
                        10
                      }%`,
                    }}
                    className="absolute border-2 border-dashed border-[#FB4934] bg-[#FB4934]/10 rounded-lg pointer-events-none flex items-start justify-start p-2"
                  >
                    <span className="px-1.5 py-0.5 rounded bg-[#FB4934] text-[#1D2021] text-[9px] font-mono font-bold">
                      Base VLM Hallucination: {activeScene.baseHallucinatedBox.label}
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Spatial Grounding IoU Telemetry & Structured JSON Output */}
        <div className="lg:col-span-5 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-[#83A598]" />
                Spatial Grounding IoU &amp; VEIL Audit Verdict
              </h4>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-[#8EC07C]/20 text-[#8EC07C]">
                Action: {evalSummary.recommendedRedactionMode}
              </span>
            </div>

            <p className="text-xs text-[#A89984] leading-relaxed">{activeScene.sceneSummary}</p>

            <div className="grid grid-cols-3 gap-2.5 text-center font-mono text-xs">
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Base Mean IoU</div>
                <div className="text-[#FB4934] font-bold mt-0.5">{evalSummary.baseMeanIoU}</div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">Fine-Tuned IoU</div>
                <div className="text-[#8EC07C] font-bold mt-0.5">
                  {evalSummary.fineTunedMeanIoU}
                </div>
              </div>
              <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]">
                <div className="text-[10px] text-[#A89984]">IoU Lift</div>
                <div className="text-[#B8BB26] font-bold mt-0.5">+{evalSummary.iouDelta}</div>
              </div>
            </div>

            <div>
              <div className="text-[11px] font-mono text-[#A89984] mb-1.5">
                Fine-Tuned VLM Structured VEIL Audit JSON Output:
              </div>
              <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#8EC07C] overflow-x-auto max-h-44">
                {JSON.stringify(
                  {
                    veil_category: activeScene.veilCategoryName,
                    suggested_action: activeScene.defaultAction,
                    safe_to_publish: activeScene.defaultAction === 'SAFE_PASS',
                    detections: activeScene.detections.map((d) => ({
                      label: d.label,
                      severity: d.riskSeverity,
                      action: d.suggestedAction,
                      box_2d: d.fineTunedBox2d,
                    })),
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          </div>

          <div className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] flex items-center justify-between">
            <span>Stage 3 Vision-GRPO GIoU Verifier:</span>
            <span className="text-[#B8BB26] inline-flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Active (Zero False Alarms)
            </span>
          </div>
        </div>
      </div>
    </section>
  );
};
