import React, { useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  Grid,
  Layers,
  Plus,
  Sparkles,
} from 'lucide-react';
import {
  NormalizedVisionSample,
  RegisteredVisionModel,
  VisionTargetFormat,
  VisionTilingMode,
  appendBoundingBoxToVisionSample,
  computeVisionTileGridAndSeamSplits,
} from '../lib/visionPipelineLogic';

interface VisionTileGridAndBboxDrawerPanelProps {
  activeSample: NormalizedVisionSample;
  activeVisionModel: RegisteredVisionModel;
  visionTargetFormat: VisionTargetFormat;
  onUpdateSample: (updated: NormalizedVisionSample, summaryMsg: string) => void;
}

export const VisionTileGridAndBboxDrawerPanel: React.FC<
  VisionTileGridAndBboxDrawerPanelProps
> = ({ activeSample, activeVisionModel, visionTargetFormat, onUpdateSample }) => {
  const defaultMode: VisionTilingMode =
    activeVisionModel.architecture === 'Gemma4'
      ? 'GEMMA4_PAN_SCAN_896'
      : activeVisionModel.architecture === 'LFM2.5'
      ? 'LFM_SIGLIP_384'
      : 'QWEN_NAVIT_28';

  const [tilingMode, setTilingMode] = useState<VisionTilingMode>(defaultMode);
  const [enableSlidingOverlap, setEnableSlidingOverlap] = useState<boolean>(false);

  // Interactive Bounding-Box Drawer state ([y1, x1, y2, x2] in [0, 1000])
  const [newLabel, setNewLabel] = useState<string>('custom_redaction_hazard');
  const [newVeilAction, setNewVeilAction] = useState<
    'REDACT' | 'BLUR' | 'MOSAIC' | 'SAFE_PASS'
  >('REDACT');
  const [draftBox, setDraftBox] = useState<[number, number, number, number]>([
    240, 320, 620, 740,
  ]);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
  const [statusBanner, setStatusBanner] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const report = computeVisionTileGridAndSeamSplits(
    activeSample,
    tilingMode,
    enableSlidingOverlap
  );

  const toNormalizedCoords = (clientX: number, clientY: number) => {
    if (!svgRef.current) return { x: 500, y: 500 };
    const rect = svgRef.current.getBoundingClientRect();
    const x = Math.max(
      0,
      Math.min(1000, Math.round(((clientX - rect.left) / Math.max(1, rect.width)) * 1000))
    );
    const y = Math.max(
      0,
      Math.min(1000, Math.round(((clientY - rect.top) / Math.max(1, rect.height)) * 1000))
    );
    return { x, y };
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const pt = toNormalizedCoords(e.clientX, e.clientY);
    setIsDragging(true);
    setDragStart(pt);
    setDraftBox([pt.y, pt.x, Math.min(1000, pt.y + 80), Math.min(1000, pt.x + 120)]);
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDragging || !dragStart) return;
    const pt = toNormalizedCoords(e.clientX, e.clientY);
    const y1 = Math.min(dragStart.y, pt.y);
    const x1 = Math.min(dragStart.x, pt.x);
    const y2 = Math.max(y1 + 30, Math.max(dragStart.y, pt.y));
    const x2 = Math.max(x1 + 30, Math.max(dragStart.x, pt.x));
    setDraftBox([y1, x1, Math.min(1000, y2), Math.min(1000, x2)]);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setDragStart(null);
  };

  const handleAppendAnnotatedBox = () => {
    const colorMap: Record<typeof newVeilAction, string> = {
      REDACT: '#FB4934',
      BLUR: '#FABD2F',
      MOSAIC: '#D3869B',
      SAFE_PASS: '#8EC07C',
    };
    const updated = appendBoundingBoxToVisionSample(
      activeSample,
      {
        label: newLabel,
        box2d: draftBox,
        colorHex: colorMap[newVeilAction],
        veilAction: newVeilAction,
      },
      visionTargetFormat
    );
    const msg = `Appended normalized [0, 1000] bounding box <ref>${newLabel}</ref><box>[${draftBox.join(', ')}]</box> (${newVeilAction}) to ${activeSample.id}.`;
    setStatusBanner(msg);
    onUpdateSample(updated, msg);
  };

  return (
    <section
      aria-label="Interactive Pan-and-Scan Tile Grid Visualizer and Live Bounding-Box Drawer"
      className="bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-5"
    >
      {/* Stacked Header + Mode Controls */}
      <div className="flex flex-col gap-3 border-b border-[#3C3836] pb-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="px-2.5 py-0.5 text-[11px] font-mono uppercase tracking-wider rounded bg-[#458588]/25 text-[#83A598] border border-[#83A598]/40">
              V10 · Spatial Tile Seam Diagnostics &amp; Grounding Studio
            </span>
            <span
              className={`px-2.5 py-0.5 text-[11px] font-mono rounded font-semibold ${
                report.unmitigatedSeamSplitCount > 0
                  ? 'bg-[#FB4934]/20 text-[#FB4934] border border-[#FB4934]/40'
                  : 'bg-[#8EC07C]/20 text-[#8EC07C] border border-[#8EC07C]/40'
              }`}
            >
              {report.unmitigatedSeamSplitCount > 0
                ? `${report.unmitigatedSeamSplitCount} Seam-Split Hazard(s) Detected`
                : '0 Unmitigated Seam Splits · Tile Overlap Protected'}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 bg-[#1D2021] p-1 rounded-lg border border-[#3C3836]">
              <button
                type="button"
                onClick={() => setTilingMode('GEMMA4_PAN_SCAN_896')}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                  tilingMode === 'GEMMA4_PAN_SCAN_896'
                    ? 'bg-[#458588] text-[#FFE9B1] font-semibold'
                    : 'text-[#A89984] hover:text-[#FFE9B1]'
                }`}
              >
                Gemma4 Pan-&amp;-Scan (896×896 Crops)
              </button>
              <button
                type="button"
                onClick={() => setTilingMode('QWEN_NAVIT_28')}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                  tilingMode === 'QWEN_NAVIT_28'
                    ? 'bg-[#458588] text-[#FFE9B1] font-semibold'
                    : 'text-[#A89984] hover:text-[#FFE9B1]'
                }`}
              >
                Qwen3.5 NaViT (28×28 Grid)
              </button>
              <button
                type="button"
                onClick={() => setTilingMode('LFM_SIGLIP_384')}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                  tilingMode === 'LFM_SIGLIP_384'
                    ? 'bg-[#458588] text-[#FFE9B1] font-semibold'
                    : 'text-[#A89984] hover:text-[#FFE9B1]'
                }`}
              >
                LFM2.5 SigLIP (384×384 Tiles)
              </button>
            </div>

            <button
              type="button"
              onClick={() => setEnableSlidingOverlap((prev) => !prev)}
              className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors whitespace-nowrap ${
                enableSlidingOverlap
                  ? 'bg-[#8EC07C] text-[#1D2021] border-[#8EC07C] font-semibold'
                  : 'bg-[#1D2021] text-[#FABD2F] border-[#FABD2F]/50 hover:bg-[#3C3836]'
              }`}
            >
              {enableSlidingOverlap
                ? '12% Sliding-Window Overlap: ON (+IoU)'
                : 'Enable 12% Sliding-Window Overlap'}
            </button>
          </div>
        </div>

        <div>
          <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
            <Grid className="w-4 h-4 text-[#83A598] shrink-0" />
            <span>
              V10. Interactive Pan-&amp;-Scan / NaViT Tile Grid Visualizer, Seam-Split Detector &amp; Live [0, 1000] Bounding-Box Drawer
            </span>
          </h3>
          <p className="text-xs text-[#A89984] mt-0.5">
            Inspect how {activeVisionModel.name} slices{' '}
            <strong className="text-[#FFE9B1]">{activeSample.imageTitle}</strong> into{' '}
            {report.gridRows}×{report.gridCols} visual tiles ({report.totalVisualTokensWithTiling}{' '}
            tokens), detect bounding boxes sliced across tile seams, and click-and-drag on the canvas to draw new normalized <code className="font-mono text-[#83A598]">[y1, x1, y2, x2]</code> grounding boxes.
          </p>
        </div>
      </div>

      {statusBanner && (
        <div
          role="status"
          className="p-3 rounded-lg bg-[#1D2021] border border-[#8EC07C]/50 text-xs font-mono text-[#8EC07C] flex items-center justify-between"
        >
          <span>{statusBanner}</span>
          <button
            type="button"
            onClick={() => setStatusBanner(null)}
            className="text-[#A89984] hover:text-[#FFE9B1] text-[11px]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* 4-Card Telemetry Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono">
        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Active Tile Topology</div>
          <div className="text-sm font-bold text-[#FFE9B1] mt-0.5 tabular-nums">
            {report.gridRows}×{report.gridCols} Grid ({report.tiles.length} Tiles)
          </div>
          <div className="text-[10px] text-[#83A598] mt-0.5">
            {tilingMode === 'GEMMA4_PAN_SCAN_896'
              ? '+256 Global Thumbnail Tok'
              : 'Dynamic 2D-RoPE Stride 28'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Tiled Visual Token Budget</div>
          <div className="text-sm font-bold text-[#83A598] mt-0.5 tabular-nums">
            {report.totalVisualTokensWithTiling} tokens
          </div>
          <div className="text-[10px] text-[#A89984] mt-0.5">
            Sliding Overlap: {report.slidingOverlapPct}%
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Boundary Seam Splits</div>
          <div
            className={`text-sm font-bold mt-0.5 tabular-nums ${
              report.unmitigatedSeamSplitCount > 0 ? 'text-[#FB4934]' : 'text-[#8EC07C]'
            }`}
          >
            {report.seamSplitHazards.length} Box(es) Cross Seams
          </div>
          <div className="text-[10px] text-[#D5C4A1] mt-0.5">
            {enableSlidingOverlap ? 'Mitigated by 12% overlap' : 'Toggle overlap to heal seams'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Projected Small-Object IoU</div>
          <div className="text-sm font-bold text-[#B8BB26] mt-0.5 tabular-nums">
            {enableSlidingOverlap
              ? `+${report.projectedSmallObjectIoUGainPct}% IoU Lift`
              : 'Baseline (0% Overlap)'}
          </div>
          <div className="text-[10px] text-[#A89984] mt-0.5">
            Fine-Print OCR &amp; BBox Recall
          </div>
        </div>
      </div>

      {/* Main 12-Col Split: Left 7 Cols Interactive Tile Grid + BBox Drawer SVG | Right 5 Cols Seam-Split Diagnostics + BBox Control Form */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 7 Cols: Interactive SVG Tile Grid + Click-and-Drag Bounding-Box Drawer */}
        <div className="lg:col-span-7 p-4 rounded-xl bg-[#1D2021] border border-[#3C3836] flex flex-col justify-between space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            <span className="text-[#FFE9B1] font-semibold flex items-center gap-1.5">
              <Crosshair className="w-3.5 h-3.5 text-[#FABD2F]" />
              Click &amp; Drag on Canvas to Draw [0, 1000] Grounding Box
            </span>
            <span className="text-[#FABD2F] tabular-nums">
              Draft Box: [{draftBox[0]}, {draftBox[1]}, {draftBox[2]}, {draftBox[3]}]
            </span>
          </div>

          <div className="relative w-full h-72 rounded-lg overflow-hidden bg-[#141617] border border-[#4A453E] select-none">
            <svg
              ref={svgRef}
              viewBox="0 0 1000 1000"
              preserveAspectRatio="none"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              className="w-full h-full cursor-crosshair"
            >
              {/* Scene Backdrop */}
              <rect width="1000" height="1000" fill="#181A1B" />
              <rect
                x="50"
                y="45"
                width="900"
                height="910"
                rx="14"
                fill="#22211F"
                stroke="#3C3836"
                strokeWidth="4"
              />
              <text
                x="85"
                y="105"
                fill="#D5C4A1"
                fontSize="26"
                fontFamily="monospace"
                fontWeight="bold"
              >
                {activeSample.imageTitle} ({activeSample.widthPx}×{activeSample.heightPx})
              </text>

              {/* Render Pan-&-Scan / NaViT Tile Boundaries */}
              {report.tiles.map((tile) => {
                const [ty1, tx1, ty2, tx2] = tile.box2d;
                return (
                  <g key={tile.tileId}>
                    <rect
                      x={tx1}
                      y={ty1}
                      width={Math.max(10, tx2 - tx1)}
                      height={Math.max(10, ty2 - ty1)}
                      fill={enableSlidingOverlap ? '#8EC07C' : '#83A598'}
                      fillOpacity={enableSlidingOverlap ? 0.05 : 0.03}
                      stroke={enableSlidingOverlap ? '#8EC07C' : '#83A598'}
                      strokeWidth="3"
                      strokeDasharray={enableSlidingOverlap ? '14,6' : '8,8'}
                    />
                    <text
                      x={tx1 + 16}
                      y={ty1 + 36}
                      fill={enableSlidingOverlap ? '#8EC07C' : '#83A598'}
                      fontSize="22"
                      fontFamily="monospace"
                    >
                      {tile.tileId} ({tile.patchTokenCount}p)
                    </text>
                  </g>
                );
              })}

              {/* Existing Sample Bounding Boxes */}
              {activeSample.boundingBoxes.map((b, idx) => {
                const [y1, x1, y2, x2] = b.box2d;
                const isSplit = report.seamSplitHazards.some(
                  (h) => h.label === b.label && !h.resolvedByOverlap
                );
                const stroke = isSplit ? '#FB4934' : b.colorHex || '#8EC07C';
                return (
                  <g key={`${b.label}-${idx}`}>
                    <rect
                      x={x1}
                      y={y1}
                      width={Math.max(20, x2 - x1)}
                      height={Math.max(20, y2 - y1)}
                      fill={stroke}
                      fillOpacity="0.16"
                      stroke={stroke}
                      strokeWidth="6"
                    />
                    <text
                      x={x1 + 10}
                      y={Math.max(30, y1 - 10)}
                      fill={stroke}
                      fontSize="23"
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {b.label} {isSplit ? '[SEAM SPLIT!]' : ''}
                    </text>
                  </g>
                );
              })}

              {/* Live User-Drawn Draft Bounding Box */}
              <g>
                <rect
                  x={draftBox[1]}
                  y={draftBox[0]}
                  width={Math.max(20, draftBox[3] - draftBox[1])}
                  height={Math.max(20, draftBox[2] - draftBox[0])}
                  fill="#FABD2F"
                  fillOpacity="0.22"
                  stroke="#FABD2F"
                  strokeWidth="7"
                  strokeDasharray="10,5"
                />
                <text
                  x={draftBox[1] + 10}
                  y={Math.max(32, draftBox[0] + 32)}
                  fill="#FABD2F"
                  fontSize="24"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  NEW: {newLabel} [{draftBox.join(',')}]
                </text>
              </g>
            </svg>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#A89984]">
            <span>
              Coordinate Convention: Canonical{' '}
              <code className="text-[#FFE9B1]">[ymin, xmin, ymax, xmax]</code> normalized to{' '}
              <code className="text-[#83A598]">[0, 1000]</code>
            </span>
            <span className="text-[#8EC07C]">
              Active Boxes on Sample: {activeSample.boundingBoxes.length}
            </span>
          </div>
        </div>

        {/* Right 5 Cols: Seam-Split Hazard Inspector + Live Bounding-Box Annotation Controls */}
        <div className="lg:col-span-5 p-4 rounded-xl bg-[#1D2021] border border-[#3C3836] flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="text-xs font-semibold text-[#FFE9B1] flex items-center justify-between">
              <span>Seam-Split Boundary Hazard Audit</span>
              <span className="font-mono text-[11px] text-[#83A598]">
                {report.seamSplitHazards.length} crossing(s)
              </span>
            </div>

            {report.seamSplitHazards.length === 0 ? (
              <div className="p-3 rounded-lg bg-[#252423] border border-[#3C3836] text-xs text-[#8EC07C] font-mono">
                All bounding boxes fit cleanly inside single tile boundaries without seam bisection.
              </div>
            ) : (
              <div className="space-y-2 max-h-36 overflow-y-auto pr-1">
                {report.seamSplitHazards.map((haz, i) => (
                  <div
                    key={`${haz.label}-${i}`}
                    className={`p-2.5 rounded-lg border text-xs font-mono space-y-1 ${
                      haz.resolvedByOverlap
                        ? 'bg-[#252423] border-[#8EC07C]/40'
                        : 'bg-[#252423] border-[#FB4934]/50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[#FFE9B1] font-semibold">{haz.label}</span>
                      <span
                        className={
                          haz.resolvedByOverlap ? 'text-[#8EC07C]' : 'text-[#FB4934]'
                        }
                      >
                        {haz.resolvedByOverlap
                          ? 'RESOLVED (12% Overlap)'
                          : `-${haz.estimatedIoUPenaltyPct}% IoU Penalty`}
                      </span>
                    </div>
                    <div className="text-[11px] text-[#A89984]">
                      Axis: {haz.splitAxis} · Spans {haz.spannedTileCount} tiles · Box [
                      {haz.box2d.join(', ')}]
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Live Bounding-Box Annotation Form */}
            <div className="pt-2 border-t border-[#3C3836] space-y-3">
              <div className="text-xs font-semibold text-[#FFE9B1]">
                Live Grounding Box Drawer &amp; VEIL Action Injector
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div>
                  <label className="block text-[11px] text-[#A89984] mb-1">
                    Region Label (<code className="font-mono">&lt;ref&gt;</code>)
                  </label>
                  <input
                    type="text"
                    aria-label="New Bounding Box Label"
                    value={newLabel}
                    onChange={(e) => setNewLabel(e.target.value)}
                    className="w-full px-2.5 py-1.5 text-xs font-mono bg-[#252423] border border-[#4A453E] rounded text-[#FFE9B1]"
                  />
                </div>
                <div>
                  <label className="block text-[11px] text-[#A89984] mb-1">
                    VEIL Remediation Policy
                  </label>
                  <select
                    aria-label="VEIL Remediation Policy"
                    value={newVeilAction}
                    onChange={(e) =>
                      setNewVeilAction(
                        e.target.value as 'REDACT' | 'BLUR' | 'MOSAIC' | 'SAFE_PASS'
                      )
                    }
                    className="w-full px-2.5 py-1.5 text-xs font-mono bg-[#252423] border border-[#4A453E] rounded text-[#FFE9B1]"
                  >
                    <option value="REDACT">REDACT (Opaque Box)</option>
                    <option value="BLUR">BLUR (Gaussian Privacy)</option>
                    <option value="MOSAIC">MOSAIC (Biometric Ink)</option>
                    <option value="SAFE_PASS">SAFE_PASS (Hard Negative)</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-2 font-mono text-xs">
                {(['ymin', 'xmin', 'ymax', 'xmax'] as const).map((coordName, idx) => (
                  <div key={coordName}>
                    <label className="block text-[10px] text-[#A89984] mb-0.5">
                      {coordName}
                    </label>
                    <input
                      type="number"
                      min={0}
                      max={1000}
                      value={draftBox[idx]}
                      onChange={(e) => {
                        const val = Math.max(0, Math.min(1000, Number(e.target.value)));
                        const next: [number, number, number, number] = [...draftBox];
                        next[idx] = val;
                        setDraftBox(next);
                      }}
                      className="w-full px-2 py-1 text-xs bg-[#252423] border border-[#4A453E] rounded text-[#FABD2F] tabular-nums"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleAppendAnnotatedBox}
            className="w-full py-2.5 px-4 text-xs font-mono font-semibold text-[#1D2021] bg-[#FABD2F] hover:bg-[#fcce4c] rounded-lg transition-colors flex items-center justify-center gap-1.5"
          >
            <Plus className="w-4 h-4" />
            <span>
              + Add Annotated [0, 1000] BBox to {activeSample.id} ({visionTargetFormat})
            </span>
          </button>
        </div>
      </div>
    </section>
  );
};
