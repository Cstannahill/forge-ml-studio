import React, { useMemo, useState } from 'react';
import {
  Check,
  Copy,
  Cpu,
  Download,
  Eye,
  Layers,
  Play,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import {
  RegisteredVisionModel,
  VisionPipelineHardwareConfig,
  computeVisionPipelineHardwarePlan,
} from '../lib/visionPipelineLogic';
import { InfoTooltip } from './InfoTooltip';

interface VisionCommandCenterPanelProps {
  visionModels: RegisteredVisionModel[];
  selectedVisionModelId: string;
  onSelectVisionModel: (id: string) => void;
  onTriggerVisionFineTune: (modelId: string) => void;
  onOpenVisionDatasetStudio: () => void;
  activeVisionSampleCount: number;
}

export const VisionCommandCenterPanel: React.FC<VisionCommandCenterPanelProps> = ({
  visionModels,
  selectedVisionModelId,
  onSelectVisionModel,
  onTriggerVisionFineTune,
  onOpenVisionDatasetStudio,
  activeVisionSampleCount,
}) => {
  const activeVisionModel = useMemo(
    () => visionModels.find((m) => m.id === selectedVisionModelId) ?? visionModels[0],
    [visionModels, selectedVisionModelId]
  );

  const [hwConfig, setHwConfig] = useState<VisionPipelineHardwareConfig>({
    freezeVisionTower: true,
    trainMlpProjector: true,
    applyLoraToLlmBackbone: true,
    minPixels: 256 * 28 * 28,
    maxPixels: 1024 * 28 * 28,
    patchSizePx: 28,
    visionProjectorLrMultiplier: 5,
    exportSeparateMmprojFp16: true,
  });

  const [loraRank, setLoraRank] = useState<number>(32);
  const [batchSize, setBatchSize] = useState<number>(8);
  const [copiedScript, setCopiedScript] = useState<boolean>(false);
  const [exportedBanner, setExportedBanner] = useState<string | null>(null);

  const vlmPlan = useMemo(
    () => computeVisionPipelineHardwarePlan(activeVisionModel, hwConfig, loraRank, batchSize),
    [activeVisionModel, hwConfig, loraRank, batchSize]
  );

  const baseBench = activeVisionModel.baseVisionBenchmarks;
  const ftBench = activeVisionModel.fineTunedVisionBenchmarks ?? baseBench;

  const handleCopyVisionScript = () => {
    navigator.clipboard.writeText(vlmPlan.unslothVisionScriptPreview);
    setCopiedScript(true);
    setTimeout(() => setCopiedScript(false), 1800);
  };

  const handleExportDualArtifact = () => {
    const manifest = {
      pipelineTrack: 'VISION_VLM_ISOLATED',
      modelId: activeVisionModel.id,
      architecture: activeVisionModel.architecture,
      visionEncoder: activeVisionModel.visionEncoder,
      projectorType: activeVisionModel.projectorType,
      backboneArtifact: vlmPlan.backboneExportFilename,
      mmprojArtifact: vlmPlan.mmprojExportFilename,
      freezeVisionTower: hwConfig.freezeVisionTower,
      trainMlpProjector: hwConfig.trainMlpProjector,
      patchTokenBudget: `${vlmPlan.minPatchTokensPerImage}..${vlmPlan.maxPatchTokensPerImage}`,
      benchmarks: {
        base: baseBench,
        fineTuned: ftBench,
      },
    };
    const blob = new Blob([JSON.stringify(manifest, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${vlmPlan.backboneExportFilename}.vlm-bundle.json`;
    a.click();
    URL.revokeObjectURL(url);
    setExportedBanner(
      `Exported Dual-Artifact VLM Bundle: ${vlmPlan.backboneExportFilename} + ${vlmPlan.mmprojExportFilename} (FP16 Vision Projector)`
    );
  };

  return (
    <section className="bg-[#252423] border-2 border-[#8EC07C]/50 rounded-xl p-6 space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#8EC07C]">
            <ShieldCheck className="w-4 h-4" />
            Isolated Vision-Language (VLM) Training Pipeline · Zero Text Interference
            <InfoTooltip glossaryKey="vlmIsolatedPipeline" />
          </div>
          <h2 className="text-lg font-bold text-[#FFE9B1]">
            Multimodal Vision Tower, Connector &amp; Dynamic Patch Studio (Qwen3.5-VL &amp; LFM2.5-VL 450M/1.6B)
          </h2>
          <p className="text-xs text-[#A89984]">
            Dedicated VLM pipeline track: freezes or tunes the SigLIP / NaViT Vision Encoder, trains the Multimodal MLP Connector, and exports dual-file{' '}
            <code className="font-mono text-[#83A598]">GGUF + mmproj-f16.gguf</code> without touching your pure Text LLM weights.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={onOpenVisionDatasetStudio}
            className="px-3.5 py-2 text-xs font-medium text-[#FFE9B1] bg-[#1D2021] hover:bg-[#3C3836] border border-[#83A598]/60 rounded-lg transition-colors flex items-center gap-1.5"
          >
            <Eye className="w-3.5 h-3.5 text-[#83A598]" />
            Open Vision Dataset Studio ({activeVisionSampleCount} Image Pairs)
          </button>

          <button
            type="button"
            onClick={() => onTriggerVisionFineTune(activeVisionModel.id)}
            className="px-4 py-2 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center gap-1.5"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Run Vision Fine-Tune &amp; Eval
          </button>
        </div>
      </div>

      {/* Vision Model Cards (LFM2.5-VL-450M, LFM2.5-VL-1.6B, Qwen3.5-VL-0.8B, Qwen3.5-VL-2B, Gemma4-E2B-Vision) */}
      <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-3">
        {visionModels.map((vm) => {
          const isSelected = vm.id === activeVisionModel.id;
          const ft = vm.fineTunedVisionBenchmarks ?? vm.baseVisionBenchmarks;
          return (
            <button
              key={vm.id}
              type="button"
              onClick={() => onSelectVisionModel(vm.id)}
              className={`text-left p-3.5 rounded-xl border transition-all flex flex-col justify-between gap-2.5 ${
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
                <div className="text-xs font-semibold text-[#FFE9B1] mt-1.5 leading-snug">
                  {vm.name}
                </div>
                <div className="text-[10px] text-[#A89984] mt-0.5 truncate">
                  {vm.visionEncoder}
                </div>
              </div>

              <div className="pt-2 border-t border-[#3C3836] flex items-center justify-between text-[11px] font-mono">
                <span className="text-[#8EC07C]">DocVQA: {ft.docVqa}%</span>
                <span className="text-[#B8BB26]">MMMU: {ft.mmmu}%</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Vision Tower & Projector Controls + Dual-Artifact Export */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#83A598]" />
              Vision Tower Freeze, MLP Connector &amp; Dynamic Patch Budget
              <InfoTooltip glossaryKey="visionTowerFreeze" />
            </h3>
            <span className="text-xs font-mono text-[#8EC07C]">
              Est. VLM VRAM: {vlmPlan.totalVisionTrainingVramGb} GB
            </span>
          </div>

          <div className="space-y-2.5">
            <label className="flex items-center justify-between p-2.5 rounded-lg bg-[#252423] border border-[#3C3836] cursor-pointer">
              <div>
                <div className="text-xs font-medium text-[#FFE9B1]">
                  Freeze Pretrained Vision Tower ({activeVisionModel.visionTowerParamsMillions}M params)
                </div>
                <div className="text-[11px] text-[#A89984]">
                  Prevents catastrophic visual forgetting on sub-2B models and saves ~58% encoder VRAM
                </div>
              </div>
              <input
                type="checkbox"
                checked={hwConfig.freezeVisionTower}
                onChange={(e) =>
                  setHwConfig((prev) => ({ ...prev, freezeVisionTower: e.target.checked }))
                }
                className="w-4 h-4 accent-[#8EC07C]"
              />
            </label>

            <label className="flex items-center justify-between p-2.5 rounded-lg bg-[#252423] border border-[#3C3836] cursor-pointer">
              <div>
                <div className="text-xs font-medium text-[#FFE9B1]">
                  Train Multimodal Projector ({activeVisionModel.projectorType})
                </div>
                <div className="text-[11px] text-[#A89984]">
                  Aligns 2D spatial visual patches with language token embeddings (5x projector LR)
                </div>
              </div>
              <input
                type="checkbox"
                checked={hwConfig.trainMlpProjector}
                onChange={(e) =>
                  setHwConfig((prev) => ({ ...prev, trainMlpProjector: e.target.checked }))
                }
                className="w-4 h-4 accent-[#8EC07C]"
              />
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">
                  Dynamic Patch Budget
                </label>
                <select
                  value={hwConfig.maxPixels}
                  onChange={(e) =>
                    setHwConfig((prev) => ({ ...prev, maxPixels: Number(e.target.value) }))
                  }
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={512 * 28 * 28}>Edge Fast (256–512 patches)</option>
                  <option value={1024 * 28 * 28}>Balanced VQA (256–1024 patches)</option>
                  <option value={1280 * 28 * 28}>Dense OCR/UI (256–1280 patches)</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">VLM LoRA Rank (r)</label>
                <select
                  value={loraRank}
                  onChange={(e) => setLoraRank(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={16}>r = 16 (Edge Lean)</option>
                  <option value={32}>r = 32 (Recommended)</option>
                  <option value={64}>r = 64 (Dense OCR/UI)</option>
                </select>
              </div>
              <div>
                <label className="block text-[11px] text-[#A89984] mb-1">Micro-Batch Size</label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full px-2.5 py-1.5 text-xs bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value={4}>batch = 4</option>
                  <option value={8}>batch = 8</option>
                  <option value={16}>batch = 16</option>
                </select>
              </div>
            </div>
          </div>

          {/* VRAM Breakdown Bar */}
          <div className="grid grid-cols-4 gap-2 pt-1 text-center font-mono text-[11px]">
            <div className="p-2 rounded bg-[#252423] border border-[#3C3836]">
              <div className="text-[#A89984]">Vision Tower</div>
              <div className="text-[#FFE9B1] font-bold mt-0.5">{vlmPlan.visionTowerVramGb} GB</div>
            </div>
            <div className="p-2 rounded bg-[#252423] border border-[#3C3836]">
              <div className="text-[#A89984]">MLP Projector</div>
              <div className="text-[#83A598] font-bold mt-0.5">{vlmPlan.projectorVramGb} GB</div>
            </div>
            <div className="p-2 rounded bg-[#252423] border border-[#3C3836]">
              <div className="text-[#A89984]">LLM Backbone</div>
              <div className="text-[#8EC07C] font-bold mt-0.5">{vlmPlan.llmBackboneVramGb} GB</div>
            </div>
            <div className="p-2 rounded bg-[#252423] border border-[#3C3836]">
              <div className="text-[#A89984]">Patch KV/Act</div>
              <div className="text-[#FABD2F] font-bold mt-0.5">
                {vlmPlan.activationPatchVramGb} GB
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Vision Benchmarks + Dual-File mmproj Exporter */}
        <div className="lg:col-span-6 bg-[#1D2021] border border-[#3C3836] rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#8EC07C]" />
              Multimodal Vision Benchmarks &amp; Dual-File mmproj Export
              <InfoTooltip glossaryKey="mmprojExport" />
            </h3>
            <button
              type="button"
              onClick={handleCopyVisionScript}
              className="px-2.5 py-1 text-[11px] font-mono rounded bg-[#252423] border border-[#4A453E] text-[#D5C4A1] hover:text-[#FFE9B1] inline-flex items-center gap-1"
            >
              {copiedScript ? (
                <>
                  <Check className="w-3 h-3 text-[#B8BB26]" />
                  Copied FastVisionModel
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  Copy FastVisionModel Script
                </>
              )}
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center font-mono">
            {[
              { label: 'MMMU', base: baseBench.mmmu, ft: ftBench.mmmu },
              { label: 'DocVQA', base: baseBench.docVqa, ft: ftBench.docVqa },
              { label: 'MathVista', base: baseBench.mathVista, ft: ftBench.mathVista },
              { label: 'RealWorldQA', base: baseBench.realWorldQa, ft: ftBench.realWorldQa },
              { label: 'BBox IoU', base: baseBench.groundingIoU, ft: ftBench.groundingIoU },
            ].map((m) => {
              const delta = Number((m.ft - m.base).toFixed(1));
              return (
                <div
                  key={m.label}
                  className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836]"
                >
                  <div className="text-[10px] text-[#A89984]">{m.label}</div>
                  <div className="text-xs font-bold text-[#FFE9B1] mt-0.5 tabular-nums">
                    {m.ft}%
                  </div>
                  <div className="text-[10px] text-[#B8BB26] tabular-nums">+{delta}%</div>
                </div>
              );
            })}
          </div>

          <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto leading-relaxed max-h-36">
            {vlmPlan.unslothVisionScriptPreview}
          </pre>

          {exportedBanner && (
            <div className="p-2 rounded bg-[#252423] border border-[#B8BB26]/40 text-[11px] font-mono text-[#B8BB26]">
              {exportedBanner}
            </div>
          )}

          <button
            type="button"
            onClick={handleExportDualArtifact}
            className="w-full py-2.5 px-4 text-xs font-semibold text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export Dual-File VLM Checkpoint ({vlmPlan.backboneExportFilename} +{' '}
            {vlmPlan.mmprojExportFilename})
          </button>
        </div>
      </div>
    </section>
  );
};
