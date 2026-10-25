import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Eye,
  FileCode2,
  Filter,
  GitCommit,
  Image as ImageIcon,
  Layers,
  RefreshCw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react';
import {
  CURATED_VISION_BLENDS,
  CuratedVisionBlend,
  NormalizedVisionSample,
  RegisteredVisionModel,
  VEIL_AUDIT_TAXONOMY,
  VEIL_COMPOSITION_DIVERSITY_RULES,
  VisionTaskCategory,
  VisionTargetFormat,
  auditVisionTokenizerSpecialTokens,
  buildVeilPrivacyAuditSeedSamples,
  buildVisionCurriculumPlan,
  calculateVisualPatchTokens,
  capExcessiveNavitPatches,
  cleanAndNormalizeVisionSample,
  computeVisionTokenizerAndPatchDensity,
  formatVisionSamplePayload,
  generateVisionTeacherDistillationBatch,
  parseCustomVisionDatasetUpload,
  pruneVisionDatasetByAlignmentAndPhash,
  validateVisionDatasetCorpus,
} from '../lib/visionPipelineLogic';
import { DatasetVersionCommit } from '../data/initialPipelineData';
import { InfoTooltip } from './InfoTooltip';
import { VisionTileGridAndBboxDrawerPanel } from './VisionTileGridAndBboxDrawerPanel';

interface VisionDatasetStudioViewProps {
  visionModels: RegisteredVisionModel[];
  selectedVisionModelId: string;
  onSelectVisionModel: (id: string) => void;
  visionSamples: NormalizedVisionSample[];
  visionTargetFormat: VisionTargetFormat;
  onChangeVisionTargetFormat: (fmt: VisionTargetFormat) => void;
  onUpdateVisionSamples: (next: NormalizedVisionSample[], logMessage: string) => void;
  visionDatasetCommits: DatasetVersionCommit[];
  onCommitVisionDatasetVersion: (notes: string) => void;
}

export const VisionDatasetStudioView: React.FC<VisionDatasetStudioViewProps> = ({
  visionModels,
  selectedVisionModelId,
  onSelectVisionModel,
  visionSamples,
  visionTargetFormat,
  onChangeVisionTargetFormat,
  onUpdateVisionSamples,
  visionDatasetCommits,
  onCommitVisionDatasetVersion,
}) => {
  const activeVisionModel = useMemo(
    () => visionModels.find((m) => m.id === selectedVisionModelId) ?? visionModels[0],
    [visionModels, selectedVisionModelId]
  );

  // Filter & Search Curated Vision Blends (25 Blends)
  const [selectedBlendCategory, setSelectedBlendCategory] = useState<VisionTaskCategory | 'ALL'>('ALL');
  const [blendSearchQuery, setBlendSearchQuery] = useState<string>('');
  const [showAllBlends, setShowAllBlends] = useState<boolean>(true);

  const filteredBlends = useMemo(() => {
    const q = blendSearchQuery.trim().toLowerCase();
    return CURATED_VISION_BLENDS.filter((b) => {
      const matchesCat =
        selectedBlendCategory === 'ALL' || b.taskCategory === selectedBlendCategory;
      if (!matchesCat) return false;
      if (!q) return true;
      return (
        b.title.toLowerCase().includes(q) ||
        b.description.toLowerCase().includes(q) ||
        b.datasets.some(
          (d) => d.hfRepoId.toLowerCase().includes(q) || d.focus.toLowerCase().includes(q)
        )
      );
    });
  }, [selectedBlendCategory, blendSearchQuery]);

  // VEIL Taxonomy Active Category Inspector
  const [selectedVeilCatNumber, setSelectedVeilCatNumber] = useState<1 | 2 | 3 | 4 | 5>(1);
  const activeVeilSpec = useMemo(
    () =>
      VEIL_AUDIT_TAXONOMY.find((c) => c.categoryNumber === selectedVeilCatNumber) ??
      VEIL_AUDIT_TAXONOMY[0],
    [selectedVeilCatNumber]
  );

  // Vision Tokenizer & Patch Density Threshold Controls
  const [maxRecommendedRatio, setMaxRecommendedRatio] = useState<number>(5.5);
  const [maxRecommendedPatches, setMaxRecommendedPatches] = useState<number>(896);
  const [lastPatchCapSummary, setLastPatchCapSummary] = useState<string | null>(null);

  const patchDensityReport = useMemo(
    () =>
      computeVisionTokenizerAndPatchDensity(
        visionSamples,
        maxRecommendedRatio,
        maxRecommendedPatches
      ),
    [visionSamples, maxRecommendedRatio, maxRecommendedPatches]
  );

  // HF Hub Multimodal Repo Ingestion
  const [hfVisionRepoInput, setHfVisionRepoInput] = useState<string>(
    'HuggingFaceM4/the_cauldron'
  );
  const [hfVisionTaskTag, setHfVisionTaskTag] = useState<VisionTaskCategory>('VEIL_PRIVACY_AUDIT');

  // VLM Teacher Distillation State
  const [teacherModel, setTeacherModel] = useState<string>('Qwen2.5-VL-72B-Instruct');
  const [distillTaskCategory, setDistillTaskCategory] =
    useState<VisionTaskCategory>('VEIL_PRIVACY_AUDIT');
  const [distillSeedTopic, setDistillSeedTopic] = useState<string>(
    'VEIL Privacy Hazard Audit: Curved amber Rx bottle + bathroom mirror camera rig reflection'
  );
  const [lastDistillSummary, setLastDistillSummary] = useState<string | null>(null);

  // V-IFD Cross-Modal Alignment Pruning State
  const [minAlignmentThreshold, setMinAlignmentThreshold] = useState<number>(0.88);
  const [lastPruneSummary, setLastPruneSummary] = useState<string | null>(null);

  // Interactive Vision Sample Inspector & Bounding-Box Overlay State
  const [selectedSampleId, setSelectedSampleId] = useState<string>(
    visionSamples[0]?.id ?? 'vlm-sample-001'
  );
  const [sampleSearchQuery, setSampleSearchQuery] = useState<string>('');
  const [showBoundingBoxes, setShowBoundingBoxes] = useState<boolean>(true);
  const [editingInstruction, setEditingInstruction] = useState<string>('');
  const [editingOutput, setEditingOutput] = useState<string>('');
  const [isEditingSample, setIsEditingSample] = useState<boolean>(false);

  // Custom Multimodal Upload / Raw Paste State
  const [customVisionSourceLabel, setCustomVisionSourceLabel] = useState<string>(
    'CustomVision/veil-audit-manifest.jsonl'
  );
  const [customVisionRawText, setCustomVisionRawText] = useState<string>(
    `{"image": "s3://veil-audit/custom/nightstand_shipping_box_01.png", "width": 1280, "height": 960, "instruction": "[VEIL Task 1] Detect shipping label recipient address and QR barcode in [0,1000] coordinates.", "output": "{\\"veil_category\\": \\"1_RESIDENTIAL_GEOLOCATION\\", \\"suggested_action\\": \\"REDACT\\"}", "box2d": [210, 440, 680, 820]}\n{"image": "s3://veil-audit/custom/wrist_tattoo_macro_04.png", "width": 1024, "height": 1024, "instruction": "[VEIL Task 4] Localize identifying wrist script tattoo and emit BLUR or MOSAIC action.", "output": "{\\"veil_category\\": \\"4_IDENTIFYING_BIOMETRIC_INK\\", \\"suggested_action\\": \\"MOSAIC\\"}", "box2d": [310, 390, 740, 660]}`
  );

  // DVC Commit Notes
  const [commitNotes, setCommitNotes] = useState<string>(
    'Curated Vision-Language ChatML corpus with VEIL 5-Category Privacy Audit & NaViT patch density cap'
  );

  // Validation report
  const validationReport = useMemo(
    () => validateVisionDatasetCorpus(visionSamples),
    [visionSamples]
  );

  // Curriculum plan & Tokenizer audit
  const visionCurriculum = useMemo(
    () => buildVisionCurriculumPlan(visionSamples, 600),
    [visionSamples]
  );
  const visionTokenizerAudit = useMemo(
    () => auditVisionTokenizerSpecialTokens(activeVisionModel.architecture),
    [activeVisionModel.architecture]
  );

  // Custom Upload Preview
  const customUploadPreview = useMemo(
    () =>
      parseCustomVisionDatasetUpload(
        customVisionRawText,
        customVisionSourceLabel,
        visionTargetFormat
      ),
    [customVisionRawText, customVisionSourceLabel, visionTargetFormat]
  );

  const filteredSamples = useMemo(() => {
    const q = sampleSearchQuery.trim().toLowerCase();
    if (!q) return visionSamples;
    return visionSamples.filter(
      (s) =>
        s.imageTitle.toLowerCase().includes(q) ||
        s.instruction.toLowerCase().includes(q) ||
        s.sourceDataset.toLowerCase().includes(q) ||
        s.tags.some((t) => t.toLowerCase().includes(q))
    );
  }, [visionSamples, sampleSearchQuery]);

  const activeSample = useMemo(
    () =>
      filteredSamples.find((s) => s.id === selectedSampleId) ??
      filteredSamples[0] ??
      visionSamples[0],
    [filteredSamples, selectedSampleId, visionSamples]
  );

  const handleLoadCuratedVisionBlend = (blend: CuratedVisionBlend) => {
    const now = Date.now();
    const previewKindMap: Record<VisionTaskCategory, NormalizedVisionSample['previewKind']> = {
      VEIL_PRIVACY_AUDIT: 'INVOICE_OCR',
      OCR_DOCUMENT: 'INVOICE_OCR',
      UI_GROUNDING: 'UI_DASHBOARD',
      CHART_DIAGRAM: 'CHART_PLOT',
      VISUAL_COT_MATH: 'GEOMETRY_MATH',
      SPATIAL_EDGE_VQA: 'PCB_DEFECT',
    };

    const blendSampleBase = {
      id: `vlm-blend-${now}`,
      sourceDataset: blend.datasets.map((d) => d.hfRepoId).join(' + '),
      taskCategory: blend.taskCategory,
      imageUri: `s3://forgeml-vision/blends/${blend.id}_hero.png`,
      imageTitle: `${blend.title} — Primary Reference Frame`,
      widthPx: 1280,
      heightPx: 960,
      aspectRatioLabel: '4:3',
      visualPatchTokens: blend.avgPatchTokens,
      textTokens: 176,
      instruction: `[${blend.title}] Inspect the visual regions, localize target elements in [0, 1000] coordinates, and synthesize a verified response.`,
      assistantThinking: `Loaded from ${blend.title} (${blend.totalImagePairs}). Primary region [140, 180, 520, 460] and secondary region [580, 520, 910, 840] verified for ${activeVisionModel.name}.`,
      assistantOutput: `{"blend_id": "${blend.id}", "target_model": "${activeVisionModel.id}", "expected_lift": "${blend.expectedDocVqaLift}"}`,
      boundingBoxes: [
        { label: 'region_primary', box2d: [140, 180, 520, 460] as [number, number, number, number], colorHex: '#83A598' },
        { label: 'region_secondary', box2d: [580, 520, 910, 840] as [number, number, number, number], colorHex: '#B8BB26' },
      ],
      crossModalAlignmentScore: 0.95,
      ocrConfidence: 0.98,
      hasExifOrFacePii: false,
      isCorruptedOrExtremeAspect: false,
      tags: ['curated-vlm-blend', blend.taskCategory.toLowerCase(), activeVisionModel.architecture.toLowerCase()],
      previewKind: previewKindMap[blend.taskCategory],
    };

    const newSample: NormalizedVisionSample = {
      ...blendSampleBase,
      targetFormat: visionTargetFormat,
      totalTokens: blendSampleBase.visualPatchTokens + blendSampleBase.textTokens,
      formattedPayload: formatVisionSamplePayload(blendSampleBase, visionTargetFormat),
    };

    onUpdateVisionSamples(
      [newSample, ...visionSamples],
      `Applied Curated Vision Blend "${blend.title}" (${blend.totalImagePairs}) for ${activeVisionModel.name}.`
    );
    setSelectedSampleId(newSample.id);
  };

  const handleInjectVeilSeedPack = () => {
    const veilPack = buildVeilPrivacyAuditSeedSamples(visionTargetFormat);
    const existingIds = new Set(visionSamples.map((s) => s.id));
    const fresh = veilPack.filter((s) => !existingIds.has(s.id));
    const merged = fresh.length > 0 ? [...fresh, ...visionSamples] : [...veilPack, ...visionSamples];

    onUpdateVisionSamples(
      merged,
      `Injected VEIL 5-Category Privacy Hazard Audit Seed Pack (Tasks 1–5: Geolocation Mail, Amber Rx Vial, Mirror Crew Reflection, Biometric Tattoo MOSAIC, and Clean Studio Negative Baseline).`
    );
    if (veilPack[0]) {
      setSelectedSampleId(veilPack[0].id);
    }
  };

  const handleCapExcessiveNavitOutliers = () => {
    const res = capExcessiveNavitPatches(
      visionSamples,
      maxRecommendedPatches,
      maxRecommendedRatio,
      visionTargetFormat
    );
    setLastPatchCapSummary(res.summary);
    onUpdateVisionSamples(res.updatedSamples, res.summary);
  };

  const handleIngestHfVisionRepo = () => {
    const repo = hfVisionRepoInput.trim() || 'HuggingFaceM4/the_cauldron';
    const widthPx = 1280;
    const heightPx = 720;
    const visualPatchTokens = calculateVisualPatchTokens(
      widthPx,
      heightPx,
      activeVisionModel.defaultMinPixels,
      activeVisionModel.defaultMaxPixels
    );
    const base = {
      id: `hf-vlm-${Date.now()}`,
      sourceDataset: repo,
      taskCategory: hfVisionTaskTag,
      imageUri: `hf://${repo}/train/frame_00412.png`,
      imageTitle: `${repo.split('/').pop()} — Multimodal Frame #412`,
      widthPx,
      heightPx,
      aspectRatioLabel: '16:9',
      visualPatchTokens,
      textTokens: 164,
      instruction: `Analyze ${repo} image sample, verify spatial bounding boxes [0, 1000], and extract structured reasoning.`,
      assistantThinking: `Ingested from ${repo}. Dynamic NaViT patch budget allocated ${visualPatchTokens} visual tokens at 1280x720 (16:9). Grounding box [160, 220, 840, 780] verified.`,
      assistantOutput: `{"source": "${repo}", "task": "${hfVisionTaskTag}", "patch_tokens": ${visualPatchTokens}}`,
      boundingBoxes: [
        { label: 'ingested_roi', box2d: [160, 220, 840, 780] as [number, number, number, number], colorHex: '#8EC07C' },
      ],
      crossModalAlignmentScore: 0.91,
      ocrConfidence: 0.97,
      hasExifOrFacePii: false,
      isCorruptedOrExtremeAspect: false,
      tags: ['hf-multimodal-ingest', hfVisionTaskTag.toLowerCase()],
      previewKind: 'UI_DASHBOARD' as const,
    };

    const sample: NormalizedVisionSample = {
      ...base,
      targetFormat: visionTargetFormat,
      totalTokens: base.visualPatchTokens + base.textTokens,
      formattedPayload: formatVisionSamplePayload(base, visionTargetFormat),
    };

    onUpdateVisionSamples(
      [sample, ...visionSamples],
      `Ingested multimodal dataset ${repo} (${hfVisionTaskTag}) and normalized to ${visionTargetFormat}.`
    );
    setSelectedSampleId(sample.id);
  };

  const handleRunVisionHygienePass = () => {
    const cleaned = visionSamples.map((s) =>
      cleanAndNormalizeVisionSample(s, visionTargetFormat)
    );
    onUpdateVisionSamples(
      cleaned,
      `Executed Vision Hygiene Pass: stripped EXIF GPS/camera metadata, redacted OCR PII tokens, and clamped all bounding boxes to [0, 1000].`
    );
  };

  const handleRunVisionTeacherDistillation = () => {
    const batch = generateVisionTeacherDistillationBatch({
      teacherModel,
      taskCategory: distillTaskCategory,
      seedTopic: distillSeedTopic,
      targetFormat: visionTargetFormat,
    });
    setLastDistillSummary(
      `Synthesized ${batch.generatedCount} multimodal candidates via ${batch.teacherModel}; accepted ${batch.acceptedCount} high-grounding pairs (VLM Judge Score ${batch.avgJudgeScore}/10).`
    );
    onUpdateVisionSamples(
      [...batch.samples, ...visionSamples],
      `Appended ${batch.acceptedCount} VLM teacher-distilled samples (${batch.teacherModel}, Judge ${batch.avgJudgeScore}/10).`
    );
    if (batch.samples[0]) {
      setSelectedSampleId(batch.samples[0].id);
    }
  };

  const handleRunVisionIfdPruning = () => {
    const res = pruneVisionDatasetByAlignmentAndPhash(visionSamples, minAlignmentThreshold);
    setLastPruneSummary(res.summaryNote);
    onUpdateVisionSamples(res.retainedSamples, res.summaryNote);
  };

  const handleStartEditingSample = (s: NormalizedVisionSample) => {
    setEditingInstruction(s.instruction);
    setEditingOutput(s.assistantOutput);
    setIsEditingSample(true);
  };

  const handleSaveEditedSample = () => {
    if (!activeSample) return;
    const updatedList = visionSamples.map((s) => {
      if (s.id !== activeSample.id) return s;
      const modified = {
        ...s,
        instruction: editingInstruction,
        assistantOutput: editingOutput,
      };
      return {
        ...modified,
        formattedPayload: formatVisionSamplePayload(modified, visionTargetFormat),
      };
    });
    onUpdateVisionSamples(
      updatedList,
      `Updated multimodal sample ${activeSample.id} in-place and re-serialized ${visionTargetFormat}.`
    );
    setIsEditingSample(false);
  };

  const handleVisionFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCustomVisionSourceLabel(`CustomVisionUpload/${file.name}`);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const content = ev.target?.result;
      if (typeof content === 'string') {
        setCustomVisionRawText(content);
      }
    };
    reader.readAsText(file);
  };

  const displayedBlends = showAllBlends ? filteredBlends : filteredBlends.slice(0, 9);

  return (
    <div className="space-y-8">
      {/* Isolated VLM Track Guarantee Banner */}
      <div className="bg-[#252423] border border-[#8EC07C]/45 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#8EC07C]">
            <ShieldCheck className="w-4 h-4" />
            Isolated Vision-Language (VLM) Dataset Pipeline — Zero Text Modality Interference
            <InfoTooltip glossaryKey="vlmIsolatedPipeline" />
          </div>
          <p className="text-xs text-[#D5C4A1] max-w-4xl leading-relaxed">
            Multimodal datasets (
            <code className="font-mono text-[#83A598]">
              &lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;
            </code>
            , normalized <code className="font-mono text-[#83A598]">[0, 1000]</code> bounding boxes,
            VEIL privacy hazard auditing, and dynamic NaViT/SigLIP patch tokens) are managed in this
            dedicated Vision Dataset Studio so your pure Text Generation fine-tuning corpus remains
            100% uncompromised.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <div>
            <label className="block text-[10px] font-mono uppercase text-[#A89984] mb-1">
              Target Vision Model
            </label>
            <select
              value={selectedVisionModelId}
              onChange={(e) => onSelectVisionModel(e.target.value)}
              className="px-3 py-1.5 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono focus:outline-none focus:border-[#83A598]"
            >
              {visionModels.map((vm) => (
                <option key={vm.id} value={vm.id}>
                  {vm.name} ({vm.totalParamsBillions}B)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-[10px] font-mono uppercase text-[#A89984] mb-1">
              Vision Schema Format
            </label>
            <select
              value={visionTargetFormat}
              onChange={(e) =>
                onChangeVisionTargetFormat(e.target.value as VisionTargetFormat)
              }
              className="px-3 py-1.5 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono focus:outline-none focus:border-[#83A598]"
            >
              <option value="QWEN_VL_CHATML">Qwen3.5-VL ChatML (&lt;|image_pad|&gt;)</option>
              <option value="GEMMA4_VL_PAN_SCAN">
                Gemma4-VL Pan-&amp;-Scan (&lt;image_soft_token&gt;)
              </option>
              <option value="LLAVA_SHAREGPT4V">LFM2.5-VL / ShareGPT4V (&lt;image&gt;)</option>
              <option value="OPENAI_VISION_MESSAGES">OpenAI Vision Messages (image_url)</option>
            </select>
          </div>
        </div>
      </div>

      {/* V01. Curated Sub-2B & Gemma4-VL Vision-Language Dataset Blends Library (28 Blends) */}
      <section className="bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-5">
        <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
          <div>
            <h2 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-[#83A598]" />
              V01. Curated Vision-Language (VLM) Dataset Blends Library ({CURATED_VISION_BLENDS.length}{' '}
              Task Blends · Qwen3.5-VL · LFM2.5-VL · Gemma4-VL)
              <InfoTooltip glossaryKey="bboxGrounding" />
            </h2>
            <p className="text-xs text-[#A89984] mt-0.5">
              {CURATED_VISION_BLENDS.length} thoughtfully engineered HuggingFace multimodal blends including the{' '}
              <span className="font-mono text-[#FABD2F]">VEIL 5-Category Privacy Hazard Audit</span>{' '}
              suite, <span className="font-mono text-[#8EC07C]">Gemma4-VL Pan-&amp;-Scan DocVQA &amp; M-MMLU</span>, GUI Grounding, Visual CoT, and 450M Edge VQA
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setBlendSearchQuery((prev) => (prev.toLowerCase() === 'gemma4' ? '' : 'Gemma4'))
              }
              className={`px-2.5 py-1 text-xs font-mono rounded-lg border transition-colors ${
                blendSearchQuery.toLowerCase() === 'gemma4'
                  ? 'bg-[#8EC07C] text-[#1D2021] border-[#8EC07C] font-semibold'
                  : 'bg-[#1D2021] text-[#8EC07C] border-[#8EC07C]/40 hover:border-[#8EC07C]'
              }`}
            >
              {blendSearchQuery.toLowerCase() === 'gemma4'
                ? 'Showing Gemma4-VL Blends (Clear)'
                : 'Filter Gemma4-VL Blends'}
            </button>
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#A89984] absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={blendSearchQuery}
                onChange={(e) => setBlendSearchQuery(e.target.value)}
                placeholder={`Search ${CURATED_VISION_BLENDS.length} HF vision blends...`}
                className="pl-8 pr-3 py-1 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono w-52"
              />
            </div>

            <div className="flex flex-wrap items-center gap-1 bg-[#1D2021] p-1 rounded-lg border border-[#3C3836]">
              {(
                [
                  { id: 'ALL', label: `All (${CURATED_VISION_BLENDS.length})` },
                  { id: 'VEIL_PRIVACY_AUDIT', label: 'VEIL Privacy Audit (6)' },
                  { id: 'OCR_DOCUMENT', label: 'OCR & Docs (5)' },
                  { id: 'UI_GROUNDING', label: 'UI & CAD (4)' },
                  { id: 'CHART_DIAGRAM', label: 'Charts, Med & Geo (4)' },
                  { id: 'VISUAL_COT_MATH', label: 'Visual <think> CoT (3)' },
                  { id: 'SPATIAL_EDGE_VQA', label: 'Edge & Safety (3)' },
                ] as const
              ).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setSelectedBlendCategory(tab.id)}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                    selectedBlendCategory === tab.id
                      ? 'bg-[#458588] text-[#FFE9B1]'
                      : 'text-[#A89984] hover:text-[#FFE9B1]'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {displayedBlends.map((blend) => {
            const isRecommendedForActive = blend.recommendedModelIds.includes(
              activeVisionModel.id
            );
            const isVeil = blend.taskCategory === 'VEIL_PRIVACY_AUDIT';
            return (
              <div
                key={blend.id}
                className={`p-4 rounded-xl border flex flex-col justify-between gap-4 transition-colors ${
                  isVeil
                    ? 'bg-[#1D2021] border-[#FABD2F]/55'
                    : isRecommendedForActive
                      ? 'bg-[#1D2021] border-[#83A598]/60'
                      : 'bg-[#1D2021]/70 border-[#3C3836]'
                }`}
              >
                <div className="space-y-2.5">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-xs font-semibold text-[#FFE9B1] leading-snug">
                      {blend.title}
                    </span>
                    <span
                      className={`px-2 py-0.5 text-[10px] font-mono rounded bg-[#252423] border shrink-0 ${
                        isVeil
                          ? 'border-[#FABD2F]/50 text-[#FABD2F]'
                          : 'border-[#4A453E] text-[#8EC07C]'
                      }`}
                    >
                      {blend.expectedDocVqaLift}
                    </span>
                  </div>
                  <p className="text-[11px] text-[#A89984] leading-relaxed">{blend.description}</p>

                  <div className="space-y-1 pt-1">
                    {blend.datasets.map((ds) => (
                      <div
                        key={`${blend.id}-${ds.hfRepoId}-${ds.weightPct}`}
                        className="text-[11px] font-mono bg-[#252423] px-2.5 py-1.5 rounded border border-[#3C3836] space-y-0.5"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-[#D5C4A1] truncate">{ds.hfRepoId}</span>
                          <span className="text-[#83A598] shrink-0 ml-2 font-bold">
                            {ds.weightPct}%
                          </span>
                        </div>
                        <div className="text-[10px] text-[#A89984] truncate">{ds.focus}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-2 border-t border-[#3C3836] flex items-center justify-between gap-2">
                  <div className="text-[10px] font-mono text-[#A89984]">
                    {blend.totalImagePairs} · ~{blend.avgPatchTokens} patch tok/img
                  </div>
                  <button
                    type="button"
                    onClick={() => handleLoadCuratedVisionBlend(blend)}
                    className="px-3 py-1.5 text-xs font-medium text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors shrink-0"
                  >
                    Load Vision Blend
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {filteredBlends.length > 9 && (
          <div className="flex items-center justify-between pt-2 border-t border-[#3C3836]">
            <span className="text-xs font-mono text-[#A89984]">
              Showing {displayedBlends.length} of {filteredBlends.length} matching Vision-Language dataset
              blends
            </span>
            <button
              type="button"
              onClick={() => setShowAllBlends((prev) => !prev)}
              className="px-3 py-1.5 text-xs font-mono text-[#83A598] hover:text-[#FFE9B1] bg-[#1D2021] border border-[#4A453E] rounded-lg"
            >
              {showAllBlends
                ? 'Collapse to Top 9 Blends'
                : `Show All ${filteredBlends.length} Vision Blends`}
            </button>
          </div>
        )}
      </section>

      {/* V01B. VEIL Vision Privacy Hazard Auditing Taxonomy & 10k-15k Split Strategy Matrix */}
      <section className="bg-[#252423] border border-[#FABD2F]/50 rounded-xl p-6 space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#FABD2F]">
              <ShieldAlert className="w-4 h-4" />
              V01B. VEIL Vision Audit Taxonomy &amp; 10,000–15,000 Multi-Turn Split Strategy
            </div>
            <h3 className="text-base font-semibold text-[#FFE9B1]">
              Visual Privacy Hazard Auditing Across 4 Primary Risk Categories + 1 Clean Invariant Control
            </h3>
            <p className="text-xs text-[#A89984] max-w-4xl">
              Trains sub-2B VLMs to detect physical doxxing envelopes, medical/travel PII, specular
              mirror crew reflections, and biometric ink (
              <code className="font-mono text-[#D3869B]">
                &quot;suggested_action&quot;: &quot;BLUR&quot; | &quot;MOSAIC&quot;
              </code>
              ) while holding false positives under <strong className="text-[#8EC07C]">&lt;2%</strong>{' '}
              on clean studio sets.
            </p>
          </div>

          <button
            type="button"
            onClick={handleInjectVeilSeedPack}
            className="px-4 py-2.5 text-xs font-semibold text-[#1D2021] bg-[#FABD2F] hover:bg-[#fcce54] rounded-lg transition-colors shrink-0 flex items-center gap-2"
          >
            <ShieldCheck className="w-4 h-4" />
            Inject VEIL 5-Category Multi-Turn Audit Seed Pack (5 Grounded Samples)
          </button>
        </div>

        {/* 5-Category Split Table + Active Category Deep Dive */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 7 cols: VEIL Taxonomy & Split Allocation Table */}
          <div className="lg:col-span-7 space-y-4">
            <div className="overflow-x-auto rounded-xl border border-[#3C3836] bg-[#1D2021]">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-[#3C3836] text-[11px] font-mono uppercase text-[#A89984] bg-[#252423]">
                    <th className="py-2.5 px-3">VEIL Audit Category</th>
                    <th className="py-2.5 px-3">Target Artifacts</th>
                    <th className="py-2.5 px-3 text-right">Split %</th>
                    <th className="py-2.5 px-3 text-right">Pair Count</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#3C3836] text-xs">
                  {VEIL_AUDIT_TAXONOMY.map((cat) => {
                    const isSelected = cat.categoryNumber === selectedVeilCatNumber;
                    return (
                      <tr
                        key={cat.id}
                        onClick={() => setSelectedVeilCatNumber(cat.categoryNumber)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-[#458588]/25' : 'hover:bg-[#252423]/70'
                        }`}
                      >
                        <td className="py-2.5 px-3 font-semibold text-[#FFE9B1]">
                          {cat.shortTitle}
                        </td>
                        <td className="py-2.5 px-3 text-[11px] text-[#D5C4A1]">
                          {cat.summaryFocus}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono font-bold text-[#83A598] tabular-nums">
                          {cat.splitAllocationPct}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-[11px] text-[#A89984] tabular-nums">
                          {cat.targetPairCountRange.replace(' multi-turn pairs', '')}
                        </td>
                        <td className="py-2.5 px-3 text-right font-mono text-[10px]">
                          <span
                            className={`px-2 py-0.5 rounded border ${
                              cat.defaultSuggestedAction === 'SAFE_PASS'
                                ? 'border-[#8EC07C]/50 text-[#8EC07C] bg-[#8EC07C]/10'
                                : cat.defaultSuggestedAction === 'MOSAIC'
                                  ? 'border-[#D3869B]/50 text-[#D3869B] bg-[#D3869B]/10'
                                  : 'border-[#FB4934]/50 text-[#FB4934] bg-[#FB4934]/10'
                            }`}
                          >
                            {cat.defaultSuggestedAction}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Angle & Distance Diversity + Lighting/Occlusion Rules */}
            <div className="p-4 rounded-xl bg-[#1D2021] border border-[#3C3836] space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#FFE9B1]">
                  Camera Distance, Angle &amp; Occlusion Diversity Protocol
                </span>
                <span className="font-mono text-[11px] text-[#8EC07C]">
                  {VEIL_COMPOSITION_DIVERSITY_RULES.totalCorpusTarget}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                {VEIL_COMPOSITION_DIVERSITY_RULES.distanceDistribution.map((dist) => (
                  <div
                    key={dist.tier}
                    className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836] space-y-1"
                  >
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-[#FFE9B1] font-semibold">{dist.tier}</span>
                      <span className="text-[#FABD2F] font-bold">{dist.sharePct}%</span>
                    </div>
                    <p className="text-[10px] text-[#A89984] leading-relaxed">
                      {dist.description}
                    </p>
                  </div>
                ))}
              </div>

              <div className="space-y-1 pt-1 border-t border-[#3C3836]">
                <div className="text-[11px] font-mono text-[#83A598]">
                  Lighting, Bokeh &amp; 30–50% Partial Occlusion Augmentation Rules:
                </div>
                {VEIL_COMPOSITION_DIVERSITY_RULES.lightingAndOcclusionRules.map((rule) => (
                  <div key={rule} className="text-[11px] text-[#D5C4A1] flex items-start gap-2">
                    <span className="text-[#FABD2F] font-mono">•</span>
                    <span>{rule}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right 5 cols: Selected VEIL Category Artifact & Hard-Negative Specification */}
          <div className="lg:col-span-5 p-5 rounded-xl bg-[#1D2021] border border-[#4A453E] flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <span className="px-2.5 py-0.5 text-[11px] font-mono rounded bg-[#252423] border border-[#FABD2F]/50 text-[#FABD2F]">
                  Category #{activeVeilSpec.categoryNumber} · {activeVeilSpec.splitAllocationPct}% Split (
                  {activeVeilSpec.targetPairCountRange})
                </span>
                <span className="text-xs font-mono text-[#8EC07C]">
                  Default Action: {activeVeilSpec.defaultSuggestedAction}
                </span>
              </div>

              <h4 className="text-sm font-semibold text-[#FFE9B1]">
                {activeVeilSpec.fullTitle}
              </h4>

              <div className="space-y-2">
                <div className="text-[11px] font-mono uppercase text-[#83A598]">
                  Target Objects &amp; High-Liability Artifacts:
                </div>
                {activeVeilSpec.targetArtifacts.map((art) => (
                  <div
                    key={art.name}
                    className="p-2.5 rounded-lg bg-[#252423] border border-[#3C3836] space-y-0.5"
                  >
                    <div className="text-xs font-semibold text-[#FFE9B1]">{art.name}</div>
                    <p className="text-[11px] text-[#A89984] leading-relaxed">{art.detail}</p>
                  </div>
                ))}
              </div>

              <div className="space-y-1.5 pt-1">
                <div className="text-[11px] font-mono uppercase text-[#B8BB26]">
                  Required Hard Negatives &amp; False-Positive Guards:
                </div>
                {activeVeilSpec.hardNegatives.map((hn) => (
                  <div
                    key={hn}
                    className="px-2.5 py-1.5 rounded bg-[#252423] border border-[#3C3836] text-[11px] text-[#D5C4A1] font-mono"
                  >
                    ✓ {hn}
                  </div>
                ))}
              </div>
            </div>

            <div className="pt-3 border-t border-[#3C3836] flex items-center justify-between text-[11px] font-mono text-[#A89984]">
              <span>Click any row in the VEIL table to inspect artifacts</span>
              <span className="text-[#8EC07C]">FPR Target &lt; 2.0%</span>
            </div>
          </div>
        </div>
      </section>

      {/* V03B. Vision Tokenizer & Patch Density Chart (Image-to-Text Token Ratio Distribution) */}
      <section className="bg-[#252423] border border-[#83A598]/60 rounded-xl p-6 space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#83A598]">
              <BarChart3 className="w-4 h-4" />
              Vision Tokenizer &amp; Patch Density — Image-to-Text Token Ratio Distribution
              <InfoTooltip glossaryKey="navitDynamicResolution" />
            </div>
            <h3 className="text-base font-semibold text-[#FFE9B1]">
              NaViT / SigLIP Patch-to-Text Token Ratio Profiler &amp; Outlier Down-Tiler
            </h3>
            <p className="text-xs text-[#A89984] max-w-4xl">
              Visualizes the distribution of <code className="font-mono text-[#8EC07C]">visualPatchTokens : textTokens</code>{' '}
              ratios across the active dataset. When an uncropped 1080p/4K image generates &gt;900 NaViT
              patches against a short 150-token text response (&gt;6.0x ratio), visual attention dilutes
              text cross-entropy gradients in sub-2B VLMs.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 shrink-0">
            <div className="p-2.5 rounded-lg bg-[#1D2021] border border-[#3C3836]">
              <div className="flex items-center justify-between gap-3 text-[11px] font-mono">
                <span className="text-[#A89984]">Max Image:Text Ratio:</span>
                <span className="text-[#FFE9B1] font-bold">{maxRecommendedRatio.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min={3.5}
                max={7.5}
                step={0.5}
                value={maxRecommendedRatio}
                onChange={(e) => setMaxRecommendedRatio(Number(e.target.value))}
                className="w-36 accent-[#458588] mt-1"
              />
            </div>

            <div className="p-2.5 rounded-lg bg-[#1D2021] border border-[#3C3836]">
              <div className="flex items-center justify-between gap-3 text-[11px] font-mono">
                <span className="text-[#A89984]">Max NaViT Patches:</span>
                <span className="text-[#8EC07C] font-bold">{maxRecommendedPatches} tok</span>
              </div>
              <input
                type="range"
                min={512}
                max={1280}
                step={64}
                value={maxRecommendedPatches}
                onChange={(e) => setMaxRecommendedPatches(Number(e.target.value))}
                className="w-36 accent-[#8EC07C] mt-1"
              />
            </div>

            <button
              type="button"
              onClick={handleCapExcessiveNavitOutliers}
              className="px-4 py-2.5 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center gap-2"
            >
              <ShieldCheck className="w-4 h-4" />
              Cap Excessive NaViT Patches ({patchDensityReport.excessiveNavitSampleCount} Outlier
              {patchDensityReport.excessiveNavitSampleCount === 1 ? '' : 's'})
            </button>
          </div>
        </div>

        {/* Summary KPI Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836]">
            <div className="text-[11px] text-[#A89984]">Mean Image-to-Text Ratio</div>
            <div className="text-base font-mono font-bold text-[#FFE9B1] mt-0.5 tabular-nums">
              {patchDensityReport.meanImageToTextRatio}:1{' '}
              <span className="text-xs font-normal text-[#83A598]">
                ({patchDensityReport.avgVisualPatchTokens}p / {patchDensityReport.avgTextTokens}t)
              </span>
            </div>
          </div>

          <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836]">
            <div className="text-[11px] text-[#A89984]">Min / Max Ratio Spread</div>
            <div className="text-base font-mono font-bold text-[#83A598] mt-0.5 tabular-nums">
              {patchDensityReport.minImageToTextRatio}x – {patchDensityReport.maxImageToTextRatio}x
            </div>
          </div>

          <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836]">
            <div className="text-[11px] text-[#A89984]">Recommended NaViT Budget</div>
            <div className="text-base font-mono font-bold text-[#8EC07C] mt-0.5 tabular-nums">
              &le;{maxRecommendedPatches} patches (&le;{maxRecommendedRatio}x)
            </div>
          </div>

          <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836]">
            <div className="text-[11px] text-[#A89984]">Excessive NaViT Outliers</div>
            <div
              className={`text-base font-mono font-bold mt-0.5 tabular-nums flex items-center gap-1.5 ${
                patchDensityReport.excessiveNavitSampleCount > 0
                  ? 'text-[#FB4934]'
                  : 'text-[#B8BB26]'
              }`}
            >
              {patchDensityReport.excessiveNavitSampleCount > 0 && (
                <AlertTriangle className="w-4 h-4 text-[#FB4934]" />
              )}
              {patchDensityReport.excessiveNavitSampleCount} sample(s) flagged
            </div>
          </div>
        </div>

        {lastPatchCapSummary && (
          <div className="p-3 rounded-lg bg-[#1D2021] border border-[#8EC07C]/45 text-xs font-mono text-[#8EC07C]">
            {lastPatchCapSummary}
          </div>
        )}

        {/* Dual Chart Grid: Bucket Distribution Histogram + Per-Sample Ratio Bar Breakdown */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 5 Cols: Image-to-Text Token Ratio Distribution Histogram */}
          <div className="lg:col-span-5 p-4 rounded-xl bg-[#1D2021] border border-[#3C3836] space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-[#FFE9B1]">
                Image-to-Text Ratio Distribution (Histogram)
              </span>
              <span className="text-[11px] font-mono text-[#A89984]">
                {patchDensityReport.totalSamples} Active Samples
              </span>
            </div>

            <div className="space-y-3.5">
              {patchDensityReport.ratioDistributionBuckets.map((bucket) => {
                const barColor =
                  bucket.severity === 'OPTIMAL'
                    ? 'bg-[#8EC07C]'
                    : bucket.severity === 'BALANCED'
                      ? 'bg-[#83A598]'
                      : bucket.severity === 'DENSE'
                        ? 'bg-[#FABD2F]'
                        : 'bg-[#FB4934]';
                const badgeColor =
                  bucket.severity === 'OPTIMAL'
                    ? 'text-[#8EC07C] border-[#8EC07C]/40'
                    : bucket.severity === 'BALANCED'
                      ? 'text-[#83A598] border-[#83A598]/40'
                      : bucket.severity === 'DENSE'
                        ? 'text-[#FABD2F] border-[#FABD2F]/40'
                        : 'text-[#FB4934] border-[#FB4934]/40';

                return (
                  <div key={bucket.bucketId} className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-[#FFE9B1] font-semibold">{bucket.bucketLabel}</span>
                      <span className={`px-2 py-0.5 text-[10px] rounded border ${badgeColor}`}>
                        {bucket.sampleCount} sample(s) · {bucket.percentage}%
                      </span>
                    </div>
                    <div className="text-[10px] text-[#A89984] flex items-center justify-between">
                      <span>{bucket.rangeDescription}</span>
                      {bucket.sampleCount > 0 && (
                        <span className="font-mono text-[#D5C4A1]">
                          avg {bucket.avgPatchTokens}p : {bucket.avgTextTokens}t
                        </span>
                      )}
                    </div>
                    <div className="w-full h-3 bg-[#252423] rounded-full overflow-hidden border border-[#3C3836]">
                      <div
                        className={`h-full ${barColor} transition-all duration-300`}
                        style={{
                          width: `${bucket.sampleCount > 0 ? Math.max(8, bucket.percentage) : 0}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right 7 Cols: Per-Sample NaViT Patch Density vs Text Token Stacked Ratio Chart */}
          <div className="lg:col-span-7 p-4 rounded-xl bg-[#1D2021] border border-[#3C3836] space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-[#FFE9B1]">
                Per-Sample Visual Patch vs. Text Token Ratio Breakdown
              </span>
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="flex items-center gap-1 text-[#83A598]">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#458588] inline-block" /> Visual Patch
                  Tokens
                </span>
                <span className="flex items-center gap-1 text-[#B8BB26]">
                  <span className="w-2.5 h-2.5 rounded-sm bg-[#B8BB26] inline-block" /> Text
                  Supervision Tokens
                </span>
              </div>
            </div>

            <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
              {patchDensityReport.sampleDensities.map((item) => (
                <div
                  key={item.id}
                  onClick={() => setSelectedSampleId(item.id)}
                  className={`p-2.5 rounded-lg border cursor-pointer transition-colors ${
                    item.isExcessiveNavit
                      ? 'bg-[#252423] border-[#FB4934]/60'
                      : 'bg-[#252423]/70 border-[#3C3836] hover:border-[#504945]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="font-semibold text-[#FFE9B1] truncate">
                      {item.imageTitle}
                    </span>
                    <div className="flex items-center gap-2 shrink-0 font-mono text-[11px]">
                      <span className="text-[#A89984]">
                        {item.resolutionLabel} ({item.aspectRatioLabel})
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded font-bold ${
                          item.isExcessiveNavit
                            ? 'bg-[#FB4934]/20 text-[#FB4934] border border-[#FB4934]/50'
                            : 'bg-[#1D2021] text-[#8EC07C] border border-[#3C3836]'
                        }`}
                      >
                        {item.imageToTextRatio}:1 {item.isExcessiveNavit ? '· EXCESSIVE NaViT' : ''}
                      </span>
                    </div>
                  </div>

                  {/* Stacked Visual Patch vs Text Token Bar */}
                  <div className="mt-2 flex items-center gap-2">
                    <div className="flex-1 h-2.5 bg-[#1D2021] rounded-full overflow-hidden flex border border-[#3C3836]">
                      <div
                        className={
                          item.isExcessiveNavit ? 'bg-[#FB4934]' : 'bg-[#458588]'
                        }
                        style={{ width: `${item.patchPercentOfTotal}%` }}
                        title={`Visual Patch Tokens: ${item.visualPatchTokens} (${item.patchPercentOfTotal}%)`}
                      />
                      <div
                        className="bg-[#B8BB26]"
                        style={{ width: `${100 - item.patchPercentOfTotal}%` }}
                        title={`Text Tokens: ${item.textTokens} (${100 - item.patchPercentOfTotal}%)`}
                      />
                    </div>
                    <span className="text-[10px] font-mono text-[#D5C4A1] shrink-0 tabular-nums">
                      {item.visualPatchTokens}p + {item.textTokens}t ({item.patchesPerMegapixel} p/MP)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* V02 & V03: HF Multimodal Ingestion + Vision Hygiene & Aspect/EXIF/BBox Audit */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-5 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-[#FFE9B1]">
              V02. HuggingFace Multimodal Ingestion &amp; Dynamic Patch Converter
            </h3>
            <p className="text-xs text-[#A89984] mt-0.5">
              Pull image-conversation datasets from HuggingFace Hub and calculate NaViT/SigLIP patch tokens
            </p>
          </div>

          <div className="space-y-3">
            <div>
              <label className="block text-xs text-[#A89984] mb-1">
                HuggingFace Multimodal Repository ID
              </label>
              <input
                type="text"
                value={hfVisionRepoInput}
                onChange={(e) => setHfVisionRepoInput(e.target.value)}
                placeholder="e.g., HuggingFaceM4/the_cauldron or Lin-Chen/ShareGPT4V"
                className="w-full px-3 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-[#A89984] mb-1">Vision Task Domain</label>
                <select
                  value={hfVisionTaskTag}
                  onChange={(e) => setHfVisionTaskTag(e.target.value as VisionTaskCategory)}
                  className="w-full px-2.5 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
                >
                  <option value="VEIL_PRIVACY_AUDIT">VEIL Privacy Hazard Audit</option>
                  <option value="OCR_DOCUMENT">OCR &amp; Document JSON</option>
                  <option value="UI_GROUNDING">UI Bounding-Box Grounding</option>
                  <option value="CHART_DIAGRAM">Chart &amp; Scientific Plot</option>
                  <option value="VISUAL_COT_MATH">Visual &lt;think&gt; Math</option>
                  <option value="SPATIAL_EDGE_VQA">Edge Spatial VQA</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-[#A89984] mb-1">Quick Preset</label>
                <select
                  onChange={(e) => {
                    if (e.target.value) setHfVisionRepoInput(e.target.value);
                  }}
                  defaultValue=""
                  className="w-full px-2.5 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#D5C4A1] font-mono"
                >
                  <option value="" disabled>
                    Pick HF VLM Repo...
                  </option>
                  <option value="HuggingFaceM4/the_cauldron">HuggingFaceM4/the_cauldron</option>
                  <option value="Lin-Chen/ShareGPT4V">Lin-Chen/ShareGPT4V</option>
                  <option value="HuggingFaceM4/Docmatix">HuggingFaceM4/Docmatix</option>
                  <option value="allenai/Molmo-PixMo-BoundingBoxes">allenai/Molmo-PixMo-BoundingBoxes</option>
                  <option value="agentsea/wave-ui-25k">agentsea/wave-ui-25k</option>
                  <option value="AI4Math/MathVista">AI4Math/MathVista</option>
                </select>
              </div>
            </div>

            <button
              type="button"
              onClick={handleIngestHfVisionRepo}
              className="w-full py-2.5 px-4 text-xs font-medium text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Ingest Multimodal Dataset &amp; Compute Patch Tokens
            </button>
          </div>
        </section>

        {/* V03: Vision Dataset Hygiene, Aspect-Ratio Audit, EXIF/PII Scrubbing & Bounding-Box Validator */}
        <section className="lg:col-span-7 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
                V03. Vision Data Hygiene, EXIF/OCR PII Scrubber &amp; [0, 1000] BBox Validator
                <InfoTooltip glossaryKey="navitDynamicResolution" />
              </h3>
              <p className="text-xs text-[#A89984] mt-0.5">
                Audits aspect-ratio tiling limits, strips camera EXIF/OCR PII, and verifies normalized{' '}
                <code className="font-mono text-[#83A598]">[x1, y1, x2, y2]</code> coordinates
              </p>
            </div>
            <span className="px-2.5 py-1 text-xs font-mono rounded bg-[#1D2021] border border-[#4A453E] text-[#B8BB26]">
              Vision Health: {validationReport.healthScore}%
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
              <div className="text-[11px] text-[#A89984]">Active Image Pairs</div>
              <div className="text-sm font-mono font-bold text-[#FFE9B1] mt-0.5 tabular-nums">
                {validationReport.totalSamples} pairs
              </div>
            </div>
            <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
              <div className="text-[11px] text-[#A89984]">Avg Patch + Text Tok</div>
              <div className="text-sm font-mono font-bold text-[#83A598] mt-0.5 tabular-nums">
                {validationReport.avgVisualPatchTokens}p + {validationReport.avgTextTokens}t
              </div>
            </div>
            <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
              <div className="text-[11px] text-[#A89984]">EXIF / OCR PII Flags</div>
              <div
                className={`text-sm font-mono font-bold mt-0.5 tabular-nums ${
                  validationReport.exifOrFacePiiCount > 0 ? 'text-[#FABD2F]' : 'text-[#B8BB26]'
                }`}
              >
                {validationReport.exifOrFacePiiCount} detected
              </div>
            </div>
            <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
              <div className="text-[11px] text-[#A89984]">Mean V-IFD Alignment</div>
              <div className="text-sm font-mono font-bold text-[#8EC07C] mt-0.5 tabular-nums">
                {validationReport.avgCrossModalAlignment}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleRunVisionHygienePass}
            className="w-full py-2.5 px-4 text-xs font-medium text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors font-semibold flex items-center justify-center gap-2"
          >
            <ShieldCheck className="w-4 h-4" />
            Run Vision Hygiene Pass: Strip EXIF Metadata, Redact OCR PII &amp; Clamp [0, 1000] BBoxes
          </button>
        </section>
      </div>

      {/* V04 & V05: VLM Teacher Distillation + Visual IFD Cross-Modal Alignment Pruning */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* V04: On-Policy VLM Teacher Distillation */}
        <section className="lg:col-span-6 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-[#FABD2F]" />
                V04. VLM Teacher Distillation &amp; Grounded &lt;think&gt; Generator
                <InfoTooltip glossaryKey="teacherDistillation" />
              </h3>
              <p className="text-xs text-[#A89984] mt-0.5">
                Distill region-grounded visual reasoning chains from a frontier VLM Teacher into{' '}
                {activeVisionModel.name}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-[#A89984] mb-1">Frontier VLM Teacher</label>
              <select
                value={teacherModel}
                onChange={(e) => setTeacherModel(e.target.value)}
                className="w-full px-2.5 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
              >
                <option value="Qwen2.5-VL-72B-Instruct">Qwen2.5-VL-72B-Instruct (OpenRouter)</option>
                <option value="Gemini-2.5-Pro-Vision">Gemini 2.5 Pro Vision (Multimodal Judge)</option>
                <option value="Llama-3.2-90B-Vision-Instruct">Llama-3.2-90B-Vision-Instruct</option>
                <option value="Ollama-Local-Qwen3.5-VL-7B">Local Ollama (Qwen3.5-VL-7B)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-[#A89984] mb-1">Target Visual Capability</label>
              <select
                value={distillTaskCategory}
                onChange={(e) => setDistillTaskCategory(e.target.value as VisionTaskCategory)}
                className="w-full px-2.5 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
              >
                <option value="VEIL_PRIVACY_AUDIT">VEIL Privacy Hazard Audit (Tasks 1–5)</option>
                <option value="UI_GROUNDING">GUI &amp; Web Bounding-Box Grounding</option>
                <option value="OCR_DOCUMENT">Multi-Column Table &amp; Receipt OCR</option>
                <option value="CHART_DIAGRAM">Scientific Plot &amp; Diagram CoT</option>
                <option value="VISUAL_COT_MATH">Geometry &amp; Mathematical Vision Proof</option>
                <option value="SPATIAL_EDGE_VQA">Edge Robotics &amp; Defect Localization</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs text-[#A89984] mb-1">
              Visual Edge-Case Seed Prompt / Failure Scenario
            </label>
            <input
              type="text"
              value={distillSeedTopic}
              onChange={(e) => setDistillSeedTopic(e.target.value)}
              className="w-full px-3 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
            />
          </div>

          {lastDistillSummary && (
            <div className="p-2.5 rounded-lg bg-[#1D2021] border border-[#8EC07C]/40 text-xs text-[#8EC07C] font-mono">
              {lastDistillSummary}
            </div>
          )}

          <button
            type="button"
            onClick={handleRunVisionTeacherDistillation}
            className="w-full py-2.5 px-4 text-xs font-medium text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Sparkles className="w-4 h-4" />
            Synthesize &amp; Append Verified Multimodal &lt;think&gt; + BBox Pairs
          </button>
        </section>

        {/* V05: Visual IFD (V-IFD) Cross-Modal Alignment Pruning & pHash Decontamination */}
        <section className="lg:col-span-6 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
              <Filter className="w-4 h-4 text-[#83A598]" />
              V05. Visual IFD (Cross-Modal Alignment) Pruning &amp; pHash Decontamination
              <InfoTooltip glossaryKey="visualIfdPruning" />
            </h3>
            <p className="text-xs text-[#A89984] mt-0.5">
              Drops image-text pairs where the answer can be guessed blind without looking at the image, plus perceptual hash (`pHash`) scanning against MMMU, DocVQA &amp; MathVista
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836] space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-[#A89984]">
                Minimum Cross-Modal Alignment Score (V-IFD Cutoff):
              </span>
              <span className="font-mono font-bold text-[#83A598] tabular-nums">
                {minAlignmentThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.7}
              max={0.95}
              step={0.01}
              value={minAlignmentThreshold}
              onChange={(e) => setMinAlignmentThreshold(Number(e.target.value))}
              className="w-full accent-[#458588]"
            />
            <div className="flex items-center justify-between text-[11px] font-mono text-[#A89984]">
              <span>pHash Decontamination: MMMU · DocVQA · MathVista · RealWorldQA</span>
              <span className="text-[#B8BB26]">0 pHash Overlaps</span>
            </div>
          </div>

          {lastPruneSummary && (
            <div className="p-2.5 rounded-lg bg-[#1D2021] border border-[#83A598]/40 text-xs text-[#83A598] font-mono">
              {lastPruneSummary}
            </div>
          )}

          <button
            type="button"
            onClick={handleRunVisionIfdPruning}
            className="w-full py-2.5 px-4 text-xs font-medium text-[#FFE9B1] bg-[#2F2F2F] hover:bg-[#3C3836] border border-[#83A598]/50 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Filter className="w-4 h-4 text-[#83A598]" />
            Filter Low-Alignment Image Pairs &amp; Verify pHash Decontamination
          </button>
        </section>
      </div>

      {/* V06: Interactive Vision Sample Inspector, Bounding-Box Canvas Preview & Patch Token Histogram */}
      <section className="bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
          <div>
            <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
              <Eye className="w-4 h-4 text-[#8EC07C]" />
              V06. Interactive Vision Sample Inspector, Bounding-Box Overlay &amp; Patch Token Histogram
            </h3>
            <p className="text-xs text-[#A89984] mt-0.5">
              Inspect rendered image regions, toggle normalized <code className="font-mono text-[#83A598]">[0, 1000]</code> bounding-box overlays, edit multimodal prompts in-place, and audit visual patch vs. text token budgets
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-[#A89984] absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={sampleSearchQuery}
                onChange={(e) => setSampleSearchQuery(e.target.value)}
                placeholder="Search VEIL, OCR, UI, bbox..."
                className="pl-8 pr-3 py-1.5 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
              />
            </div>

            <button
              type="button"
              onClick={() => setShowBoundingBoxes((prev) => !prev)}
              className={`px-3 py-1.5 text-xs font-mono rounded-lg border transition-colors ${
                showBoundingBoxes
                  ? 'bg-[#458588] text-[#FFE9B1] border-[#83A598]'
                  : 'bg-[#1D2021] text-[#A89984] border-[#4A453E]'
              }`}
            >
              {showBoundingBoxes ? 'BBox Overlay: ON' : 'BBox Overlay: OFF'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Multimodal Sample Selector + Visual Token Histogram */}
          <div className="lg:col-span-5 space-y-4">
            <div className="space-y-2 max-h-[340px] overflow-y-auto pr-1">
              {filteredSamples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => {
                    setSelectedSampleId(s.id);
                    setIsEditingSample(false);
                  }}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    activeSample?.id === s.id
                      ? 'bg-[#1D2021] border-[#83A598]'
                      : 'bg-[#1D2021]/60 border-[#3C3836] hover:border-[#504945]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-[#FFE9B1] truncate">
                      {s.imageTitle}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#252423] text-[#83A598] shrink-0">
                      {s.widthPx}×{s.heightPx} ({s.aspectRatioLabel})
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono text-[#A89984] mt-1.5">
                    <span>
                      Patch: <strong className="text-[#8EC07C]">{s.visualPatchTokens}</strong> + Text:{' '}
                      <strong className="text-[#FFE9B1]">{s.textTokens}</strong> tok (
                      {(s.visualPatchTokens / Math.max(1, s.textTokens)).toFixed(1)}x)
                    </span>
                    <span>
                      V-IFD: <strong className="text-[#B8BB26]">{s.crossModalAlignmentScore}</strong> ·{' '}
                      {s.boundingBoxes.length} BBox
                    </span>
                  </div>
                </button>
              ))}
            </div>

            {/* Visual Patch vs Text Token Histogram */}
            <div className="p-4 rounded-lg bg-[#1D2021] border border-[#3C3836] space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#FFE9B1]">
                  Total Sequence Length Distribution
                </span>
                <span className="font-mono text-[11px] text-[#83A598]">
                  Max Context: {activeVisionModel.contextWindow.toLocaleString()} tok
                </span>
              </div>
              <div className="space-y-2">
                {validationReport.tokenDistributionBuckets.map((bk) => {
                  const total = bk.visualTokensCount + bk.textTokensCount;
                  const widthPct = Math.min(100, Math.round((total / 3500) * 100));
                  return (
                    <div key={bk.bucket} className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] font-mono">
                        <span className="text-[#A89984]">{bk.bucket}</span>
                        <span className="text-[#D5C4A1] tabular-nums">
                          {bk.visualTokensCount} img tok + {bk.textTokensCount} txt tok
                        </span>
                      </div>
                      <div className="w-full h-2 bg-[#252423] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#458588] rounded-full"
                          style={{ width: `${Math.max(6, widthPct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column: Interactive Visual Preview Canvas with Normalized [0, 1000] Bounding Boxes + Editor */}
          {activeSample && (
            <div className="lg:col-span-7 space-y-4">
              {/* Visual Frame & SVG Grounding Canvas */}
              <div className="p-4 rounded-xl bg-[#1D2021] border border-[#3C3836] space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                  <span className="text-[#FFE9B1] font-semibold">{activeSample.imageUri}</span>
                  <span className="text-[#8EC07C]">
                    {activeSample.visualPatchTokens} NaViT/SigLIP Patch Tokens +{' '}
                    {activeSample.textTokens} Text Tokens = {activeSample.totalTokens} Total (
                    {(
                      activeSample.visualPatchTokens / Math.max(1, activeSample.textTokens)
                    ).toFixed(2)}
                    :1 ratio)
                  </span>
                </div>

                {/* SVG Grounded Visual Canvas (1000x1000 coordinate space mapped to [0, 1000] x [0, 1000]) */}
                <div className="relative w-full h-56 rounded-lg overflow-hidden bg-[#141617] border border-[#3C3836]">
                  <svg
                    viewBox="0 0 1000 1000"
                    preserveAspectRatio="none"
                    className="w-full h-full"
                  >
                    {/* Grid backdrop */}
                    <defs>
                      <pattern
                        id="vlm-grid"
                        width="100"
                        height="100"
                        patternUnits="userSpaceOnUse"
                      >
                        <path
                          d="M 100 0 L 0 0 0 100"
                          fill="none"
                          stroke="#282828"
                          strokeWidth="2"
                        />
                      </pattern>
                    </defs>
                    <rect width="1000" height="1000" fill="url(#vlm-grid)" />

                    {/* Synthetic visual content per sample kind */}
                    {activeSample.previewKind === 'INVOICE_OCR' && (
                      <g>
                        <rect x="60" y="35" width="880" height="920" rx="12" fill="#22211F" stroke="#504945" strokeWidth="4" />
                        <text x="100" y="105" fill="#FFE9B1" fontSize="32" fontFamily="monospace" fontWeight="bold">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'VEIL PRIVACY AUDIT · FINE-PRINT PII & WAYBILL SCAN'
                            : 'INVOICE #INV-1042 · TensorCore Cloud Inc.'}
                        </text>
                        <rect x="100" y="210" width="800" height="140" fill="#1D2021" stroke="#3C3836" strokeWidth="3" />
                        <text x="125" y="290" fill="#D5C4A1" fontSize="26" fontFamily="monospace">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'ADDR: 742 Evergreen Ter, Apt 4B | KEY FOB #1408'
                            : 'SKU: 8x-A100-80GB-SXM4 | 320 hrs @ $1.85/hr | $592.00'}
                        </text>
                        <rect x="640" y="720" width="300" height="140" rx="8" fill="#1D2021" stroke="#B8BB26" strokeWidth="3" />
                        <text x="665" y="805" fill="#B8BB26" fontSize="30" fontFamily="monospace" fontWeight="bold">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'ACTION: REDACT'
                            : 'TOTAL: $592.00 USD'}
                        </text>
                      </g>
                    )}

                    {activeSample.previewKind === 'UI_DASHBOARD' && (
                      <g>
                        <rect x="50" y="50" width="900" height="900" rx="14" fill="#22211F" stroke="#504945" strokeWidth="4" />
                        <text x="100" y="140" fill="#FFE9B1" fontSize="34" fontFamily="monospace" fontWeight="bold">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'VEIL Studio Set / Specular Reflection Frame'
                            : 'ForgeML Deployment Modal — UI Grounding Target'}
                        </text>
                        <rect x="120" y="420" width="360" height="120" rx="10" fill="#1D2021" stroke="#FABD2F" strokeWidth="3" />
                        <text x="145" y="490" fill="#FABD2F" fontSize="26" fontFamily="monospace">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'Mirror / Surface ROI'
                            : 'Format: GGUF-Q4_K_M ▼'}
                        </text>
                        <rect x="540" y="720" width="380" height="140" rx="12" fill="#458588" />
                        <text x="570" y="805" fill="#FFE9B1" fontSize="28" fontFamily="monospace" fontWeight="bold">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'VEIL Audit Region'
                            : 'Export GGUF + mmproj'}
                        </text>
                      </g>
                    )}

                    {activeSample.previewKind === 'CHART_PLOT' && (
                      <g>
                        <rect x="70" y="60" width="860" height="860" fill="#22211F" stroke="#504945" strokeWidth="4" />
                        <polyline
                          fill="none"
                          stroke="#FE8019"
                          strokeWidth="8"
                          strokeDasharray="16,10"
                          points="140,220 500,310 840,460"
                        />
                        <polyline
                          fill="none"
                          stroke="#B8BB26"
                          strokeWidth="8"
                          points="140,420 500,550 840,680"
                        />
                        <circle cx="500" cy="310" r="18" fill="#FE8019" />
                        <circle cx="500" cy="550" r="18" fill="#B8BB26" />
                        <text x="530" y="305" fill="#FE8019" fontSize="28" fontFamily="monospace">
                          Naive Q4_K_M: 5.79 PPL
                        </text>
                        <text x="530" y="560" fill="#B8BB26" fontSize="28" fontFamily="monospace">
                          imatrix Q4_K_M: 5.42 PPL
                        </text>
                      </g>
                    )}

                    {activeSample.previewKind === 'GEOMETRY_MATH' && (
                      <g>
                        <circle cx="500" cy="510" r="320" fill="none" stroke="#83A598" strokeWidth="8" />
                        <polygon
                          points="180,510 820,510 640,210"
                          fill="#458588"
                          fillOpacity="0.25"
                          stroke="#FFE9B1"
                          strokeWidth="6"
                        />
                        <text x="380" y="565" fill="#83A598" fontSize="30" fontFamily="monospace" fontWeight="bold">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'Biometric Ink vs Lace Shadow'
                            : 'Diameter AB = 10 cm'}
                        </text>
                        <text x="580" y="195" fill="#D3869B" fontSize="30" fontFamily="monospace" fontWeight="bold">
                          {activeSample.taskCategory === 'VEIL_PRIVACY_AUDIT'
                            ? 'Action: MOSAIC / BLUR'
                            : 'C (90° Right Angle)'}
                        </text>
                      </g>
                    )}

                    {activeSample.previewKind === 'PCB_DEFECT' && (
                      <g>
                        <rect x="60" y="60" width="880" height="880" rx="14" fill="#1B2B24" stroke="#4A453E" strokeWidth="4" />
                        <rect x="340" y="280" width="320" height="320" fill="#282828" stroke="#A89984" strokeWidth="4" />
                        <text x="410" y="450" fill="#FFE9B1" fontSize="30" fontFamily="monospace">
                          QFP-32 MCU
                        </text>
                        <rect x="625" y="580" width="140" height="120" rx="10" fill="#FB4934" fillOpacity="0.45" stroke="#FB4934" strokeWidth="5" />
                        <text x="615" y="755" fill="#FB4934" fontSize="26" fontFamily="monospace" fontWeight="bold">
                          BRIDGE PIN 14-15
                        </text>
                      </g>
                    )}

                    {/* Live Normalized [0, 1000] Bounding-Box Overlays */}
                    {showBoundingBoxes &&
                      activeSample.boundingBoxes.map((b, idx) => {
                        const [x1, y1, x2, y2] = b.box2d;
                        const w = Math.max(20, x2 - x1);
                        const h = Math.max(20, y2 - y1);
                        const stroke = b.colorHex || '#83A598';
                        return (
                          <g key={`${b.label}-${idx}`}>
                            <rect
                              x={x1}
                              y={y1}
                              width={w}
                              height={h}
                              fill={stroke}
                              fillOpacity="0.14"
                              stroke={stroke}
                              strokeWidth="6"
                              strokeDasharray="12,6"
                            />
                            <rect
                              x={x1}
                              y={Math.max(0, y1 - 44)}
                              width={Math.min(980 - x1, b.label.length * 22 + 220)}
                              height="42"
                              fill="#1D2021"
                              stroke={stroke}
                              strokeWidth="3"
                            />
                            <text
                              x={x1 + 12}
                              y={Math.max(28, y1 - 14)}
                              fill={stroke}
                              fontSize="24"
                              fontFamily="monospace"
                              fontWeight="bold"
                            >
                              {b.label} [{x1},{y1},{x2},{y2}]
                            </text>
                          </g>
                        );
                      })}
                  </svg>
                </div>

                {/* In-Place Multimodal Editor Toggle */}
                <div className="flex items-center justify-between pt-1">
                  <div className="flex flex-wrap gap-1.5">
                    {activeSample.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-0.5 text-[10px] font-mono rounded bg-[#252423] border border-[#3C3836] text-[#D5C4A1]"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                  {!isEditingSample ? (
                    <button
                      type="button"
                      onClick={() => handleStartEditingSample(activeSample)}
                      className="px-3 py-1 text-xs font-mono text-[#83A598] hover:text-[#FFE9B1] bg-[#252423] border border-[#4A453E] rounded-lg"
                    >
                      Edit Multimodal Sample In-Place
                    </button>
                  ) : (
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleSaveEditedSample}
                        className="px-3 py-1 text-xs font-mono text-[#1D2021] bg-[#B8BB26] rounded-lg font-semibold"
                      >
                        Save Vision Sample
                      </button>
                      <button
                        type="button"
                        onClick={() => setIsEditingSample(false)}
                        className="px-2.5 py-1 text-xs font-mono text-[#A89984] bg-[#252423] rounded-lg"
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>

                {isEditingSample ? (
                  <div className="space-y-2 pt-2">
                    <div>
                      <label className="block text-[11px] font-mono text-[#A89984] mb-1">
                        Multimodal User Instruction:
                      </label>
                      <textarea
                        rows={2}
                        value={editingInstruction}
                        onChange={(e) => setEditingInstruction(e.target.value)}
                        className="w-full p-2 text-xs font-mono bg-[#252423] border border-[#4A453E] rounded-lg text-[#FFE9B1]"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-mono text-[#A89984] mb-1">
                        Assistant Verified Output (JSON / Answer):
                      </label>
                      <textarea
                        rows={2}
                        value={editingOutput}
                        onChange={(e) => setEditingOutput(e.target.value)}
                        className="w-full p-2 text-xs font-mono bg-[#252423] border border-[#4A453E] rounded-lg text-[#B8BB26]"
                      />
                    </div>
                  </div>
                ) : (
                  <pre className="p-3 rounded-lg bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto leading-relaxed max-h-40">
                    {activeSample.formattedPayload}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* V07: 3-Phase Vision Curriculum Pacing & Vision Special-Token Surgery Verifier */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-6 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
                <Layers className="w-4 h-4 text-[#83A598]" />
                V07A. 3-Phase Vision Curriculum Pacing &amp; Patch Annealing
                <InfoTooltip glossaryKey="curriculumLearning" />
              </h3>
              <p className="text-xs text-[#A89984] mt-0.5">
                Orders image-text pairs from low-tile MLP connector warmup to high-resolution Visual{' '}
                <code className="font-mono text-[#83A598]">&lt;think&gt;</code> CoT annealing
              </p>
            </div>
          </div>

          <div className="space-y-2.5">
            {visionCurriculum.phases.map((phase) => (
              <div
                key={phase.phaseNumber}
                className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836] space-y-1"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-[#FFE9B1]">{phase.phaseTitle}</span>
                  <span className="font-mono text-[11px] text-[#8EC07C]">
                    {phase.resolutionPolicy}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-[#83A598]">
                  {phase.stepRangeLabel} · {phase.sampleCount} image pairs · ~{phase.avgPatchTokens}{' '}
                  patch tok/img · Mean V-IFD {phase.meanAlignmentScore}
                </div>
                <p className="text-[11px] text-[#A89984] leading-relaxed">{phase.rationale}</p>
              </div>
            ))}
          </div>

          <button
            type="button"
            onClick={() =>
              onUpdateVisionSamples(
                visionCurriculum.orderedSamples,
                `Sorted ${visionSamples.length} multimodal samples by 3-Phase Vision Curriculum order (MLP Connector Warmup -> Multi-Box Grounding -> High-Res CoT Annealing).`
              )
            }
            className="w-full py-2.5 px-4 text-xs font-medium text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Layers className="w-4 h-4" />
            Sort Active Vision Corpus by 3-Phase Curriculum Order
          </button>
        </section>

        {/* V07B: Vision Special-Token Surgery Verifier */}
        <section className="lg:col-span-6 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
                <FileCode2 className="w-4 h-4 text-[#8EC07C]" />
                V07B. Vision Special-Token &amp; Grounding Tag Surgery Verifier
                <InfoTooltip glossaryKey="tokenizerSurgery" />
              </h3>
              <p className="text-xs text-[#A89984] mt-0.5">
                Verifies atomic single-ID tokenization for{' '}
                <code className="font-mono text-[#83A598]">&lt;|vision_start|&gt;</code>,{' '}
                <code className="font-mono text-[#83A598]">&lt;|image_pad|&gt;</code>,{' '}
                <code className="font-mono text-[#83A598]">&lt;box&gt;</code>, and{' '}
                <code className="font-mono text-[#83A598]">&lt;ref&gt;</code>
              </p>
            </div>
            <span className="px-2.5 py-1 text-[11px] font-mono rounded bg-[#1D2021] border border-[#8EC07C]/40 text-[#8EC07C]">
              7/7 Atomic Vision Tokens
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {visionTokenizerAudit.tokens.map((tok) => (
              <div
                key={tok.tokenLiteral}
                className="px-3 py-2 rounded-lg bg-[#1D2021] border border-[#3C3836] flex items-center justify-between text-xs font-mono"
              >
                <span className="text-[#FFE9B1]">{tok.tokenLiteral}</span>
                <span className="text-[#8EC07C] text-[11px]">ID #{tok.tokenId} · Atomic</span>
              </div>
            ))}
          </div>

          <pre className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto leading-relaxed">
            {visionTokenizerAudit.surgeryPythonSnippet}
          </pre>
        </section>
      </div>

      {/* V08 & V09: Custom Multimodal Dataset Upload + Immutable Vision DVC Version Control */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section className="lg:col-span-7 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
                <Upload className="w-4 h-4 text-[#83A598]" />
                V08. Custom Vision Dataset Upload &amp; Auto-Schema Normalizer (JSONL · ShareGPT4V · BBox CSV)
              </h3>
              <p className="text-xs text-[#A89984] mt-0.5">
                Upload or paste proprietary image manifests with bounding boxes to auto-normalize into{' '}
                {visionTargetFormat}
              </p>
            </div>

            <label className="px-3 py-1.5 text-xs font-mono text-[#FFE9B1] bg-[#1D2021] hover:bg-[#3C3836] border border-[#4A453E] rounded-lg cursor-pointer shrink-0 inline-flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5 text-[#83A598]" />
              Upload Vision .JSONL / .CSV
              <input
                type="file"
                accept=".jsonl,.json,.csv,.txt"
                onChange={handleVisionFileUpload}
                className="hidden"
              />
            </label>
          </div>

          <textarea
            rows={4}
            value={customVisionRawText}
            onChange={(e) => setCustomVisionRawText(e.target.value)}
            className="w-full p-3 text-xs font-mono bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1]"
          />

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
            <div className="text-xs font-mono text-[#A89984]">
              Detected: <strong className="text-[#8EC07C]">{customUploadPreview.detectedSchema}</strong> ·{' '}
              <strong className="text-[#FFE9B1]">{customUploadPreview.samples.length}</strong> valid
              multimodal record(s)
            </div>
            <button
              type="button"
              onClick={() => {
                if (customUploadPreview.samples.length === 0) return;
                onUpdateVisionSamples(
                  [...customUploadPreview.samples, ...visionSamples],
                  customUploadPreview.summary
                );
              }}
              className="px-4 py-2 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors"
            >
              + Append {customUploadPreview.samples.length} Vision Record(s) to Active Corpus
            </button>
          </div>
        </section>

        {/* V09: Immutable Vision Dataset Version Control (DVC) */}
        <section className="lg:col-span-5 bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-4">
          <div>
            <h3 className="text-base font-semibold text-[#FFE9B1] flex items-center gap-2">
              <GitCommit className="w-4 h-4 text-[#FABD2F]" />
              V09. Vision Dataset Version Control (DVC + Image Manifest Hash)
            </h3>
            <p className="text-xs text-[#A89984] mt-0.5">
              Snapshot image manifest URIs, patch budgets, and normalized bounding boxes with SHA-256 lineage
            </p>
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={commitNotes}
              onChange={(e) => setCommitNotes(e.target.value)}
              placeholder="Vision dataset commit message..."
              className="flex-1 px-3 py-2 text-xs bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1] font-mono"
            />
            <button
              type="button"
              onClick={() => onCommitVisionDatasetVersion(commitNotes)}
              className="px-3.5 py-2 text-xs font-medium text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors shrink-0"
            >
              Commit Vision DVC
            </button>
          </div>

          <div className="space-y-2 max-h-44 overflow-y-auto pr-1">
            {visionDatasetCommits.map((c) => (
              <div
                key={c.commitHash}
                className="p-2.5 rounded-lg bg-[#1D2021] border border-[#3C3836] text-xs space-y-1"
              >
                <div className="flex items-center justify-between font-mono">
                  <span className="text-[#83A598] font-semibold">
                    {c.versionId} ({c.commitHash})
                  </span>
                  <span className="text-[#B8BB26] flex items-center gap-1 text-[11px]">
                    <CheckCircle2 className="w-3 h-3" />
                    {c.sampleCount} img pairs · Health {c.healthScore}%
                  </span>
                </div>
                <div className="text-[11px] text-[#D5C4A1]">{c.notes}</div>
              </div>
            ))}
          </div>
        </section>
      </div>

      {/* V10: Interactive Pan-&-Scan / NaViT Tile Grid Visualizer, Seam-Split Detector & Live [0, 1000] Bounding-Box Drawer */}
      {activeSample && (
        <VisionTileGridAndBboxDrawerPanel
          activeSample={activeSample}
          activeVisionModel={activeVisionModel}
          visionTargetFormat={visionTargetFormat}
          onUpdateSample={(updatedSample, summaryMsg) => {
            const nextSamples = visionSamples.map((s) =>
              s.id === updatedSample.id ? updatedSample : s
            );
            onUpdateVisionSamples(nextSamples, summaryMsg);
          }}
        />
      )}
    </div>
  );
};

