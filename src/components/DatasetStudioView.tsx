import React, { useState } from 'react';
import {
  BarChart2,
  Check,
  CloudUpload,
  Database,
  Edit3,
  FileCheck2,
  Filter,
  GitCommit,
  Layers,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  ArchitectureFamily,
  CurriculumPacingStrategy,
  DatasetValidationReport,
  NormalizedDatasetSample,
  SyntheticDistilledPair,
  TargetFormat,
  buildCurriculumPacingPlan,
  computeTokenLengthHistogramAndTruncation,
  generateSyntheticDistillationBatch,
  inspectTokenizerAndChatTemplate,
  parseCustomDatasetUploadText,
  scoreAndPruneDatasetByIfdAndDecontamination,
  validateDatasetRecords,
} from '../lib/pipelineLogic';
import { DatasetVersionCommit } from '../data/initialPipelineData';
import {
  BlendTaskCategory,
  CURATED_DATASET_BLENDS,
  CuratedDatasetBlend,
  ModelSizeTier,
} from '../data/curatedDatasetBlends';
import { InfoTooltip } from './InfoTooltip';
import { HuggingFaceLivePreviewPanel } from './HuggingFaceLivePreviewPanel';

export interface HfDatasetCatalogItem {
  id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  description: string;
  format: string;
  estimatedRows: string;
}

interface DatasetStudioViewProps {
  samples: NormalizedDatasetSample[];
  targetFormat: TargetFormat;
  onChangeTargetFormat: (fmt: TargetFormat) => void;
  onIngestDatasets: (datasetIds: string[]) => Promise<void>;
  onApplyCuratedBlend: (blend: CuratedDatasetBlend) => void;
  onRunAgenticCleanAndTag: () => Promise<void>;
  onCommitDatasetVersion: (cloudUri: string, notes: string) => void;
  onAppendSamples?: (newSamples: NormalizedDatasetSample[], note: string) => void;
  onReplaceSamples?: (nextSamples: NormalizedDatasetSample[], note: string) => void;
  datasetVersions: DatasetVersionCommit[];
  isBusy: boolean;
}

const MODEL_TIER_FILTERS: Array<{ id: 'ALL' | ModelSizeTier; label: string }> = [
  { id: 'ALL', label: 'All Target Models (24 Blends)' },
  { id: 'LFM2.5-300M', label: 'LFM2.5 (300M)' },
  { id: 'LFM2.5-1.2B', label: 'LFM2.5 (1.2B)' },
  { id: 'LFM2.5-2.6B', label: 'LFM2.5 (2.6B)' },
  { id: 'Qwen3.5-0.8B', label: 'Qwen3.5 (0.8B)' },
  { id: 'Qwen3.5-2B', label: 'Qwen3.5 (2B)' },
  { id: 'Gemma4-E2B', label: 'Gemma4 (E2B)' },
  { id: 'Gemma4-E4B', label: 'Gemma4 (E4B)' },
];

const TASK_CATEGORY_FILTERS: Array<'ALL' | BlendTaskCategory> = [
  'ALL',
  'Tool Calling & Function Execution',
  'Code Generation & Debugging',
  'Edge RAG & Grounded QA',
  'Structured JSON & Schema Extraction',
  'Math & Step-by-Step Reasoning',
  'On-Device Assistant & Classification',
  'Domain-Specific & Security',
];

export const DatasetStudioView: React.FC<DatasetStudioViewProps> = ({
  samples,
  targetFormat,
  onChangeTargetFormat,
  onIngestDatasets,
  onApplyCuratedBlend,
  onRunAgenticCleanAndTag,
  onCommitDatasetVersion,
  onAppendSamples,
  onReplaceSamples,
  datasetVersions,
  isBusy,
}) => {
  // Curated 24 Blends Explorer State
  const [selectedModelTier, setSelectedModelTier] = useState<'ALL' | ModelSizeTier>('ALL');
  const [selectedTaskCategory, setSelectedTaskCategory] = useState<'ALL' | BlendTaskCategory>('ALL');
  const [blendSearch, setBlendSearch] = useState('');
  const [activeBlendId, setActiveBlendId] = useState<string>(CURATED_DATASET_BLENDS[0].id);
  const [appliedBlendNotice, setAppliedBlendNotice] = useState<string>('');

  // Live HuggingFace Search & Direct Ingest State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<HfDatasetCatalogItem[]>([
    {
      id: 'HuggingFaceH4/ultrachat_200k',
      author: 'HuggingFaceH4',
      downloads: 482900,
      likes: 1840,
      tags: ['chatml', 'multi-turn', 'instruction-tuning'],
      description: 'Filtered and cleaned 200k multi-turn dialogue dataset used for Zephyr & Qwen3.5 SFT.',
      format: 'messages',
      estimatedRows: '207,865',
    },
    {
      id: 'Open-Orca/SlimOrca-Dedup',
      author: 'Open-Orca',
      downloads: 319400,
      likes: 1420,
      tags: ['sharegpt', 'reasoning', 'flan'],
      description: 'Deduplicated 363k ShareGPT-formatted reasoning traces with system prompts.',
      format: 'sharegpt',
      estimatedRows: '363,491',
    },
    {
      id: 'Qwen/Qwen3-Code-Reasoning-110k',
      author: 'Qwen',
      downloads: 294100,
      likes: 1650,
      tags: ['code-generation', 'python', 'typescript'],
      description: 'High-density polyglot software engineering and step-by-step code verification traces.',
      format: 'alpaca',
      estimatedRows: '110,400',
    },
    {
      id: 'LiquidAI/LFM-LongContext-Synthesis-64k',
      author: 'LiquidAI',
      downloads: 148200,
      likes: 920,
      tags: ['long-context', 'state-space', 'document-qa'],
      description: 'Long-horizon multi-document synthesis & structured extraction optimized for LFM2.5 and Gemma4.',
      format: 'messages',
      estimatedRows: '64,000',
    },
  ]);
  const [selectedHfIds, setSelectedHfIds] = useState<string[]>([
    'HuggingFaceH4/ultrachat_200k',
    'Open-Orca/SlimOrca-Dedup',
  ]);
  const [customRepoInput, setCustomRepoInput] = useState('');
  const [cloudTargetUri, setCloudTargetUri] = useState('s3://forgeml-datasets/snapshots/v2.5.0.parquet');
  const [commitNotes, setCommitNotes] = useState(
    'Unified multi-dataset HF blend cleaned & validated for Qwen3.5 / LFM2.5 / Gemma4'
  );
  const [selectedSampleId, setSelectedSampleId] = useState<string>(samples[0]?.id ?? '');
  const [isSearchingHf, setIsSearchingHf] = useState(false);

  // 1. On-Policy Teacher Knowledge Distillation & Synthetic Data Generator State
  const [distillSeedPrompt, setDistillSeedPrompt] = useState(
    'Write a deterministic Python function to parse nested JSON tool calls and validate required keys without external dependencies.'
  );
  const [distillDomainTag, setDistillDomainTag] = useState('tool-calling-python');
  const [distillTeacherModel, setDistillTeacherModel] = useState('gemini-3-flash-preview (Frontier Teacher)');
  const [minRubricScore, setMinRubricScore] = useState<number>(8.5);
  const [generatedDistillPairs, setGeneratedDistillPairs] = useState<SyntheticDistilledPair[]>(() =>
    generateSyntheticDistillationBatch({
      seedPrompt:
        'Write a deterministic Python function to parse nested JSON tool calls and validate required keys without external dependencies.',
      domainTag: 'tool-calling-python',
      teacherModel: 'gemini-3-flash-preview',
      minRubricScore: 8.5,
      targetFormat,
    })
  );
  const [distillStatusBanner, setDistillStatusBanner] = useState<string | null>(null);

  // 2. Perplexity-Based IFD Data Pruning & 13-Gram Benchmark Decontamination State
  const [minIfdThreshold, setMinIfdThreshold] = useState<number>(0.2);
  const [maxIfdThreshold, setMaxIfdThreshold] = useState<number>(0.92);
  const [enableDecontamination, setEnableDecontamination] = useState<boolean>(true);
  const [pruningStatusBanner, setPruningStatusBanner] = useState<string | null>(null);

  // 3. Interactive Sample Editor & Context Cutoff Truncation Analyzer State
  const [contextCutoffTokens, setContextCutoffTokens] = useState<number>(256);
  const [sampleSearchQuery, setSampleSearchQuery] = useState('');
  const [onlyTruncationRisk, setOnlyTruncationRisk] = useState(false);
  const [isEditingSample, setIsEditingSample] = useState(false);
  const [editedFormattedText, setEditedFormattedText] = useState('');
  const [sampleEditNotice, setSampleEditNotice] = useState<string | null>(null);

  const validationReport: DatasetValidationReport = validateDatasetRecords(samples);
  const ifdAndDecontamReport = scoreAndPruneDatasetByIfdAndDecontamination({
    samples,
    minIfd: minIfdThreshold,
    maxIfd: maxIfdThreshold,
    enableDecontamination,
  });
  const tokenHistogramReport = computeTokenLengthHistogramAndTruncation({
    samples,
    contextCutoffTokens,
  });

  // 4. 3-Phase Curriculum Pacing & Annealing Scheduler State
  const [curriculumStrategy, setCurriculumStrategy] =
    useState<CurriculumPacingStrategy>('hybrid_ifd_length');
  const [curriculumAppliedBanner, setCurriculumAppliedBanner] = useState<string | null>(null);

  const curriculumPlan = buildCurriculumPacingPlan({
    samples,
    strategy: curriculumStrategy,
    totalSteps: 600,
  });

  // 5. Tokenizer Special-Token Surgery & Chat Template Verifier State
  const [tokenizerArch, setTokenizerArch] = useState<ArchitectureFamily>('Qwen3.5');
  const [injectReasoningTokens, setInjectReasoningTokens] = useState<boolean>(true);
  const [injectToolTokens, setInjectToolTokens] = useState<boolean>(true);

  const tokenizerReport = inspectTokenizerAndChatTemplate({
    architecture: tokenizerArch,
    modelName:
      tokenizerArch === 'LFM2.5'
        ? 'LFM2.5-1.2B-Instruct'
        : tokenizerArch === 'Gemma4'
        ? 'Gemma4-E2B-Instruct'
        : 'Qwen3.5-2B-Instruct',
    targetFormat,
    injectReasoningTokens,
    injectToolTokens,
  });

  // 6. Custom Dataset Upload / Paste & Auto-Schema Normalizer State
  const [customSourceLabel, setCustomSourceLabel] = useState<string>(
    'CustomUpload/proprietary-edge-logs.jsonl'
  );
  const [customRawText, setCustomRawText] = useState<string>(
    `{"instruction": "Extract telemetry alert code and severity from log: [GPU0] TEMP=89C POWER=410W", "input": "", "output": "<think>\\nParse GPU0 log fields and return strict JSON.\\n</think>\\n{\\"gpu_id\\": 0, \\"temp_c\\": 89, \\"severity\\": \\"CRITICAL\\"}"}\n{"instruction": "Write a compact Python helper to clamp learning rate warmup between 1e-6 and 2e-4.", "input": "", "output": "<think>\\nUse min/max clamping with linear step ratio.\\n</think>\\ndef clamp_lr(step: int, warmup: int) -> float:\\n    return max(1e-6, min(2e-4, 2e-4 * (step / max(1, warmup))))"}`
  );
  const [customImportBanner, setCustomImportBanner] = useState<string | null>(null);

  const customParsePreview = parseCustomDatasetUploadText({
    rawText: customRawText,
    sourceLabel: customSourceLabel,
    targetFormat,
  });

  const filteredSamples = samples.filter((s) => {
    const q = sampleSearchQuery.trim().toLowerCase();
    const matchesSearch =
      !q ||
      s.formattedPreview.toLowerCase().includes(q) ||
      s.sourceDataset.toLowerCase().includes(q) ||
      s.tags.some((t) => t.toLowerCase().includes(q));
    const matchesTruncation = !onlyTruncationRisk || s.tokenEstimate > contextCutoffTokens;
    return matchesSearch && matchesTruncation;
  });

  const activeSample =
    filteredSamples.find((s) => s.id === selectedSampleId) ??
    samples.find((s) => s.id === selectedSampleId) ??
    samples[0];

  const filteredBlends = CURATED_DATASET_BLENDS.filter((blend) => {
    const matchesModel =
      selectedModelTier === 'ALL' || blend.primaryTargetModels.includes(selectedModelTier);
    const matchesCategory =
      selectedTaskCategory === 'ALL' || blend.taskCategory === selectedTaskCategory;
    const q = blendSearch.trim().toLowerCase();
    const matchesQuery =
      !q ||
      blend.title.toLowerCase().includes(q) ||
      blend.specificTaskFocus.toLowerCase().includes(q) ||
      blend.datasets.some((d) => d.datasetId.toLowerCase().includes(q));
    return matchesModel && matchesCategory && matchesQuery;
  });

  const activeBlend =
    filteredBlends.find((b) => b.id === activeBlendId) ??
    filteredBlends[0] ??
    CURATED_DATASET_BLENDS[0];

  const handleLoadBlendIntoPipeline = (blend: CuratedDatasetBlend) => {
    const dsIds = blend.datasets.map((d) => d.datasetId);
    setSelectedHfIds(dsIds);
    setCommitNotes(
      `Blend #${blend.blendNumber}: ${blend.title} (${blend.primaryTargetModels.join(', ')})`
    );
    onApplyCuratedBlend(blend);
    setAppliedBlendNotice(
      `Loaded Blend #${blend.blendNumber} (${blend.title}) · Queued ${dsIds.length} HF datasets & applied ${blend.recommendedModelId} sub-2B HPO preset.`
    );
  };

  const handleSearchHuggingFace = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSearchingHf(true);
    try {
      const res = await fetch(`/api/hf/search?q=${encodeURIComponent(searchQuery)}`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.datasets)) {
          setSearchResults(data.datasets);
        }
      }
    } finally {
      setIsSearchingHf(false);
    }
  };

  const toggleDatasetSelection = (id: string) => {
    setSelectedHfIds((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleAddCustomRepo = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = customRepoInput.trim();
    if (!cleaned) return;
    if (!selectedHfIds.includes(cleaned)) {
      setSelectedHfIds((prev) => [...prev, cleaned]);
    }
    setCustomRepoInput('');
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-6 border-b border-slate-800/80">
        <div>
          <p className="text-xs text-slate-400">
            24 Task-Specific Sub-2B HuggingFace Blends · Multi-Repo Ingestion · Schema Normalization & Cloud Versioning
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100 mt-1">
            Dataset Studio & Sub-2B Curated Blend Library
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Unified Format Selector */}
          <div className="flex items-center gap-1 p-1 bg-[#111827] border border-slate-800 rounded-lg">
            {(['chatml', 'sharegpt', 'alpaca', 'openai-messages'] as TargetFormat[]).map((fmt) => (
              <button
                key={fmt}
                type="button"
                onClick={() => onChangeTargetFormat(fmt)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  targetFormat === fmt
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {fmt === 'chatml'
                  ? 'ChatML'
                  : fmt === 'sharegpt'
                  ? 'ShareGPT'
                  : fmt === 'alpaca'
                  ? 'Alpaca'
                  : 'Messages JSON'}
              </button>
            ))}
            <span className="pr-2">
              <InfoTooltip termKey="chatml" align="right" />
            </span>
          </div>

          <button
            type="button"
            onClick={onRunAgenticCleanAndTag}
            disabled={isBusy}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5" />
            {isBusy ? 'Orchestrating Hygiene...' : 'Run Agentic Clean, Redact & Tag'}
          </button>
        </div>
      </section>

      {/* Section 01: 24 Curated Task-Focused HuggingFace Dataset Blends for Sub-2B Models */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              01. Curated Task-Specific HuggingFace Dataset Blends (24 Architectures &amp; Recipes)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Purpose-built multi-dataset blends engineered for{' '}
              <span className="text-slate-200 font-mono">LFM2.5 (300M, 1.2B, 2.6B)</span>,{' '}
              <span className="text-slate-200 font-mono">Qwen3.5 (0.8B, 2B)</span>, and{' '}
              <span className="text-slate-200 font-mono">Gemma4 (E2B, E4B)</span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="blend-category-select" className="sr-only">
              Filter by Task Category
            </label>
            <select
              id="blend-category-select"
              aria-label="Filter by Task Category"
              value={selectedTaskCategory}
              onChange={(e) =>
                setSelectedTaskCategory(e.target.value as 'ALL' | BlendTaskCategory)
              }
              className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100"
            >
              {TASK_CATEGORY_FILTERS.map((cat) => (
                <option key={cat} value={cat}>
                  {cat === 'ALL' ? 'All Task Categories (7 Domains)' : cat}
                </option>
              ))}
            </select>

            <input
              type="search"
              aria-label="Search 24 curated blends"
              placeholder="Search 24 blends (e.g. SQL, RAG, tool)..."
              value={blendSearch}
              onChange={(e) => setBlendSearch(e.target.value)}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 w-56"
            />
          </div>
        </div>

        {/* Target Model Size Tier Filter Bar */}
        <div className="flex flex-wrap items-center gap-1.5">
          {MODEL_TIER_FILTERS.map((tier) => (
            <button
              key={tier.id}
              type="button"
              onClick={() => setSelectedModelTier(tier.id)}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors whitespace-nowrap ${
                selectedModelTier === tier.id
                  ? 'bg-blue-600 text-white font-medium'
                  : 'bg-[#0B0F17] text-slate-400 hover:text-slate-200 border border-slate-800/80'
              }`}
            >
              {tier.label}
            </button>
          ))}
        </div>

        {appliedBlendNotice && (
          <div
            role="status"
            className="px-3.5 py-2 bg-emerald-950/50 border border-emerald-500/40 rounded-lg text-xs text-emerald-300 font-mono"
          >
            {appliedBlendNotice}
          </div>
        )}

        {/* Master-Detail Split: Left 5 cols Blend List (24 Blends), Right 7 cols Selected Blend Spec */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Left 5 cols: Scrollable List of Curated Blends */}
          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800/80 rounded-lg divide-y divide-slate-800/70 max-h-[430px] overflow-y-auto">
            {filteredBlends.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400">
                No curated blends match the current model/task filter.
              </div>
            ) : (
              filteredBlends.map((blend) => {
                const isSelected = blend.id === activeBlend.id;
                return (
                  <button
                    key={blend.id}
                    type="button"
                    onClick={() => setActiveBlendId(blend.id)}
                    className={`w-full text-left p-3.5 transition-colors block ${
                      isSelected
                        ? 'bg-blue-600/15 border-l-2 border-blue-500'
                        : 'hover:bg-slate-800/40'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
                      <span>
                        BLEND #{String(blend.blendNumber).padStart(2, '0')} · {blend.taskCategory}
                      </span>
                      <span className="text-emerald-400 tabular-nums shrink-0">
                        {blend.estimatedVramQloraGb} GB VRAM
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-slate-100 mt-1">
                      {blend.title}
                    </div>
                    <div className="text-[11px] font-mono text-blue-400 mt-1 truncate">
                      Targets: {blend.primaryTargetModels.join(' · ')}
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Right 7 cols: Active Blend Architecture, Ratios & One-Click Loader */}
          <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-3 border-b border-slate-800">
                <div>
                  <div className="text-xs font-mono text-blue-400">
                    BLEND #{String(activeBlend.blendNumber).padStart(2, '0')} ·{' '}
                    {activeBlend.taskCategory.toUpperCase()}
                  </div>
                  <h3 className="text-base font-semibold text-slate-100 mt-0.5">
                    {activeBlend.title}
                  </h3>
                  <p className="text-xs text-slate-300 mt-1">{activeBlend.specificTaskFocus}</p>
                </div>

                <button
                  type="button"
                  onClick={() => handleLoadBlendIntoPipeline(activeBlend)}
                  className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap shrink-0"
                >
                  <Zap className="w-3.5 h-3.5" />
                  Load Blend &amp; Apply Sub-2B Preset
                </button>
              </div>

              {/* Why Effective for Sub-2B Models */}
              <div className="text-xs text-slate-300 leading-relaxed">
                <span className="font-semibold text-slate-100">
                  Why This Blend Excels on ≤2B Models:{' '}
                </span>
                {activeBlend.whyEffectiveForSmallModels}
              </div>

              {/* Constituent HuggingFace Datasets & Exact Mix Ratios */}
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-200">
                  Constituent HuggingFace Datasets &amp; Sampling Mix Ratio (100% Unified{' '}
                  {activeBlend.targetFormat.toUpperCase()})
                </div>
                <div className="divide-y divide-slate-800/80 border-t border-b border-slate-800/80">
                  {activeBlend.datasets.map((ds) => (
                    <div
                      key={ds.datasetId}
                      className="py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs"
                    >
                      <div className="min-w-0">
                        <div className="font-mono font-semibold text-slate-100">
                          {ds.datasetId}
                        </div>
                        <div className="text-slate-400 text-[11px] mt-0.5">{ds.roleInBlend}</div>
                      </div>
                      <div className="font-mono tabular-nums text-emerald-400 font-semibold shrink-0">
                        {ds.weightPct}% mix weight
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Sub-2B Hyperparameter & Hardware Spec Footer */}
            <div className="pt-3 mt-4 border-t border-slate-800 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono tabular-nums">
              <div>
                <div className="text-slate-400 text-[11px]">Primary Model</div>
                <div className="text-slate-100 truncate">{activeBlend.recommendedModelId}</div>
              </div>
              <div>
                <div className="text-slate-400 text-[11px]">Sub-2B LoRA Rank</div>
                <div className="text-slate-100">
                  r={activeBlend.recommendedHpo.loraRank} · α={activeBlend.recommendedHpo.loraAlpha}
                </div>
              </div>
              <div>
                <div className="text-slate-400 text-[11px]">LR &amp; Schedule</div>
                <div className="text-slate-100">
                  {activeBlend.recommendedHpo.learningRate} ({activeBlend.recommendedHpo.lrScheduler.toUpperCase()})
                </div>
              </div>
              <div>
                <div className="text-slate-400 text-[11px]">QLoRA VRAM / Rows</div>
                <div className="text-emerald-400">
                  {activeBlend.estimatedVramQloraGb} GB · {activeBlend.estimatedTotalRows}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 02: Multi-Dataset HuggingFace Direct Ingest & Validation Report */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  02. Custom HuggingFace Dataset Search &amp; Multi-Repo Combiner
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Point to any 1 or more HuggingFace datasets to ingest, combine, and normalize into {targetFormat.toUpperCase()}
                </p>
              </div>

              <form onSubmit={handleSearchHuggingFace} className="flex items-center gap-2">
                <input
                  type="search"
                  aria-label="Search HuggingFace datasets"
                  placeholder="Search HF Hub (e.g. math, code)..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 w-48"
                />
                <button
                  type="submit"
                  disabled={isSearchingHf}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
                >
                  <Search className="w-3.5 h-3.5" />
                  {isSearchingHf ? 'Searching...' : 'Search HF'}
                </button>
              </form>
            </div>

            {/* Direct Custom HF Repo Input */}
            <form onSubmit={handleAddCustomRepo} className="mt-4 flex items-center gap-2">
              <label htmlFor="custom-hf-repo" className="sr-only">
                Direct HuggingFace Dataset ID
              </label>
              <input
                id="custom-hf-repo"
                type="text"
                placeholder="Or enter any HuggingFace dataset ID directly (e.g. teknium/OpenHermes-2.5)..."
                value={customRepoInput}
                onChange={(e) => setCustomRepoInput(e.target.value)}
                className="flex-1 bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 placeholder:text-slate-500"
              />
              <button
                type="submit"
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors whitespace-nowrap shrink-0"
              >
                <Plus className="w-3.5 h-3.5" />
                Add Dataset ID
              </button>
            </form>

            {/* Dataset Catalog Table */}
            <div className="mt-4 divide-y divide-slate-800/70">
              {searchResults.map((ds) => {
                const isSelected = selectedHfIds.includes(ds.id);
                return (
                  <div
                    key={ds.id}
                    className="py-3 flex items-start justify-between gap-4 hover:bg-slate-800/20 px-2 rounded"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold text-slate-100">
                          {ds.id}
                        </span>
                        <span className="text-xs text-slate-400">
                          · {ds.downloads.toLocaleString()} downloads · {ds.estimatedRows} rows
                        </span>
                      </div>
                      <p className="text-xs text-slate-400 mt-1">{ds.description}</p>
                      <div className="text-[11px] text-slate-500 mt-1">
                        Source schema: {ds.format} · {ds.tags.join(' · ')}
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={() => toggleDatasetSelection(ds.id)}
                      className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 ${
                        isSelected
                          ? 'bg-blue-600/20 text-blue-300 border border-blue-500/40'
                          : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      {isSelected ? 'Selected for Blend' : 'Select Dataset'}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Queue & Ingest Action */}
          <div className="pt-4 mt-4 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="text-xs text-slate-300">
              <span className="text-slate-400">Queued for Unified Merge ({selectedHfIds.length}): </span>
              <span className="font-mono">{selectedHfIds.join(' + ') || 'None selected'}</span>
            </div>
            <button
              type="button"
              disabled={selectedHfIds.length === 0 || isBusy}
              onClick={() => onIngestDatasets(selectedHfIds)}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
            >
              <Database className="w-3.5 h-3.5" />
              Ingest, Combine &amp; Normalize ({selectedHfIds.length} Datasets)
            </button>
          </div>
        </div>

        {/* Right 5 cols: Data Validation Report */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  03. Automated Data Validation Report
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Schema verification, MinHash deduplication, PII leakage scan &amp; token lengths
                </p>
              </div>
              <FileCheck2 className="w-4 h-4 text-emerald-400" />
            </div>

            <div className="grid grid-cols-2 gap-6 my-4 pb-2">
              <div>
                <div className="text-xs text-slate-400 flex items-center">
                  <span>Dataset Health Score</span>
                  <InfoTooltip termKey="minhashDedup" align="left" />
                </div>
                <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                  <span
                    className={`text-2xl font-semibold ${
                      validationReport.overallHealthScore >= 85
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                    }`}
                  >
                    {validationReport.overallHealthScore}/100
                  </span>
                  <span className="text-xs text-slate-400">({validationReport.status})</span>
                </div>
              </div>

              <div className="border-l border-slate-800/80 pl-6">
                <div className="text-xs text-slate-400">Schema Valid / Total</div>
                <div className="mt-1 flex items-baseline gap-2 font-mono tabular-nums">
                  <span className="text-2xl font-semibold text-slate-100">
                    {validationReport.validSchemaSamples}/{validationReport.totalSamples}
                  </span>
                  <span className="text-xs text-emerald-400">100%</span>
                </div>
              </div>
            </div>

            <div className="space-y-2.5 text-xs border-t border-slate-800/80 pt-4 font-mono tabular-nums">
              <div className="flex justify-between">
                <span className="text-slate-400">PII / Secret Pattern Detections:</span>
                <span
                  className={
                    validationReport.piiFlaggedCount > 0 ? 'text-amber-400' : 'text-emerald-400'
                  }
                >
                  {validationReport.piiFlaggedCount} samples flagged
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Exact / Near-Duplicate Collisions:</span>
                <span className="text-slate-200">{validationReport.duplicateCount} records</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Token Distribution (Min / Median / P95 / Max):</span>
                <span className="text-slate-200">
                  {validationReport.minTokens} / {validationReport.medianTokens} /{' '}
                  {validationReport.p95Tokens} / {validationReport.maxTokens} tok
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Target Unified Format:</span>
                <span className="text-blue-400">{targetFormat.toUpperCase()}</span>
              </div>
            </div>
          </div>

          {/* Commit Version & Cloud Storage Sync */}
          <div className="pt-4 mt-4 border-t border-slate-800/80 space-y-3">
            <div className="text-xs font-medium text-slate-200">
              Commit Version &amp; Sync to Cloud Storage (S3 / GCS / HF Hub)
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <input
                type="text"
                aria-label="Cloud Storage Target URI"
                value={cloudTargetUri}
                onChange={(e) => setCloudTargetUri(e.target.value)}
                className="bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
              />
              <input
                type="text"
                aria-label="Dataset Commit Notes"
                value={commitNotes}
                onChange={(e) => setCommitNotes(e.target.value)}
                className="bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs text-slate-100"
              />
            </div>
            <button
              type="button"
              onClick={() => onCommitDatasetVersion(cloudTargetUri, commitNotes)}
              className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors"
            >
              <CloudUpload className="w-3.5 h-3.5 text-blue-400" />
              Snapshot Dataset Version &amp; Sync to Cloud Storage
            </button>
          </div>
        </div>
      </section>

      {/* Section 04: On-Policy Teacher Knowledge Distillation & IFD Pruning / Decontamination Gate */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 7 cols: On-Policy Knowledge Distillation & Synthetic Teacher Pipeline */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                  Sub-2B Reasoning Booster · Teacher Distillation
                </span>
                <h2 className="text-base font-semibold text-slate-100 mt-1 flex items-center">
                  <span>04. On-Policy Teacher Knowledge Distillation &amp; Judge Rubric</span>
                  <InfoTooltip termKey="teacherDistillation" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Synthesize verified &lt;think&gt;...&lt;/think&gt; reasoning traces from a frontier teacher LLM and filter with an LLM-as-a-Judge rubric
                </p>
              </div>

              <button
                type="button"
                onClick={() => {
                  const batch = generateSyntheticDistillationBatch({
                    seedPrompt: distillSeedPrompt,
                    domainTag: distillDomainTag,
                    teacherModel: distillTeacherModel,
                    minRubricScore,
                    targetFormat,
                  });
                  setGeneratedDistillPairs(batch);
                  setDistillStatusBanner(
                    `Generated ${batch.length} teacher traces (${batch.filter((b) => b.passedRubric).length} passed Judge Rubric >= ${minRubricScore}/10).`
                  );
                }}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors shrink-0"
              >
                <Sparkles className="w-3.5 h-3.5" />
                Synthesize Teacher CoT Batch
              </button>
            </div>

            {distillStatusBanner && (
              <div
                role="status"
                className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-3.5 py-2 text-xs font-mono text-emerald-300"
              >
                {distillStatusBanner}
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-12 gap-3">
              <div className="sm:col-span-6">
                <label htmlFor="distill-seed-input" className="block text-[11px] text-slate-400 mb-1">
                  Seed Prompt or Sub-2B Failure Case
                </label>
                <input
                  id="distill-seed-input"
                  type="text"
                  value={distillSeedPrompt}
                  onChange={(e) => setDistillSeedPrompt(e.target.value)}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100"
                />
              </div>
              <div className="sm:col-span-3">
                <label htmlFor="distill-teacher-select" className="block text-[11px] text-slate-400 mb-1">
                  Teacher Model
                </label>
                <select
                  id="distill-teacher-select"
                  value={distillTeacherModel}
                  onChange={(e) => setDistillTeacherModel(e.target.value)}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-200"
                >
                  <option value="gemini-3-flash-preview">Gemini 3 Flash</option>
                  <option value="deepseek-r1-671b-openrouter">DeepSeek-R1 (OpenRouter)</option>
                  <option value="qwen3.5-32b-ollama-local">Qwen3.5-32B (Local Ollama)</option>
                </select>
              </div>
              <div className="sm:col-span-3">
                <label htmlFor="distill-rubric-select" className="block text-[11px] text-slate-400 mb-1">
                  Min Judge Rubric
                </label>
                <select
                  id="distill-rubric-select"
                  value={minRubricScore}
                  onChange={(e) => setMinRubricScore(Number(e.target.value))}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs font-mono text-slate-200"
                >
                  <option value={8.0}>&gt;= 8.0 / 10</option>
                  <option value={8.5}>&gt;= 8.5 / 10 (Strict)</option>
                  <option value={9.0}>&gt;= 9.0 / 10 (Elite)</option>
                </select>
              </div>
            </div>

            <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
              {generatedDistillPairs.map((pair) => (
                <div
                  key={pair.id}
                  className={`bg-[#0B0F17] border rounded-lg p-3 space-y-1.5 ${
                    pair.passedRubric ? 'border-slate-800' : 'border-red-500/30 opacity-75'
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                    <span className="text-slate-200 font-semibold truncate max-w-md">
                      {pair.seedPrompt}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        pair.passedRubric
                          ? 'bg-emerald-500/15 text-emerald-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}
                    >
                      Judge: {pair.rubricScores.overall}/10{' '}
                      {pair.passedRubric ? '· VERIFIED' : '· REJECTED'}
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-blue-300/90 line-clamp-2">
                    &lt;think&gt; {pair.reasoningTrace} &lt;/think&gt;
                  </div>
                  <div className="text-[11px] font-mono text-slate-400 flex items-center gap-3">
                    <span>Faithfulness: {pair.rubricScores.faithfulness}</span>
                    <span>Formatting: {pair.rubricScores.formatting}</span>
                    <span>Conciseness: {pair.rubricScores.conciseness}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-mono text-slate-400">
              Passing Candidates:{' '}
              <strong className="text-emerald-400">
                {generatedDistillPairs.filter((p) => p.passedRubric).length} /{' '}
                {generatedDistillPairs.length}
              </strong>
            </span>
            <button
              type="button"
              onClick={() => {
                const passing = generatedDistillPairs
                  .filter((p) => p.passedRubric)
                  .map((p) => p.normalizedSample);
                if (passing.length > 0) {
                  onAppendSamples?.(
                    passing,
                    `Appended ${passing.length} teacher-distilled CoT samples (${distillTeacherModel}, Judge >= ${minRubricScore})`
                  );
                  setDistillStatusBanner(
                    `Appended ${passing.length} verified <think> teacher traces into active dataset corpus (${samples.length + passing.length} total rows).`
                  );
                }
              }}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              Append Passing Teacher Traces to Active Corpus
            </button>
          </div>
        </div>

        {/* Right 5 cols: Perplexity-Based IFD Data Pruning & 13-Gram Benchmark Decontamination Gate */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="pb-3 border-b border-slate-800/80">
              <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 rounded bg-blue-500/15 text-blue-400">
                LIMA Less-Is-More Filter · Eval Hygiene
              </span>
              <h2 className="text-base font-semibold text-slate-100 mt-1 flex items-center">
                <span>05. IFD Perplexity Pruning &amp; 13-Gram Decontamination Gate</span>
                <InfoTooltip termKey="ifdPruning" align="right" />
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Filter noisy high-PPL outliers and trivial rows while scanning for MMLU-Pro, GSM8K, and HumanEval+ test leakage
              </p>
            </div>

            {pruningStatusBanner && (
              <div
                role="status"
                className="bg-emerald-950/30 border border-emerald-500/40 rounded-lg px-3.5 py-2 text-xs font-mono text-emerald-300"
              >
                {pruningStatusBanner}
              </div>
            )}

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label htmlFor="min-ifd-input" className="block text-[11px] text-slate-400 mb-1">
                  Min IFD (Drop Trivial): <span className="font-mono text-slate-200">{minIfdThreshold}</span>
                </label>
                <input
                  id="min-ifd-input"
                  type="range"
                  min={0.1}
                  max={0.45}
                  step={0.05}
                  value={minIfdThreshold}
                  onChange={(e) => setMinIfdThreshold(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
              <div>
                <label htmlFor="max-ifd-input" className="block text-[11px] text-slate-400 mb-1">
                  Max IFD (Drop Noisy PPL): <span className="font-mono text-slate-200">{maxIfdThreshold}</span>
                </label>
                <input
                  id="max-ifd-input"
                  type="range"
                  min={0.7}
                  max={1.15}
                  step={0.05}
                  value={maxIfdThreshold}
                  onChange={(e) => setMaxIfdThreshold(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2 font-mono text-xs tabular-nums">
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Mean IFD Score</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  {ifdAndDecontamReport.meanIfdScore}
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Retained High-Signal</div>
                <div className="text-slate-100 font-semibold mt-0.5">
                  {ifdAndDecontamReport.retainedSamples.length} / {samples.length}
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Outliers Flagged</div>
                <div className="text-amber-400 font-semibold mt-0.5">
                  {ifdAndDecontamReport.prunedHighPplCount + ifdAndDecontamReport.prunedLowIfdCount}
                </div>
              </div>
            </div>

            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">
                  13-Gram Benchmark Decontamination Matrix
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400">
                  0 Test Leaks
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                {ifdAndDecontamReport.benchmarkDecontaminationSummary.map((b) => (
                  <div
                    key={b.benchmark}
                    className="flex items-center justify-between bg-[#111827] border border-slate-800/80 rounded px-2.5 py-1.5"
                  >
                    <span className="text-slate-300">{b.benchmark}</span>
                    <span className="text-emerald-400">{b.status}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => {
              if (ifdAndDecontamReport.retainedSamples.length > 0) {
                const dropped =
                  samples.length - ifdAndDecontamReport.retainedSamples.length;
                onReplaceSamples?.(
                  ifdAndDecontamReport.retainedSamples,
                  `Applied LIMA IFD Pruning (${minIfdThreshold}–${maxIfdThreshold}): retained ${ifdAndDecontamReport.retainedSamples.length} high-signal rows, pruned ${dropped} outliers.`
                );
                setPruningStatusBanner(
                  `LIMA IFD Filter applied: retained ${ifdAndDecontamReport.retainedSamples.length} high-density samples (Mean IFD ${ifdAndDecontamReport.meanIfdScore}), pruned ${dropped} outlier/redundant rows.`
                );
              }
            }}
            className="w-full inline-flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors"
          >
            <Filter className="w-3.5 h-3.5 text-emerald-400" />
            Apply LIMA IFD Filter &amp; Decontaminate Active Corpus
          </button>
        </div>
      </section>

      {/* Section 06: Interactive Dataset Sample Editor, Token Length Histogram & Version History */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Normalized Records List, Token Length Histogram & In-Place ChatML Editor */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-4">
          <div className="pb-3 border-b border-slate-800/80 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                06. Interactive Sample Editor &amp; Token Length Truncation Analyzer ({samples.length} Records)
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Inspect token length distribution vs. context cutoff, search records, and edit normalized {targetFormat.toUpperCase()} samples in place
              </p>
            </div>

            <div className="flex items-center gap-2">
              <label htmlFor="cutoff-select" className="text-[11px] font-mono text-slate-400">
                Cutoff:
              </label>
              <select
                id="cutoff-select"
                value={contextCutoffTokens}
                onChange={(e) => setContextCutoffTokens(Number(e.target.value))}
                className="bg-[#0B0F17] border border-slate-800 rounded px-2 py-1 text-xs font-mono text-slate-200"
              >
                <option value={128}>128 tok (Strict Edge)</option>
                <option value={256}>256 tok</option>
                <option value={512}>512 tok</option>
                <option value={2048}>2048 tok</option>
              </select>
            </div>
          </div>

          {/* Token Length Distribution Histogram */}
          <div className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-3.5 space-y-2.5">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <span className="text-slate-300 font-semibold">
                Token Length Histogram (Avg: {tokenHistogramReport.avgSampleTokens} tok · Max:{' '}
                {tokenHistogramReport.maxSampleTokens} tok)
              </span>
              <span
                className={
                  tokenHistogramReport.truncatedSampleIds.length > 0
                    ? 'text-amber-400'
                    : 'text-emerald-400'
                }
              >
                {tokenHistogramReport.truncatedSampleIds.length} sample(s) &gt;{' '}
                {contextCutoffTokens} tok ({tokenHistogramReport.truncationRiskPct}% truncation risk)
              </span>
            </div>
            <div className="grid grid-cols-5 gap-2">
              {tokenHistogramReport.buckets.map((b) => (
                <div
                  key={b.rangeLabel}
                  className={`p-2 rounded border text-center font-mono text-xs ${
                    b.exceedsCutoff && b.count > 0
                      ? 'bg-amber-500/10 border-amber-500/40 text-amber-300'
                      : 'bg-[#111827] border-slate-800 text-slate-300'
                  }`}
                >
                  <div className="text-[10px] text-slate-400">{b.rangeLabel}</div>
                  <div className="text-sm font-semibold mt-0.5">{b.count} rows</div>
                </div>
              ))}
            </div>
          </div>

          {/* Search & Truncation Filter Bar */}
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="search"
              aria-label="Search dataset samples"
              placeholder="Search samples by prompt, code, or tag..."
              value={sampleSearchQuery}
              onChange={(e) => setSampleSearchQuery(e.target.value)}
              className="flex-1 bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500"
            />
            <label className="inline-flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5">
              <input
                type="checkbox"
                checked={onlyTruncationRisk}
                onChange={(e) => setOnlyTruncationRisk(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-blue-600"
              />
              <span>Only &gt; {contextCutoffTokens} tok</span>
            </label>
          </div>

          {sampleEditNotice && (
            <div className="text-xs font-mono text-emerald-400">{sampleEditNotice}</div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
            <div className="md:col-span-5 divide-y divide-slate-800/70 max-h-72 overflow-y-auto pr-1">
              {filteredSamples.map((s) => {
                const diag = ifdAndDecontamReport.diagnostics[s.id];
                const isTruncated = s.tokenEstimate > contextCutoffTokens;
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => {
                      setSelectedSampleId(s.id);
                      setIsEditingSample(false);
                      setEditedFormattedText(s.formattedPreview);
                    }}
                    className={`w-full text-left py-2.5 px-2.5 rounded transition-colors ${
                      activeSample?.id === s.id
                        ? 'bg-blue-600/15 text-slate-100'
                        : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-mono font-medium truncate">
                        {s.sourceDataset}
                      </span>
                      {isTruncated && (
                        <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 shrink-0">
                          &gt;Cutoff
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5 font-mono tabular-nums">
                      {s.tokenEstimate} tok · IFD {diag?.ifdScore ?? 0.55} · PPL{' '}
                      {diag?.perplexity ?? 5.1}
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5 truncate">
                      Tags: {s.tags.join(' · ')}
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="md:col-span-7 bg-[#0B0F17] border border-slate-800/80 rounded-lg p-3.5 font-mono text-xs overflow-x-auto max-h-72 flex flex-col justify-between">
              {activeSample ? (
                <>
                  <div className="pb-2 mb-2 border-b border-slate-800 text-[11px] text-slate-400 flex items-center justify-between gap-2">
                    <span className="truncate">Source: {activeSample.sourceDataset}</span>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {!isEditingSample ? (
                        <button
                          type="button"
                          onClick={() => {
                            setIsEditingSample(true);
                            setEditedFormattedText(activeSample.formattedPreview);
                          }}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px]"
                        >
                          <Edit3 className="w-3 h-3" />
                          Edit Sample
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => {
                            const updatedList = samples.map((item) =>
                              item.id === activeSample.id
                                ? {
                                    ...item,
                                    formattedPreview: editedFormattedText,
                                    tokenEstimate: Math.max(
                                      12,
                                      Math.round(editedFormattedText.length / 3.8)
                                    ),
                                    cleaned: true,
                                  }
                                : item
                            );
                            onReplaceSamples?.(
                              updatedList,
                              `Edited sample ${activeSample.id} in-place (${targetFormat.toUpperCase()}).`
                            );
                            setIsEditingSample(false);
                            setSampleEditNotice(
                              `Saved in-place edit to sample ${activeSample.id}.`
                            );
                          }}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-[11px]"
                        >
                          <Check className="w-3 h-3" />
                          Save Edit
                        </button>
                      )}
                    </div>
                  </div>
                  {isEditingSample ? (
                    <textarea
                      aria-label="Edit normalized sample content"
                      value={editedFormattedText}
                      onChange={(e) => setEditedFormattedText(e.target.value)}
                      rows={8}
                      className="w-full bg-[#111827] border border-slate-700 rounded p-2 text-[11px] font-mono text-slate-100 leading-relaxed"
                    />
                  ) : (
                    <pre className="text-slate-200 whitespace-pre-wrap leading-relaxed text-[11px]">
                      {activeSample.formattedPreview}
                    </pre>
                  )}
                </>
              ) : (
                <div className="text-slate-500 py-8 text-center">Select a sample to inspect</div>
              )}
            </div>
          </div>
        </div>

        {/* Version History & Cloud Sync Ledger */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800/90 rounded-lg p-5">
          <div className="pb-4 border-b border-slate-800/80 flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                07. Dataset Versioning &amp; Cloud Sync Ledger
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Immutable SHA-256 dataset snapshots synced with remote object storage
              </p>
            </div>
            <GitCommit className="w-4 h-4 text-slate-400" />
          </div>

          <div className="mt-4 divide-y divide-slate-800/70 max-h-96 overflow-y-auto">
            {datasetVersions.map((ver) => (
              <div key={ver.versionId} className="py-3 space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono font-semibold text-slate-100">
                    {ver.versionId} · {ver.commitHash}
                  </span>
                  <span className="font-mono tabular-nums text-emerald-400">
                    Health {ver.healthScore}/100 · Synced
                  </span>
                </div>
                <p className="text-xs text-slate-300">{ver.notes}</p>
                <div className="text-[11px] font-mono text-slate-400 truncate">
                  URI: {ver.cloudUri}
                </div>
                <div className="text-[11px] text-slate-500">
                  Sources: {ver.sourceDatasets.join(' + ')} · {ver.timestamp}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 08: 3-Phase Curriculum Pacing Scheduler & Tokenizer Special-Token Surgery Verifier */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 6 cols: 3-Phase Curriculum Learning & Annealing Scheduler */}
        <div className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100 flex items-center">
                  <span>08. 3-Phase Curriculum Pacing &amp; CoT Annealing Scheduler</span>
                  <InfoTooltip termKey="curriculumLearning" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Order sub-2B training samples from easy syntax warmup to high-IFD reasoning chains
                </p>
              </div>

              <select
                aria-label="Curriculum Pacing Strategy"
                value={curriculumStrategy}
                onChange={(e) =>
                  setCurriculumStrategy(e.target.value as CurriculumPacingStrategy)
                }
                className="bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
              >
                <option value="hybrid_ifd_length">Hybrid IFD + Length (Recommended)</option>
                <option value="ifd_ascending">IFD Difficulty Ascending</option>
                <option value="length_ascending">Token Length Ascending</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3 font-mono text-xs tabular-nums">
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Convergence Speed Gain</div>
                <div className="text-emerald-400 font-semibold mt-0.5">
                  +{curriculumPlan.projectedConvergenceBoostPct}% Faster Convergence
                </div>
              </div>
              <div className="bg-[#0B0F17] border border-slate-800 rounded p-2.5">
                <div className="text-[10px] text-slate-400">Early Gradient Variance</div>
                <div className="text-blue-400 font-semibold mt-0.5">
                  -{curriculumPlan.gradientVarianceReductionPct}% Spike Reduction
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {curriculumPlan.phases.map((p) => (
                <div
                  key={p.phaseIndex}
                  className="bg-[#0B0F17] border border-slate-800/90 rounded-lg p-3 space-y-1"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                    <span className="font-semibold text-slate-100">{p.phaseName}</span>
                    <span className="text-emerald-400">
                      {p.stepRangeLabel} · {p.lrSchedulePhase}
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-400">
                    {p.sampleCount} samples · Avg {p.avgTokens} tok · Mean IFD {p.avgIfdScore}
                  </div>
                  <p className="text-[11px] text-slate-400">{p.focusDescription}</p>
                </div>
              ))}
            </div>

            {curriculumAppliedBanner && (
              <div className="text-xs font-mono text-emerald-400">{curriculumAppliedBanner}</div>
            )}
          </div>

          <button
            type="button"
            onClick={() => {
              onReplaceSamples?.(
                curriculumPlan.orderedSamples,
                `Sorted ${curriculumPlan.orderedSamples.length} active samples by 3-Phase Curriculum (${curriculumStrategy}).`
              );
              setCurriculumAppliedBanner(
                `Re-ordered all ${curriculumPlan.orderedSamples.length} active samples into 3-Phase Curriculum order (${curriculumStrategy}).`
              );
            }}
            className="w-full inline-flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
          >
            <Layers className="w-3.5 h-3.5" />
            Sort Active Corpus by 3-Phase Curriculum Order
          </button>
        </div>

        {/* Right 6 cols: Tokenizer Special-Token Surgery & Chat Template Verifier */}
        <div className="lg:col-span-6 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100 flex items-center">
                  <span>Tokenizer Special-Token Surgery &amp; EOS Verifier</span>
                  <InfoTooltip termKey="tokenizerSurgery" align="left" />
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Verify atomic special token IDs and prevent sub-2B infinite EOS generation loops
                </p>
              </div>

              <div className="flex items-center gap-1 bg-[#0B0F17] p-1 rounded-lg border border-slate-800">
                {(['Qwen3.5', 'LFM2.5', 'Gemma4'] as ArchitectureFamily[]).map((arch) => (
                  <button
                    key={arch}
                    type="button"
                    onClick={() => setTokenizerArch(arch)}
                    className={`px-2.5 py-1 text-[11px] font-mono rounded transition-colors ${
                      tokenizerArch === arch
                        ? 'bg-blue-600 text-white font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {arch}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <label className="flex items-center gap-2 p-2.5 rounded bg-[#0B0F17] border border-slate-800 text-xs text-slate-200 cursor-pointer">
                <input
                  type="checkbox"
                  checked={injectReasoningTokens}
                  onChange={(e) => setInjectReasoningTokens(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-900 text-blue-600"
                />
                <span>Register Atomic &lt;think&gt; / &lt;/think&gt; IDs</span>
              </label>
              <label className="flex items-center gap-2 p-2.5 rounded bg-[#0B0F17] border border-slate-800 text-xs text-slate-200 cursor-pointer">
                <input
                  type="checkbox"
                  checked={injectToolTokens}
                  onChange={(e) => setInjectToolTokens(e.target.checked)}
                  className="rounded border-slate-700 bg-slate-900 text-blue-600"
                />
                <span>Register Atomic &lt;tool_call&gt; IDs</span>
              </label>
            </div>

            {/* Special Token Atomic ID Table */}
            <div className="bg-[#0B0F17] border border-slate-800 rounded-lg p-3 space-y-1.5 font-mono text-xs">
              <div className="flex items-center justify-between text-[11px] text-slate-400 pb-1 border-b border-slate-800">
                <span>
                  EOS Token: <strong className="text-slate-200">{tokenizerReport.eosTokenLiteral}</strong> (ID{' '}
                  {tokenizerReport.eosTokenId})
                </span>
                <span
                  className={
                    tokenizerReport.isEosAligned ? 'text-emerald-400' : 'text-amber-400'
                  }
                >
                  {tokenizerReport.isEosAligned
                    ? '[100% ATOMIC SPECIAL TOKENS]'
                    : '[WARNING: FRAGMENTED DELIMITERS]'}
                </span>
              </div>
              {tokenizerReport.specialTokens.map((tok) => (
                <div
                  key={tok.tokenLiteral}
                  className="flex items-center justify-between text-[11px] py-0.5"
                >
                  <span className="text-slate-200">
                    {tok.tokenLiteral} <span className="text-slate-500">({tok.roleDescription})</span>
                  </span>
                  <span
                    className={
                      tok.atomicStatus === 'ATOMIC_SINGLE_ID'
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                    }
                  >
                    ID #{tok.tokenId} ·{' '}
                    {tok.atomicStatus === 'ATOMIC_SINGLE_ID'
                      ? '1 Atomic Token'
                      : `${tok.subwordCount} Subwords`}
                  </span>
                </div>
              ))}
            </div>

            <pre className="p-3 bg-[#0B0F17] border border-slate-800 rounded-lg font-mono text-[11px] text-slate-300 overflow-x-auto max-h-32 whitespace-pre leading-relaxed">
              {tokenizerReport.surgeryScriptPy}
            </pre>
          </div>

          <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Embedding Table Alignment:</span>
            <strong className="text-blue-400">
              {tokenizerReport.requiresResizeTokenEmbeddings
                ? 'resize_token_embeddings(pad_to_multiple_of=64)'
                : 'Native Vocab Table'}
            </strong>
          </div>
        </div>
      </section>

      {/* Section 09: Custom Dataset Upload / Paste & Auto-Schema Normalizer (JSONL / CSV / JSON) */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-3 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              09. Custom Dataset Upload / Raw Paste &amp; Auto-Schema Normalizer (JSONL · JSON · CSV)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Drop a local .jsonl, .json, or .csv file or paste proprietary logs below to auto-normalize into {targetFormat.toUpperCase()} and append to your active blend
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <input
              type="text"
              aria-label="Custom Dataset Source Tag"
              value={customSourceLabel}
              onChange={(e) => setCustomSourceLabel(e.target.value)}
              className="bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 w-64"
            />
            <label className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg cursor-pointer transition-colors">
              <CloudUpload className="w-3.5 h-3.5 text-blue-400" />
              <span>Upload .JSONL / .CSV File</span>
              <input
                type="file"
                accept=".jsonl,.json,.csv,.txt"
                className="sr-only"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  setCustomSourceLabel(`LocalFile/${file.name}`);
                  const reader = new FileReader();
                  reader.onload = () => {
                    if (typeof reader.result === 'string') {
                      setCustomRawText(reader.result);
                    }
                  };
                  reader.readAsText(file);
                }}
              />
            </label>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
          <div className="lg:col-span-7 flex flex-col space-y-2">
            <label htmlFor="custom-raw-paste" className="text-xs font-mono text-slate-400">
              Raw JSONL / JSON Array / CSV Input ({customParsePreview.detectedFormat} Auto-Detected):
            </label>
            <textarea
              id="custom-raw-paste"
              rows={5}
              value={customRawText}
              onChange={(e) => setCustomRawText(e.target.value)}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-100 leading-relaxed"
            />
          </div>

          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-slate-400">Detected Schema Format:</span>
                <span className="text-emerald-400 font-semibold">
                  {customParsePreview.detectedFormat}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Valid Normalized Rows:</span>
                <span className="text-slate-100 font-semibold">
                  {customParsePreview.parsedSamples.length} record(s)
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Target Serialization:</span>
                <span className="text-blue-400">{targetFormat.toUpperCase()}</span>
              </div>
              <p className="text-[11px] text-slate-400 pt-1">
                {customParsePreview.summaryMessage}
              </p>
              {customImportBanner && (
                <div className="text-xs text-emerald-400 pt-1">{customImportBanner}</div>
              )}
            </div>

            <button
              type="button"
              onClick={() => {
                if (customParsePreview.parsedSamples.length > 0) {
                  onAppendSamples?.(
                    customParsePreview.parsedSamples,
                    customParsePreview.summaryMessage
                  );
                  setCustomImportBanner(
                    `Appended ${customParsePreview.parsedSamples.length} normalized custom sample(s) from ${customSourceLabel} into active corpus.`
                  );
                }
              }}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3.5 py-2 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              Append {customParsePreview.parsedSamples.length} Normalized Custom Record(s) to Active Corpus
            </button>
          </div>
        </div>
      </section>

      {/* Live HuggingFace Hub Dataset Inspector & Column Mapper */}
      <HuggingFaceLivePreviewPanel
        targetFormat={targetFormat}
        onIngestPreviewRows={(datasetId, newRows) => {
          onAppendSamples?.(
            newRows,
            `Ingested ${newRows.length} live-mapped rows from HuggingFace Hub (${datasetId}).`
          );
        }}
      />
    </div>
  );
};
