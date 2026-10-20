import React, { useState } from 'react';
import {
  CloudUpload,
  Database,
  FileCheck2,
  GitCommit,
  Layers,
  Plus,
  Search,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  DatasetValidationReport,
  HyperparameterConfig,
  NormalizedDatasetSample,
  TargetFormat,
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

  const validationReport: DatasetValidationReport = validateDatasetRecords(samples);
  const activeSample = samples.find((s) => s.id === selectedSampleId) ?? samples[0];

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

      {/* Section 03: Normalized Sample Inspector & Version History */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Normalized Records List & Formatted Preview */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800/90 rounded-lg p-5">
          <div className="pb-4 border-b border-slate-800/80 flex items-center justify-between">
            <div>
              <h2 className="text-base font-semibold text-slate-100">
                04. Unified Normalized Corpus Inspector ({samples.length} Active Records)
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Inspect how heterogeneous Alpaca, ShareGPT, and Messages rows normalize into {targetFormat.toUpperCase()}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 mt-4">
            <div className="md:col-span-5 divide-y divide-slate-800/70 max-h-72 overflow-y-auto pr-1">
              {samples.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  onClick={() => setSelectedSampleId(s.id)}
                  className={`w-full text-left py-2.5 px-2.5 rounded transition-colors ${
                    activeSample?.id === s.id
                      ? 'bg-blue-600/15 text-slate-100'
                      : 'hover:bg-slate-800/40 text-slate-300'
                  }`}
                >
                  <div className="text-xs font-mono font-medium truncate">{s.sourceDataset}</div>
                  <div className="text-[11px] text-slate-400 mt-0.5 font-mono tabular-nums">
                    {s.tokenEstimate} tok · Quality {s.qualityScore}% ·{' '}
                    {s.piiDetected ? 'PII Flagged' : 'Clean'}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-0.5 truncate">
                    Tags: {s.tags.join(' · ')}
                  </div>
                </button>
              ))}
            </div>

            <div className="md:col-span-7 bg-[#0B0F17] border border-slate-800/80 rounded-lg p-3.5 font-mono text-xs overflow-x-auto max-h-72">
              {activeSample ? (
                <>
                  <div className="pb-2 mb-2 border-b border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
                    <span>Source: {activeSample.sourceDataset}</span>
                    <span>
                      Status:{' '}
                      {activeSample.piiDetected
                        ? 'Needs PII Redaction'
                        : 'Normalized & Verified'}
                    </span>
                  </div>
                  <pre className="text-slate-200 whitespace-pre-wrap leading-relaxed text-[11px]">
                    {activeSample.formattedPreview}
                  </pre>
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
                05. Dataset Versioning &amp; Cloud Sync Ledger
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Immutable SHA-256 dataset snapshots synced with remote object storage
              </p>
            </div>
            <GitCommit className="w-4 h-4 text-slate-400" />
          </div>

          <div className="mt-4 divide-y divide-slate-800/70 max-h-72 overflow-y-auto">
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
    </div>
  );
};
