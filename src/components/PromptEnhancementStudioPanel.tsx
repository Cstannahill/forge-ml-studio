import React, { useMemo, useState } from 'react';
import {
  Check,
  Code2,
  Cpu,
  FileText,
  Film,
  Layers,
  Music,
  ShieldCheck,
  Sparkles,
  Wand2,
} from 'lucide-react';
import {
  ALL_REGISTERED_PROMPT_ENHANCEMENT_MODELS,
  NormalizedDatasetSample,
  PROMPT_ENHANCEMENT_TAXONOMY,
  PromptEnhancementCategoryId,
  TargetFormat,
  computeModelPromptEnhancementConfig,
  normalizeDatasetRecord,
  runNaturalLanguagePromptUpsampler,
} from '../lib/pipelineLogic';
import {
  CURATED_DATASET_BLENDS,
  CuratedDatasetBlend,
} from '../data/curatedDatasetBlends';

interface PromptEnhancementStudioPanelProps {
  targetFormat: TargetFormat;
  onApplyCuratedBlend: (blend: CuratedDatasetBlend) => void;
  onAppendSamples?: (newSamples: NormalizedDatasetSample[], note: string) => void;
}

export const PromptEnhancementStudioPanel: React.FC<
  PromptEnhancementStudioPanelProps
> = ({ targetFormat, onApplyCuratedBlend, onAppendSamples }) => {
  const [selectedCategoryId, setSelectedCategoryId] =
    useState<PromptEnhancementCategoryId>('IMAGE_GEN_NATURAL_PROSE');
  const [selectedModelId, setSelectedModelId] = useState<string>(
    'Qwen/Qwen3.5-2B-Instruct'
  );
  const [modalityFilter, setModalityFilter] = useState<
    'ALL' | 'TEXT_LLM' | 'VISION_VLM'
  >('ALL');
  const [customSeedPrompt, setCustomSeedPrompt] = useState<string>(
    PROMPT_ENHANCEMENT_TAXONOMY[0].defaultSeedPrompt
  );
  const [statusBanner, setStatusBanner] = useState<string>('');

  const activeCategory = useMemo(
    () =>
      PROMPT_ENHANCEMENT_TAXONOMY.find((c) => c.id === selectedCategoryId) ??
      PROMPT_ENHANCEMENT_TAXONOMY[0],
    [selectedCategoryId]
  );

  const filteredModels = useMemo(
    () =>
      ALL_REGISTERED_PROMPT_ENHANCEMENT_MODELS.filter((m) =>
        modalityFilter === 'ALL' ? true : m.modalityTrack === modalityFilter
      ),
    [modalityFilter]
  );

  const activeModelPlan = useMemo(
    () => computeModelPromptEnhancementConfig(selectedModelId, selectedCategoryId),
    [selectedModelId, selectedCategoryId]
  );

  const upsampleResult = useMemo(
    () =>
      runNaturalLanguagePromptUpsampler({
        rawPrompt: customSeedPrompt,
        categoryId: selectedCategoryId,
      }),
    [customSeedPrompt, selectedCategoryId]
  );

  const handleSelectCategory = (catId: PromptEnhancementCategoryId) => {
    setSelectedCategoryId(catId);
    const nextCat =
      PROMPT_ENHANCEMENT_TAXONOMY.find((c) => c.id === catId) ??
      PROMPT_ENHANCEMENT_TAXONOMY[0];
    setCustomSeedPrompt(nextCat.defaultSeedPrompt);
    setStatusBanner('');
  };

  const handleApplyBlendAndConfig = () => {
    const linkedBlend = CURATED_DATASET_BLENDS.find(
      (b) => b.id === activeCategory.linkedTextBlendId
    );
    if (!linkedBlend) return;

    // Override blend recommendedHpo and recommendedModelId with the user's selected model plan
    const customizedBlend: CuratedDatasetBlend = {
      ...linkedBlend,
      recommendedModelId:
        activeModelPlan.modalityTrack === 'TEXT_LLM'
          ? activeModelPlan.modelId
          : linkedBlend.recommendedModelId,
      recommendedHpo: activeModelPlan.hpoConfig,
    };

    onApplyCuratedBlend(customizedBlend);
    setStatusBanner(
      `Applied "${linkedBlend.title}" + tailored DoRA r=${activeModelPlan.hpoConfig.loraRank} (lr=${activeModelPlan.hpoConfig.learningRate}, ${activeModelPlan.hpoConfig.lrScheduler.toUpperCase()}) configuration for ${activeModelPlan.modelDisplayName}.`
    );
  };

  const handlePushSampleToCorpus = () => {
    if (!onAppendSamples) return;
    const newSample = normalizeDatasetRecord(
      {
        instruction: upsampleResult.trainingPairPreview.instruction,
        input: upsampleResult.trainingPairPreview.input,
        output: upsampleResult.trainingPairPreview.output,
      },
      `PromptEnhancer/${activeCategory.id}`,
      Date.now() % 10000,
      targetFormat
    );
    const enriched: NormalizedDatasetSample = {
      ...newSample,
      id: `prompt-upsample-${Date.now()}`,
      tags: Array.from(
        new Set([
          ...newSample.tags,
          'prompt-enhancement',
          activeCategory.id.toLowerCase(),
          'zero-booru-prose',
        ])
      ),
    };
    onAppendSamples(
      [enriched],
      `Appended verified ${activeCategory.shortLabel} natural-language prompt upsampling pair to active Dataset Studio corpus.`
    );
    setStatusBanner(
      `Pushed compiled natural-language training pair (${enriched.tokenEstimate} tokens) into active Dataset Studio corpus.`
    );
  };

  return (
    <section
      aria-label="Multi-Model Prompt Enhancement & Natural-Language Upsampling Suite"
      className="bg-[#252423] border border-[#4A453E] rounded-xl p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#FABD2F]">
            <Wand2 className="w-4 h-4 text-[#FABD2F]" />
            <span>
              Multi-Model Prompt Enhancement &amp; Natural-Language Upsampling Studio (All 15 Models · Incl. LFM3-1.7B)
            </span>
          </div>
          <h3 className="text-lg font-semibold text-[#FFE9B1] mt-1">
            Pre-Built Prompt Enhancement Dataset Combinations &amp; Zero-Booru Natural Prose Configurations
          </h3>
          <p className="text-xs text-[#A89984] mt-1 max-w-4xl leading-relaxed">
            Modern generative architectures—including{' '}
            <strong className="text-[#FFE9B1]">
              FLUX.1/2/3 (Klein-PUP), Qwen-Image 2/3, Imagen 3/4, Recraft V3, Wan 2.1/2.2, and Veo 3
            </strong>
            —use T5-XXL, Qwen-VL, or Gemma4 text encoders with full sentence self-attention. Legacy
            SD1.5 Danbooru comma tags (<code className="text-[#FB4934]">1girl, masterpiece, best quality, 8k</code>)
            degrade spatial composition and text rendering. Select any of the{' '}
            <strong className="text-[#8EC07C]">8 Prompt Enhancement Paradigms</strong> and any of the{' '}
            <strong className="text-[#83A598]">15 Registered Models (9 Text LLMs incl. LFM3-1.7B &amp; 6 Vision VLMs)</strong>{' '}
            below to load pre-built HuggingFace blends and tailored DoRA/QLoRA hyperparameters.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0 font-mono text-xs">
          <div className="bg-[#1D2021] border border-[#3C3836] rounded-lg px-3 py-2">
            <div className="text-[10px] text-[#A89984] uppercase">Enhancement Tracks</div>
            <div className="text-sm font-semibold text-[#FABD2F] tabular-nums">
              8 Curated Paradigms
            </div>
          </div>
          <div className="bg-[#1D2021] border border-[#3C3836] rounded-lg px-3 py-2">
            <div className="text-[10px] text-[#A89984] uppercase">Supported Models</div>
            <div className="text-sm font-semibold text-[#8EC07C] tabular-nums">
              15 Models (9 LLM + 6 VLM)
            </div>
          </div>
        </div>
      </div>

      {/* Status Notice */}
      {statusBanner && (
        <div className="bg-[#1D2021] border border-[#8EC07C]/60 rounded-lg px-4 py-2.5 flex items-center justify-between gap-3 text-xs text-[#8EC07C]">
          <div className="flex items-center gap-2">
            <Check className="w-4 h-4 shrink-0" />
            <span>{statusBanner}</span>
          </div>
          <button
            type="button"
            onClick={() => setStatusBanner('')}
            className="text-[11px] font-mono text-[#A89984] hover:text-[#FFE9B1]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* 8 Prompt Enhancement Category Selector Cards */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono uppercase tracking-wider text-[#D5C4A1]">
            Step 1 · Select Prompt Enhancement Paradigm (Research-Backed Taxonomy)
          </span>
          <span className="text-[11px] font-mono text-[#A89984]">
            Linked Blend: <strong className="text-[#83A598]">{activeCategory.linkedTextBlendId}</strong>
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {PROMPT_ENHANCEMENT_TAXONOMY.map((cat) => {
            const isSelected = cat.id === selectedCategoryId;
            return (
              <button
                key={cat.id}
                type="button"
                onClick={() => handleSelectCategory(cat.id)}
                className={`text-left p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-[#1D2021] border-[#FABD2F] shadow-sm'
                    : 'bg-[#1D2021]/60 border-[#3C3836] hover:border-[#665C54]'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span
                    className={`text-xs font-semibold ${
                      isSelected ? 'text-[#FFE9B1]' : 'text-[#D5C4A1]'
                    }`}
                  >
                    {cat.shortLabel}
                  </span>
                  <span className="text-[10px] font-mono text-[#83A598] shrink-0">
                    {cat.modalitySupport === 'VISION_AND_TEXT' ? 'LLM + VLM' : 'TEXT LLM'}
                  </span>
                </div>
                <p className="text-[11px] text-[#A89984] mt-1.5 line-clamp-2">
                  {cat.targetDownstreamGenerators.slice(0, 2).join(' · ')}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Category Research Breakdown & Dataset Weights */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 bg-[#1D2021] border border-[#3C3836] rounded-xl p-4">
        <div className="lg:col-span-7 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-[#FABD2F]" />
              <span>{activeCategory.title}</span>
            </h4>
            <span className="text-[11px] font-mono text-[#8EC07C]">
              Targets: {activeCategory.targetDownstreamGenerators.join(' · ')}
            </span>
          </div>

          <p className="text-xs text-[#D5C4A1] leading-relaxed">
            {activeCategory.researchSummary}
          </p>

          <div className="bg-[#252423] border border-[#3C3836] rounded-lg p-3 space-y-1.5">
            <div className="text-[11px] font-mono uppercase text-[#FABD2F]">
              Why Natural-Language Prose &amp; Structured Clauses Outperform Tag Salad
            </div>
            <p className="text-xs text-[#A89984] leading-relaxed">
              {activeCategory.whyNaturalProseBeatsTags}
            </p>
          </div>

          <div className="space-y-1.5">
            <div className="text-[11px] font-mono uppercase text-[#83A598]">
              Target Output Structural Clauses
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
              {activeCategory.structuralClauses.map((clause) => (
                <div
                  key={clause}
                  className="text-xs font-mono text-[#D5C4A1] bg-[#252423] border border-[#3C3836] rounded px-2.5 py-1.5"
                >
                  {clause}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Pre-Built HuggingFace Dataset Combination & Verifiers */}
        <div className="lg:col-span-5 space-y-3 border-t lg:border-t-0 lg:border-l border-[#3C3836] pt-3 lg:pt-0 lg:pl-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase text-[#8EC07C] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5" />
              <span>Pre-Built HuggingFace Dataset Blend (100%)</span>
            </span>
            <span className="text-[11px] font-mono text-[#A89984]">
              {activeCategory.datasets.length} Sources
            </span>
          </div>

          <div className="space-y-2">
            {activeCategory.datasets.map((ds) => (
              <div
                key={ds.hfRepoId}
                className="bg-[#252423] border border-[#3C3836] rounded-lg p-2.5 space-y-1"
              >
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[#FFE9B1] font-semibold truncate">{ds.hfRepoId}</span>
                  <span className="text-[#8EC07C] font-semibold tabular-nums shrink-0">
                    {ds.weightPct}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-[#1D2021] rounded overflow-hidden">
                  <div
                    className="h-full bg-[#458588]"
                    style={{ width: `${ds.weightPct}%` }}
                  />
                </div>
                <p className="text-[11px] text-[#A89984]">{ds.focus}</p>
              </div>
            ))}
          </div>

          <div className="space-y-1">
            <div className="text-[11px] font-mono uppercase text-[#FE8019] flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Stage 2 ORPO &amp; Stage 3 RLVR Reward Verifiers</span>
            </div>
            <ul className="space-y-1">
              {activeCategory.verifiersAndPenalties.map((v) => (
                <li
                  key={v}
                  className="text-[11px] font-mono text-[#D5C4A1] bg-[#252423] border border-[#3C3836] rounded px-2.5 py-1"
                >
                  • {v}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* Step 2: All 14 Models Configuration Matrix */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-[#D5C4A1] flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-[#83A598]" />
              <span>
                Step 2 · Select Any Registered Model (All 9 Text LLMs incl. LFM3-1.7B &amp; 6 Vision VLMs Supported)
              </span>
            </span>
          </div>

          <div className="flex items-center gap-1.5 font-mono text-xs">
            {(
              [
                { id: 'ALL', label: 'All 15 Models' },
                { id: 'TEXT_LLM', label: 'Text LLMs (9)' },
                { id: 'VISION_VLM', label: 'Vision VLMs (6)' },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setModalityFilter(tab.id)}
                className={`px-2.5 py-1 rounded border transition-colors ${
                  modalityFilter === tab.id
                    ? 'bg-[#458588] text-[#FFE9B1] border-[#83A598]'
                    : 'bg-[#1D2021] text-[#A89984] border-[#3C3836] hover:text-[#FFE9B1]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Grid of all 14 models */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
          {filteredModels.map((m) => {
            const isSelected = m.modelId === selectedModelId;
            const plan = computeModelPromptEnhancementConfig(
              m.modelId,
              selectedCategoryId
            );
            return (
              <button
                key={m.modelId}
                type="button"
                onClick={() => setSelectedModelId(m.modelId)}
                className={`text-left p-2.5 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-[#1D2021] border-[#8EC07C] shadow-sm'
                    : 'bg-[#1D2021]/50 border-[#3C3836] hover:border-[#665C54]'
                }`}
              >
                <div className="flex items-center justify-between gap-1">
                  <span
                    className={`text-xs font-mono font-semibold truncate ${
                      isSelected ? 'text-[#8EC07C]' : 'text-[#FFE9B1]'
                    }`}
                  >
                    {m.displayName}
                  </span>
                  <span className="text-[10px] font-mono text-[#FABD2F] tabular-nums shrink-0">
                    {m.modalityTrack === 'VISION_VLM' ? 'VLM' : 'LLM'}
                  </span>
                </div>
                <p className="text-[11px] text-[#A89984] mt-1 line-clamp-1">
                  {m.roleStrength}
                </p>
                <div className="flex items-center justify-between text-[10px] font-mono text-[#83A598] mt-1.5 pt-1.5 border-t border-[#3C3836]/70 tabular-nums">
                  <span>
                    DoRA r={plan.hpoConfig.loraRank} · α={plan.hpoConfig.loraAlpha}
                  </span>
                  <span>{plan.estimatedVramGb} GB VRAM</span>
                  <span>~{plan.estimatedUpsampleLatencyMs}ms</span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Active Model Configuration Detail Bar + 1-Click Apply */}
        <div className="bg-[#1D2021] border border-[#458588]/60 rounded-xl p-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
              <span className="text-[#FFE9B1] font-semibold">
                Active Model Plan: {activeModelPlan.modelDisplayName}
              </span>
              <span className="text-[#83A598]">
                · {activeModelPlan.quantization.toUpperCase()} · Context{' '}
                {activeModelPlan.recommendedContextLen}
              </span>
              {activeModelPlan.visionTargetFormat && (
                <span className="text-[#8EC07C]">
                  · VLM Format: {activeModelPlan.visionTargetFormat}
                </span>
              )}
            </div>
            <p className="text-xs text-[#D5C4A1]">{activeModelPlan.rationale}</p>
            <div className="text-[11px] font-mono text-[#A89984] bg-[#252423] border border-[#3C3836] rounded px-3 py-1.5">
              <strong className="text-[#FABD2F]">System Prompt Template:</strong>{' '}
              {activeModelPlan.systemPromptTemplate}
            </div>
          </div>

          <button
            type="button"
            onClick={handleApplyBlendAndConfig}
            className="px-4 py-2.5 rounded-lg bg-[#458588] hover:bg-[#83A598] text-[#1D2021] font-semibold text-xs font-mono transition-colors shrink-0 flex items-center gap-2"
          >
            <Wand2 className="w-4 h-4" />
            <span>Apply Blend &amp; Model Config to Workspace</span>
          </button>
        </div>
      </div>

      {/* Step 3: Live Natural-Language Prompt Upsampler & Anti-Booru Tag Linter */}
      <div className="bg-[#1D2021] border border-[#3C3836] rounded-xl p-4 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#3C3836] pb-3">
          <div>
            <h4 className="text-sm font-semibold text-[#FFE9B1] flex items-center gap-2">
              <Code2 className="w-4 h-4 text-[#8EC07C]" />
              <span>
                Step 3 · Interactive Natural-Language Prompt Compiler &amp; Anti-Booru Tag Linter
              </span>
            </h4>
            <p className="text-xs text-[#A89984] mt-0.5">
              Test how the fine-tuned prompt enhancer strips legacy SD1.5/Danbooru comma tags and
              compiles short ideas into structured natural English prose for{' '}
              <strong className="text-[#FFE9B1]">
                {activeCategory.targetDownstreamGenerators[0]}
              </strong>
              .
            </p>
          </div>

          <div className="flex items-center gap-3 font-mono text-xs shrink-0">
            <span className="text-[#A89984]">
              Booru Tags Stripped:{' '}
              <strong
                className={
                  upsampleResult.legacyTagContaminationCount > 0
                    ? 'text-[#FE8019]'
                    : 'text-[#8EC07C]'
                }
              >
                {upsampleResult.legacyTagContaminationCount}
              </strong>
            </span>
            <span className="text-[#A89984]">
              Prose Compliance:{' '}
              <strong className="text-[#8EC07C] tabular-nums">
                {upsampleResult.antiBooruComplianceScore}%
              </strong>
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          {/* Input Seed Box */}
          <div className="lg:col-span-5 space-y-2.5">
            <label
              htmlFor="prompt-enhancer-seed-input"
              className="block text-xs font-mono uppercase text-[#D5C4A1]"
            >
              Raw User Seed Prompt (or Legacy Tag-Salad Input)
            </label>
            <textarea
              id="prompt-enhancer-seed-input"
              rows={4}
              value={customSeedPrompt}
              onChange={(e) => setCustomSeedPrompt(e.target.value)}
              placeholder="Enter a short concept or legacy Danbooru tag prompt..."
              className="w-full rounded-lg bg-[#252423] border border-[#4A453E] px-3 py-2 text-xs text-[#FFE9B1] font-mono focus:outline-none focus:border-[#83A598]"
            />

            {upsampleResult.strippedBooruTags.length > 0 && (
              <div className="bg-[#252423] border border-[#FE8019]/50 rounded-lg p-2.5 space-y-1">
                <div className="text-[11px] font-mono text-[#FE8019] font-semibold">
                  Stripped Legacy Danbooru / SD1.5 Tags (Replaced with Natural Prose):
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {upsampleResult.strippedBooruTags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[11px] font-mono line-through text-[#FB4934] bg-[#1D2021] px-2 py-0.5 rounded border border-[#FB4934]/40"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2 pt-1">
              <button
                type="button"
                onClick={() =>
                  setCustomSeedPrompt(
                    '1girl, astronaut botanist on mars, glowing blue plants, sign "ARES LAB 04", masterpiece, best quality, absurdres, 8k, bokeh'
                  )
                }
                className="px-2.5 py-1 rounded bg-[#252423] border border-[#3C3836] hover:border-[#83A598] text-[11px] font-mono text-[#D5C4A1]"
              >
                Load Flux 3 / Qwen-Image 3 Tag-to-Prose Test
              </button>
              <button
                type="button"
                onClick={handlePushSampleToCorpus}
                className="px-3 py-1.5 rounded bg-[#8EC07C] hover:bg-[#B8BB26] text-[#1D2021] font-semibold text-xs font-mono transition-colors ml-auto"
              >
                + Push Upsampled Pair to Dataset Corpus
              </button>
            </div>
          </div>

          {/* Compiled Natural-Language Output */}
          <div className="lg:col-span-7 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-[#8EC07C]">
                Compiled Natural-Language Enhanced Prompt ({activeCategory.shortLabel})
              </span>
              <span className="text-[11px] font-mono text-[#A89984] tabular-nums">
                {upsampleResult.enhancedNaturalLanguagePrompt.split(/\s+/).length} words · Zero Booru Tags
              </span>
            </div>

            <pre className="w-full rounded-lg bg-[#252423] border border-[#8EC07C]/50 p-3 text-xs text-[#FFE9B1] font-mono whitespace-pre-wrap leading-relaxed">
              {upsampleResult.enhancedNaturalLanguagePrompt}
            </pre>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
              {upsampleResult.clauseBreakdown.map((c) => (
                <div
                  key={c.clauseTitle}
                  className="bg-[#252423] border border-[#3C3836] rounded p-2 space-y-0.5"
                >
                  <div className="text-[10px] font-mono uppercase text-[#83A598] font-semibold">
                    {c.clauseTitle}
                  </div>
                  <div className="text-[11px] text-[#D5C4A1] line-clamp-2">
                    {c.clauseContent}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
