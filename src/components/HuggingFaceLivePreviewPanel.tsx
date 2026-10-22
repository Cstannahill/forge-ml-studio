import React, { useState } from 'react';
import {
  CheckCircle2,
  Database,
  DownloadCloud,
  RefreshCw,
  Search,
  Sparkles,
} from 'lucide-react';
import { NormalizedDatasetSample, TargetFormat, normalizeDatasetRecord } from '../lib/pipelineLogic';

interface HfPreviewRow {
  rowIndex: number;
  detectedColumns: string[];
  instruction: string;
  input: string;
  output: string;
}

interface HfPreviewResponse {
  datasetId: string;
  sourceMode: 'LIVE_HF_HUB_API' | 'WORKSTATION_SNAPSHOT_CACHE';
  detectedColumns: string[];
  rows: HfPreviewRow[];
}

interface HuggingFaceLivePreviewPanelProps {
  targetFormat: TargetFormat;
  onIngestPreviewRows: (datasetId: string, samples: NormalizedDatasetSample[]) => void;
}

export const HuggingFaceLivePreviewPanel: React.FC<HuggingFaceLivePreviewPanelProps> = ({
  targetFormat,
  onIngestPreviewRows,
}) => {
  const [repoQuery, setRepoQuery] = useState<string>('Qwen/Qwen3-Code-Reasoning-110k');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [previewData, setPreviewData] = useState<HfPreviewResponse | null>(null);
  const [ingestedBanner, setIngestedBanner] = useState<string | null>(null);

  const quickPresets = [
    'Qwen/Qwen3-Code-Reasoning-110k',
    'HuggingFaceH4/ultrachat_200k',
    'open-r1/OpenR1-Math-220k',
    'argilla/ultrafeedback-binarized-preferences-cleaned',
  ];

  const handleFetchPreview = async (datasetIdOverride?: string) => {
    const targetRepo = (datasetIdOverride ?? repoQuery).trim();
    if (!targetRepo) return;
    if (datasetIdOverride) setRepoQuery(datasetIdOverride);

    setIsLoading(true);
    setIngestedBanner(null);
    try {
      const res = await fetch(
        `/api/hf/preview?dataset=${encodeURIComponent(targetRepo)}&limit=3`
      );
      if (res.ok) {
        const data = (await res.json()) as HfPreviewResponse;
        setPreviewData(data);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleIngestIntoCorpus = () => {
    if (!previewData || previewData.rows.length === 0) return;
    const normalized = previewData.rows.map((r, idx) =>
      normalizeDatasetRecord(
        {
          instruction: r.instruction,
          input: r.input,
          output: r.output,
        },
        previewData.datasetId,
        idx + 50,
        targetFormat
      )
    );
    onIngestPreviewRows(previewData.datasetId, normalized);
    setIngestedBanner(
      `Ingested ${normalized.length} normalized ${targetFormat.toUpperCase()} row(s) from ${previewData.datasetId} into active corpus.`
    );
  };

  return (
    <section className="bg-[#252423] border border-[#83A598]/50 rounded-xl p-5 space-y-4">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-[#3C3836] pb-3">
        <div>
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#83A598]">
            <Database className="w-4 h-4 text-[#8EC07C]" />
            Live HuggingFace Hub Dataset Inspector &amp; Automatic Schema Mapper (`/api/hf/preview`)
          </div>
          <h3 className="text-sm font-semibold text-[#FFE9B1] mt-0.5">
            Stream &amp; Map Arbitrary HuggingFace Hub Repos Directly into Active Training Blend
          </h3>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {quickPresets.map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => handleFetchPreview(preset)}
              className="px-2.5 py-1 text-[11px] font-mono rounded bg-[#1D2021] hover:bg-[#3C3836] border border-[#4A453E] text-[#D5C4A1]"
            >
              {preset.split('/')[1]}
            </button>
          ))}
        </div>
      </div>

      {/* Search & Fetch Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-[#A89984] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            aria-label="HuggingFace dataset repository ID"
            value={repoQuery}
            onChange={(e) => setRepoQuery(e.target.value)}
            placeholder="e.g. HuggingFaceH4/ultrachat_200k or open-r1/OpenR1-Math-220k"
            className="w-full pl-9 pr-3 py-2 text-xs font-mono bg-[#1D2021] border border-[#4A453E] rounded-lg text-[#FFE9B1]"
          />
        </div>

        <button
          type="button"
          onClick={() => handleFetchPreview()}
          disabled={isLoading}
          className="px-4 py-2 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors inline-flex items-center justify-center gap-1.5 shrink-0"
        >
          {isLoading ? (
            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <DownloadCloud className="w-3.5 h-3.5" />
          )}
          Preview Live HuggingFace Rows
        </button>

        {previewData && (
          <button
            type="button"
            onClick={handleIngestIntoCorpus}
            className="px-4 py-2 text-xs font-semibold text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors inline-flex items-center justify-center gap-1.5 shrink-0"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Ingest Mapped Rows into Active Corpus (+{previewData.rows.length})
          </button>
        )}
      </div>

      {ingestedBanner && (
        <div className="p-2.5 rounded-lg bg-[#1D2021] border border-[#B8BB26]/50 text-xs font-mono text-[#B8BB26] flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{ingestedBanner}</span>
        </div>
      )}

      {previewData && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono bg-[#1D2021] px-3 py-2 rounded-lg border border-[#3C3836]">
            <span className="text-[#FFE9B1]">
              Repo: <strong className="text-[#8EC07C]">{previewData.datasetId}</strong> · Source:{' '}
              <span className="text-[#83A598]">{previewData.sourceMode}</span>
            </span>
            <span className="text-[#A89984]">
              Auto-Mapped Columns:{' '}
              <code className="text-[#FABD2F]">{previewData.detectedColumns.join(', ')}</code> →{' '}
              <code className="text-[#B8BB26]">{targetFormat.toUpperCase()}</code>
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {previewData.rows.map((row) => (
              <div
                key={`${previewData.datasetId}-row-${row.rowIndex}`}
                className="p-3 rounded-xl bg-[#1D2021] border border-[#3C3836] space-y-2 text-xs"
              >
                <div className="flex items-center justify-between font-mono text-[11px] text-[#83A598]">
                  <span>Row #{row.rowIndex + 1}</span>
                  <span>Mapped → {targetFormat}</span>
                </div>
                <div className="text-[#FFE9B1] font-medium line-clamp-2">{row.instruction}</div>
                <pre className="p-2 rounded bg-[#141617] border border-[#3C3836] text-[10px] font-mono text-[#D5C4A1] overflow-x-auto max-h-28">
                  {row.output}
                </pre>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
};
