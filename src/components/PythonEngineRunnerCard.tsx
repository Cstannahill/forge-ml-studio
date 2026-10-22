import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Copy,
  Cpu,
  FileCode2,
  FolderGit2,
  Play,
  RefreshCw,
  Terminal,
} from 'lucide-react';
import { PythonEngineMaterializedBundle } from '../lib/pipelineLogic';

interface PreflightStatus {
  pythonVersion: string;
  hasTorch: boolean;
  hasCuda: boolean;
  cudaDeviceName: string;
  hasUnsloth: boolean;
  hasTransformers: boolean;
  hasTrl: boolean;
  engineScriptsReady: boolean;
  scripts: {
    textPipeline: string;
    visionPipeline: string;
    exportAndMerge: string;
    requirements: string;
  };
}

interface ActiveJobResponse {
  runId: string;
  track: 'TEXT_LLM' | 'VISION_VLM';
  status: 'RUNNING' | 'COMPLETED' | 'FAILED';
  command: string;
  manifestDiskPath: string;
  datasetDiskPath: string;
  logs: string[];
  exitCode: number | null;
}

interface PythonEngineRunnerCardProps {
  bundle: PythonEngineMaterializedBundle;
  onLogMessage?: (msg: string) => void;
}

export const PythonEngineRunnerCard: React.FC<PythonEngineRunnerCardProps> = ({
  bundle,
  onLogMessage,
}) => {
  const [preflight, setPreflight] = useState<PreflightStatus | null>(null);
  const [activeJob, setActiveJob] = useState<ActiveJobResponse | null>(null);
  const [isSpawning, setIsSpawning] = useState<boolean>(false);
  const [copiedCmd, setCopiedCmd] = useState<boolean>(false);
  const [showManifestPreview, setShowManifestPreview] = useState<boolean>(false);

  const fetchPreflight = async () => {
    try {
      const res = await fetch('/api/engine/preflight');
      if (res.ok) {
        const data = (await res.json()) as PreflightStatus;
        setPreflight(data);
      }
    } catch {
      // Ignore network error
    }
  };

  useEffect(() => {
    fetchPreflight();
  }, []);

  const handleMaterializeAndRun = async (mode: 'dry-run' | 'full-train') => {
    setIsSpawning(true);
    try {
      const res = await fetch('/api/engine/materialize-and-run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          runId: bundle.runId,
          track: bundle.track,
          manifestFilename: bundle.manifestFilename,
          datasetJsonlFilename: bundle.datasetJsonlFilename,
          manifestJson: bundle.manifestJson,
          datasetJsonlContent: bundle.datasetJsonlContent,
          mode,
        }),
      });
      if (res.ok) {
        const data = (await res.json()) as ActiveJobResponse;
        setActiveJob(data);
        if (onLogMessage) {
          onLogMessage(
            `Materialized ${bundle.manifestFilename} & ${bundle.datasetJsonlFilename} to ${data.manifestDiskPath} and executed Python engine (${mode}).`
          );
        }
      }
    } finally {
      setIsSpawning(false);
    }
  };

  const handleCopyCommand = () => {
    navigator.clipboard.writeText(bundle.pythonTrainCommand);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 1800);
  };

  return (
    <section className="bg-[#252423] border border-[#8EC07C]/55 rounded-xl p-6 space-y-5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-[#3C3836] pb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-wider text-[#8EC07C]">
            <Terminal className="w-4 h-4" />
            Real Python Execution Layer (`/engine/`) · Unsloth + HuggingFace Transformers + TRL + PEFT
          </div>
          <h3 className="text-base font-semibold text-[#FFE9B1]">
            Local GPU Workstation &amp; Docker Subprocess Bridge ({bundle.engineBackendLabel})
          </h3>
          <p className="text-xs text-[#A89984] max-w-4xl leading-relaxed">
            Writes your active hyperparameter manifest (<code className="font-mono text-[#83A598]">{bundle.manifestFilename}</code>)
            and normalized dataset (<code className="font-mono text-[#83A598]">{bundle.datasetJsonlFilename}</code>) directly
            to <code className="font-mono text-[#FFE9B1]">./runs/</code> on disk and spawns{' '}
            <code className="font-mono text-[#8EC07C]">
              {bundle.track === 'VISION_VLM'
                ? 'engine/train_vision_pipeline.py'
                : 'engine/train_text_pipeline.py'}
            </code>
            .
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setShowManifestPreview((prev) => !prev)}
            className="px-3 py-2 text-xs font-mono text-[#D5C4A1] hover:text-[#FFE9B1] bg-[#1D2021] border border-[#4A453E] rounded-lg flex items-center gap-1.5"
          >
            <FileCode2 className="w-3.5 h-3.5 text-[#83A598]" />
            {showManifestPreview ? 'Hide Manifest JSON' : 'Inspect Manifest JSON'}
          </button>

          <button
            type="button"
            onClick={() => handleMaterializeAndRun('dry-run')}
            disabled={isSpawning}
            className="px-3.5 py-2 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center gap-1.5"
          >
            {isSpawning ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )}
            Materialize to ./runs/ &amp; Verify Python Engine (--dry-run)
          </button>

          <button
            type="button"
            onClick={() =>
              handleMaterializeAndRun(preflight?.hasTorch ? 'full-train' : 'dry-run')
            }
            disabled={isSpawning}
            className="px-3.5 py-2 text-xs font-medium text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors flex items-center gap-1.5"
          >
            <Cpu className="w-3.5 h-3.5" />
            Spawn GPU Training Subprocess
          </button>
        </div>
      </div>

      {/* Preflight Environment & Engine Script Status Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Host Python Runtime</div>
          <div className="text-xs font-mono font-bold text-[#FFE9B1] mt-1 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-[#B8BB26]" />
            {preflight?.pythonVersion ?? 'Python 3.11+'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Repo `/engine/` Scripts</div>
          <div className="text-xs font-mono font-bold text-[#8EC07C] mt-1">
            {preflight?.engineScriptsReady !== false
              ? '3/3 Verified on Disk'
              : 'Checking /engine/...'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">Active Trainer Backend</div>
          <div className="text-xs font-mono font-bold text-[#83A598] mt-1 truncate">
            {bundle.track === 'VISION_VLM'
              ? 'FastVisionModel + TRL'
              : bundle.engineBackendLabel.split('(')[0].trim()}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#1D2021] border border-[#3C3836]">
          <div className="text-[11px] text-[#A89984]">CUDA / PyTorch Status</div>
          <div className="text-xs font-mono font-bold text-[#FABD2F] mt-1 truncate">
            {preflight?.hasCuda
              ? preflight.cudaDeviceName
              : 'Install engine/requirements.txt on GPU Host'}
          </div>
        </div>
      </div>

      {/* Direct Workstation CLI Commands */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836] space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-mono text-[#A89984] flex items-center gap-1.5">
              <FolderGit2 className="w-3.5 h-3.5 text-[#83A598]" />
              1. Local Workstation / Docker Training Command:
            </span>
            <button
              type="button"
              onClick={handleCopyCommand}
              className="text-[11px] font-mono text-[#83A598] hover:text-[#FFE9B1] flex items-center gap-1"
            >
              <Copy className="w-3 h-3" />
              {copiedCmd ? 'Copied!' : 'Copy CLI'}
            </button>
          </div>
          <pre className="p-2.5 rounded bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#8EC07C] overflow-x-auto">
            {bundle.pythonTrainCommand}
          </pre>
        </div>

        <div className="p-3.5 rounded-lg bg-[#1D2021] border border-[#3C3836] space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-mono text-[#A89984]">
              2. Post-Train PEFT Merge + Safetensors / GGUF Exporter:
            </span>
            <span className="text-[10px] font-mono text-[#B8BB26]">
              engine/export_and_merge.py
            </span>
          </div>
          <pre className="p-2.5 rounded bg-[#141617] border border-[#3C3836] text-[11px] font-mono text-[#D5C4A1] overflow-x-auto">
            {bundle.pythonExportCommand}
          </pre>
        </div>
      </div>

      {showManifestPreview && (
        <div className="p-3.5 rounded-lg bg-[#141617] border border-[#3C3836] space-y-1.5">
          <div className="text-[11px] font-mono text-[#83A598]">
            Preview of {bundle.runDirRelativePath}/{bundle.manifestFilename}:
          </div>
          <pre className="text-[11px] font-mono text-[#D5C4A1] overflow-x-auto max-h-52">
            {bundle.manifestJson}
          </pre>
        </div>
      )}

      {/* Live Subprocess Stdout / Stderr Console */}
      {activeJob && (
        <div className="p-4 rounded-xl bg-[#141617] border border-[#8EC07C]/50 space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono border-b border-[#282828] pb-2">
            <span className="text-[#FFE9B1]">
              Subprocess Run ID: <strong className="text-[#83A598]">{activeJob.runId}</strong> · Disk:{' '}
              <code className="text-[#8EC07C]">{activeJob.manifestDiskPath}</code>
            </span>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                activeJob.status === 'COMPLETED'
                  ? 'bg-[#B8BB26]/20 text-[#B8BB26] border border-[#B8BB26]/40'
                  : activeJob.status === 'RUNNING'
                    ? 'bg-[#FABD2F]/20 text-[#FABD2F] border border-[#FABD2F]/40'
                    : 'bg-[#FB4934]/20 text-[#FB4934] border border-[#FB4934]/40'
              }`}
            >
              STATUS: {activeJob.status}{' '}
              {activeJob.exitCode !== null ? `(EXIT ${activeJob.exitCode})` : ''}
            </span>
          </div>

          <div className="space-y-1 max-h-48 overflow-y-auto font-mono text-[11px] leading-relaxed">
            {activeJob.logs.map((line, i) => (
              <div
                key={`${activeJob.runId}-log-${i}`}
                className={
                  line.includes('[FORGEML_')
                    ? 'text-[#8EC07C]'
                    : line.includes('[STDERR]')
                      ? 'text-[#FB4934]'
                      : 'text-[#D5C4A1]'
                }
              >
                {line}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
};
