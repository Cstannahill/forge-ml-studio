import React, { useState } from 'react';
import {
  Check,
  Cloud,
  Container,
  Copy,
  Download,
  Server,
  Terminal,
} from 'lucide-react';
import {
  ArchitectureFamily,
  LocalInferenceRuntime,
  LoggingFramework,
  QuantizationMode,
  estimateVramRequirements,
  generateContainerArtifacts,
  generateLocalInferenceAndSdkBundle,
} from '../lib/pipelineLogic';
import {
  CLOUD_PROVIDER_GUIDES,
  RegisteredModel,
} from '../data/initialPipelineData';
import { InfoTooltip } from './InfoTooltip';

interface CloudAndContainerViewProps {
  models: RegisteredModel[];
  selectedModelId: string;
  activeLogger: LoggingFramework;
}

export const CloudAndContainerView: React.FC<CloudAndContainerViewProps> = ({
  models,
  selectedModelId,
  activeLogger,
}) => {
  const selectedModel = models.find((m) => m.id === selectedModelId) ?? models[0];

  const [quantization, setQuantization] = useState<QuantizationMode>('qlora-4bit');
  const [contextLength, setContextLength] = useState<number>(8192);
  const [batchSize, setBatchSize] = useState<number>(4);
  const [gradCheckpointing, setGradCheckpointing] = useState<boolean>(true);
  const [cloudStorageUri, setCloudStorageUri] = useState<string>(
    's3://forgeml-datasets/snapshots/v2.4.0'
  );
  const [selectedProviderId, setSelectedProviderId] = useState<string>('runpod');
  const [activeArtifactTab, setActiveArtifactTab] = useState<
    'dockerfile' | 'compose' | 'launch'
  >('dockerfile');
  const [copiedKey, setCopiedKey] = useState<string>('');

  // Local Inference Endpoint Launcher & Client SDK Generator State
  const [localRuntime, setLocalRuntime] = useState<LocalInferenceRuntime>('llama_cpp');
  const [localPort, setLocalPort] = useState<number>(8080);
  const [localTemperature, setLocalTemperature] = useState<number>(0.2);
  const [localRepeatPenalty, setLocalRepeatPenalty] = useState<number>(1.08);
  const [sdkLangTab, setSdkLangTab] = useState<'python' | 'typescript' | 'curl'>('python');

  const localInferenceBundle = generateLocalInferenceAndSdkBundle({
    modelName: selectedModel.name,
    checkpointTag:
      selectedModel.fineTunedCheckpointId ??
      `ft-${selectedModel.name.toLowerCase().replace(/[^a-z0-9.-]+/g, '-')}`,
    exportFormat: 'GGUF-Q4_K_M',
    contextWindow: contextLength,
    port: localPort,
    temperature: localTemperature,
    repeatPenalty: localRepeatPenalty,
    runtime: localRuntime,
  });

  const vramEstimate = estimateVramRequirements({
    paramsBillions: selectedModel.paramsBillions,
    quantization,
    contextLength,
    batchSize,
    gradientCheckpointing: gradCheckpointing,
    architecture: selectedModel.architecture,
  });

  const containerArtifacts = generateContainerArtifacts({
    modelId: selectedModel.id,
    architecture: selectedModel.architecture,
    quantization,
    loggingFramework: activeLogger,
    cloudStorageUri,
  });

  const selectedProvider =
    CLOUD_PROVIDER_GUIDES.find((p) => p.id === selectedProviderId) ??
    CLOUD_PROVIDER_GUIDES[0];

  const activeCodeContent =
    activeArtifactTab === 'dockerfile'
      ? containerArtifacts.dockerfile
      : activeArtifactTab === 'compose'
      ? containerArtifacts.dockerCompose
      : containerArtifacts.cloudLaunchScript;

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard?.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(''), 1800);
  };

  const handleDownloadArtifact = () => {
    const filename =
      activeArtifactTab === 'dockerfile'
        ? 'Dockerfile'
        : activeArtifactTab === 'compose'
        ? 'docker-compose.yml'
        : 'cloud-launch.sh';
    const blob = new Blob([activeCodeContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-6 border-b border-slate-800/80">
        <div>
          <p className="text-xs text-slate-400">
            Local-to-Cloud Compute Scaling · Docker & CUDA 12.6 Containerization · Provider Migration Guides
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100 mt-1">
            Cloud Compute Guides & Containerization Hub
          </h1>
        </div>
      </section>

      {/* Section 1: Interactive VRAM Sizing & Local-vs-Cloud Estimator + Container Generator */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 5 cols: VRAM & Local/Cloud Resource Sizing Calculator */}
        <div className="lg:col-span-5 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="pb-4 border-b border-slate-800/80">
              <h2 className="text-base font-semibold text-slate-100">
                01. Local vs. Cloud VRAM Sizing Calculator
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Target: {selectedModel.name} ({selectedModel.paramsBillions}B · {selectedModel.architecture})
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              <div>
                <label htmlFor="calc-quant" className="flex items-center text-xs text-slate-400 mb-1">
                  <span>Training Precision / Adapter</span>
                  <InfoTooltip termKey="qlora4bit" align="left" />
                </label>
                <select
                  id="calc-quant"
                  value={quantization}
                  onChange={(e) => setQuantization(e.target.value as QuantizationMode)}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
                >
                  <option value="qlora-4bit">QLoRA 4-Bit NF4</option>
                  <option value="lora-8bit">LoRA 8-Bit Int8</option>
                  <option value="lora-16bit">LoRA 16-Bit BF16</option>
                  <option value="full-bf16">Full Parameter BF16</option>
                </select>
              </div>

              <div>
                <label htmlFor="calc-ctx" className="block text-xs text-slate-400 mb-1">
                  Sequence Context Length
                </label>
                <select
                  id="calc-ctx"
                  value={contextLength}
                  onChange={(e) => setContextLength(Number(e.target.value))}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
                >
                  <option value={2048}>2,048 tokens</option>
                  <option value={4096}>4,096 tokens</option>
                  <option value={8192}>8,192 tokens</option>
                  <option value={16384}>16,384 tokens</option>
                  <option value={32768}>32,768 tokens</option>
                  <option value={65536}>65,536 tokens</option>
                </select>
              </div>

              <div>
                <label htmlFor="calc-batch" className="block text-xs text-slate-400 mb-1">
                  Per-GPU Micro-Batch
                </label>
                <select
                  id="calc-batch"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono tabular-nums text-slate-100"
                >
                  {[1, 2, 4, 8, 16].map((b) => (
                    <option key={b} value={b}>
                      Batch = {b}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-end pb-1">
                <label className="inline-flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={gradCheckpointing}
                    onChange={(e) => setGradCheckpointing(e.target.checked)}
                    className="rounded border-slate-700 bg-[#0B0F17] text-blue-600"
                  />
                  <span>Gradient Checkpointing</span>
                </label>
                <InfoTooltip termKey="gradCheckpointing" />
              </div>
            </div>

            {/* VRAM Breakdown */}
            <div className="mt-5 pt-4 border-t border-slate-800/80 space-y-2.5 text-xs font-mono tabular-nums">
              <div className="flex justify-between">
                <span className="text-slate-400">Model Weights Footprint:</span>
                <span className="text-slate-200">{vramEstimate.weightsGb} GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Optimizer States & Gradients:</span>
                <span className="text-slate-200">{vramEstimate.optimizerAndGradsGb} GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">
                  Activations & KV ({selectedModel.architecture} Kernel):
                </span>
                <span className="text-slate-200">{vramEstimate.kvCacheAndActivationsGb} GB</span>
              </div>
              <div className="flex justify-between pt-2 border-t border-slate-800 text-sm font-semibold">
                <span className="text-slate-100">Total Required VRAM:</span>
                <span
                  className={
                    vramEstimate.canRunLocally24Gb ? 'text-emerald-400' : 'text-amber-400'
                  }
                >
                  {vramEstimate.totalRequiredGb} GB
                </span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 text-xs">
            <div className="text-slate-400">
              Hardware Verdict:{' '}
              <span className="text-slate-100 font-medium">{vramEstimate.recommendedGpu}</span>
            </div>
            <div className="text-slate-400 mt-1">
              {vramEstimate.canRunLocally24Gb
                ? 'Fits on local 24 GB GPU (RTX 3090 / 4090) without offloading.'
                : vramEstimate.canRunLocally48Gb
                ? 'Exceeds 24 GB local card; fits on 48 GB workstation or single cloud A100/H100.'
                : 'Exceeds local workstation VRAM — migrate container to Cloud H100/A100 cluster below.'}
            </div>
          </div>
        </div>

        {/* Right 7 cols: Containerization Generator (Dockerfile / docker-compose.yml / launch.sh) */}
        <div className="lg:col-span-7 bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  02. Portable Containerization & Migration Artifacts
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Zero-friction migration between local workstation and cloud GPU instances
                </p>
              </div>

              <div className="flex items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
                {(
                  [
                    { id: 'dockerfile', label: 'Dockerfile' },
                    { id: 'compose', label: 'docker-compose.yml' },
                    { id: 'launch', label: 'cloud-launch.sh' },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveArtifactTab(tab.id)}
                    className={`px-2.5 py-1 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                      activeArtifactTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="mt-4 bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5 font-mono text-xs overflow-x-auto max-h-72">
              <pre className="text-slate-200 whitespace-pre leading-relaxed text-[11px]">
                {activeCodeContent}
              </pre>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span>Cloud Sync Target:</span>
              <input
                type="text"
                aria-label="Container Cloud Storage Sync URI"
                value={cloudStorageUri}
                onChange={(e) => setCloudStorageUri(e.target.value)}
                className="bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1 text-xs font-mono text-slate-200 w-64"
              />
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => handleCopy(activeCodeContent, activeArtifactTab)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-200 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
              >
                {copiedKey === activeArtifactTab ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    Copy File
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={handleDownloadArtifact}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Download File
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: Recommended Cloud Compute Providers & Detailed Step-by-Step Setup Guides */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              03. Recommended Cloud Compute Providers & Step-by-Step Deployment Guides
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Select a cloud GPU provider when local compute is insufficient for large-parameter or multi-node runs
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
            {CLOUD_PROVIDER_GUIDES.map((prov) => (
              <button
                key={prov.id}
                type="button"
                onClick={() => setSelectedProviderId(prov.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  selectedProviderId === prov.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {prov.name}
              </button>
            ))}
          </div>
        </div>

        {/* Selected Provider Overview */}
        <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 pb-5 border-b border-slate-800/80 text-xs">
          <div>
            <div className="text-slate-400">Provider & Architecture Fit</div>
            <div className="text-sm font-semibold text-slate-100 mt-0.5">
              {selectedProvider.name}
            </div>
            <div className="text-slate-400 mt-1">{selectedProvider.tagline}</div>
          </div>
          <div>
            <div className="text-slate-400">Recommended GPU Spec</div>
            <div className="font-mono text-slate-100 font-medium mt-0.5">
              {selectedProvider.recommendedInstance}
            </div>
            <div className="font-mono tabular-nums text-emerald-400 mt-1">
              {selectedProvider.vramTotalGb} GB High-Bandwidth VRAM
            </div>
          </div>
          <div>
            <div className="text-slate-400">Estimated Compute Rate</div>
            <div className="font-mono tabular-nums text-lg font-semibold text-slate-100 mt-0.5">
              ${selectedProvider.hourlyRateUsd.toFixed(2)} / hr
            </div>
            <div className="text-slate-400 mt-0.5">On-demand container billing</div>
          </div>
          <div>
            <div className="text-slate-400">Optimal Workload</div>
            <div className="text-slate-200 mt-0.5">{selectedProvider.bestFor}</div>
          </div>
        </div>

        {/* Step-by-Step CLI Setup Guide */}
        <div className="mt-5 space-y-4">
          {selectedProvider.setupSteps.map((step, index) => (
            <div
              key={step.stepTitle}
              className="bg-[#0B0F17] border border-slate-800/80 rounded-lg p-4"
            >
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-xs font-semibold text-slate-100">{step.stepTitle}</h3>
                <button
                  type="button"
                  onClick={() => handleCopy(step.command, `step-${index}`)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  {copiedKey === `step-${index}` ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      Copy CLI Command
                    </>
                  )}
                </button>
              </div>
              <p className="text-xs text-slate-400 mt-1">{step.description}</p>
              <pre className="mt-3 p-3 bg-[#111827] border border-slate-800/90 rounded font-mono text-[11px] text-blue-300 overflow-x-auto whitespace-pre">
                {step.command}
              </pre>
            </div>
          ))}
        </div>
      </section>

      {/* Section 04: Local Inference Endpoint Launcher & Client SDK Snippet Generator */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              04. Local Inference Endpoint Launcher &amp; OpenAI-Compatible Client SDK Generator
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Serve your exported Safetensors / GGUF-Q4_K_M checkpoint locally via Ollama, llama.cpp, vLLM, or LM Studio and call it from Python, TypeScript, or cURL
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
            {(
              [
                { id: 'llama_cpp', label: 'llama.cpp Server' },
                { id: 'ollama', label: 'Ollama' },
                { id: 'vllm', label: 'vLLM' },
                { id: 'lm_studio', label: 'LM Studio CLI' },
              ] as Array<{ id: LocalInferenceRuntime; label: string }>
            ).map((rt) => (
              <button
                key={rt.id}
                type="button"
                onClick={() => setLocalRuntime(rt.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  localRuntime === rt.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {rt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Sampling & Endpoint Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label htmlFor="local-port-input" className="block text-xs text-slate-400 mb-1">
              Local HTTP Server Port
            </label>
            <input
              id="local-port-input"
              type="number"
              value={localPort}
              onChange={(e) => setLocalPort(Number(e.target.value) || 8080)}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
            />
          </div>
          <div>
            <label htmlFor="local-temp-select" className="block text-xs text-slate-400 mb-1">
              Sampling Temperature
            </label>
            <select
              id="local-temp-select"
              value={localTemperature}
              onChange={(e) => setLocalTemperature(Number(e.target.value))}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
            >
              <option value={0.0}>0.0 (Deterministic JSON / Code)</option>
              <option value={0.2}>0.2 (Recommended Sub-2B CoT)</option>
              <option value={0.6}>0.6 (Balanced Creative / Chat)</option>
            </select>
          </div>
          <div>
            <label htmlFor="local-rep-select" className="block text-xs text-slate-400 mb-1">
              Repeat Penalty (Sub-2B Loop Guard)
            </label>
            <select
              id="local-rep-select"
              value={localRepeatPenalty}
              onChange={(e) => setLocalRepeatPenalty(Number(e.target.value))}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded px-2.5 py-1.5 text-xs font-mono text-slate-100"
            >
              <option value={1.05}>1.05 (Gentle)</option>
              <option value={1.08}>1.08 (Recommended Sub-2B)</option>
              <option value={1.15}>1.15 (Strict Anti-Loop)</option>
            </select>
          </div>
        </div>

        {/* Two-Column Server Launch vs Client SDK Snippets */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left 5 cols: Server Launch Command */}
          <div className="lg:col-span-5 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-semibold text-slate-200">
                  1. Start Local Inference Server
                </span>
                <button
                  type="button"
                  onClick={() =>
                    handleCopy(localInferenceBundle.serverLaunchCommand, 'local-srv')
                  }
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  {copiedKey === 'local-srv' ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      Copy Server Command
                    </>
                  )}
                </button>
              </div>
              <pre className="font-mono text-[11px] text-emerald-300 overflow-x-auto whitespace-pre leading-relaxed pt-1">
                {localInferenceBundle.serverLaunchCommand}
              </pre>
            </div>
            <div className="pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-400">
              Endpoint: http://localhost:{localRuntime === 'ollama' ? 11434 : localPort}/v1
            </div>
          </div>

          {/* Right 7 cols: Client SDK Snippet (Python / TypeScript / cURL) */}
          <div className="lg:col-span-7 bg-[#0B0F17] border border-slate-800 rounded-lg p-4 flex flex-col justify-between space-y-3">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800">
                <div className="flex items-center gap-1">
                  {(
                    [
                      { id: 'python', label: 'Python (openai SDK)' },
                      { id: 'typescript', label: 'TypeScript (fetch)' },
                      { id: 'curl', label: 'cURL CLI' },
                    ] as const
                  ).map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setSdkLangTab(tab.id)}
                      className={`px-2.5 py-1 text-[11px] font-mono rounded transition-colors ${
                        sdkLangTab === tab.id
                          ? 'bg-blue-600 text-white font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => {
                    const snippet =
                      sdkLangTab === 'python'
                        ? localInferenceBundle.pythonSdkSnippet
                        : sdkLangTab === 'typescript'
                        ? localInferenceBundle.typescriptSdkSnippet
                        : localInferenceBundle.curlSnippet;
                    handleCopy(snippet, 'local-sdk');
                  }}
                  className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono text-slate-200 bg-slate-800 hover:bg-slate-700 rounded transition-colors"
                >
                  {copiedKey === 'local-sdk' ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      Copied Snippet
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      Copy Client Snippet
                    </>
                  )}
                </button>
              </div>

              <pre className="font-mono text-[11px] text-slate-200 overflow-x-auto whitespace-pre leading-relaxed pt-1">
                {sdkLangTab === 'python'
                  ? localInferenceBundle.pythonSdkSnippet
                  : sdkLangTab === 'typescript'
                  ? localInferenceBundle.typescriptSdkSnippet
                  : localInferenceBundle.curlSnippet}
              </pre>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

