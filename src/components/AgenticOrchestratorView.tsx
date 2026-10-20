import React, { useState } from 'react';
import {
  Bot,
  CheckCircle2,
  Cpu,
  FileText,
  Play,
  Sliders,
  Sparkles,
  Wand2,
} from 'lucide-react';
import { HyperparameterConfig } from '../lib/pipelineLogic';
import { RegisteredModel } from '../data/initialPipelineData';

export type AgentProviderId = 'gemini' | 'openrouter' | 'llama-server' | 'ollama';

export interface AgentExecutionRecord {
  id: string;
  timestamp: string;
  agentName: string;
  providerUsed: string;
  summary: string;
  actionsTaken: string[];
  recommendedDatasets?: string[];
  suggestedHyperparameters?: Partial<HyperparameterConfig>;
  markdownReport?: string;
}

interface AgenticOrchestratorViewProps {
  models: RegisteredModel[];
  selectedModelId: string;
  activeProvider: AgentProviderId;
  onSelectProvider: (p: AgentProviderId) => void;
  providerModelName: string;
  onChangeProviderModelName: (name: string) => void;
  localEndpoint: string;
  onChangeLocalEndpoint: (url: string) => void;
  onExecuteAgentTask: (
    role: 'clean_and_tag' | 'goal_to_pipeline' | 'refine_hpo' | 'generate_report',
    goal?: string
  ) => Promise<void>;
  agentHistory: AgentExecutionRecord[];
  isAgentRunning: boolean;
}

export const AgenticOrchestratorView: React.FC<AgenticOrchestratorViewProps> = ({
  models,
  selectedModelId,
  activeProvider,
  onSelectProvider,
  providerModelName,
  onChangeProviderModelName,
  localEndpoint,
  onChangeLocalEndpoint,
  onExecuteAgentTask,
  agentHistory,
  isAgentRunning,
}) => {
  const [trainingGoal, setTrainingGoal] = useState(
    'Train Qwen3.5-14B for high-precision TypeScript/Python code synthesis and multi-step mathematical verification with zero format drift.'
  );

  const selectedModel = models.find((m) => m.id === selectedModelId) ?? models[0];

  const defaultModelsByProvider: Record<AgentProviderId, string[]> = {
    gemini: ['gemini-3.8-flash', 'gemini-3.1-flash-lite', 'gemini-4-argon'],
    openrouter: [
      'qwen/qwen3.5-72b-instruct',
      'liquid/lfm-2.5-40b',
      'google/gemma-4-27b-it',
      'anthropic/claude-3.7-sonnet',
    ],
    'llama-server': [
      'qwen3.5-14b-instruct-q4_k_m.gguf',
      'lfm2.5-8b-instruct-q8_0.gguf',
      'gemma-4-27b-it-q4_k_m.gguf',
    ],
    ollama: ['qwen3.5:14b', 'lfm2.5:8b', 'gemma4:27b', 'llama3.3:70b'],
  };

  const handleProviderSwitch = (provider: AgentProviderId) => {
    onSelectProvider(provider);
    onChangeProviderModelName(defaultModelsByProvider[provider][0]);
    if (provider === 'llama-server') {
      onChangeLocalEndpoint('http://localhost:8080/v1');
    } else if (provider === 'ollama') {
      onChangeLocalEndpoint('http://localhost:11434');
    }
  };

  const handleGoalSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!trainingGoal.trim()) return;
    await onExecuteAgentTask('goal_to_pipeline', trainingGoal);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <section className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-6 border-b border-slate-800/80">
        <div>
          <p className="text-xs text-slate-400">
            Autonomous Multi-Agent Pipeline · Supports Google Gemini, OpenRouter, Local llama-server & Ollama
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-100 mt-1">
            Agentic AI Orchestration Layer
          </h1>
        </div>
      </section>

      {/* Multi-Provider Engine Configuration Bar */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              01. Agentic Inference Provider & Model Backend
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Select cloud (Google AI / Gemini, OpenRouter) or local inference daemon (llama.cpp llama-server, Ollama)
            </p>
          </div>

          {/* Segmented Provider Selector */}
          <div className="flex flex-wrap items-center gap-1 p-1 bg-[#0B0F17] border border-slate-800 rounded-lg">
            {(
              [
                { id: 'gemini', label: 'Google AI / Gemini' },
                { id: 'openrouter', label: 'OpenRouter' },
                { id: 'llama-server', label: 'Local llama-server' },
                { id: 'ollama', label: 'Local Ollama' },
              ] as Array<{ id: AgentProviderId; label: string }>
            ).map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => handleProviderSwitch(p.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap shrink-0 ${
                  activeProvider === p.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 mt-4 items-end">
          <div className="md:col-span-5">
            <label htmlFor="agent-model-select" className="block text-xs text-slate-400 mb-1">
              Orchestrator Model Identifier ({activeProvider})
            </label>
            <select
              id="agent-model-select"
              value={providerModelName}
              onChange={(e) => onChangeProviderModelName(e.target.value)}
              className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100"
            >
              {defaultModelsByProvider[activeProvider].map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {(activeProvider === 'llama-server' || activeProvider === 'ollama') && (
            <div className="md:col-span-4">
              <label htmlFor="local-daemon-url" className="block text-xs text-slate-400 mb-1">
                Local Daemon Endpoint URL
              </label>
              <input
                id="local-daemon-url"
                type="text"
                value={localEndpoint}
                onChange={(e) => onChangeLocalEndpoint(e.target.value)}
                className="w-full bg-[#0B0F17] border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100"
              />
            </div>
          )}

          <div className="md:col-span-3 text-xs text-slate-400 font-mono">
            Target Pipeline Model: {selectedModel.name}
          </div>
        </div>
      </section>

      {/* Agent 2 Spotlight: Autonomous Goal-to-Dataset & Preprocessing Pipeline */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="pb-4 border-b border-slate-800/80">
          <h2 className="text-base font-semibold text-slate-100">
            02. Autonomous Goal-to-Dataset & Preprocessing Agent
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Describe your training objective in natural language. The agent searches HuggingFace Hub for matching datasets, ingests and merges them, normalizes schemas, runs PII hygiene, and commits a new dataset snapshot automatically.
          </p>
        </div>

        <form onSubmit={handleGoalSubmit} className="mt-4 flex flex-col sm:flex-row gap-3">
          <label htmlFor="user-training-goal" className="sr-only">
            Training Goal Description
          </label>
          <input
            id="user-training-goal"
            type="text"
            value={trainingGoal}
            onChange={(e) => setTrainingGoal(e.target.value)}
            placeholder="Describe your training goal (e.g. Fine-tune LFM2.5-8B for 64k long-context legal contract extraction)..."
            className="flex-1 bg-[#0B0F17] border border-slate-800 rounded-lg px-3.5 py-2.5 text-xs text-slate-100 placeholder:text-slate-500"
          />
          <button
            type="submit"
            disabled={isAgentRunning}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors whitespace-nowrap shrink-0 disabled:opacity-50"
          >
            <Wand2 className="w-3.5 h-3.5" />
            {isAgentRunning ? 'Agent Executing...' : 'Search HF & Auto-Run Preprocessing Pipeline'}
          </button>
        </form>
      </section>

      {/* Specialized Agent Layers Grid (All 4 Requested Agent Roles) */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Agent 1: Data Cleaning & Tagging */}
        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="text-xs text-slate-400 font-mono">Agent Layer 01 · Data Hygiene</div>
            <h3 className="text-sm font-semibold text-slate-100 mt-1">
              Automated Data Cleaning & Semantic Tagging Agent
            </h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Scans ingested HuggingFace records, strips malformed turns, redacts PII/secrets, deduplicates via MinHash LSH, and labels domain & reasoning complexity.
            </p>
          </div>
          <button
            type="button"
            disabled={isAgentRunning}
            onClick={() => onExecuteAgentTask('clean_and_tag')}
            className="mt-4 w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            Execute Data Cleaning & Tagging
          </button>
        </div>

        {/* Agent 3: Iterative HPO & Performance Refiner */}
        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="text-xs text-slate-400 font-mono">Agent Layer 03 · HPO Optimizer</div>
            <h3 className="text-sm font-semibold text-slate-100 mt-1">
              Iterative Performance & Hyperparameter Refiner
            </h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Evaluates training vs. validation loss divergence, gradient norms, and benchmark scores to iteratively tune LoRA rank, learning rate, and NEFTune noise.
            </p>
          </div>
          <button
            type="button"
            disabled={isAgentRunning}
            onClick={() => onExecuteAgentTask('refine_hpo')}
            className="mt-4 w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <Sliders className="w-3.5 h-3.5 text-blue-400" />
            Evaluate & Refine Hyperparameters
          </button>
        </div>

        {/* Agent 4: Automated Summary Reports & Actionable Insights */}
        <div className="bg-[#111827] border border-slate-800/90 rounded-lg p-5 flex flex-col justify-between">
          <div>
            <div className="text-xs text-slate-400 font-mono">Agent Layer 04 · Executive Insights</div>
            <h3 className="text-sm font-semibold text-slate-100 mt-1">
              Automated Summary Report & Insights Generator
            </h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Synthesizes Base vs. Fine-Tuned benchmark comparisons, training stability metrics, and concrete ablation recommendations into an executive report.
            </p>
          </div>
          <button
            type="button"
            disabled={isAgentRunning}
            onClick={() => onExecuteAgentTask('generate_report')}
            className="mt-4 w-full inline-flex items-center justify-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-100 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors disabled:opacity-50"
          >
            <FileText className="w-3.5 h-3.5 text-amber-400" />
            Generate Summary & Actionable Report
          </button>
        </div>
      </section>

      {/* Agent Execution Trace & Generated Reports */}
      <section className="bg-[#111827] border border-slate-800/90 rounded-lg p-5">
        <div className="pb-4 border-b border-slate-800/80 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100">
              03. Autonomous Agent Execution Ledger & Actionable Insights
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Real-time audit log of dataset orchestrations, hyperparameter refinements, and benchmark reports
            </p>
          </div>
          <Bot className="w-4 h-4 text-blue-400" />
        </div>

        <div className="mt-4 divide-y divide-slate-800/80 space-y-4">
          {agentHistory.map((item) => (
            <div key={item.id} className="pt-4 first:pt-0 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="font-semibold text-slate-100">
                  {item.agentName} · Provider: {item.providerUsed}
                </span>
                <span className="font-mono tabular-nums text-slate-400">{item.timestamp}</span>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">{item.summary}</p>

              <ul className="space-y-1 text-xs text-slate-400 font-mono">
                {item.actionsTaken.map((act, idx) => (
                  <li key={idx}>• {act}</li>
                ))}
              </ul>

              {item.markdownReport && (
                <div className="mt-3 bg-[#0B0F17] border border-slate-800 rounded-lg p-3.5 text-xs text-slate-200 font-mono whitespace-pre-wrap leading-relaxed">
                  {item.markdownReport}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
