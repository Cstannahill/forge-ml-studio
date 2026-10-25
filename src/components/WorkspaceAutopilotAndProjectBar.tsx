import React, { useState } from 'react';
import {
  Check,
  Copy,
  Download,
  FolderGit2,
  GitCompare,
  Plus,
  RotateCcw,
  Sparkles,
  Upload,
  Wand2,
  X,
} from 'lucide-react';
import {
  DEFAULT_WORKSPACE_PROJECTS,
  END_TO_END_RECIPE_PRESETS,
  EndToEndRecipePreset,
  EndToEndRecipePresetId,
  HyperparameterConfig,
  TargetFormat,
  WorkspaceProjectProfile,
  computeRecipePresetDiffPreview,
  exportForgeMlWorkspaceJson,
  parseAndValidateForgeMlWorkspaceJson,
} from '../lib/pipelineLogic';
import { PipelineModalityTrack, VisionTargetFormat } from '../lib/visionPipelineLogic';

interface WorkspaceAutopilotAndProjectBarProps {
  projects: WorkspaceProjectProfile[];
  activeProjectId: string;
  onSelectProject: (project: WorkspaceProjectProfile) => void;
  onSaveNewProject: (projectName: string, domainTag: string) => void;
  onImportWorkspaceSnapshot: (
    activeProject: WorkspaceProjectProfile,
    allProjects: WorkspaceProjectProfile[]
  ) => void;
  onResetFactoryDefaults: () => void;
  onApplyRecipePreset: (preset: EndToEndRecipePreset) => void;
  currentState: {
    pipelineTrack: PipelineModalityTrack;
    selectedModelId: string;
    selectedVisionModelId: string;
    targetFormat: TargetFormat;
    visionTargetFormat: VisionTargetFormat;
    hpoConfig: HyperparameterConfig;
    samplePackingEnabled: boolean;
    completionOnlyLossEnabled: boolean;
  };
  lastAutoSavedTime: string;
}

export const WorkspaceAutopilotAndProjectBar: React.FC<
  WorkspaceAutopilotAndProjectBarProps
> = ({
  projects,
  activeProjectId,
  onSelectProject,
  onSaveNewProject,
  onImportWorkspaceSnapshot,
  onResetFactoryDefaults,
  onApplyRecipePreset,
  currentState,
  lastAutoSavedTime,
}) => {
  const activeProject =
    projects.find((p) => p.projectId === activeProjectId) ??
    projects[0] ??
    DEFAULT_WORKSPACE_PROJECTS[0];

  const [previewRecipeId, setPreviewRecipeId] = useState<EndToEndRecipePresetId | null>(null);
  const [appliedRecipeBanner, setAppliedRecipeBanner] = useState<string | null>(null);
  const [isJsonDrawerOpen, setIsJsonDrawerOpen] = useState<boolean>(false);
  const [jsonInputText, setJsonInputText] = useState<string>('');
  const [jsonImportStatus, setJsonImportStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const [copiedJson, setCopiedJson] = useState<boolean>(false);

  const [isNewProjectOpen, setIsNewProjectOpen] = useState<boolean>(false);
  const [newProjectName, setNewProjectName] = useState<string>('');
  const [newProjectDomain, setNewProjectDomain] = useState<string>('Custom Domain');

  const previewPreset = previewRecipeId
    ? END_TO_END_RECIPE_PRESETS.find((r) => r.id === previewRecipeId) ?? null
    : null;

  const diffPreview = previewPreset
    ? computeRecipePresetDiffPreview(currentState, previewPreset)
    : null;

  const handleOpenJsonDrawer = () => {
    const exported = exportForgeMlWorkspaceJson(activeProject, projects);
    setJsonInputText(exported);
    setJsonImportStatus(null);
    setIsJsonDrawerOpen((prev) => !prev);
  };

  const handleDownloadForgeMlJson = () => {
    const exported = exportForgeMlWorkspaceJson(activeProject, projects);
    const slug = activeProject.projectName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/(^-|-$)/g, '');
    const blob = new Blob([exported], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slug || 'workspace'}.forgeml.json`;
    a.click();
    URL.revokeObjectURL(url);
    setJsonImportStatus({
      type: 'success',
      message: `Downloaded ${slug || 'workspace'}.forgeml.json (${projects.length} workspace profiles).`,
    });
  };

  const handleImportFromText = (rawText: string) => {
    const validation = parseAndValidateForgeMlWorkspaceJson(rawText);
    if (!validation.valid || !validation.snapshot) {
      setJsonImportStatus({
        type: 'error',
        message: validation.error || 'Invalid .forgeml.json file.',
      });
      return;
    }
    onImportWorkspaceSnapshot(
      validation.snapshot.activeProject,
      validation.snapshot.projects
    );
    setJsonImportStatus({
      type: 'success',
      message: `Restored project "${validation.snapshot.activeProject.projectName}" and ${validation.snapshot.projects.length} workspace profile(s) from .forgeml.json.`,
    });
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const content = String(reader.result ?? '');
      setJsonInputText(content);
      handleImportFromText(content);
    };
    reader.readAsText(file);
  };

  return (
    <section
      aria-label="Workspace Project Switcher and End-to-End Recipe Autopilot"
      className="bg-[#22211F] border border-[#4A453E] rounded-xl p-4 space-y-4"
    >
      {/* Top Row: Persistent Workspace Project Switcher + Auto-Save Status + .forgeml.json Import/Export + Factory Reset */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-3 pb-3.5 border-b border-[#3C3836]">
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#FFE9B1]">
            <FolderGit2 className="w-4 h-4 text-[#8EC07C]" />
            <span>Workspace Project Profile:</span>
          </div>

          <select
            aria-label="Select Active Workspace Project Profile"
            value={activeProject.projectId}
            onChange={(e) => {
              const found = projects.find((p) => p.projectId === e.target.value);
              if (found) {
                onSelectProject(found);
                setAppliedRecipeBanner(
                  `Switched workspace to "${found.projectName}" (${found.pipelineTrack} · ${
                    found.pipelineTrack === 'VISION_VLM'
                      ? found.selectedVisionModelId
                      : found.selectedModelId
                  }).`
                );
              }
            }}
            className="bg-[#1D2021] border border-[#504945] rounded-lg px-3 py-1.5 text-xs font-mono text-[#FFE9B1] focus:outline-none focus:border-[#8EC07C]"
          >
            {projects.map((proj) => (
              <option key={proj.projectId} value={proj.projectId}>
                {proj.projectName} [{proj.domainTag}]
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => setIsNewProjectOpen((prev) => !prev)}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-mono text-[#D5C4A1] bg-[#1D2021] hover:bg-[#32302F] border border-[#504945] rounded-lg transition-colors"
          >
            <Plus className="w-3.5 h-3.5 text-[#8EC07C]" />
            <span>New Profile</span>
          </button>

          <span
            title="Workspace state is automatically persisted to browser localStorage"
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#1D2021] border border-[#8EC07C]/40 text-[11px] font-mono text-[#8EC07C]"
          >
            <span className="w-2 h-2 rounded-full bg-[#8EC07C] animate-pulse" />
            <span>Auto-Saved · {lastAutoSavedTime}</span>
          </span>
        </div>

        {/* Right Controls: .forgeml.json Snapshot Export/Import & Factory Reset */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleDownloadForgeMlJson}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium text-[#FFE9B1] bg-[#1D2021] hover:bg-[#32302F] border border-[#83A598]/50 rounded-lg transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-[#83A598]" />
            <span>Export .forgeml.json</span>
          </button>

          <button
            type="button"
            onClick={handleOpenJsonDrawer}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium text-[#8EC07C] bg-[#1D2021] hover:bg-[#32302F] border border-[#8EC07C]/50 rounded-lg transition-colors"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>{isJsonDrawerOpen ? 'Close .forgeml.json Studio' : 'Import / Inspect .forgeml.json'}</span>
          </button>

          <button
            type="button"
            onClick={() => {
              onResetFactoryDefaults();
              setPreviewRecipeId(null);
              setAppliedRecipeBanner(
                'Reset workspace to Factory Defaults (Qwen3.5-2B-Instruct · Track A · r=32 · WSD).'
              );
            }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-medium text-[#FB4934] bg-[#1D2021] hover:bg-[#FB4934]/15 border border-[#FB4934]/40 rounded-lg transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset to Factory Defaults</span>
          </button>
        </div>
      </div>

      {/* Optional New Project Creator Row */}
      {isNewProjectOpen && (
        <div className="bg-[#1D2021] border border-[#504945] rounded-lg p-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2 flex-1">
            <input
              type="text"
              aria-label="New Workspace Project Name"
              placeholder="e.g. Project D: Medical Clinical Note Summarizer"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              className="bg-[#282828] border border-[#504945] rounded px-3 py-1.5 text-xs text-[#FFE9B1] w-72"
            />
            <input
              type="text"
              aria-label="New Workspace Project Domain Tag"
              placeholder="Domain tag (e.g. Clinical NLP)"
              value={newProjectDomain}
              onChange={(e) => setNewProjectDomain(e.target.value)}
              className="bg-[#282828] border border-[#504945] rounded px-3 py-1.5 text-xs font-mono text-[#D5C4A1] w-48"
            />
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                if (!newProjectName.trim()) return;
                onSaveNewProject(newProjectName.trim(), newProjectDomain.trim() || 'Custom');
                setAppliedRecipeBanner(`Saved new workspace profile "${newProjectName.trim()}".`);
                setNewProjectName('');
                setIsNewProjectOpen(false);
              }}
              className="px-3 py-1.5 text-xs font-medium bg-[#8EC07C] text-[#1D2021] font-semibold rounded-lg hover:bg-[#A0D08E] transition-colors"
            >
              Save Profile
            </button>
            <button
              type="button"
              onClick={() => setIsNewProjectOpen(false)}
              className="px-2.5 py-1.5 text-xs text-[#A89984] hover:text-[#FFE9B1]"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Optional .forgeml.json Import / Export Inspector Drawer */}
      {isJsonDrawerOpen && (
        <div className="bg-[#1D2021] border border-[#83A598]/50 rounded-xl p-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-[#3C3836]">
            <div>
              <h3 className="text-xs font-semibold text-[#FFE9B1] font-mono">
                Full Workspace Snapshot (.forgeml.json · Schema v1.0)
              </h3>
              <p className="text-[11px] text-[#A89984]">
                Copy, edit, or upload a .forgeml.json project file to migrate entire multi-track configurations across machines.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono text-[#1D2021] bg-[#8EC07C] hover:bg-[#A0D08E] font-semibold rounded-lg cursor-pointer transition-colors">
                <Upload className="w-3.5 h-3.5" />
                <span>Upload .forgeml.json File</span>
                <input
                  type="file"
                  accept=".json,.forgeml.json"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard?.writeText(jsonInputText);
                  setCopiedJson(true);
                  setTimeout(() => setCopiedJson(false), 1800);
                }}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-mono text-[#FFE9B1] bg-[#282828] border border-[#504945] rounded-lg"
              >
                {copiedJson ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-[#B8BB26]" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy JSON</span>
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={() => handleImportFromText(jsonInputText)}
                className="px-3 py-1.5 text-xs font-mono font-semibold text-[#FFE9B1] bg-[#458588] hover:bg-[#529497] rounded-lg transition-colors"
              >
                Validate &amp; Load JSON
              </button>
            </div>
          </div>

          {jsonImportStatus && (
            <div
              role="status"
              className={`px-3 py-2 rounded-lg text-xs font-mono border ${
                jsonImportStatus.type === 'success'
                  ? 'bg-[#B8BB26]/10 border-[#B8BB26]/40 text-[#B8BB26]'
                  : 'bg-[#FB4934]/15 border-[#FB4934]/50 text-[#FB4934]'
              }`}
            >
              {jsonImportStatus.message}
            </div>
          )}

          <textarea
            aria-label="Workspace .forgeml.json Payload Editor"
            rows={7}
            value={jsonInputText}
            onChange={(e) => setJsonInputText(e.target.value)}
            className="w-full bg-[#141617] border border-[#3C3836] rounded-lg p-3 font-mono text-[11px] text-[#D5C4A1] focus:outline-none focus:border-[#83A598]"
          />
        </div>
      )}

      {/* Row 2: One-Click End-to-End Recipe Presets (Goal-Driven Autopilot) */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-[#FABD2F]" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#FFE9B1]">
              One-Click End-to-End Recipe Presets (Goal-Driven Autopilot)
            </h2>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#1D2021] border border-[#FABD2F]/40 text-[#FABD2F]">
              6 Battle-Tested Profiles · Diff Preview Enabled
            </span>
          </div>
          <span className="text-[11px] text-[#A89984]">
            Click any recipe card to inspect a side-by-side Diff Preview or apply all 15+ pipeline knobs in 1 click
          </span>
        </div>

        {appliedRecipeBanner && (
          <div
            role="status"
            className="bg-[#1D2021] border border-[#8EC07C]/50 rounded-lg px-3.5 py-2 text-xs font-mono text-[#8EC07C] flex items-center justify-between gap-2"
          >
            <span>{appliedRecipeBanner}</span>
            <button
              type="button"
              onClick={() => setAppliedRecipeBanner(null)}
              className="text-[#A89984] hover:text-[#FFE9B1] text-[11px]"
            >
              Dismiss
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-2.5">
          {END_TO_END_RECIPE_PRESETS.map((recipe) => {
            const isPreviewing = previewRecipeId === recipe.id;
            const isVision = recipe.pipelineTrack === 'VISION_VLM';
            return (
              <div
                key={recipe.id}
                className={`rounded-xl p-3 border transition-all flex flex-col justify-between gap-2.5 ${
                  isPreviewing
                    ? 'bg-[#1D2021] border-[#FABD2F] shadow-sm'
                    : 'bg-[#1D2021]/80 border-[#3C3836] hover:border-[#665C54]'
                }`}
              >
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between gap-1">
                    <span
                      className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-semibold ${
                        isVision
                          ? 'bg-[#8EC07C]/15 text-[#8EC07C] border border-[#8EC07C]/30'
                          : 'bg-[#83A598]/15 text-[#83A598] border border-[#83A598]/30'
                      }`}
                    >
                      {recipe.pipelineTrack}
                    </span>
                    <span className="text-[10px] font-mono text-[#A89984]">
                      {recipe.estimatedVramGb}GB · {recipe.estimatedRuntimeMinutes}m
                    </span>
                  </div>

                  <div className="text-xs font-semibold text-[#FFE9B1] leading-snug">
                    {recipe.title}
                  </div>

                  <div className="text-[10px] font-mono text-[#FABD2F] truncate">
                    {recipe.shortBadge} · r={recipe.hpoConfig.loraRank}
                  </div>

                  <p className="text-[11px] text-[#A89984] line-clamp-2 leading-relaxed">
                    {recipe.goalDescription}
                  </p>
                </div>

                <div className="flex items-center gap-1.5 pt-2 border-t border-[#32302F]">
                  <button
                    type="button"
                    onClick={() =>
                      setPreviewRecipeId((prev) => (prev === recipe.id ? null : recipe.id))
                    }
                    className={`flex-1 px-2 py-1 text-[11px] font-mono rounded border transition-colors ${
                      isPreviewing
                        ? 'bg-[#FABD2F]/20 border-[#FABD2F] text-[#FABD2F] font-semibold'
                        : 'bg-[#282828] border-[#504945] text-[#D5C4A1] hover:text-[#FFE9B1]'
                    }`}
                  >
                    {isPreviewing ? 'Hide Diff' : 'Diff Preview'}
                  </button>
                  <button
                    type="button"
                    aria-label={`Apply Recipe ${recipe.title}`}
                    onClick={() => {
                      onApplyRecipePreset(recipe);
                      setPreviewRecipeId(null);
                      setAppliedRecipeBanner(
                        `Applied Autopilot Recipe "${recipe.title}" -> Track: ${recipe.pipelineTrack}, Model: ${recipe.targetModelId}, DoRA r=${recipe.hpoConfig.loraRank}, Stage 2: ${recipe.stage2AlignmentMethod}, Export: ${recipe.quantizationExportTarget}.`
                      );
                    }}
                    className="px-2.5 py-1 text-[11px] font-mono font-semibold rounded bg-[#458588] hover:bg-[#529497] text-[#FFE9B1] transition-colors shrink-0"
                  >
                    Apply
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Interactive Diff Preview Before Applying Panel */}
        {diffPreview && previewPreset && (
          <div
            role="region"
            aria-label="Recipe Autopilot Configuration Diff Preview"
            className="bg-[#1D2021] border border-[#FABD2F]/60 rounded-xl p-4 space-y-3"
          >
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-3 border-b border-[#3C3836]">
              <div>
                <div className="flex items-center gap-2">
                  <GitCompare className="w-4 h-4 text-[#FABD2F]" />
                  <h3 className="text-xs font-semibold text-[#FFE9B1]">
                    Recipe Autopilot Diff Preview: {previewPreset.title}
                  </h3>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#FABD2F]/15 text-[#FABD2F]">
                    {diffPreview.changedCount} of {diffPreview.diffs.length} settings will update
                  </span>
                </div>
                <p className="text-xs text-[#A89984] mt-0.5">
                  {previewPreset.goalDescription}
                </p>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => {
                    onApplyRecipePreset(previewPreset);
                    setAppliedRecipeBanner(
                      `Applied Autopilot Recipe "${previewPreset.title}" (${diffPreview.changedCount} configuration changes committed).`
                    );
                    setPreviewRecipeId(null);
                  }}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-mono font-semibold text-[#1D2021] bg-[#FABD2F] hover:bg-[#FCCA46] rounded-lg transition-colors"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Confirm &amp; Apply Recipe Autopilot</span>
                </button>
                <button
                  type="button"
                  aria-label="Close Recipe Diff Preview"
                  onClick={() => setPreviewRecipeId(null)}
                  className="p-1.5 text-[#A89984] hover:text-[#FFE9B1] rounded-lg"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 font-mono text-xs">
              {diffPreview.diffs.map((item) => (
                <div
                  key={item.field}
                  className={`p-2.5 rounded-lg border ${
                    item.changed
                      ? 'bg-[#282828] border-[#FABD2F]/40'
                      : 'bg-[#181A1B] border-[#32302F] opacity-75'
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] text-[#A89984]">
                    <span>{item.field}</span>
                    <span
                      className={
                        item.changed ? 'text-[#FABD2F] font-semibold' : 'text-[#8EC07C]'
                      }
                    >
                      {item.changed ? 'CHANGED' : 'UNCHANGED'}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] text-[#FB4934]/90 line-through truncate">
                    - {item.currentValue}
                  </div>
                  <div className="text-[11px] text-[#B8BB26] font-semibold truncate">
                    + {item.nextValue}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
