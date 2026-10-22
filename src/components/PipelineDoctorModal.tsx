import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Wrench,
  X,
  XCircle,
} from 'lucide-react';
import {
  PipelineDoctorAuditReport,
  PipelineDoctorCheckItem,
} from '../lib/pipelineLogic';

interface PipelineDoctorModalProps {
  isOpen: boolean;
  onClose: () => void;
  report: PipelineDoctorAuditReport;
  onApplyAutoFix: (
    fixKey: NonNullable<PipelineDoctorCheckItem['autoFixKey']>
  ) => void;
  onApplyAllFixes: () => void;
}

export const PipelineDoctorModal: React.FC<PipelineDoctorModalProps> = ({
  isOpen,
  onClose,
  report,
  onApplyAutoFix,
  onApplyAllFixes,
}) => {
  if (!isOpen) return null;

  const hasFixableIssues = report.checks.some((c) => c.status !== 'PASS' && c.autoFixKey);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/75 backdrop-blur-xs flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="pipeline-doctor-title"
    >
      <div className="bg-[#252423] border-2 border-[#8EC07C]/70 rounded-2xl max-w-4xl w-full max-h-[88vh] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="p-4 sm:p-5 border-b border-[#3C3836] flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#1D2021]">
          <div className="flex items-start sm:items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#8EC07C]/15 border border-[#8EC07C]/40 flex items-center justify-center text-[#8EC07C] shrink-0">
              <Stethoscope className="w-5 h-5" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 id="pipeline-doctor-title" className="text-sm sm:text-base font-bold text-[#FFE9B1]">
                  Pre-Flight Pipeline Doctor · 12-Invariant Readiness &amp; Bottleneck Audit
                </h2>
                <span
                  className={`px-2 py-0.5 text-[11px] font-mono font-bold rounded ${
                    report.readinessVerdict === 'PRODUCTION_READY'
                      ? 'bg-[#B8BB26]/20 text-[#B8BB26] border border-[#B8BB26]/40'
                      : report.readinessVerdict === 'WARNINGS_PRESENT'
                        ? 'bg-[#FABD2F]/20 text-[#FABD2F] border border-[#FABD2F]/40'
                        : 'bg-[#FB4934]/20 text-[#FB4934] border border-[#FB4934]/40'
                  }`}
                >
                  SCORE: {report.overallScore}/100 ({report.readinessVerdict})
                </span>
              </div>
              <p className="text-xs text-[#A89984] mt-0.5">
                Audits Dataset Hygiene, Tokenizer EOS Alignment, Sub-2B LR Bounds, Multipack Masking, ORPO/GRPO, Vision Tower Freeze &amp; NaViT Patch Density.
              </p>
            </div>
          </div>

          <div className="flex items-center justify-end gap-2 shrink-0">
            {hasFixableIssues && (
              <button
                type="button"
                onClick={onApplyAllFixes}
                className="px-3.5 py-2 text-xs font-semibold text-[#1D2021] bg-[#8EC07C] hover:bg-[#9ed08c] rounded-lg transition-colors flex items-center gap-1.5"
              >
                <Sparkles className="w-3.5 h-3.5" />
                1-Click Auto-Fix All Warnings
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Pipeline Doctor modal"
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-[#A89984] hover:text-[#FFE9B1] bg-[#252423] hover:bg-[#3C3836] border border-[#3C3836] transition-colors flex items-center gap-1"
            >
              <X className="w-3.5 h-3.5" />
              <span>Close Audit</span>
            </button>
          </div>
        </div>

        {/* Summary Strip */}
        <div className="grid grid-cols-3 gap-3 px-5 py-3 bg-[#1D2021]/60 border-b border-[#3C3836] font-mono text-xs">
          <div className="flex items-center gap-2 text-[#B8BB26]">
            <CheckCircle2 className="w-4 h-4" />
            <span>Passed Checks: {report.passedCount} / 12</span>
          </div>
          <div className="flex items-center gap-2 text-[#FABD2F]">
            <AlertTriangle className="w-4 h-4" />
            <span>Warnings: {report.warningCount}</span>
          </div>
          <div className="flex items-center gap-2 text-[#FB4934]">
            <XCircle className="w-4 h-4" />
            <span>Critical Blockers: {report.criticalCount}</span>
          </div>
        </div>

        {/* 12 Check Cards */}
        <div className="p-5 overflow-y-auto space-y-2.5">
          {report.checks.map((chk) => (
            <div
              key={chk.id}
              className={`p-3.5 rounded-xl border flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                chk.status === 'PASS'
                  ? 'bg-[#1D2021] border-[#3C3836]'
                  : chk.status === 'WARN'
                    ? 'bg-[#1D2021] border-[#FABD2F]/50'
                    : 'bg-[#1D2021] border-[#FB4934]/60'
              }`}
            >
              <div className="space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`px-1.5 py-0.5 text-[10px] font-mono font-bold rounded ${
                      chk.status === 'PASS'
                        ? 'bg-[#B8BB26]/20 text-[#B8BB26]'
                        : chk.status === 'WARN'
                          ? 'bg-[#FABD2F]/20 text-[#FABD2F]'
                          : 'bg-[#FB4934]/20 text-[#FB4934]'
                    }`}
                  >
                    {chk.status}
                  </span>
                  <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-[#252423] text-[#83A598]">
                    {chk.category}
                  </span>
                  <span className="text-xs font-semibold text-[#FFE9B1]">{chk.title}</span>
                  <span className="text-[11px] font-mono text-[#8EC07C]">
                    [{chk.metricSummary}]
                  </span>
                </div>
                <p className="text-xs text-[#A89984]">{chk.diagnosis}</p>
              </div>

              {chk.status !== 'PASS' && chk.autoFixKey && chk.autoFixLabel && (
                <button
                  type="button"
                  onClick={() => onApplyAutoFix(chk.autoFixKey!)}
                  className="px-3 py-1.5 text-xs font-semibold text-[#1D2021] bg-[#FABD2F] hover:bg-[#fcc93d] rounded-lg transition-colors shrink-0 inline-flex items-center gap-1.5"
                >
                  <Wrench className="w-3.5 h-3.5" />
                  {chk.autoFixLabel}
                </button>
              )}
              {chk.status === 'PASS' && (
                <span className="text-[11px] font-mono text-[#B8BB26] shrink-0 inline-flex items-center gap-1">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Verified
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
