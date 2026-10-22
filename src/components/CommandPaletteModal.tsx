import React, { useEffect, useState } from 'react';
import { Command, CornerDownLeft, Search, X } from 'lucide-react';
import {
  CommandPaletteActionItem,
  filterCommandPaletteActions,
} from '../lib/pipelineLogic';

interface CommandPaletteModalProps {
  isOpen: boolean;
  onClose: () => void;
  actions: CommandPaletteActionItem[];
  onExecuteAction: (actionId: string) => void;
}

export const CommandPaletteModal: React.FC<CommandPaletteModalProps> = ({
  isOpen,
  onClose,
  actions,
  onExecuteAction,
}) => {
  const [query, setQuery] = useState<string>('');
  const [selectedIndex, setSelectedIndex] = useState<number>(0);

  const filtered = filterCommandPaletteActions(actions, query);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query, isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) =>
          filtered.length > 0 ? (prev + 1) % filtered.length : 0
        );
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) =>
          filtered.length > 0
            ? (prev - 1 + filtered.length) % filtered.length
            : 0
        );
      } else if (e.key === 'Enter' && filtered[selectedIndex]) {
        e.preventDefault();
        onExecuteAction(filtered[selectedIndex].id);
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filtered, selectedIndex, onClose, onExecuteAction]);

  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Global Command Palette"
      className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl bg-[#111827] border border-slate-700 rounded-lg shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-slate-800 bg-[#0B0F17]">
          <Search className="w-4 h-4 text-slate-400 shrink-0" />
          <input
            type="text"
            autoFocus
            aria-label="Search Command Palette Actions"
            placeholder="Type a command or search studios (e.g., GRPO, imatrix, Curriculum, DoRA, Red-Team, LFM2.5)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none"
          />
          <button
            type="button"
            aria-label="Close Command Palette"
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-200 rounded"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="max-h-80 overflow-y-auto divide-y divide-slate-800/70 p-2">
          {filtered.length === 0 ? (
            <div className="py-8 text-center text-xs font-mono text-slate-400">
              No matching pipeline commands for &ldquo;{query}&rdquo;.
            </div>
          ) : (
            filtered.map((item, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    onExecuteAction(item.id);
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between gap-3 px-3 py-2.5 rounded text-left transition-colors ${
                    isSelected
                      ? 'bg-blue-600/20 text-slate-100'
                      : 'text-slate-300 hover:bg-slate-800/50'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <Command className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                    <span className="text-xs font-medium truncate">
                      {item.title}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 font-mono text-[10px]">
                    <span className="text-slate-400">{item.category}</span>
                    {item.shortcutHint && (
                      <span className="px-1.5 py-0.5 rounded bg-[#0B0F17] border border-slate-800 text-slate-300">
                        {item.shortcutHint}
                      </span>
                    )}
                    <CornerDownLeft className="w-3 h-3 text-slate-500" />
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="px-4 py-2 border-t border-slate-800 bg-[#0B0F17] flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Use ↑↓ arrows to navigate · Enter to execute · Esc to close</span>
          <span>Shortcut: Ctrl+K / Cmd+K</span>
        </div>
      </div>
    </div>
  );
};
