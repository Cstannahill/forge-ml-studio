import React, { useEffect, useRef, useState } from 'react';
import { HelpCircle } from 'lucide-react';
import { ML_GLOSSARY } from '../lib/glossaryData';

interface InfoTooltipProps {
  termKey: keyof typeof ML_GLOSSARY | string;
  customLabel?: string;
  align?: 'left' | 'center' | 'right';
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({
  termKey,
  customLabel,
  align = 'center',
}) => {
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);
  const entry = ML_GLOSSARY[termKey];

  useEffect(() => {
    if (!pinned) return;
    const handleOutsideClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setPinned(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [pinned]);

  if (!entry) return null;

  const isOpen = hovered || pinned;

  const alignmentClass =
    align === 'left'
      ? 'left-0'
      : align === 'right'
      ? 'right-0'
      : 'left-1/2 -translate-x-1/2';

  return (
    <span
      ref={containerRef}
      className="relative inline-flex items-center align-middle ml-1.5"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        aria-label={`What is ${customLabel || entry.term}?`}
        aria-expanded={isOpen}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setPinned((prev) => !prev);
        }}
        className={`inline-flex items-center justify-center rounded-full p-0.5 transition-colors cursor-help focus:outline-none ${
          isOpen
            ? 'text-[#8EC07C] bg-[#363431]'
            : 'text-[#A89984] hover:text-[#FFE9B1]'
        }`}
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>

      {isOpen && (
        <div
          role="tooltip"
          className={`absolute bottom-full mb-2 ${alignmentClass} z-50 w-72 sm:w-80 p-3 rounded-lg bg-[#22211F] border border-[#83A598]/60 shadow-2xl text-left font-mono text-[11px] leading-relaxed text-[#FFE9B1]`}
        >
          <div className="flex items-center justify-between gap-2 pb-1.5 mb-1.5 border-b border-[#3F3B35]">
            <span className="font-semibold text-[#83A598]">{entry.term}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#363431] text-[#D5C4A1] shrink-0">
              {entry.category}
            </span>
          </div>

          <div className="space-y-1.5">
            <div>
              <span className="text-[#FABD2F] font-semibold">What it is: </span>
              <span className="text-[#D5C4A1]">{entry.whatItIs}</span>
            </div>
            <div>
              <span className="text-[#8EC07C] font-semibold">What it tells us: </span>
              <span className="text-[#FFE9B1]">{entry.whatItTellsUs}</span>
            </div>
            {entry.idealDirection && (
              <div className="pt-1 border-t border-[#3F3B35] text-[10px] text-[#83A598]">
                Target: {entry.idealDirection}
              </div>
            )}
          </div>
        </div>
      )}
    </span>
  );
};
