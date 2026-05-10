import { useState } from 'react'
import { convictionTier, CONVICTION_TIERS } from '../utils/formatters'

/**
 * Inline badge showing conviction score + tier label.
 * Clicking it toggles the legend popover.
 *
 * compact=true  →  shows score number only (used in dense table rows)
 * compact=false →  shows "72 · High Conviction" (used in detail pages)
 */
export default function ConvictionBadge({ score, compact = false }) {
  const [open, setOpen] = useState(false)
  const tier = convictionTier(score)

  if (!tier || score == null) return <span className="text-terminal-muted font-mono text-xs">—</span>

  return (
    <div className="relative inline-block">
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v) }}
        className={`text-xs font-mono px-2 py-0.5 rounded border ${tier.bg} ${tier.color} hover:opacity-80 transition-opacity cursor-pointer tabular-nums`}
      >
        {compact
          ? `${Math.round(score)}`
          : `${Math.round(score)} · ${tier.label}`}
      </button>

      {open && (
        <>
          {/* backdrop */}
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 mt-1 z-20 w-80 bg-terminal-elevated border border-terminal-border rounded-lg shadow-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] font-mono text-terminal-muted uppercase tracking-wider">
                Conviction Score
              </p>
              <span className={`text-lg font-mono font-bold ${tier.color}`}>{Math.round(score)}</span>
            </div>
            <p className="text-[11px] text-terminal-dim mb-3 leading-relaxed">
              Composite of model agreement, forecast interval width, and valuation gap magnitude.
            </p>
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-terminal-muted text-[10px]">
                  <th className="text-left pb-1">Label</th>
                  <th className="text-center pb-1">Range</th>
                  <th className="text-left pb-1 pl-2">Meaning</th>
                </tr>
              </thead>
              <tbody>
                {CONVICTION_TIERS.map((t) => (
                  <tr
                    key={t.label}
                    className={`border-t border-terminal-border/40 ${t.label === tier.label ? 'opacity-100' : 'opacity-50'}`}
                  >
                    <td className={`py-1.5 pr-2 font-semibold ${t.color}`}>{t.label}</td>
                    <td className="py-1.5 text-center text-terminal-muted whitespace-nowrap">{t.min}–{t.max}</td>
                    <td className="py-1.5 pl-2 text-terminal-dim leading-snug">{t.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
