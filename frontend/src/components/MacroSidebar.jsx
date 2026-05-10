import { useState } from 'react'
import { LineChart, Line, ResponsiveContainer } from 'recharts'

// Order in which we prefer to display indicators
const PRIORITY = [
  'Fed Funds Rate',
  '10Y Treasury',
  'Yield Curve (10Y-2Y)',
  'CPI YoY',
  'VIX',
  'Unemployment',
]

// Synthesize 5 data points ending at `value` whose shape reflects `direction`.
// macro_context is snapshot-only (no time series), so these are trend estimates.
function trendPoints(value, direction) {
  if (value == null) return []
  const factors =
    direction === 'rising'  ? [0.93, 0.95, 0.97, 0.99, 1.00]
  : direction === 'falling' ? [1.07, 1.05, 1.03, 1.01, 1.00]
  :                           [0.998, 1.002, 0.999, 1.001, 1.00]
  return factors.map((f, i) => ({ i, v: +(value * f).toFixed(4) }))
}

const STROKE = {
  rising:  '#10b981',  // accent-green
  falling: '#ef4444',  // accent-red
  flat:    '#475569',  // terminal-muted
}
const TEXT_COLOR = {
  rising:  'text-accent-green',
  falling: 'text-accent-red',
  flat:    'text-terminal-muted',
}
const ARROW = { rising: '↑', falling: '↓', flat: '→' }

function Sparkline({ value, direction }) {
  const data = trendPoints(value, direction)
  if (!data.length) return null
  return (
    <ResponsiveContainer width="100%" height={32}>
      <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <Line
          type="monotone"
          dataKey="v"
          stroke={STROKE[direction] ?? '#475569'}
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default function MacroSidebar({ macroContext }) {
  const [open, setOpen] = useState(false)

  if (!macroContext?.length) return null

  // Pick up to 4 indicators in priority order; fall back to first 4 if names differ
  const prioritised = PRIORITY
    .map((name) => macroContext.find((m) => m.indicator === name))
    .filter(Boolean)
    .slice(0, 4)
  const display = prioritised.length ? prioritised : macroContext.slice(0, 4)

  return (
    <div className="card">
      {/* ── Toggle header ── */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="card-header w-full text-left focus:outline-none hover:bg-terminal-elevated/40 transition-colors rounded-t-lg"
        aria-expanded={open}
      >
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-mono font-semibold text-terminal-text">Macro Context</h2>
          <span className="text-[10px] font-mono text-terminal-muted border border-terminal-border px-1.5 py-0.5 rounded">
            {display.length} indicators
          </span>
        </div>
        <span
          className="text-terminal-muted text-xs font-mono transition-transform duration-200 select-none"
          style={{ display: 'inline-block', transform: open ? 'rotate(180deg)' : 'none' }}
        >
          ▾
        </span>
      </button>

      {/* ── Expanded panel ── */}
      {open && (
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {display.map((m, i) => (
              <div key={i} className="bg-terminal-elevated rounded-lg p-3">
                <div className="data-label mb-1 truncate" title={m.indicator}>
                  {m.indicator}
                </div>
                <div className="flex items-baseline gap-1.5 mb-1">
                  <span className="text-sm font-mono font-bold text-terminal-text">
                    {m.value?.toFixed(2)}
                  </span>
                  <span className={`text-xs font-mono ${TEXT_COLOR[m.direction] ?? 'text-terminal-muted'}`}>
                    {ARROW[m.direction] ?? '→'} {m.direction}
                  </span>
                </div>
                <Sparkline value={m.value} direction={m.direction} />
              </div>
            ))}
          </div>
          <p className="text-[10px] font-mono text-terminal-muted mt-3 leading-relaxed">
            Trend lines are directional estimates from the report snapshot — not historical series.
          </p>
        </div>
      )}
    </div>
  )
}
