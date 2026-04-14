import { useMemo } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts'

/**
 * Price chart with optional fair value overlay and confidence bands.
 *
 * Props:
 *   bars - array of { date, close, fair_value?, upper?, lower? }
 *   height - chart height in px (default 400)
 */
export default function PriceChart({ bars = [], height = 400 }) {
  const data = useMemo(() => {
    if (!bars.length) return []
    return bars.map((b) => ({
      date: b.date,
      close: b.close,
      fairValue: b.fair_value || null,
      upper: b.upper || null,
      lower: b.lower || null,
    }))
  }, [bars])

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-64 text-terminal-muted font-mono text-sm">
        No price data available
      </div>
    )
  }

  const hasFV = data.some((d) => d.fairValue != null)
  const hasBands = data.some((d) => d.upper != null)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={{ stroke: '#1e293b' }}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          domain={['auto', 'auto']}
          tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={false}
          width={60}
          tickFormatter={(v) => `$${v.toFixed(0)}`}
        />
        <Tooltip
          contentStyle={{
            background: '#111827',
            border: '1px solid #1e293b',
            borderRadius: '8px',
            fontFamily: 'JetBrains Mono',
            fontSize: '12px',
          }}
          labelStyle={{ color: '#94a3b8' }}
          formatter={(value, name) => {
            const labels = { close: 'Price', fairValue: 'Fair Value', upper: 'Upper', lower: 'Lower' }
            return [`$${value?.toFixed(2)}`, labels[name] || name]
          }}
        />

        {/* Confidence bands */}
        {hasBands && (
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.06}
            connectNulls
          />
        )}
        {hasBands && (
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#0a0e17"
            fillOpacity={1}
            connectNulls
          />
        )}

        {/* Fair value line */}
        {hasFV && (
          <Line
            type="monotone"
            dataKey="fairValue"
            stroke="#f59e0b"
            strokeWidth={1.5}
            strokeDasharray="6 3"
            dot={false}
            connectNulls
          />
        )}

        {/* Price line */}
        <Line
          type="monotone"
          dataKey="close"
          stroke="#06b6d4"
          strokeWidth={2}
          dot={false}
          connectNulls
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
