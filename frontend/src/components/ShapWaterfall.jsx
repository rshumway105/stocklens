import { useMemo } from 'react'
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, Tooltip, Cell, ReferenceLine,
} from 'recharts'

/**
 * Horizontal bar chart showing top feature contributions (SHAP-style).
 *
 * Props:
 *   drivers - array of { feature, shap_value?, importance?, direction, explanation }
 *   height - chart height (default 320)
 */
export default function ShapWaterfall({ drivers = [], height = 320 }) {
  const data = useMemo(() => {
    return drivers.slice(0, 10).map((d) => ({
      feature: formatFeatureName(d.feature),
      value: d.shap_value ?? d.importance ?? 0,
      direction: d.direction,
      explanation: d.explanation,
      raw: d.feature,
    }))
  }, [drivers])

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-48 text-terminal-muted font-mono text-sm">
        No feature explanations available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 120 }}>
        <XAxis
          type="number"
          tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={{ stroke: '#1e293b' }}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: '#94a3b8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={false}
          width={115}
        />
        <Tooltip
          contentStyle={{
            background: '#111827',
            border: '1px solid #1e293b',
            borderRadius: '8px',
            fontFamily: 'JetBrains Mono',
            fontSize: '11px',
            maxWidth: '300px',
          }}
          formatter={(value, name, entry) => {
            return [value?.toFixed(5), entry.payload.explanation || 'Contribution']
          }}
        />
        <ReferenceLine x={0} stroke="#1e293b" />
        <Bar dataKey="value" radius={[0, 3, 3, 0]} maxBarSize={20}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.direction === 'positive' ? '#10b981' : entry.direction === 'negative' ? '#ef4444' : '#475569'}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/Sma/g, 'SMA')
    .replace(/Rsi/g, 'RSI')
    .replace(/Macd/g, 'MACD')
    .replace(/Atr/g, 'ATR')
    .replace(/Obv/g, 'OBV')
    .replace(/Zscore/g, 'Z-Score')
    .replace(/Pct/g, '%')
    .substring(0, 20)
}
