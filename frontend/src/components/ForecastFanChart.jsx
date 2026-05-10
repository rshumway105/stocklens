import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  Tooltip, Cell, ErrorBar, ReferenceLine,
} from 'recharts'
import { formatPercent } from '../utils/formatters'

const HORIZON_LABELS = {
  '5d': '1W',
  '21d': '1M',
  '63d': '3M',
  '126d': '6M',
}

/**
 * Bar chart showing predicted returns by horizon with symmetric error bars.
 *
 * Props:
 *   forecasts - array of { horizon, predicted_return, lower_bound, upper_bound }
 */
export default function ForecastFanChart({ forecasts = [] }) {
  const data = forecasts.map((f) => {
    const ret = f.predicted_return || 0
    // Both bounds needed to draw whiskers; store as [downside, upside] from center
    const errorDown = f.lower_bound != null ? Math.abs(ret - f.lower_bound) : 0
    const errorUp   = f.upper_bound != null ? Math.abs(f.upper_bound - ret) : 0
    return {
      horizon: HORIZON_LABELS[f.horizon] || f.horizon,
      return: ret,
      // Recharts ErrorBar uses a single value for symmetric bars; we pass the max of both
      // to give a conservative representation of the full interval width
      error: [errorDown, errorUp],
      hasInterval: f.lower_bound != null && f.upper_bound != null,
    }
  })

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-48 text-terminal-muted font-mono text-sm">
        No forecasts available
      </div>
    )
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 20, right: 20, bottom: 4, left: 20 }}>
          <XAxis
            dataKey="horizon"
            tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            axisLine={{ stroke: '#1e293b' }}
          />
          <YAxis
            tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => formatPercent(v)}
          />
          <Tooltip
            contentStyle={{
              background: '#111827',
              border: '1px solid #1e293b',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono',
              fontSize: '12px',
            }}
            formatter={(value, name, entry) => {
              if (name === 'return') {
                const d = entry.payload
                if (d.hasInterval) {
                  const low = formatPercent(d.return - d.error[0])
                  const high = formatPercent(d.return + d.error[1])
                  return [
                    `${formatPercent(value)}  (${low} to ${high})`,
                    'Predicted Return',
                  ]
                }
                return [formatPercent(value), 'Predicted Return']
              }
              return null
            }}
          />
          <ReferenceLine y={0} stroke="#1e293b" strokeWidth={1} />
          <Bar dataKey="return" radius={[4, 4, 0, 0]} maxBarSize={48}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.return >= 0 ? '#10b981' : '#ef4444'}
                fillOpacity={0.75}
              />
            ))}
            <ErrorBar
              dataKey="error"
              direction="y"
              width={8}
              stroke="#94a3b8"
              strokeWidth={1.5}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {data.some((d) => d.hasInterval) && (
        <p className="text-[10px] font-mono text-terminal-muted text-center -mt-1">
          Whiskers show 80% prediction interval
        </p>
      )}
    </div>
  )
}
