import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from 'recharts'

/**
 * Sentiment timeline showing sentiment scores over time.
 *
 * Props:
 *   data - array of { date, news_sentiment, social_sentiment, combined }
 *   height - chart height (default 200)
 */
export default function SentimentTimeline({ data = [], height = 200 }) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-32 text-terminal-muted font-mono text-sm">
        No sentiment data available
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
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
          domain={[-1, 1]}
          tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={false}
          width={36}
        />
        <Tooltip
          contentStyle={{
            background: '#111827',
            border: '1px solid #1e293b',
            borderRadius: '8px',
            fontFamily: 'JetBrains Mono',
            fontSize: '11px',
          }}
          formatter={(value, name) => [value?.toFixed(3), name]}
        />
        <ReferenceLine y={0} stroke="#334155" strokeDasharray="3 3" />
        <Area
          type="monotone"
          dataKey="combined"
          name="Combined"
          stroke="#f59e0b"
          fill="#f59e0b"
          fillOpacity={0.1}
          strokeWidth={2}
          dot={false}
        />
        <Area
          type="monotone"
          dataKey="news_sentiment"
          name="News"
          stroke="#06b6d4"
          fill="none"
          strokeWidth={1}
          strokeDasharray="4 2"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
