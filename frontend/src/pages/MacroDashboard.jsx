import { useState } from 'react'
import { useApi } from '../hooks/useApi'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid,
} from 'recharts'

const CATEGORY_MAP = {
  'Interest Rates': ['fed_funds_rate', 'treasury_2y', 'treasury_10y', 'treasury_30y', 'yield_curve_10y2y'],
  'Inflation': ['cpi_yoy', 'core_cpi', 'pce'],
  'Employment': ['unemployment_rate', 'initial_claims', 'nonfarm_payrolls'],
  'Activity': ['real_gdp', 'ism_manufacturing', 'consumer_confidence'],
  'Market': ['vix', 'credit_spread_baa', 'usd_index'],
}

const FRIENDLY_NAMES = {
  fed_funds_rate: 'Fed Funds Rate',
  treasury_2y: '2Y Treasury',
  treasury_10y: '10Y Treasury',
  treasury_30y: '30Y Treasury',
  yield_curve_10y2y: 'Yield Curve (10Y-2Y)',
  cpi_yoy: 'CPI YoY',
  core_cpi: 'Core CPI',
  pce: 'PCE',
  unemployment_rate: 'Unemployment',
  initial_claims: 'Initial Claims',
  nonfarm_payrolls: 'Nonfarm Payrolls',
  real_gdp: 'Real GDP',
  ism_manufacturing: 'ISM Manufacturing',
  consumer_confidence: 'Consumer Confidence',
  vix: 'VIX',
  credit_spread_baa: 'BAA Credit Spread',
  usd_index: 'USD Index',
}

export default function MacroDashboard() {
  const { data: catalog } = useApi('/macro/catalog', { defaultData: [] })
  const [selectedSeries, setSelectedSeries] = useState(null)
  const { data: seriesData, loading: seriesLoading } = useApi(
    selectedSeries ? `/macro/${selectedSeries}` : null,
    { autoFetch: !!selectedSeries }
  )

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-display font-bold text-terminal-text">Macro Dashboard</h1>
        <p className="text-sm font-body text-terminal-muted mt-1">
          Macroeconomic indicators influencing the model's view
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Indicator grid ── */}
        <div className="lg:col-span-1 space-y-4">
          {Object.entries(CATEGORY_MAP).map(([category, keys]) => (
            <div key={category} className="card">
              <div className="card-header">
                <h3 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
                  {category}
                </h3>
              </div>
              <div className="divide-y divide-terminal-border/30">
                {keys.map((key) => {
                  const item = catalog?.find?.((c) => c.key === key)
                  const isActive = selectedSeries === key
                  return (
                    <button
                      key={key}
                      onClick={() => setSelectedSeries(key)}
                      className={`w-full text-left px-4 py-2.5 flex items-center justify-between transition-colors ${
                        isActive
                          ? 'bg-accent-amber/10 border-l-2 border-accent-amber'
                          : 'hover:bg-terminal-elevated/50 border-l-2 border-transparent'
                      }`}
                    >
                      <span className={`text-sm font-mono ${isActive ? 'text-accent-amber' : 'text-terminal-dim'}`}>
                        {FRIENDLY_NAMES[key] || key}
                      </span>
                      {item && (
                        <span className="text-[10px] font-mono text-terminal-muted">
                          {item.series_id}
                        </span>
                      )}
                    </button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>

        {/* ── Chart area ── */}
        <div className="lg:col-span-2">
          <div className="card sticky top-20">
            <div className="card-header">
              <h2 className="text-sm font-mono font-semibold text-terminal-text">
                {selectedSeries ? FRIENDLY_NAMES[selectedSeries] || selectedSeries : 'Select an indicator'}
              </h2>
              {seriesData && (
                <span className="text-xs font-mono text-terminal-muted">
                  {seriesData.count} observations
                </span>
              )}
            </div>
            <div className="card-body">
              {!selectedSeries ? (
                <div className="flex items-center justify-center h-96 text-terminal-muted font-mono text-sm">
                  ← Select a macro indicator to view its history
                </div>
              ) : seriesLoading ? (
                <div className="flex items-center justify-center h-96">
                  <div className="skeleton h-full w-full rounded" />
                </div>
              ) : seriesData?.data?.length ? (
                <MacroChart
                  data={seriesData.data}
                  name={FRIENDLY_NAMES[selectedSeries] || selectedSeries}
                />
              ) : (
                <div className="flex items-center justify-center h-96 text-terminal-muted font-mono text-sm">
                  No data available for this series.
                  <br />
                  Make sure FRED_API_KEY is set and data has been seeded.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function MacroChart({ data, name }) {
  // Show last 5 years max
  const recent = data.slice(-1260)

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={recent} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={{ stroke: '#1e293b' }}
          interval="preserveStartEnd"
          minTickGap={80}
        />
        <YAxis
          tick={{ fill: '#475569', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={false}
          width={50}
        />
        <Tooltip
          contentStyle={{
            background: '#111827',
            border: '1px solid #1e293b',
            borderRadius: '8px',
            fontFamily: 'JetBrains Mono',
            fontSize: '12px',
          }}
          formatter={(value) => [value?.toFixed(3), name]}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#06b6d4"
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
