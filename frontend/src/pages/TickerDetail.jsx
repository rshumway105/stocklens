import { useParams, Link } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import ValuationBadge from '../components/ValuationBadge'
import PriceChart from '../components/PriceChart'
import ShapWaterfall from '../components/ShapWaterfall'
import ForecastFanChart from '../components/ForecastFanChart'
import { formatPrice, formatPercentRaw, formatPercent, valueColor } from '../utils/formatters'

export default function TickerDetail() {
  const { ticker } = useParams()
  const { data: report, loading, error } = useApi(`/reports/${ticker}`)
  const { data: priceData } = useApi(`/prices/${ticker}?years=2`)

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="skeleton h-8 w-48" />
        <div className="skeleton h-[400px] w-full rounded-lg" />
        <div className="grid grid-cols-3 gap-4">
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card p-8 text-center">
        <p className="text-accent-red font-mono mb-2">Failed to load report for {ticker}</p>
        <p className="text-terminal-muted text-sm">{error}</p>
        <Link to="/" className="text-accent-amber text-sm font-mono mt-4 inline-block hover:underline">
          ← Back to Watchlist
        </Link>
      </div>
    )
  }

  if (!report) return null

  const bars = priceData?.bars || []

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div>
          <Link to="/" className="text-terminal-muted text-xs font-mono hover:text-accent-amber transition-colors">
            ← Watchlist
          </Link>
          <div className="flex items-center gap-3 mt-2">
            <h1 className="text-3xl font-display font-bold text-accent-cyan">{report.ticker}</h1>
            <ValuationBadge signal={report.signal} />
            {report.confidence != null && (
              <span className="text-xs font-mono text-terminal-muted border border-terminal-border px-2 py-0.5 rounded">
                Confidence: {report.confidence.toFixed(0)}
              </span>
            )}
          </div>
          <p className="text-terminal-dim font-body mt-1">
            {report.name} {report.sector && `• ${report.sector}`}
          </p>
        </div>
        <div className="text-right">
          <div className="data-label">Current Price</div>
          <div className="text-2xl font-mono font-bold text-terminal-text">
            {formatPrice(report.current_price)}
          </div>
        </div>
      </div>

      {/* ── Key metrics row ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Fair Value"
          value={report.fair_value ? formatPrice(report.fair_value) : '—'}
        />
        <MetricCard
          label="Valuation Gap"
          value={report.valuation_gap_pct != null ? formatPercentRaw(report.valuation_gap_pct) : '—'}
          color={report.valuation_gap_pct != null ? valueColor(report.valuation_gap_pct) : ''}
        />
        <MetricCard
          label="1M Forecast"
          value={
            report.forecasts?.find((f) => f.horizon === '21d')
              ? formatPercent(report.forecasts.find((f) => f.horizon === '21d').predicted_return)
              : '—'
          }
          color={valueColor(report.forecasts?.find((f) => f.horizon === '21d')?.predicted_return)}
        />
        <MetricCard
          label="Signal"
          value={report.signal?.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase()) || 'Unknown'}
        />
      </div>

      {/* ── Price chart ── */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-sm font-mono font-semibold text-terminal-text">Price History</h2>
          <span className="text-xs font-mono text-terminal-muted">
            {bars.length} bars • {bars[0]?.date} → {bars[bars.length - 1]?.date}
          </span>
        </div>
        <div className="card-body">
          <PriceChart bars={bars} height={380} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Return forecasts ── */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-sm font-mono font-semibold text-terminal-text">Forward Return Forecasts</h2>
          </div>
          <div className="card-body">
            <ForecastFanChart forecasts={report.forecasts} />
          </div>
        </div>

        {/* ── Top drivers ── */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-sm font-mono font-semibold text-terminal-text">Top Drivers</h2>
          </div>
          <div className="card-body">
            <ShapWaterfall drivers={report.top_drivers} height={280} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Fundamentals ── */}
        {report.fundamentals?.length > 0 && (
          <div className="card">
            <div className="card-header">
              <h2 className="text-sm font-mono font-semibold text-terminal-text">Fundamentals</h2>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                {report.fundamentals.map((f) => (
                  <div key={f.metric} className="flex justify-between py-1 border-b border-terminal-border/30">
                    <span className="text-xs font-mono text-terminal-muted">
                      {f.metric.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                    </span>
                    <span className="text-xs font-mono text-terminal-text">
                      {f.value != null ? f.value.toFixed(3) : '—'}
                      {f.zscore != null && (
                        <span className={`ml-2 ${valueColor(f.zscore)}`}>
                          z:{f.zscore.toFixed(1)}
                        </span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ── Risk flags ── */}
        {report.risk_flags?.length > 0 && (
          <div className="card">
            <div className="card-header">
              <h2 className="text-sm font-mono font-semibold text-terminal-text">Risk Flags</h2>
            </div>
            <div className="card-body space-y-2">
              {report.risk_flags.map((flag, i) => (
                <div
                  key={i}
                  className={`flex items-start gap-2 p-2 rounded text-xs font-mono ${
                    flag.severity === 'critical' ? 'bg-accent-red/10 border border-accent-red/20' :
                    flag.severity === 'warning' ? 'bg-accent-amber/10 border border-accent-amber/20' :
                    'bg-terminal-elevated border border-terminal-border'
                  }`}
                >
                  <span className={`mt-0.5 ${
                    flag.severity === 'critical' ? 'text-accent-red' :
                    flag.severity === 'warning' ? 'text-accent-amber' :
                    'text-accent-blue'
                  }`}>
                    {flag.severity === 'critical' ? '⚠' : flag.severity === 'warning' ? '△' : 'ℹ'}
                  </span>
                  <span className="text-terminal-dim">{flag.description}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Macro context ── */}
      {report.macro_context?.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="text-sm font-mono font-semibold text-terminal-text">Macro Context</h2>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {report.macro_context.map((m, i) => (
                <div key={i} className="text-center">
                  <div className="data-label mb-1">{m.indicator}</div>
                  <div className="data-value-sm">{m.value?.toFixed(2)}</div>
                  <div className={`text-xs font-mono mt-0.5 ${
                    m.direction === 'rising' ? 'text-accent-green' :
                    m.direction === 'falling' ? 'text-accent-red' :
                    'text-terminal-muted'
                  }`}>
                    {m.direction === 'rising' ? '↑' : m.direction === 'falling' ? '↓' : '→'} {m.direction}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, color = '' }) {
  return (
    <div className="card p-4">
      <div className="data-label mb-1">{label}</div>
      <div className={`data-value ${color}`}>{value}</div>
    </div>
  )
}
