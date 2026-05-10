import { useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import ValuationBadge from '../components/ValuationBadge'
import ConvictionBadge from '../components/ConvictionBadge'
import PriceChart from '../components/PriceChart'
import ShapWaterfall from '../components/ShapWaterfall'
import ForecastFanChart from '../components/ForecastFanChart'
import MacroSidebar from '../components/MacroSidebar'
import { formatPrice, formatPercentRaw, formatPercent, valueColor } from '../utils/formatters'
import { buildSummary } from '../utils/summaryBuilder'
import LastUpdated from '../components/LastUpdated'
import AccuracyCallout from '../components/AccuracyCallout'

export default function TickerDetail() {
  const { ticker } = useParams()
  const { data: report, loading, error } = useApi(`/reports/${ticker}`)
  const { data: priceData } = useApi(`/prices/${ticker}?years=2`)
  const { data: backtestData } = useApi(`/backtest/${ticker}`, { defaultData: {} })

  const bars = useMemo(() => {
    const raw = priceData?.bars || []
    if (!raw.length) return []
    const WINDOW = 63
    return raw.map((b, i) => {
      let fair_value = null
      if (i >= WINDOW - 1) {
        const slice = raw.slice(i - WINDOW + 1, i + 1)
        fair_value = slice.reduce((sum, x) => sum + x.close, 0) / WINDOW
      }
      return { ...b, fair_value }
    })
  }, [priceData])

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
        <Link to="/watchlist" className="text-accent-amber text-sm font-mono mt-4 inline-block hover:underline">
          ← Back to Watchlist
        </Link>
      </div>
    )
  }

  if (!report) return null

  // Detect demo mode (models not yet trained for this ticker)
  const isDemoMode = report.risk_flags?.some((f) => f.flag === 'demo_mode')
  const summary = buildSummary(report)

  const rfMetrics = backtestData?.return_forecaster?.metrics
  const calloutAccuracy = rfMetrics?.direction_accuracy_21d != null
    ? rfMetrics.direction_accuracy_21d * 100
    : null
  const calloutWithinPct = rfMetrics?.interval_coverage_21d != null
    ? rfMetrics.interval_coverage_21d * 100
    : null

  const SUMMARY_STYLES = {
    undervalued: {
      border: 'border-l-4 border-accent-green',
      bg: 'bg-accent-green/5',
      accent: 'text-accent-green',
    },
    overvalued: {
      border: 'border-l-4 border-accent-red',
      bg: 'bg-accent-red/5',
      accent: 'text-accent-red',
    },
    fairly_valued: {
      border: 'border-l-4 border-terminal-muted',
      bg: 'bg-terminal-elevated',
      accent: 'text-terminal-dim',
    },
  }
  const summaryStyle = SUMMARY_STYLES[report.signal] ?? SUMMARY_STYLES.fairly_valued

  return (
    <div className="space-y-6">
      {/* ── Model summary banner ── */}
      {summary && (
        <div className={`${summaryStyle.border} ${summaryStyle.bg} rounded-r-lg px-4 py-3`}>
          <p className={`text-[11px] font-mono uppercase tracking-wider mb-1 ${summaryStyle.accent}`}>
            Model Summary
          </p>
          <p className="text-sm text-terminal-text leading-relaxed">{summary}</p>
        </div>
      )}

      {/* ── Demo mode banner ── */}
      {isDemoMode && (
        <div className="border border-accent-amber/30 bg-accent-amber/5 rounded-lg px-4 py-3 flex items-start gap-3">
          <span className="text-accent-amber text-sm mt-0.5">△</span>
          <div>
            <p className="text-accent-amber font-mono text-sm font-semibold">Models not trained</p>
            <p className="text-terminal-dim text-xs mt-0.5">
              Forecasts and signals shown below are placeholders. Go to the Watchlist, add this ticker, and wait for training to complete.
            </p>
          </div>
        </div>
      )}

      {/* ── Header ── */}
      <div className="flex items-start justify-between">
        <div>
          <Link to="/watchlist" className="text-terminal-muted text-xs font-mono hover:text-accent-amber transition-colors">
            ← Watchlist
          </Link>
          <div className="flex items-center gap-3 mt-2">
            <h1 className="text-3xl font-mono font-bold text-accent-cyan">{report.ticker}</h1>
            <ValuationBadge signal={report.signal} />
            {report.confidence != null && (
              <ConvictionBadge score={report.confidence} />
            )}
          </div>
          <p className="text-terminal-dim mt-1">
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
          value={!isDemoMode && report.fair_value ? formatPrice(report.fair_value) : '—'}
        />
        <MetricCard
          label="Valuation Gap"
          value={!isDemoMode && report.valuation_gap_pct != null ? formatPercentRaw(report.valuation_gap_pct) : '—'}
          color={!isDemoMode && report.valuation_gap_pct != null ? valueColor(report.valuation_gap_pct) : ''}
        />
        <MetricCard
          label="1M Forecast"
          value={(() => {
            if (isDemoMode) return '—'
            const f = report.forecasts?.find((f) => f.horizon === '21d')
            return f ? formatPercent(f.predicted_return) : '—'
          })()}
          color={isDemoMode ? '' : valueColor(report.forecasts?.find((f) => f.horizon === '21d')?.predicted_return)}
        />
        <MetricCard
          label="6M Forecast"
          value={(() => {
            if (isDemoMode) return '—'
            const f = report.forecasts?.find((f) => f.horizon === '126d')
            return f ? formatPercent(f.predicted_return) : '—'
          })()}
          color={isDemoMode ? '' : valueColor(report.forecasts?.find((f) => f.horizon === '126d')?.predicted_return)}
        />
      </div>

      {/* ── Price chart ── */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-sm font-mono font-semibold text-terminal-text">Price History</h2>
          <div className="flex items-center gap-4 text-xs font-mono text-terminal-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 h-0.5 bg-cyan-400" />
              Price
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-4 border-t-2 border-dashed border-amber-400" />
              63-Day Moving Average
            </span>
            <span>{bars[0]?.date} → {bars[bars.length - 1]?.date}</span>
          </div>
        </div>
        <div className="card-body">
          <PriceChart bars={bars} height={380} />
          <LastUpdated timestamp={priceData?.last_date} className="mt-2 block text-right" />
        </div>
      </div>

      {!isDemoMode && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ── Return forecasts ── */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-sm font-mono font-semibold text-terminal-text">Forward Return Forecasts</h2>
              <span className="text-[11px] font-mono text-terminal-muted">80% prediction interval</span>
            </div>
            <div className="card-body">
              <ForecastFanChart forecasts={report.forecasts} />
              <AccuracyCallout
                accuracy={calloutAccuracy}
                overDays={21}
                withinPct={calloutWithinPct}
              />
            </div>
          </div>

          {/* ── Top drivers ── */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-sm font-mono font-semibold text-terminal-text">Top Drivers</h2>
              <span className="text-[11px] font-mono text-terminal-muted">SHAP feature contributions</span>
            </div>
            <div className="card-body">
              <ShapWaterfall drivers={report.top_drivers} height={280} />
            </div>
          </div>
        </div>
      )}

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
                      {formatFundamentalLabel(f.metric)}
                    </span>
                    <span className="text-xs font-mono text-terminal-text">
                      {formatFundamentalValue(f.metric, f.value)}
                      {f.zscore != null && (
                        <span className={`ml-2 ${valueColor(f.zscore)}`}>
                          z:{f.zscore.toFixed(1)}
                        </span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
              <LastUpdated timestamp={priceData?.last_date} className="mt-3 block" />
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

      {/* ── Macro context sidebar (collapsible) ── */}
      <MacroSidebar macroContext={report.macro_context} />
    </div>
  )
}

const FUNDAMENTAL_LABELS = {
  pe_ratio: 'P/E Ratio', forward_pe: 'Fwd P/E', pb_ratio: 'P/B Ratio',
  ps_ratio: 'P/S Ratio', ev_ebitda: 'EV/EBITDA', peg_ratio: 'PEG',
  gross_margin: 'Gross Margin', operating_margin: 'Operating Margin',
  net_margin: 'Net Margin', roe: 'ROE', roa: 'ROA', roic: 'ROIC',
  revenue_growth: 'Rev Growth', earnings_growth: 'EPS Growth',
  debt_equity: 'Debt/Equity', current_ratio: 'Current Ratio',
  quick_ratio: 'Quick Ratio', fcf_yield: 'FCF Yield',
  dividend_yield: 'Div Yield', beta: 'Beta',
}

const PCT_METRICS = new Set([
  'gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa', 'roic',
  'revenue_growth', 'earnings_growth', 'fcf_yield', 'dividend_yield',
])

function formatFundamentalLabel(metric) {
  return FUNDAMENTAL_LABELS[metric] ||
    metric.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatFundamentalValue(metric, value) {
  if (value == null || isNaN(value)) return '—'
  if (PCT_METRICS.has(metric)) {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${(value * 100).toFixed(1)}%`
  }
  // Ratios: show 1–2 decimals, no leading zeros
  if (Math.abs(value) >= 100) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

function MetricCard({ label, value, color = '' }) {
  return (
    <div className="card p-4">
      <div className="data-label mb-1">{label}</div>
      <div className={`data-value ${color}`}>{value}</div>
    </div>
  )
}
