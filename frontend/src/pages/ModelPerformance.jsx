import { useMemo } from 'react'
import { useApi } from '../hooks/useApi'
import { CONVICTION_TIERS } from '../utils/formatters'

export default function ModelPerformance() {
  const { data: health } = useApi('/health')
  const { data: backtests } = useApi('/backtest', { defaultData: {} })

  // Compute averaged metrics across all tickers
  const { tickers, avgRf, avgFv, totalFolds, perTickerRows } = useMemo(() => {
    const tickers = Object.keys(backtests || {})
    if (tickers.length === 0) return { tickers: [], avgRf: {}, avgFv: {}, totalFolds: null, perTickerRows: [] }

    const rfKeys = [
      'direction_accuracy_5d', 'direction_accuracy_21d', 'direction_accuracy_63d', 'direction_accuracy_126d',
      'ic_5d', 'ic_21d', 'ic_63d', 'ic_126d',
      'rmse_5d', 'rmse_21d', 'rmse_63d', 'rmse_126d',
      'interval_coverage_21d',
    ]
    const fvKeys = ['mae', 'rmse', 'r2', 'mean_valuation_gap']

    const rfSums = {}
    const fvSums = {}
    const rfCounts = {}
    const fvCounts = {}
    rfKeys.forEach(k => { rfSums[k] = 0; rfCounts[k] = 0 })
    fvKeys.forEach(k => { fvSums[k] = 0; fvCounts[k] = 0 })

    let totalFolds = 0

    const perTickerRows = tickers.map(ticker => {
      const bt = backtests[ticker]
      const rf = bt?.return_forecaster?.metrics || {}
      const fv = bt?.fair_value_estimator?.metrics || {}
      const folds = bt?.return_forecaster?.n_folds ?? 0
      totalFolds += folds

      rfKeys.forEach(k => {
        if (rf[k] != null && !isNaN(rf[k])) { rfSums[k] += rf[k]; rfCounts[k]++ }
      })
      fvKeys.forEach(k => {
        if (fv[k] != null && !isNaN(fv[k])) { fvSums[k] += fv[k]; fvCounts[k]++ }
      })

      return { ticker, rf, fv, folds }
    })

    const avgRf = {}
    rfKeys.forEach(k => { avgRf[k] = rfCounts[k] > 0 ? rfSums[k] / rfCounts[k] : null })
    const avgFv = {}
    fvKeys.forEach(k => { avgFv[k] = fvCounts[k] > 0 ? fvSums[k] / fvCounts[k] : null })

    return { tickers, avgRf, avgFv, totalFolds: totalFolds || null, perTickerRows }
  }, [backtests])

  const fmt = (v, decimals = 3) => (v != null && !isNaN(v) ? Number(v).toFixed(decimals) : '—')
  const fmtPct = (v) => (v != null && !isNaN(v) ? (v * 100).toFixed(1) + '%' : '—')

  const lastUpdateEST = health?.timestamp
    ? new Date(health.timestamp).toLocaleString('en-US', {
        timeZone: 'America/New_York',
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit', hour12: true,
      }) + ' EST'
    : '—'

  const noData = tickers.length === 0

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-mono font-bold text-terminal-text">Model Performance</h1>
        <p className="text-sm text-terminal-muted mt-1">
          Walk-forward backtest results and model accuracy metrics
        </p>
      </div>

      {/* ── System status ── */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
            System Status
          </h2>
          <span className={`text-xs font-mono px-2 py-0.5 rounded ${
            health?.status === 'ok'
              ? 'bg-accent-green/15 text-accent-green'
              : 'bg-accent-red/15 text-accent-red'
          }`}>
            {health?.status === 'ok' ? '● Online' : '● Offline'}
          </span>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <StatusItem label="API Version" value={health?.version || '—'} />
            <StatusItem label="Tickers Trained" value={tickers.length > 0 ? `${tickers.length} (${tickers.join(', ')})` : 'None'} />
            <StatusItem label="Total Folds" value={totalFolds != null ? `${totalFolds} folds` : '—'} />
            <StatusItem label="Last Update" value={lastUpdateEST} />
          </div>
        </div>
      </div>

      {/* ── Averaged metrics ── */}
      {!noData && (
        <div className="card mb-6">
          <div className="card-header">
            <h2 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
              Averaged Backtest Metrics
            </h2>
            <span className="text-xs font-mono text-terminal-muted">
              across {tickers.length} ticker{tickers.length !== 1 ? 's' : ''}
            </span>
          </div>
          <div className="card-body space-y-5">
            {/* Return Forecaster averaged */}
            <div>
              <div className="data-label mb-3">Return Forecaster — Direction Accuracy</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['5d', '1-Week'], ['21d', '1-Month'], ['63d', '3-Month'], ['126d', '6-Month']].map(([h, label]) => (
                  <MetricTile
                    key={h}
                    label={label}
                    value={fmtPct(avgRf[`direction_accuracy_${h}`])}
                    sub={`IC: ${fmt(avgRf[`ic_${h}`])}`}
                    highlight={avgRf[`direction_accuracy_${h}`] != null && avgRf[`direction_accuracy_${h}`] > 0.52}
                  />
                ))}
              </div>
            </div>

            <div>
              <div className="data-label mb-3">Return Forecaster — RMSE (log return)</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {[['5d', '1-Week'], ['21d', '1-Month'], ['63d', '3-Month'], ['126d', '6-Month']].map(([h, label]) => (
                  <MetricTile
                    key={h}
                    label={label}
                    value={fmt(avgRf[`rmse_${h}`])}
                    sub={h === '21d' ? `Coverage: ${fmtPct(avgRf.interval_coverage_21d)}` : ''}
                  />
                ))}
              </div>
            </div>

            <div>
              <div className="data-label mb-3">Fair Value Estimator</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <MetricTile label="MAE (price)" value={avgFv.mae != null ? `$${fmt(avgFv.mae, 2)}` : '—'} sub="Mean abs error" />
                <MetricTile label="RMSE (price)" value={avgFv.rmse != null ? `$${fmt(avgFv.rmse, 2)}` : '—'} sub="Root mean sq error" />
                <MetricTile label="R²" value={fmt(avgFv.r2)} sub="Explained variance" />
                <MetricTile label="Mean Val. Gap" value={fmtPct(avgFv.mean_valuation_gap)} sub="Avg price vs fair value" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Per-ticker breakdown ── */}
      {!noData && (
        <div className="card mb-6">
          <div className="card-header">
            <h2 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
              Per-Ticker Breakdown
            </h2>
          </div>
          <div className="card-body overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="bg-terminal-elevated/60">
                  <th className="text-left px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">Ticker</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">Folds</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">Dir Acc 5d</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">Dir Acc 21d</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">Dir Acc 63d</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">IC 21d</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">FV MAE</th>
                  <th className="text-right px-3 py-2 text-terminal-muted font-semibold uppercase tracking-wider text-[10px]">FV R²</th>
                </tr>
              </thead>
              <tbody>
                {perTickerRows.map((row, i) => (
                  <tr key={row.ticker} className={`border-t border-terminal-border/40 ${i % 2 === 0 ? '' : 'bg-terminal-elevated/20'}`}>
                    <td className="px-3 py-2 text-accent-cyan font-semibold">{row.ticker}</td>
                    <td className="px-3 py-2 text-right text-terminal-dim">{row.folds || '—'}</td>
                    <td className={`px-3 py-2 text-right ${accuracyColor(row.rf.direction_accuracy_5d)}`}>
                      {fmtPct(row.rf.direction_accuracy_5d)}
                    </td>
                    <td className={`px-3 py-2 text-right ${accuracyColor(row.rf.direction_accuracy_21d)}`}>
                      {fmtPct(row.rf.direction_accuracy_21d)}
                    </td>
                    <td className={`px-3 py-2 text-right ${accuracyColor(row.rf.direction_accuracy_63d)}`}>
                      {fmtPct(row.rf.direction_accuracy_63d)}
                    </td>
                    <td className={`px-3 py-2 text-right ${icColor(row.rf.ic_21d)}`}>
                      {fmt(row.rf.ic_21d)}
                    </td>
                    <td className="px-3 py-2 text-right text-terminal-dim">
                      {row.fv.mae != null ? `$${fmt(row.fv.mae, 2)}` : '—'}
                    </td>
                    <td className={`px-3 py-2 text-right ${icColor(row.fv.r2)}`}>
                      {fmt(row.fv.r2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {noData && (
        <div className="card mb-6 p-6 text-center">
          <p className="text-terminal-muted font-mono text-sm">
            No backtest results yet. Add tickers to your watchlist and train models to see metrics.
          </p>
        </div>
      )}

      {/* ── Model architecture ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ModelCard
          title="Return Forecaster"
          subtitle="Model 1 — Forward Return Prediction"
          description="XGBoost + LightGBM ensemble predicting forward log returns at 4 horizons (1-week, 1-month, 3-month, 6-month). Predictions are a 50/50 blend of both models, reducing variance through algorithm diversity. Trained with walk-forward expanding window validation."
          features={[
            'Algorithm: XGBoost + LightGBM ensemble (50/50 blend)',
            'Horizons: 5d, 21d, 63d, 126d',
            'Features: Technical + Fundamental + Macro + Sentiment',
            'Validation: Walk-forward, 3-year minimum train window',
            'Output: Point estimate + 80% prediction interval',
          ]}
        />

        <ModelCard
          title="Fair Value Estimator"
          subtitle="Model 2 — Intrinsic Value Estimation"
          description="XGBoost regression predicting smoothed price (63-day MA) from fundamentals and macro features only. Excludes technicals and current price to capture intrinsic value."
          features={[
            'Algorithm: XGBoost regression',
            'Target: 63-day smoothed price',
            'Features: Fundamentals + Macro only (no technicals)',
            'Signal: Overvalued if gap > +15%, Undervalued if < -15%',
            'Sector-relative z-scores for all fundamental inputs',
          ]}
        />
      </div>

      {/* ── Ensemble logic ── */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
            Ensemble Logic
          </h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h3 className="text-sm font-mono font-semibold text-terminal-text mb-2">Signal Blending</h3>
              <p className="text-xs text-terminal-dim leading-relaxed">
                When both models agree (e.g., negative predicted returns AND overvalued fair value),
                the signal is stronger. Disagreement reduces confidence.
              </p>
            </div>
            <div>
              <h3 className="text-sm font-mono font-semibold text-terminal-text mb-2">Confidence Score</h3>
              <p className="text-xs text-terminal-dim leading-relaxed">
                0–100 composite based on: model agreement (35%), prediction interval width (25%),
                signal magnitude (25%), and historical accuracy (15%).
              </p>
            </div>
            <div>
              <h3 className="text-sm font-mono font-semibold text-terminal-text mb-2">Explainability</h3>
              <p className="text-xs text-terminal-dim leading-relaxed">
                Every prediction includes SHAP explanations showing the top 10 features
                driving the model's view, with plain-English descriptions.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Conviction score — compact reference ── */}
      <div className="card mb-6">
        <div className="card-header">
          <h2 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
            Conviction Score Reference
          </h2>
          <span className="text-xs font-mono text-terminal-muted">
            Weighted composite: 35% agreement · 25% precision · 25% gap · 15% accuracy
          </span>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {CONVICTION_TIERS.map((t) => (
              <div key={t.label} className={`rounded-md p-3 border ${t.bg}`}>
                <div className={`text-sm font-mono font-bold ${t.color}`}>{t.min}–{t.max}</div>
                <div className={`text-xs font-mono font-semibold mt-0.5 ${t.color}`}>{t.label}</div>
                <div className="text-[11px] text-terminal-muted mt-1 leading-snug">{t.description}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Walk-forward methodology ── */}
      <div className="card">
        <div className="card-header">
          <h2 className="text-xs font-mono font-semibold text-accent-amber uppercase tracking-wider">
            Walk-Forward Methodology
          </h2>
        </div>
        <div className="card-body">
          <div className="flex gap-3 overflow-x-auto pb-2">
            {[
              { step: '1', title: 'Train', desc: 'Expanding window from T₀ to T', color: 'accent-cyan' },
              { step: '2', title: 'Purge', desc: '5-day gap prevents leakage', color: 'accent-amber' },
              { step: '3', title: 'Test', desc: 'Predict T+5 to T+68 (3 months)', color: 'accent-green' },
              { step: '4', title: 'Record', desc: 'Store out-of-sample predictions', color: 'terminal-dim' },
              { step: '5', title: 'Slide', desc: 'Advance window by 126 days', color: 'terminal-dim' },
              { step: '6', title: 'Repeat', desc: 'Until all data is covered', color: 'terminal-dim' },
            ].map((s) => (
              <div key={s.step} className="flex-shrink-0 w-40">
                <div className={`w-8 h-8 rounded-full bg-${s.color}/20 border border-${s.color}/40 flex items-center justify-center text-xs font-mono font-bold text-${s.color} mb-2`}>
                  {s.step}
                </div>
                <div className="text-sm font-mono font-semibold text-terminal-text">{s.title}</div>
                <div className="text-[11px] text-terminal-muted mt-0.5">{s.desc}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 bg-terminal-elevated rounded-md">
            <p className="text-xs font-mono text-terminal-muted">
              <span className="text-accent-amber">No lookahead bias:</span> Every prediction is made using only data available
              at prediction time. The purge gap ensures forward return windows don't overlap with the training set.
              Random seeds are fixed for reproducibility.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function accuracyColor(v) {
  if (v == null) return 'text-terminal-muted'
  if (v > 0.55) return 'text-accent-green'
  if (v > 0.50) return 'text-accent-amber'
  return 'text-accent-red'
}

function icColor(v) {
  if (v == null) return 'text-terminal-muted'
  if (v > 0.05) return 'text-accent-green'
  if (v > 0) return 'text-accent-amber'
  return 'text-accent-red'
}

function StatusItem({ label, value }) {
  return (
    <div>
      <div className="data-label mb-1">{label}</div>
      <div className="data-value-sm">{value}</div>
    </div>
  )
}

function MetricTile({ label, value, sub, highlight }) {
  return (
    <div className="bg-terminal-elevated rounded px-3 py-2.5">
      <div className="text-[10px] font-mono text-terminal-muted">{label}</div>
      <div className={`text-sm font-mono font-semibold mt-0.5 ${highlight ? 'text-accent-green' : 'text-terminal-text'}`}>
        {value}
      </div>
      {sub && <div className="text-[10px] font-mono text-terminal-muted mt-0.5">{sub}</div>}
    </div>
  )
}

function ModelCard({ title, subtitle, description, features }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="text-sm font-mono font-semibold text-terminal-text">{title}</h2>
          <p className="text-[11px] font-mono text-terminal-muted mt-0.5">{subtitle}</p>
        </div>
      </div>
      <div className="card-body space-y-4">
        <p className="text-xs text-terminal-dim leading-relaxed">{description}</p>
        <div>
          <div className="data-label mb-2">Configuration</div>
          <ul className="space-y-1">
            {features.map((f, i) => (
              <li key={i} className="text-xs font-mono text-terminal-dim flex items-start gap-2">
                <span className="text-accent-amber mt-0.5">›</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
