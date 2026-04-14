import { useApi } from '../hooks/useApi'

/**
 * Model Performance page.
 *
 * Displays backtest results, accuracy metrics, and calibration info.
 * In demo mode (no trained models), shows placeholder explanations.
 */
export default function ModelPerformance() {
  const { data: health } = useApi('/health')

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-display font-bold text-terminal-text">Model Performance</h1>
        <p className="text-sm font-body text-terminal-muted mt-1">
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
            <StatusItem label="Models Loaded" value={health?.models_loaded ? 'Yes' : 'Not yet'} />
            <StatusItem label="Status" value={health?.status || 'Unknown'} />
            <StatusItem label="Last Update" value={health?.timestamp ? new Date(health.timestamp).toLocaleString() : '—'} />
          </div>
        </div>
      </div>

      {/* ── Model architecture ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ModelCard
          title="Return Forecaster"
          subtitle="Model 1 — Forward Return Prediction"
          description="XGBoost regression predicting forward log returns at 4 horizons (1-week, 1-month, 3-month, 6-month). Trained with walk-forward expanding window validation."
          features={[
            'Algorithm: XGBoost with quantile regression',
            'Horizons: 5d, 21d, 63d, 126d',
            'Features: Technical + Fundamental + Macro + Sentiment',
            'Validation: Walk-forward, 3-year minimum train window',
            'Output: Point estimate + 80% prediction interval',
          ]}
          metrics={[
            { label: 'Direction Accuracy', value: '—', note: 'Train models to see results' },
            { label: 'RMSE (21d)', value: '—', note: '' },
            { label: 'Information Coeff.', value: '—', note: '' },
            { label: 'Interval Coverage', value: '—', note: '' },
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
          metrics={[
            { label: 'MAE', value: '—', note: 'Train models to see results' },
            { label: 'RMSE', value: '—', note: '' },
            { label: 'Signal Accuracy', value: '—', note: '' },
            { label: 'Mean Valuation Gap', value: '—', note: '' },
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
              <p className="text-xs font-body text-terminal-dim leading-relaxed">
                When both models agree (e.g., negative predicted returns AND overvalued fair value),
                the signal is stronger. Disagreement reduces confidence.
              </p>
            </div>
            <div>
              <h3 className="text-sm font-mono font-semibold text-terminal-text mb-2">Confidence Score</h3>
              <p className="text-xs font-body text-terminal-dim leading-relaxed">
                0–100 composite based on: model agreement (35%), prediction interval width (25%),
                signal magnitude (25%), and historical accuracy (15%).
              </p>
            </div>
            <div>
              <h3 className="text-sm font-mono font-semibold text-terminal-text mb-2">Explainability</h3>
              <p className="text-xs font-body text-terminal-dim leading-relaxed">
                Every prediction includes SHAP explanations showing the top 10 features
                driving the model's view, with plain-English descriptions.
              </p>
            </div>
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
              { step: '5', title: 'Slide', desc: 'Advance window by 63 days', color: 'terminal-dim' },
              { step: '6', title: 'Repeat', desc: 'Until all data is covered', color: 'terminal-dim' },
            ].map((s) => (
              <div key={s.step} className="flex-shrink-0 w-40">
                <div className={`w-8 h-8 rounded-full bg-${s.color}/20 border border-${s.color}/40 flex items-center justify-center text-xs font-mono font-bold text-${s.color} mb-2`}>
                  {s.step}
                </div>
                <div className="text-sm font-mono font-semibold text-terminal-text">{s.title}</div>
                <div className="text-[11px] font-body text-terminal-muted mt-0.5">{s.desc}</div>
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

function StatusItem({ label, value }) {
  return (
    <div>
      <div className="data-label mb-1">{label}</div>
      <div className="data-value-sm">{value}</div>
    </div>
  )
}

function ModelCard({ title, subtitle, description, features, metrics }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="text-sm font-mono font-semibold text-terminal-text">{title}</h2>
          <p className="text-[11px] font-mono text-terminal-muted mt-0.5">{subtitle}</p>
        </div>
      </div>
      <div className="card-body space-y-4">
        <p className="text-xs font-body text-terminal-dim leading-relaxed">{description}</p>

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

        <div>
          <div className="data-label mb-2">Backtest Metrics</div>
          <div className="grid grid-cols-2 gap-2">
            {metrics.map((m, i) => (
              <div key={i} className="bg-terminal-elevated rounded px-3 py-2">
                <div className="text-[10px] font-mono text-terminal-muted">{m.label}</div>
                <div className="text-sm font-mono font-semibold text-terminal-text">{m.value}</div>
                {m.note && <div className="text-[10px] font-mono text-terminal-muted mt-0.5">{m.note}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
