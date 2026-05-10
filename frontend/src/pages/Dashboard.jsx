import { useState, useEffect, useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import ValuationBadge from '../components/ValuationBadge'
import { formatPrice, formatPercentRaw, valueColor } from '../utils/formatters'

// How often to tick the clock (ms)
const CLOCK_INTERVAL = 30_000

export default function Dashboard() {
  const navigate = useNavigate()
  const { data: overview } = useApi('/reports', { defaultData: { items: [] } })
  const { data: moversData, loading: moversLoading } = useApi('/movers?top_n=8', { defaultData: { gainers: [], losers: [] } })
  const { data: newsData, loading: newsLoading } = useApi('/news?limit=12', { defaultData: { articles: [] } })
  const { data: macroCatalog } = useApi('/macro/catalog', { defaultData: [] })

  const [clock, setClock] = useState(() => nowEST())
  useEffect(() => {
    const id = setInterval(() => setClock(nowEST()), CLOCK_INTERVAL)
    return () => clearInterval(id)
  }, [])

  const items = overview?.items || []
  const articles = newsData?.articles || []
  const gainers = moversData?.gainers || []
  const losers = moversData?.losers || []

  // Key macro indicators for the ticker tape
  const macroHighlights = useMemo(() => {
    const keys = ['fed_funds_rate', 'treasury_10y', 'vix', 'unemployment_rate', 'cpi_yoy']
    const labels = {
      fed_funds_rate: 'Fed Funds',
      treasury_10y: '10Y Treasury',
      vix: 'VIX',
      unemployment_rate: 'Unemployment',
      cpi_yoy: 'CPI YoY',
    }
    return keys
      .map((k) => {
        const item = macroCatalog?.find?.((c) => c.key === k)
        return item ? { key: k, label: labels[k], value: item.last_value, direction: item.direction } : null
      })
      .filter(Boolean)
  }, [macroCatalog])

  return (
    <div className="space-y-6">
      {/* ── Header ── */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold text-terminal-text">Market Dashboard</h1>
          <p className="text-sm text-terminal-muted mt-1">
            Live signals, top movers, and market news
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs font-mono text-terminal-muted">Market Time</div>
          <div className="text-sm font-mono text-accent-amber">{clock}</div>
        </div>
      </div>

      {/* ── Macro ticker tape ── */}
      {macroHighlights.length > 0 && (
        <div className="card">
          <div className="card-body py-3">
            <div className="flex items-center gap-6 overflow-x-auto no-scrollbar">
              {macroHighlights.map((m) => (
                <div key={m.key} className="flex items-center gap-2 shrink-0">
                  <span className="text-[11px] font-mono text-terminal-muted">{m.label}</span>
                  <span className="text-sm font-mono font-semibold text-terminal-text">
                    {m.value?.toFixed(2)}
                  </span>
                  <span className={`text-xs font-mono ${
                    m.direction === 'rising' ? 'text-accent-green' :
                    m.direction === 'falling' ? 'text-accent-red' :
                    'text-terminal-muted'
                  }`}>
                    {m.direction === 'rising' ? '↑' : m.direction === 'falling' ? '↓' : '→'}
                  </span>
                </div>
              ))}
              <Link to="/macro" className="shrink-0 text-[11px] font-mono text-accent-amber hover:underline ml-2">
                Full Macro →
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* ── Top Movers ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MoversCard title="Top Gainers" icon="↑" items={gainers} loading={moversLoading} navigate={navigate} universe={moversData?.universe_size} />
        <MoversCard title="Top Losers" icon="↓" items={losers} loading={moversLoading} navigate={navigate} flip universe={moversData?.universe_size} />
      </div>

      {/* ── Watchlist summary + News ── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Watchlist summary (2/5) */}
        <div className="lg:col-span-2 card">
          <div className="card-header">
            <h2 className="text-sm font-mono font-semibold text-terminal-text">Watchlist Signals</h2>
            <Link to="/watchlist" className="text-[11px] font-mono text-accent-amber hover:underline">
              Full list →
            </Link>
          </div>
          <div className="divide-y divide-terminal-border/30">
            {items.length === 0 ? (
              <div className="px-4 py-6 text-center text-terminal-muted font-mono text-xs">
                No tickers on your watchlist.{' '}
                <Link to="/watchlist" className="text-accent-amber hover:underline">Add one →</Link>
              </div>
            ) : (
              items.slice(0, 8).map((item) => (
                <button
                  key={item.ticker}
                  onClick={() => navigate(`/ticker/${item.ticker}`)}
                  className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-terminal-elevated/50 transition-colors text-left"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-sm font-mono font-semibold text-accent-cyan w-14 shrink-0">{item.ticker}</span>
                    <span className="text-xs font-mono text-terminal-muted truncate">{item.name || ''}</span>
                  </div>
                  <div className="flex items-center gap-3 shrink-0 ml-2">
                    <span className={`text-xs font-mono ${valueColor(item.price_change_1d_pct)}`}>
                      {formatPercentRaw(item.price_change_1d_pct)}
                    </span>
                    <ValuationBadge signal={item.signal} />
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        {/* News feed (3/5) */}
        <div className="lg:col-span-3 card">
          <div className="card-header">
            <h2 className="text-sm font-mono font-semibold text-terminal-text">Market News</h2>
            <span className="text-[11px] font-mono text-terminal-muted">Yahoo Finance · Reuters</span>
          </div>
          <div className="divide-y divide-terminal-border/30 max-h-[520px] overflow-y-auto">
            {newsLoading ? (
              Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="px-4 py-3 space-y-1.5">
                  <div className="skeleton h-3 w-4/5 rounded" />
                  <div className="skeleton h-2.5 w-2/3 rounded" />
                </div>
              ))
            ) : articles.length === 0 ? (
              <div className="px-4 py-8 text-center text-terminal-muted font-mono text-xs">
                No news available. RSS feeds may be temporarily unavailable.
              </div>
            ) : (
              articles.map((a, i) => (
                <a
                  key={i}
                  href={a.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block px-4 py-3 hover:bg-terminal-elevated/50 transition-colors group"
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-xs font-mono text-terminal-text group-hover:text-accent-cyan transition-colors leading-snug flex-1">
                      {a.title}
                    </p>
                    <span className="text-[10px] font-mono text-terminal-muted shrink-0 mt-0.5">↗</span>
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-[10px] font-mono text-accent-amber">{a.source}</span>
                    {a.published_at && (
                      <span className="text-[10px] font-mono text-terminal-muted">
                        {relativeTime(a.published_at)}
                      </span>
                    )}
                  </div>
                </a>
              ))
            )}
          </div>
        </div>
      </div>

    </div>
  )
}

function MoversCard({ title, icon, items, loading, navigate, flip = false, universe }) {
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="text-sm font-mono font-semibold text-terminal-text">
          <span className={flip ? 'text-accent-red' : 'text-accent-green'}>{icon} </span>
          {title}
        </h2>
        <span className="text-[11px] font-mono text-terminal-muted">
          {universe ? `1-day · ${universe} tickers scanned` : '1-day change'}
        </span>
      </div>
      <div className="divide-y divide-terminal-border/30">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="px-4 py-3 flex justify-between items-center">
              <div className="skeleton h-4 w-24 rounded" />
              <div className="skeleton h-4 w-14 rounded" />
            </div>
          ))
        ) : items.length === 0 ? (
          <div className="px-4 py-6 text-center text-terminal-muted font-mono text-xs">
            No data — market may be closed
          </div>
        ) : (
          items.map((item) => (
            <button
              key={item.ticker}
              onClick={() => navigate(`/ticker/${item.ticker}`)}
              className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-terminal-elevated/50 transition-colors"
            >
              <div className="text-left min-w-0">
                <div className="text-sm font-mono font-semibold text-accent-cyan">{item.ticker}</div>
                <div className="text-[11px] font-mono text-terminal-muted truncate max-w-[160px]">{item.name}</div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <span className="text-sm font-mono text-terminal-dim">{formatPrice(item.price)}</span>
                <span className={`text-sm font-mono font-semibold tabular-nums w-16 text-right ${item.change_pct >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                  {item.change_pct >= 0 ? '+' : ''}{item.change_pct?.toFixed(2)}%
                </span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  )
}

function nowEST() {
  return new Date().toLocaleString('en-US', {
    timeZone: 'America/New_York',
    weekday: 'short',
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
    hour12: true,
  }) + ' EST'
}

function relativeTime(iso) {
  try {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
    return `${Math.floor(diff / 86400)}d ago`
  } catch {
    return ''
  }
}
