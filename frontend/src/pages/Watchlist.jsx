import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApi, apiPost, apiDelete } from '../hooks/useApi'
import ValuationBadge from '../components/ValuationBadge'
import { formatPrice, formatPercentRaw, valueColor } from '../utils/formatters'

export default function Watchlist() {
  const navigate = useNavigate()
  const { data, loading, error, refetch } = useApi('/reports', { defaultData: { items: [] } })
  const [addTicker, setAddTicker] = useState('')
  const [adding, setAdding] = useState(false)
  const [sortKey, setSortKey] = useState('ticker')
  const [sortDir, setSortDir] = useState('asc')

  const items = data?.items || []

  const sorted = [...items].sort((a, b) => {
    let aVal = a[sortKey] ?? ''
    let bVal = b[sortKey] ?? ''
    if (typeof aVal === 'string') aVal = aVal.toLowerCase()
    if (typeof bVal === 'string') bVal = bVal.toLowerCase()
    if (aVal < bVal) return sortDir === 'asc' ? -1 : 1
    if (aVal > bVal) return sortDir === 'asc' ? 1 : -1
    return 0
  })

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const handleAdd = async (e) => {
    e.preventDefault()
    if (!addTicker.trim()) return
    setAdding(true)
    try {
      await apiPost('/watchlist', { ticker: addTicker.trim().toUpperCase() })
      setAddTicker('')
      refetch()
    } catch (err) {
      alert(err.response?.data?.detail || 'Failed to add ticker')
    } finally {
      setAdding(false)
    }
  }

  const handleRemove = async (ticker, e) => {
    e.stopPropagation()
    if (!confirm(`Remove ${ticker} from watchlist?`)) return
    try {
      await apiDelete(`/watchlist/${ticker}`)
      refetch()
    } catch (err) {
      alert('Failed to remove ticker')
    }
  }

  const SortIcon = ({ col }) => {
    if (sortKey !== col) return <span className="text-terminal-border ml-1">↕</span>
    return <span className="text-accent-amber ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-display font-bold text-terminal-text">Watchlist</h1>
          <p className="text-sm font-body text-terminal-muted mt-1">
            {items.length} ticker{items.length !== 1 ? 's' : ''} tracked
          </p>
        </div>
        <form onSubmit={handleAdd} className="flex items-center gap-2">
          <input
            type="text"
            value={addTicker}
            onChange={(e) => setAddTicker(e.target.value.toUpperCase())}
            placeholder="Add ticker..."
            maxLength={10}
            className="bg-terminal-elevated border border-terminal-border rounded-md px-3 py-1.5 text-sm font-mono text-terminal-text placeholder-terminal-muted focus:outline-none focus:border-accent-amber/50 w-32"
          />
          <button
            type="submit"
            disabled={adding || !addTicker.trim()}
            className="bg-accent-amber/20 border border-accent-amber/30 text-accent-amber text-sm font-mono px-3 py-1.5 rounded-md hover:bg-accent-amber/30 disabled:opacity-40 transition-colors"
          >
            {adding ? '...' : '+ Add'}
          </button>
        </form>
      </div>

      {/* Error state */}
      {error && (
        <div className="card mb-4 p-4 border-accent-red/30">
          <p className="text-accent-red text-sm font-mono">
            Failed to load watchlist: {error}
          </p>
          <p className="text-terminal-muted text-xs font-mono mt-1">
            Make sure the backend is running at localhost:8000
          </p>
        </div>
      )}

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-terminal-elevated/50">
                <th className="table-header cursor-pointer" onClick={() => handleSort('ticker')}>
                  Ticker <SortIcon col="ticker" />
                </th>
                <th className="table-header">Name</th>
                <th className="table-header">Sector</th>
                <th className="table-header text-right cursor-pointer" onClick={() => handleSort('current_price')}>
                  Price <SortIcon col="current_price" />
                </th>
                <th className="table-header text-right cursor-pointer" onClick={() => handleSort('price_change_1d_pct')}>
                  1D Chg <SortIcon col="price_change_1d_pct" />
                </th>
                <th className="table-header text-right cursor-pointer" onClick={() => handleSort('price_change_1m_pct')}>
                  1M Chg <SortIcon col="price_change_1m_pct" />
                </th>
                <th className="table-header text-center">Signal</th>
                <th className="table-header text-right cursor-pointer" onClick={() => handleSort('confidence')}>
                  Confidence <SortIcon col="confidence" />
                </th>
                <th className="table-header w-10"></th>
              </tr>
            </thead>
            <tbody>
              {loading && !items.length ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="table-row">
                    {Array.from({ length: 9 }).map((_, j) => (
                      <td key={j} className="table-cell">
                        <div className="skeleton h-4 w-16" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : sorted.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-4 py-12 text-center text-terminal-muted font-mono text-sm">
                    No tickers on your watchlist yet. Add one above to get started.
                  </td>
                </tr>
              ) : (
                sorted.map((item) => (
                  <tr
                    key={item.ticker}
                    className="table-row"
                    onClick={() => navigate(`/ticker/${item.ticker}`)}
                  >
                    <td className="table-cell font-semibold text-accent-cyan">{item.ticker}</td>
                    <td className="table-cell text-terminal-dim truncate max-w-[200px]">{item.name || '—'}</td>
                    <td className="table-cell text-terminal-dim text-xs">{item.sector || '—'}</td>
                    <td className="table-cell text-right">{formatPrice(item.current_price)}</td>
                    <td className={`table-cell text-right ${valueColor(item.price_change_1d_pct)}`}>
                      {formatPercentRaw(item.price_change_1d_pct)}
                    </td>
                    <td className={`table-cell text-right ${valueColor(item.price_change_1m_pct)}`}>
                      {formatPercentRaw(item.price_change_1m_pct)}
                    </td>
                    <td className="table-cell text-center">
                      <ValuationBadge signal={item.signal} />
                    </td>
                    <td className="table-cell text-right font-mono">
                      {item.confidence != null ? `${item.confidence.toFixed(0)}` : '—'}
                    </td>
                    <td className="table-cell text-center">
                      <button
                        onClick={(e) => handleRemove(item.ticker, e)}
                        className="text-terminal-muted hover:text-accent-red transition-colors text-xs"
                        title="Remove"
                      >
                        ✕
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
