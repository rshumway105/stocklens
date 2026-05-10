import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApi, apiPost, apiDelete } from '../hooks/useApi'
import api from '../hooks/useApi'
import ValuationBadge from '../components/ValuationBadge'
import ConvictionBadge from '../components/ConvictionBadge'
import { formatPrice, formatPercentRaw, valueColor } from '../utils/formatters'
import { searchTickers } from '../utils/tickers'
import LastUpdated from '../components/LastUpdated'

export default function Watchlist() {
  const navigate = useNavigate()
  const { data, loading, error, refetch } = useApi('/reports', { defaultData: { items: [] } })
  const [addTicker, setAddTicker] = useState('')
  const [adding, setAdding] = useState(false)
  const [sortKey, setSortKey] = useState('ticker')
  const [sortDir, setSortDir] = useState('asc')
  const [trainingStatuses, setTrainingStatuses] = useState({})
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const pollRef = useRef(null)
  const comboRef = useRef(null)

  const suggestions = searchTickers(addTicker)

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (comboRef.current && !comboRef.current.contains(e.target)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const items = data?.items || []

  // Poll training status every 3 seconds while any ticker is in-flight
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await api.get('/status')
        const statuses = res.data
        setTrainingStatuses(statuses)

        const anyInProgress = Object.values(statuses).some(
          (s) => s.state === 'fetching' || s.state === 'training'
        )
        const anyJustFinished = Object.values(statuses).some(
          (s) => s.state === 'ready'
        )

        if (anyJustFinished) {
          refetch()
        }

        if (!anyInProgress) {
          clearInterval(pollRef.current)
          pollRef.current = null
        }
      } catch (_) {}
    }

    pollRef.current = setInterval(poll, 3000)
    poll() // run immediately on mount
    return () => clearInterval(pollRef.current)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const startPolling = () => {
    if (!pollRef.current) {
      pollRef.current = setInterval(async () => {
        try {
          const res = await api.get('/status')
          const statuses = res.data
          setTrainingStatuses(statuses)
          const anyInProgress = Object.values(statuses).some(
            (s) => s.state === 'fetching' || s.state === 'training'
          )
          if (!anyInProgress) {
            clearInterval(pollRef.current)
            pollRef.current = null
            refetch()
          }
        } catch (_) {}
      }, 3000)
    }
  }

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
      startPolling()
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

  const TrainingBadge = ({ ticker }) => {
    const s = trainingStatuses[ticker]
    if (!s || s.state === 'idle' || s.state === 'ready') return null
    if (s.state === 'error') {
      return (
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-accent-red/15 text-accent-red ml-1" title={s.message}>
          Error
        </span>
      )
    }
    const label = s.state === 'fetching' ? 'Fetching...' : 'Training...'
    return (
      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-accent-amber/15 text-accent-amber ml-1 animate-pulse">
        {label}
      </span>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-mono font-bold text-terminal-text">Watchlist</h1>
          <p className="text-sm text-terminal-muted mt-1">
            {items.length} ticker{items.length !== 1 ? 's' : ''} tracked
          </p>
        </div>
        <form onSubmit={handleAdd} className="flex items-center gap-2">
          <div ref={comboRef} className="relative">
            <input
              type="text"
              value={addTicker}
              onChange={(e) => {
                setAddTicker(e.target.value.toUpperCase())
                setDropdownOpen(true)
                setActiveIndex(-1)
              }}
              onFocus={() => { if (addTicker) setDropdownOpen(true) }}
              onKeyDown={(e) => {
                if (!dropdownOpen || suggestions.length === 0) return
                if (e.key === 'ArrowDown') {
                  e.preventDefault()
                  setActiveIndex((i) => Math.min(i + 1, suggestions.length - 1))
                } else if (e.key === 'ArrowUp') {
                  e.preventDefault()
                  setActiveIndex((i) => Math.max(i - 1, -1))
                } else if (e.key === 'Enter' && activeIndex >= 0) {
                  e.preventDefault()
                  setAddTicker(suggestions[activeIndex].ticker)
                  setDropdownOpen(false)
                  setActiveIndex(-1)
                } else if (e.key === 'Escape') {
                  setDropdownOpen(false)
                  setActiveIndex(-1)
                }
              }}
              placeholder="Add ticker..."
              maxLength={10}
              autoComplete="off"
              className="bg-terminal-elevated border border-terminal-border rounded-md px-3 py-1.5 text-sm font-mono text-terminal-text placeholder-terminal-muted focus:outline-none focus:border-accent-amber/50 w-44"
            />
            {dropdownOpen && suggestions.length > 0 && (
              <ul className="absolute z-50 top-full mt-1 left-0 w-64 bg-terminal-elevated border border-terminal-border rounded-md shadow-lg overflow-hidden">
                {suggestions.map((s, i) => (
                  <li
                    key={s.ticker}
                    onMouseDown={(e) => {
                      e.preventDefault()
                      setAddTicker(s.ticker)
                      setDropdownOpen(false)
                      setActiveIndex(-1)
                    }}
                    onMouseEnter={() => setActiveIndex(i)}
                    className={`flex items-center justify-between px-3 py-2 cursor-pointer text-sm font-mono ${
                      i === activeIndex
                        ? 'bg-accent-amber/15 text-terminal-text'
                        : 'text-terminal-dim hover:bg-terminal-border/30'
                    }`}
                  >
                    <span className="text-accent-cyan font-semibold w-16 shrink-0">{s.ticker}</span>
                    <span className="text-terminal-muted text-xs truncate">{s.name}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button
            type="submit"
            disabled={adding || !addTicker.trim()}
            className="bg-accent-amber/20 border border-accent-amber/30 text-accent-amber text-sm font-mono px-3 py-1.5 rounded-md hover:bg-accent-amber/30 disabled:opacity-40 transition-colors"
          >
            {adding ? '...' : '+ Add'}
          </button>
        </form>
      </div>

      {/* Training notice */}
      {Object.values(trainingStatuses).some(s => s.state === 'fetching' || s.state === 'training') && (
        <div className="card mb-4 p-4 border-accent-amber/30 bg-accent-amber/5">
          <p className="text-accent-amber text-sm font-mono">
            Fetching data and training models in the background — this takes about 30–60 seconds.
            The page will update automatically when ready.
          </p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div className="card mb-4 p-4 border-accent-red/30">
          <p className="text-accent-red text-sm font-mono">
            Failed to load watchlist: {error}
          </p>
          <p className="text-terminal-muted text-xs font-mono mt-1">
            Make sure the backend API is reachable at {import.meta.env.VITE_API_URL || 'http://localhost:8000'}
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
                <th className="table-header text-right cursor-pointer" onClick={() => handleSort('confidence')} title="Conviction score 0–100. Click badge for full legend.">
                  Score <SortIcon col="confidence" />
                </th>
                <th className="table-header text-right">As Of</th>
                <th className="table-header w-10"></th>
              </tr>
            </thead>
            <tbody>
              {loading && !items.length ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="table-row">
                    {Array.from({ length: 10 }).map((_, j) => (
                      <td key={j} className="table-cell">
                        <div className="skeleton h-4 w-16" />
                      </td>
                    ))}
                  </tr>
                ))
              ) : sorted.length === 0 ? (
                <tr>
                  <td colSpan={10} className="px-4 py-12 text-center text-terminal-muted font-mono text-sm">
                    No tickers on your watchlist yet. Add one above to get started.
                  </td>
                </tr>
              ) : (
                sorted.map((item) => {
                  const status = trainingStatuses[item.ticker]
                  const isTraining = status?.state === 'fetching' || status?.state === 'training'
                  return (
                    <tr
                      key={item.ticker}
                      className="table-row"
                      onClick={() => navigate(`/ticker/${item.ticker}`)}
                    >
                      <td className="table-cell font-semibold text-accent-cyan">
                        {item.ticker}
                        <TrainingBadge ticker={item.ticker} />
                      </td>
                      <td className="table-cell text-terminal-dim truncate max-w-[200px]" title={item.name || ''}>{item.name || '—'}</td>
                      <td className="table-cell text-terminal-dim text-xs">{item.sector || '—'}</td>
                      <td className="table-cell text-right">
                        {isTraining ? <span className="text-terminal-muted text-xs">—</span> : formatPrice(item.current_price)}
                      </td>
                      <td className={`table-cell text-right ${valueColor(item.price_change_1d_pct)}`}>
                        {isTraining ? '—' : formatPercentRaw(item.price_change_1d_pct)}
                      </td>
                      <td className={`table-cell text-right ${valueColor(item.price_change_1m_pct)}`}>
                        {isTraining ? '—' : formatPercentRaw(item.price_change_1m_pct)}
                      </td>
                      <td className="table-cell text-center">
                        {isTraining
                          ? <span className="text-[10px] font-mono text-terminal-muted">training</span>
                          : <ValuationBadge signal={item.signal} />}
                      </td>
                      <td className="table-cell text-right font-mono">
                        {isTraining ? '—' : <ConvictionBadge score={item.confidence} compact />}
                      </td>
                      <td className="table-cell text-right">
                        {isTraining ? '—' : <LastUpdated timestamp={data?.updated_at} />}
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
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
