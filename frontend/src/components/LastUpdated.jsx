/**
 * Renders a relative-time "updated X ago" label for a given timestamp.
 *
 * Color thresholds:
 *   < 24 h  →  terminal-muted  (fresh)
 *   24–72 h →  accent-amber    (getting stale)
 *   > 72 h  →  accent-red      (stale)
 *
 * Accepts any string parseable by Date(): ISO datetimes ("2024-01-15T12:00:00")
 * or bare date strings ("2024-01-15"). Bare dates are treated as UTC noon to
 * avoid ±1-day timezone drift in the staleness calculation.
 */

function parseTimestamp(raw) {
  if (!raw) return null
  // Bare date (no T): treat as noon UTC so timezone offsets don't flip the day
  const normalised = /^\d{4}-\d{2}-\d{2}$/.test(raw)
    ? `${raw}T12:00:00Z`
    : raw
  const d = new Date(normalised)
  return isNaN(d.getTime()) ? null : d
}

function toRelativeTime(date) {
  const diffMs = Date.now() - date.getTime()
  const s = Math.floor(diffMs / 1000)
  const m = Math.floor(s / 60)
  const h = Math.floor(m / 60)
  const d = Math.floor(h / 24)

  if (s < 60)  return 'just now'
  if (m < 60)  return `${m} minute${m !== 1 ? 's' : ''} ago`
  if (h < 24)  return `${h} hour${h !== 1 ? 's' : ''} ago`
  return `${d} day${d !== 1 ? 's' : ''} ago`
}

export default function LastUpdated({ timestamp, className = '' }) {
  const date = parseTimestamp(timestamp)
  if (!date) return null

  const diffHours = (Date.now() - date.getTime()) / 3_600_000

  const colorClass =
    diffHours > 72 ? 'text-accent-red' :
    diffHours > 24 ? 'text-accent-amber' :
    'text-terminal-muted'

  return (
    <span className={`text-[10px] font-mono ${colorClass} ${className}`}>
      updated {toRelativeTime(date)}
    </span>
  )
}
