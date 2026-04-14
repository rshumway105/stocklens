/**
 * Format a number as currency (USD).
 */
export function formatPrice(value) {
  if (value == null || isNaN(value)) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

/**
 * Format a number as a percentage with sign.
 * e.g., 0.0532 → "+5.32%", -0.12 → "-12.00%"
 */
export function formatPercent(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—'
  const pct = value * 100
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${pct.toFixed(decimals)}%`
}

/**
 * Format a percentage that's already in % form (e.g., 5.32 → "+5.32%").
 */
export function formatPercentRaw(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(decimals)}%`
}

/**
 * Format large numbers compactly (e.g., 2.5T, 150B, 3.2M).
 */
export function formatCompact(value) {
  if (value == null || isNaN(value)) return '—'
  const abs = Math.abs(value)
  if (abs >= 1e12) return `${(value / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return value.toFixed(2)
}

/**
 * Format a date string to a short readable form.
 */
export function formatDate(dateStr) {
  if (!dateStr) return '—'
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

/**
 * Get the color class for a value (green for positive, red for negative).
 */
export function valueColor(value) {
  if (value == null || isNaN(value)) return 'text-terminal-dim'
  if (value > 0) return 'text-accent-green'
  if (value < 0) return 'text-accent-red'
  return 'text-terminal-dim'
}

/**
 * Get the signal badge class name.
 */
export function signalBadgeClass(signal) {
  switch (signal) {
    case 'overvalued': return 'badge-overvalued'
    case 'undervalued': return 'badge-undervalued'
    case 'fairly_valued': return 'badge-fairly-valued'
    default: return 'badge-unknown'
  }
}

/**
 * Format signal label for display.
 */
export function signalLabel(signal) {
  switch (signal) {
    case 'overvalued': return 'Overvalued'
    case 'undervalued': return 'Undervalued'
    case 'fairly_valued': return 'Fair Value'
    default: return 'Unknown'
  }
}
