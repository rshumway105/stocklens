import { signalBadgeClass, signalLabel } from '../utils/formatters'

/**
 * Color-coded badge showing overvalued / fairly valued / undervalued.
 */
export default function ValuationBadge({ signal, className = '' }) {
  return (
    <span className={`${signalBadgeClass(signal)} ${className}`}>
      {signalLabel(signal)}
    </span>
  )
}
