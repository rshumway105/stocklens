/**
 * Generates a one-to-two sentence plain-English summary from a ValuationReport.
 * Returns null when models haven't been trained or signal is unknown.
 *
 * Logic:
 *  - Signal determines the opening clause.
 *  - valuation_gap_pct gives the magnitude (negative = below fair value).
 *  - Top 2 SHAP drivers with non-empty explanations form the second sentence.
 */
export function buildSummary(report) {
  if (!report) return null

  const { ticker, signal, valuation_gap_pct, top_drivers, risk_flags } = report

  if (risk_flags?.some((f) => f.flag === 'demo_mode')) return null
  if (!signal || signal === 'unknown') return null

  // Gap magnitude as a rounded whole-number percentage
  const gap =
    valuation_gap_pct != null ? Math.abs(Math.round(valuation_gap_pct)) : null
  const gapStr = gap != null ? ` by ~${gap}%` : ''

  // Top 2 SHAP drivers that have a plain-English explanation
  const drivers = (top_drivers ?? [])
    .filter((d) => d.explanation?.trim())
    .slice(0, 2)
    .map((d) => d.explanation.trim().replace(/\.$/, '').toLowerCase())

  const driverSentence =
    drivers.length === 0
      ? ''
      : drivers.length === 1
      ? `The primary driver is ${drivers[0]}.`
      : `The primary drivers are ${drivers[0]} and ${drivers[1]}.`

  if (signal === 'undervalued') {
    const lead = `${ticker} appears undervalued${gapStr} relative to its fair value estimate.`
    return driverSentence ? `${lead} ${driverSentence}` : lead
  }

  if (signal === 'overvalued') {
    const lead = `${ticker} appears overvalued${gapStr}.`
    return driverSentence ? `${lead} ${driverSentence}` : lead
  }

  // fairly_valued
  const lead = `${ticker} is trading close to fair value.`
  return driverSentence
    ? `${lead} ${driverSentence}`
    : `${lead} No strong signal in either direction.`
}
