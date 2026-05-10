/**
 * Compact accuracy callout placed below the forecast fan chart.
 *
 * Props
 *   accuracy   – 0-100 float: % of backtested 21-day windows where the model
 *                predicted the correct return direction (direction_accuracy_21d)
 *   overDays   – int: the horizon this accuracy applies to (always 21 here)
 *   withinPct  – 0-100 float | null: % of backtested windows where the actual
 *                return fell inside the 80% prediction interval
 *                (interval_coverage_21d). Optional — omitted when not available.
 *
 * Returns null when accuracy is null (no backtest data for this ticker yet).
 */
export default function AccuracyCallout({ accuracy, overDays, withinPct }) {
  if (accuracy == null) return null

  const accStr = accuracy.toFixed(1)

  const primaryText =
    withinPct != null
      ? `Correctly predicted ${overDays}-day direction in ${accStr}% of backtested periods — 80% intervals captured actual returns ${withinPct.toFixed(1)}% of the time`
      : `Correctly predicted ${overDays}-day return direction in ${accStr}% of backtested periods`

  return (
    <div className="mt-4 pt-3 border-t border-terminal-border/50 space-y-1.5">
      <p className="text-xs font-mono text-terminal-dim leading-relaxed">{primaryText}</p>
      <p className="text-[10px] font-mono text-terminal-muted leading-relaxed">
        Past model accuracy does not predict future performance.
      </p>
    </div>
  )
}
