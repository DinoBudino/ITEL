export function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`
}

export function formatMs(value) {
  return `${Number(value || 0).toFixed(1)} ms`
}
