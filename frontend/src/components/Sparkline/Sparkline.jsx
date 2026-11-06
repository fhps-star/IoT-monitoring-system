import "./Sparkline.css";

export function Sparkline({ values, color = 'var(--cyan)' }) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max === min ? 1 : max - min
  const points = values.map((value, index) => `${(index / Math.max(values.length - 1, 1)) * 100},${90 - ((value - min) / span) * 80}`).join(' ')
  return <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline" aria-hidden="true"><polyline points={points} fill="none" stroke={color} strokeWidth="2.5" vectorEffect="non-scaling-stroke" /></svg>
}
