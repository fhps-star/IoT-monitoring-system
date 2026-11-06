import { Gauge, ArrowDownRight, ArrowUpRight } from "lucide-react";
import "./KPICard.css";

export function KPICard({ label, value, suffix, icon: Icon = Gauge, tone = 'cyan', detail, trend, delay = 0 }) {
  const down = trend?.startsWith('−') || trend?.startsWith('-')
  return (
    <article className="metric-card" style={{ '--enter-delay': `${delay}ms` }}>
      <div className="metric-top">
        <span>{label}</span>
        <span className={`metric-icon ${tone}`}><Icon size={16} strokeWidth={1.8} /></span>
      </div>
      <div className="metric-value-row"><strong>{value}</strong>{suffix && <span className="metric-suffix">{suffix}</span>}</div>
      <div className="metric-bottom">
        <span>{detail}</span>
        {trend && <span className={`metric-trend ${down ? 'trend-down' : ''}`}>{down ? <ArrowDownRight size={12} /> : <ArrowUpRight size={12} />}{trend}</span>}
      </div>
    </article>
  )
}
