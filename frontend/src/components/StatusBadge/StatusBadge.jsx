import { statusLabel } from "../../utils/status.js";
import "./StatusBadge.css";

export function Badge({ value }) {
  const key = value.toLowerCase().replaceAll(' ', '-')
  return <span className={`badge badge-${key}`}><i />{value}</span>
}

export function StatusDot({ status }) {
  return <i className={`status-dot ${status}`} aria-label={statusLabel(status)} title={statusLabel(status)} />
}
