import { Activity, AlertOctagon, AlertTriangle, Check, ChevronDown, Filter, Gauge, Search, Thermometer } from "lucide-react";
import { StatusDot } from "./StatusBadge/StatusBadge.jsx";
import { statusLabel } from "../utils/status.js";
import "./shared.css";

export function Panel({ children, className = "" }) {
  return <section className={`panel ${className}`}>{children}</section>;
}

export function MiniStat({ label, value, icon: Icon }) {
  return (
    <div className="mini-stat">
      <span className="mini-stat-icon">
        <Icon size={16} />
      </span>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

export function TopologyLegend({ compact = false }) {
  return (
    <div className={`topology-legend ${compact ? "" : "full-legend"}`}>
      <span>
        <i className="legend-line healthy-line" />
        Healthy
      </span>
      <span>
        <i className="legend-line warning-line" />
        Warning
      </span>
      <span>
        <i className="legend-line critical-line" />
        Critical
      </span>
      {!compact && (
        <>
          <span>
            <i className="legend-line offline-line" />
            Disconnected
          </span>
          {["healthy", "warning", "critical", "offline"].map((s) => (
            <span key={s}>
              <StatusDot status={s} /> {statusLabel(s)} node
            </span>
          ))}
        </>
      )}
    </div>
  );
}

export function EventRow({ event }) {
  const Icon =
    event.kind === "critical"
      ? AlertOctagon
      : event.kind === "warning"
        ? AlertTriangle
        : event.kind === "recovery"
          ? Check
          : Activity;
  return (
    <div className="event-row">
      <span className={`event-symbol ${event.kind}`}>
        <Icon size={15} />
      </span>
      <div className="event-content">
        <strong>{event.title}</strong>
        <span>
          {event.device} · {event.message}
        </span>
      </div>
      <time>{event.time}</time>
    </div>
  );
}

export function TableToolbar({ title, count, controls }) {
  return (
    <div className="module-toolbar">
      <div>
        <h2>{title}</h2>
        <span>{count} records</span>
      </div>
      <div className="toolbar-filters">{controls}</div>
    </div>
  );
}

export function FilterSelect({ value, onChange, options }) {
  return (
    <label className="filter-control">
      <Filter size={14} />
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label={`Filter ${options[0]}`}
      >
        {options.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </select>
      <ChevronDown size={13} />
    </label>
  );
}

export function RangeSelect({ range, setRange }) {
  return (
    <select
      className="select-control range-select"
      value={range}
      onChange={(e) => setRange(e.target.value)}
      aria-label="Select time range"
    >
      {["Last hour", "Last 24 hours", "Last 7 days"].map((item) => (
        <option key={item}>{item}</option>
      ))}
    </select>
  );
}

export function LinkButton({ id, deviceId, openDevice }) {
  return (
    <button className="link-button" onClick={() => openDevice(deviceId || id)}>
      {id}
    </button>
  );
}

export function EmptyState({ title, detail }) {
  return (
    <div className="empty-state">
      <div>
        <Search size={18} />
      </div>
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

export function SensorIcon({ kind }) {
  const Icon =
    kind === "temperature"
      ? Thermometer
      : kind === "vibration"
        ? Activity
        : Gauge;
  return <Icon size={17} />;
}

export function LineChart({ values, label }) {
  const width = 1000;
  const height = 260;
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const min = rawMin === rawMax ? rawMin - 1 : rawMin;
  const max = rawMin === rawMax ? rawMax + 1 : rawMax;
  const span = max - min;
  const yFor = (value) => height - 12 - ((value - min) / span) * (height - 24);
  const points = values
    .map((value, index) => `${(index / Math.max(values.length - 1, 1)) * width},${yFor(value)}`)
    .join(" ");
  const unit = label.match(/\(([^)]+)\)/)?.[1] ?? "";
  const axisValues = Array.from({ length: 5 }, (_, index) => max - (span * index) / 4);
  return (
    <div className="line-chart-wrap">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${label} trend line chart`}
      >
        <defs>
          <linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#47b9d5" stopOpacity=".17" />
            <stop offset="100%" stopColor="#47b9d5" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[35, 90, 145, 200, 255].map((y) => (
          <line
            key={y}
            x1="0"
            x2={width}
            y1={y}
            y2={y}
            className="chart-gridline"
          />
        ))}
        <polygon
          points={`0,${height} ${points} ${width},${height}`}
          fill="url(#chartFill)"
        />
        <polyline
          points={points}
          fill="none"
          stroke="#52b8d0"
          strokeWidth="2.4"
          vectorEffect="non-scaling-stroke"
        />
        {values.map((value, index) =>
            (index % 5 === 0 || index === values.length - 1) && (
              <circle
                key={index}
                cx={(index / Math.max(values.length - 1, 1)) * width}
                cy={yFor(value)}
                r="3.8"
                className="chart-point"
              >
                <title>{`${label}: ${value}`}</title>
              </circle>
            ),
        )}
      </svg>
      <div className="chart-y-axis">
        {axisValues.map((value, index) => (
          <span key={index}>{`${Number(value.toFixed(span < 1 ? 2 : 1))} ${unit}`.trim()}</span>
        ))}
      </div>
    </div>
  );
}
