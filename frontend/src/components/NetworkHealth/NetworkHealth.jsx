import { ChevronRight, ShieldCheck } from "lucide-react";
import "./NetworkHealth.css";

export function NetworkHealth({ devices = [], metrics, dataState }) {
  const online = devices.filter((device) => device.online).length;
  const configured = metrics?.total_devices ?? devices.length;
  const availability = configured ? Math.round((online / configured) * 100) : 0;

  // Strict industrial state: only ONLINE, WARNING, OFFLINE
  let resolvedStatus = "offline";
  let resolvedLabel = "OFFLINE";

  if (online > 0) {
    if (availability === 100) {
      resolvedStatus = "healthy";
      resolvedLabel = "ONLINE";
    } else {
      resolvedStatus = "warning";
      resolvedLabel = "WARNING";
    }
  } else {
    resolvedStatus = "offline";
    resolvedLabel = "OFFLINE";
  }

  const averageLatency =
    metrics?.avg_latency_ms !== null &&
    metrics?.avg_latency_ms !== undefined &&
    Number.isFinite(Number(metrics.avg_latency_ms))
      ? `${Math.round(Number(metrics.avg_latency_ms))} ms`
      : "—";

  const factors = [
    ["Online devices", `${online} / ${configured}`, availability],
    ["Average latency", averageLatency, null],
  ];

  return (
    <section className="health-card">
      <div className="health-card-top">
        <div>
          <span className="health-eyebrow">
            <ShieldCheck size={14} /> Network Health
          </span>
          <h2>Network status</h2>
        </div>
        <button className="icon-button health-more" title="Network analytics">
          <ChevronRight size={17} />
        </button>
      </div>
      <div className="health-score-row">
        <div className="health-score">
          <strong>{availability}</strong>
          <span>%</span>
        </div>
        <span className={`health-state ${resolvedStatus}`}>
          <i />
          {resolvedLabel}
        </span>
      </div>
      <div className="health-breakdown">
        {factors.map(([name, value, percent]) => (
          <div className="health-factor" key={name}>
            <div>
              <span>{name}</span>
              <strong>{value}</strong>
            </div>
            {percent !== null && (
              <span className="health-track">
                <i style={{ width: `${percent}%` }} />
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="health-card-foot">
        <span className="foot-status">
          <i className={`status-dot ${resolvedStatus}`} />
          {online > 0
            ? `${online} Connected Device${online > 1 ? "s" : ""}`
            : "No Active Nodes"}
        </span>
        <span className="foot-time">
          {metrics?.last_message_at
            ? `Last sync: ${new Date(metrics.last_message_at).toLocaleTimeString()}`
            : "No telemetry received"}
        </span>
      </div>
    </section>
  );
}
