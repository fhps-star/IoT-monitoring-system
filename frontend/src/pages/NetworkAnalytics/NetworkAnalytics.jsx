import { useState } from "react";
import { PanelHeading, MetricCard, StatusDot } from "../../components/index.jsx";
import { Panel, RangeSelect, LineChart, EmptyState } from "../../components/shared.jsx";
import { Activity, Clock, Database, Radio, Thermometer, Zap } from "lucide-react";
import "./NetworkAnalytics.css";

const chartMetrics = {
  Temperature: { field: "temperature_c", unit: "°C", icon: Thermometer },
  Vibration: { field: "vibration", unit: "m/s²", icon: Activity },
  Current: { field: "current_a", unit: "A", icon: Zap },
};

export function NetworkAnalytics({
  range,
  setRange,
  metrics,
  telemetry = [],
  now,
  dataState,
}) {
  const [metric, setMetric] = useState("Temperature");
  const { field, unit } = chartMetrics[metric];
  const maxAgeMs =
    range === "Last hour"
      ? 60 * 60 * 1000
      : range === "Last 24 hours"
      ? 24 * 60 * 60 * 1000
      : 7 * 24 * 60 * 60 * 1000;
  const cutoff = now - maxAgeMs;

  const visibleTelemetry = telemetry.filter((item) => {
    const receivedAt = Date.parse(item.received_at ?? item.timestamp);
    return Number.isFinite(receivedAt) && receivedAt >= cutoff;
  });

  const values = visibleTelemetry
    .filter(
      (item) =>
        item[field] !== null &&
        item[field] !== undefined &&
        Number.isFinite(Number(item[field])),
    )
    .map((item) => Number(item[field]))
    .reverse();

  const currentValue = values.length
    ? `${values.at(-1).toFixed(metric === "Vibration" ? 3 : 2)} ${unit}`
    : "—";

  return (
    <div className="page-stack network-analytics-page">
      <div className="analytics-controls">
        <span
          className={`updated-label status-${
            dataState === "live"
              ? "healthy"
              : dataState === "stale"
              ? "stale"
              : "offline"
          }`}
        >
          <StatusDot
            status={
              dataState === "live"
                ? "healthy"
                : dataState === "stale"
                ? "stale"
                : "offline"
            }
          />
          {dataState === "live"
            ? "LIVE TELEMETRY STREAM"
            : dataState === "stale"
            ? "HISTORICAL TELEMETRY · INGEST PAUSED"
            : "NO LIVE DATA RECEIVED"}
        </span>

        <div className="analytics-toolbar-right">
          <div
            className="segmented-control"
            role="tablist"
            aria-label="Select telemetry metric"
          >
            {Object.entries(chartMetrics).map(([name, config]) => {
              const Icon = config.icon;
              return (
                <button
                  key={name}
                  className={metric === name ? "selected" : ""}
                  onClick={() => setMetric(name)}
                  type="button"
                  role="tab"
                  aria-selected={metric === name}
                >
                  <Icon size={14} />
                  <span>{name}</span>
                </button>
              );
            })}
          </div>
          <RangeSelect range={range} setRange={setRange} />
        </div>
      </div>

      <div className="analytics-metrics">
        <MetricCard
          label="MQTT Broker"
          value={metrics?.broker_connected ? "Connected" : "Disconnected"}
          icon={Radio}
          tone={metrics?.broker_connected ? "green" : "slate"}
          detail="Live backend connection status"
        />
        <MetricCard
          label="Telemetry Snapshots"
          value={metrics?.total_samples ?? 0}
          icon={Database}
          tone="cyan"
          detail="MQTT telemetry successfully stored in Firebase"
        />
      </div>

      <Panel className="analytics-chart-panel">
        <PanelHeading
          title={`${metric} from received telemetry`}
          meta={range.toUpperCase()}
          action={
            <span className="chart-current">
              <i
                className={
                  dataState === "live" ? "dot-live" : "dot-historical"
                }
              />
              <span className="current-source-label">
                {dataState === "live" ? "LIVE" : "HISTORICAL"}
              </span>
              <strong className="current-val-text">{currentValue}</strong>
            </span>
          }
        />
        {values.length ? (
          <>
            <LineChart values={values} label={`${metric} (${unit})`} />
            <div className="chart-axis">
              <span className="axis-point">
                <Clock size={13} />
                <span>Oldest received ({range})</span>
              </span>
              <span className="axis-point">
                <Clock size={13} />
                <span>Newest received</span>
              </span>
            </div>
          </>
        ) : (
          <EmptyState
            title="No measurements in this range"
            detail="Telemetry readings received in this time range will appear here."
          />
        )}
      </Panel>
    </div>
  );
}
