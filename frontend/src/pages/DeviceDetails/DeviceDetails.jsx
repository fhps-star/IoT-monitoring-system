import { useState } from "react";
import { PanelHeading, MetricCard, Badge } from "../../components/index.jsx";
import { Panel, SensorIcon } from "../../components/shared.jsx";
import {
  Activity,
  ArrowLeft,
  Check,
  Copy,
  Cpu,
  Gauge,
  Network,
  Radio,
  Server,
  Wifi,
} from "lucide-react";
import { statusLabel } from "../../utils/status.js";
import "./DeviceDetails.css";

function getSensorMeta(sensor) {
  const nameLower = (sensor.name || "").toLowerCase();
  if (sensor.kind === "temperature" || nameLower.includes("temp")) {
    return { channel: "CH-01", type: "PT100 Core Thermal RTD" };
  }
  if (sensor.kind === "vibration" || nameLower.includes("vib")) {
    return { channel: "CH-02", type: "Piezoelectric Accelerometer" };
  }
  if (nameLower.includes("speed") || nameLower.includes("rpm")) {
    return { channel: "CH-04", type: "Optical Pulse Tachometer" };
  }
  if (sensor.kind === "current" || nameLower.includes("current")) {
    return { channel: "CH-03", type: "Hall-Effect Current (CT)" };
  }
  return {
    channel: sensor.channel || "CH-0X",
    type: sensor.model || "Telemetry Ingest Feed",
  };
}

export function DeviceDetails({ device, onBack }) {
  const [copiedIp, setCopiedIp] = useState(false);

  const handleCopyIp = (ip) => {
    if (!ip || ip === "—") return;
    navigator.clipboard?.writeText(ip);
    setCopiedIp(true);
    setTimeout(() => setCopiedIp(false), 2000);
  };

  return (
    <div className="page-stack device-details-page">
      <div className="detail-header-bar">
        <button className="back-link" onClick={onBack}>
          <ArrowLeft size={14} /> Back to devices
        </button>
        <div className="detail-title-row">
          <div className="title-group">
            <h1>{device.name}</h1>
            <span className="asset-type-badge">Industrial Asset · Motor Node</span>
          </div>
          <div className="detail-actions">
            <Badge value={statusLabel(device.status)} />
          </div>
        </div>
      </div>

      <div className="detail-metrics">
        <MetricCard
          label="Device health"
          value={device.deviceHealth ? statusLabel(device.deviceHealth) : "—"}
          icon={Gauge}
          tone={
            device.status === "critical"
              ? "red"
              : device.status === "warning"
                ? "amber"
                : device.status === "online"
                  ? "green"
                  : "slate"
          }
          detail="Connectivity and packet-loss status"
        />
        <MetricCard
          label="Latency"
          value={Number.isFinite(device.latency) ? `${device.latency} ms` : "—"}
          icon={Activity}
          tone="cyan"
          detail="Latest ESP32 MQTT round-trip"
        />
        <MetricCard
          label="Packet loss"
          value={Number.isFinite(device.packetLoss) ? `${device.packetLoss.toFixed(2)}%` : "—"}
          icon={Network}
          tone={
            !Number.isFinite(device.packetLoss)
              ? "slate"
              : device.packetLoss > 1
                ? "amber"
                : "green"
          }
          detail="Estimated from missing sequence numbers"
        />
        <MetricCard
          label="MQTT connection"
          value={
            device.mqttConnected === null
              ? "—"
              : device.mqttConnected
                ? "Connected"
                : "Disconnected"
          }
          icon={Network}
          tone={device.mqttConnected === false ? "amber" : device.mqttConnected ? "green" : "slate"}
          detail="ESP32-reported status"
        />
      </div>

      <div className="details-grid">
        <div className="details-main">
          <Panel>
            <PanelHeading
              title="Attached sensors"
              meta={`${device.sensors.length} MEASURED`}
            />
            <div className="section-explainer-bar">
              <div className="explainer-source">
                <Radio size={13} className="explainer-icon" />
                <span>
                  {!device.telemetryRecordId
                    ? "No sensor telemetry received for this device yet."
                    : device.historical
                    ? `Historical telemetry · Record ${device.telemetryRecordId}`
                    : `Live MQTT telemetry · Record ${device.telemetryRecordId}`}
                </span>
              </div>
              <span className="telemetry-protocol-badge">MQTT Payload</span>
            </div>
            <div className="sensor-list">
              {device.sensors.map((sensor) => {
                const meta = getSensorMeta(sensor);
                return (
                  <div
                    className={`sensor-row ${sensor.status === "critical" ? "sensor-alert" : ""}`}
                    key={sensor.name}
                  >
                    <div className={`sensor-icon ${sensor.status} kind-${sensor.kind}`}>
                      <SensorIcon kind={sensor.kind} />
                    </div>
                    <div className="sensor-copy">
                      <div className="sensor-name-line">
                        <strong>{sensor.name}</strong>
                        <span className="sensor-channel-tag">{meta.channel}</span>
                      </div>
                      <span className="sensor-meta-caption">{meta.type}</span>
                    </div>
                    <div className="sensor-reading">
                      <strong className="reading-value">{sensor.value}</strong>
                      <span className="reading-unit">{sensor.unit}</span>
                    </div>
                    <Badge value={statusLabel(sensor.status)} />
                  </div>
                );
              })}
            </div>
          </Panel>
        </div>

        <div className="details-side">
          <Panel>
            <PanelHeading title="Device information" />
            <div className="info-sections">
              <div className="info-group">
                <div className="info-group-title">
                  <Cpu size={12} />
                  <span>Asset & Hardware</span>
                </div>
                <div className="info-row">
                  <span>Industrial asset</span>
                  <strong>{device.asset || device.name || "—"}</strong>
                </div>
                <div className="info-row">
                  <span>Device model</span>
                  <strong>{device.model || "ESP32"}</strong>
                </div>
              </div>

              <div className="info-group">
                <div className="info-group-title">
                  <Wifi size={12} />
                  <span>Network & Connectivity</span>
                </div>
                <div className="info-row">
                  <span>IP address</span>
                  <div className="ip-cell">
                    <code className="mono-ip">{device.ip || "—"}</code>
                    {device.ip && device.ip !== "—" && (
                      <button
                        className="copy-ip-btn"
                        onClick={() => handleCopyIp(device.ip)}
                        title={copiedIp ? "Copied!" : "Copy IP address"}
                        type="button"
                        aria-label="Copy IP address"
                      >
                        {copiedIp ? <Check size={12} className="copy-ok" /> : <Copy size={12} />}
                      </button>
                    )}
                  </div>
                </div>
                <div className="info-row">
                  <span>Wi-Fi</span>
                  {device.wifiConnected === null ? (
                    <strong>—</strong>
                  ) : (
                    <span className={`status-pill ${device.wifiConnected ? "online" : "offline"}`}>
                      <i className="status-dot" />
                      {device.wifiConnected ? "Connected" : "Disconnected"}
                    </span>
                  )}
                </div>
                <div className="info-row">
                  <span>MQTT broker</span>
                  {device.mqttConnected === null ? (
                    <strong>—</strong>
                  ) : (
                    <span className={`status-pill ${device.mqttConnected ? "online" : "offline"}`}>
                      <i className="status-dot" />
                      {device.mqttConnected ? "Connected" : "Disconnected"}
                    </span>
                  )}
                </div>
                <div className="info-row">
                  <span>Protocol</span>
                  <strong>{device.protocol || "MQTT"}</strong>
                </div>
              </div>

              <div className="info-group">
                <div className="info-group-title">
                  <Server size={12} />
                  <span>Telemetry & Diagnostics</span>
                </div>
                <div className="info-row">
                  <span>Latest MQTT RTT</span>
                  <strong>
                    {Number.isFinite(device.latency) ? `${device.latency} ms` : "—"}
                  </strong>
                </div>
                <div className="info-row">
                  <span>Average MQTT RTT</span>
                  <strong>
                    {Number.isFinite(device.averageLatency)
                      ? `${device.averageLatency.toFixed(1)} ms`
                      : "—"}
                  </strong>
                </div>
                <div className="info-row">
                  <span>Estimated packet loss</span>
                  <strong className={device.packetLoss > 1 ? "loss-warning" : ""}>
                    {Number.isFinite(device.packetLoss)
                      ? `${device.packetLoss.toFixed(2)}%`
                      : "—"}
                  </strong>
                </div>
                <div className="info-row">
                  <span>Last sequence</span>
                  <strong>
                    {Number.isFinite(device.sequence) ? `#${device.sequence}` : "—"}
                  </strong>
                </div>
                <div className="info-row">
                  <span>Last heartbeat</span>
                  <strong className="timestamp-value">{device.lastSeen || "—"}</strong>
                </div>
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
