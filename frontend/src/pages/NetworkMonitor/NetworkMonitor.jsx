import { PanelHeading, MetricCard, StatusDot, NetworkTopology } from "../../components/index.jsx";
import { Panel, TopologyLegend } from "../../components/shared.jsx";
import { Activity, Cable, Clock, Network, Server } from "lucide-react";
import "./NetworkMonitor.css";

export function NetworkMonitor({ devices, metrics, openDevice }) {
  const online = devices.filter((device) => device.online).length;
  const brokerConnected = metrics?.broker_connected === true;
  const streamStatus = online
    ? "LIVE DATA"
    : brokerConnected
      ? "WAITING FOR DEVICE"
      : "MQTT DISCONNECTED";

  return (
    <div className="page-stack network-monitor-page">
      <div className="network-metrics-grid">
        <MetricCard
          label="Connected Devices"
          value={`${online} / ${devices.length}`}
          icon={Network}
          tone={online > 0 ? "green" : "slate"}
          detail="Online nodes in fleet"
        />
        <MetricCard
          label="Avg. Latency"
          value={
            Number.isFinite(metrics?.avg_latency_ms)
              ? `${Math.round(metrics.avg_latency_ms)} ms`
              : "—"
          }
          icon={Server}
          tone="cyan"
          detail="Gateway round-trip time"
        />
        <MetricCard
          label="Messages Received"
          value={metrics?.total_messages ?? "—"}
          icon={Cable}
          tone="blue"
          detail="Broker ingest counter"
        />
        <MetricCard
          label="Last Telemetry"
          value={
            metrics?.last_message_at
              ? new Date(metrics.last_message_at).toLocaleTimeString()
              : "None"
          }
          icon={Activity}
          tone="amber"
          detail="Latest heartbeat timestamp"
        />
      </div>

      <Panel className="full-topology-panel">
        <PanelHeading
          title="IIoT Edge Topology"
          meta={
            <span
              className={`topology-stream-pill ${
                online ? "online" : brokerConnected ? "stale" : "offline"
              }`}
            >
              <StatusDot
                status={online ? "healthy" : brokerConnected ? "stale" : "offline"}
              />
              {streamStatus}
            </span>
          }
          action={
            <div className="topology-tools">
              <span className="text-note">
                <Clock size={13} />
                <span>
                  {metrics?.last_message_at
                    ? `Last message: ${new Date(
                        metrics.last_message_at,
                      ).toLocaleString()}`
                    : "No telemetry received yet"}
                </span>
              </span>
            </div>
          }
        />
        <NetworkTopology
          devices={devices}
          onOpenDevice={openDevice}
          gatewayConnected={brokerConnected}
        />
        <TopologyLegend />
      </Panel>
    </div>
  );
}
