import { SectionTitle, PanelHeading, StatusDot, HealthScore, NetworkTopology } from "../../components/index.jsx";
import { EmptyState, Panel, TopologyLegend } from "../../components/shared.jsx";
import { AttentionRequired } from "../../components/AttentionRequired/AttentionRequired.jsx";
import { KPICards } from "../../components/KPICards/KPICards.jsx";
import { ExternalLink } from "lucide-react";
import { descriptions } from "../../data/pageContent.js";
import "./Dashboard.css";

export function DashboardSkeleton() {
  return (
    <div className="page-stack" aria-label="Loading dashboard" aria-busy="true">
      <SectionTitle title="Dashboard" subtitle={descriptions.Dashboard}>
        <span className="skeleton skeleton-live" />
      </SectionTitle>
      <div className="metrics-grid">
        {Array.from({ length: 5 }, (_, index) => (
          <div className="metric-card skeleton-card" key={index}>
            <span className="skeleton skeleton-label" />
            <span className="skeleton skeleton-value" />
            <span className="skeleton skeleton-detail" />
          </div>
        ))}
      </div>
      <div className="dashboard-feature-grid">
        <div className="health-card skeleton-panel">
          <span className="skeleton skeleton-label" />
          <span className="skeleton skeleton-value" />
          <span className="skeleton skeleton-detail" />
        </div>
        <div className="panel skeleton-panel">
          <span className="skeleton skeleton-label" />
          <span className="skeleton skeleton-value" />
        </div>
        <div className="panel skeleton-panel">
          <span className="skeleton skeleton-label" />
          <span className="skeleton skeleton-detail" />
        </div>
      </div>
    </div>
  );
}

export function Dashboard({
  devices,
  alerts,
  networkMetrics,
  dataState,
  hasHistoricalData,
  historyLoading,
  historyError,
  firebaseReady,
  navigate,
  openDevice,
}) {
  return (
    <div className="page-stack dashboard-page">
      {!devices.length &&
        !historyLoading &&
        !historyError &&
        firebaseReady &&
        !hasHistoricalData && (
          <Panel>
            <EmptyState
              title="No Telemetry Data Available"
              detail={`Backend received ${
                networkMetrics?.total_messages ?? 0
              } MQTT messages. It is subscribed to ${
                networkMetrics?.mqtt_broker ?? "the configured broker"
              }:${networkMetrics?.mqtt_port ?? "?"} / ${
                networkMetrics?.mqtt_topic ?? "the configured topic"
              }. Verify the ESP32 publishes to this exact endpoint.`}
            />
          </Panel>
        )}
      <KPICards devices={devices} alerts={alerts} />
      <section className="dashboard-feature-grid">
        <HealthScore
          devices={devices}
          metrics={networkMetrics}
          dataState={dataState}
        />
        <Panel className="dashboard-topology-panel">
          <PanelHeading
            title="Motor network topology"
            meta={
              <span className={`status-pill ${dataState === "live" ? "online" : "offline"}`}>
                <StatusDot status={dataState === "live" ? "healthy" : "offline"} />
                {dataState === "live" ? "LIVE" : "OFFLINE"}
              </span>
            }
            action={
              <button
                className="icon-button"
                title="Open network monitor"
                onClick={() => navigate("Network Monitor")}
                type="button"
              >
                <ExternalLink size={15} />
              </button>
            }
          />
          <NetworkTopology
            devices={devices.slice(0, 5)}
            dashboard
            gatewayConnected={networkMetrics?.broker_connected}
            onOpenDevice={(id) => openDevice(id, "Dashboard")}
          />
          <TopologyLegend compact />
        </Panel>
        <AttentionRequired
          devices={devices}
          navigate={navigate}
          openDevice={openDevice}
        />
      </section>
    </div>
  );
}
