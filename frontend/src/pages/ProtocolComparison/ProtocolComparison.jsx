import { useState, useEffect } from "react";
import { PanelHeading } from "../../components/index.jsx";
import { Panel } from "../../components/shared.jsx";
import { fetchProtocolComparison } from "../../api/client.js";
import "./ProtocolComparison.css";

const COMPARISON_ROWS = [
  {
    key: "transport",
    metric: "Transport Protocol",
    aliases: ["transport", "transport_protocol"],
    mqtt: "TCP",
    http: "TCP",
    coap: "UDP",
    isTransport: true,
  },
  {
    key: "avg_latency",
    metric: "Average Latency (ms)",
    aliases: ["avg_latency", "average_latency"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "min_latency",
    metric: "Minimum Latency (ms)",
    aliases: ["min_latency", "minimum_latency"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "max_latency",
    metric: "Maximum Latency (ms)",
    aliases: ["max_latency", "maximum_latency"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "packet_loss",
    metric: "Packet Loss (%)",
    aliases: ["packet_loss"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "throughput",
    metric: "Throughput (msg/sec)",
    aliases: ["throughput"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "bandwidth",
    metric: "Bandwidth Usage (KB)",
    aliases: ["bandwidth", "bandwidth_usage"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "overhead",
    metric: "Message Overhead (bytes)",
    aliases: ["overhead", "message_overhead"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "memory",
    metric: "Memory Usage (bytes)",
    aliases: ["memory", "memory_usage"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "cpu",
    metric: "CPU Usage (%)",
    aliases: ["cpu", "cpu_usage"],
    mqtt: null,
    http: null,
    coap: null,
  },
  {
    key: "reconnect_time",
    metric: "Reconnection Time (ms)",
    aliases: ["reconnect_time", "reconnection_time"],
    mqtt: null,
    http: null,
    coap: null,
  },
];

export function ProtocolComparison({ protocolData: propData }) {
  const [liveData, setLiveData] = useState(null);

  useEffect(() => {
    let mounted = true;
    const controller = new AbortController();

    async function loadMetrics() {
      try {
        const data = await fetchProtocolComparison(controller.signal);
        if (mounted && data) {
          setLiveData(data);
        }
      } catch (err) {
        if (mounted && err.name !== "AbortError") {
          // silently handle polling network errors
        }
      }
    }

    loadMetrics();
    const intervalId = setInterval(loadMetrics, 2000);

    return () => {
      mounted = false;
      controller.abort();
      clearInterval(intervalId);
    };
  }, []);

  const activeData = propData && Object.keys(propData).length > 0 ? propData : liveData;

  const resolveBackendValue = (row, protocolKey) => {
    if (!activeData || typeof activeData !== "object") return null;

    // Check direct key-indexed structure: activeData[row.key]?.[protocolKey]
    if (activeData[row.key]?.[protocolKey] !== undefined && activeData[row.key]?.[protocolKey] !== null) {
      return activeData[row.key][protocolKey];
    }

    // Check protocol-indexed structure: activeData[protocolKey] or activeData.protocols?.[protocolKey]
    const protoObj = activeData[protocolKey] || activeData.protocols?.[protocolKey];
    if (protoObj && typeof protoObj === "object") {
      const keysToCheck = row.aliases || [row.key];
      for (const k of keysToCheck) {
        if (protoObj[k] !== undefined && protoObj[k] !== null) {
          return protoObj[k];
        }
      }
    }

    return null;
  };

  const renderValue = (row, protocolKey) => {
    const backendVal = resolveBackendValue(row, protocolKey);
    if (backendVal !== null && backendVal !== undefined) {
      if (row.isTransport) {
        return (
          <span className={`transport-pill ${String(backendVal).toLowerCase()}`}>
            {backendVal}
          </span>
        );
      }
      return <span className="metric-val">{backendVal}</span>;
    }

    // Static default value (e.g. TCP / UDP for transport row)
    const val = row[protocolKey];
    if (val !== null && val !== undefined) {
      if (row.isTransport) {
        return (
          <span className={`transport-pill ${String(val).toLowerCase()}`}>
            {val}
          </span>
        );
      }
      return <span className="metric-val">{val}</span>;
    }

    // Unavailable measurement placeholder
    return <span className="metric-empty">—</span>;
  };

  return (
    <div className="page-stack protocol-comparison-page">
      <Panel className="protocol-table-panel">
        <PanelHeading
          title="Communication Protocol Performance"
          meta="3 PROTOCOLS"
        />
        <div className="table-responsive">
          <table className="protocol-comparison-table">
            <thead>
              <tr>
                <th className="th-metric">Performance Metrics</th>
                <th className="th-protocol">
                  <div className="protocol-header-cell">
                    <span className="proto-name">MQTT</span>
                    <span className="proto-type-tag">Pub / Sub</span>
                  </div>
                </th>
                <th className="th-protocol">
                  <div className="protocol-header-cell">
                    <span className="proto-name">HTTP</span>
                    <span className="proto-type-tag">Request / Response</span>
                  </div>
                </th>
                <th className="th-protocol">
                  <div className="protocol-header-cell">
                    <span className="proto-name">CoAP</span>
                    <span className="proto-type-tag">Constrained REST</span>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {COMPARISON_ROWS.map((row) => (
                <tr key={row.key}>
                  <td className="td-metric">
                    <strong>{row.metric}</strong>
                  </td>
                  <td className="td-val">{renderValue(row, "mqtt")}</td>
                  <td className="td-val">{renderValue(row, "http")}</td>
                  <td className="td-val">{renderValue(row, "coap")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

