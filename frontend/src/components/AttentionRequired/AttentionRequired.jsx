import { PanelHeading } from "../PageHeading/PageHeading.jsx";
import { Panel } from "../shared.jsx";
import { AlertTriangle, ArrowRight, ChevronRight } from "lucide-react";
import { statusLabel } from "../../utils/status.js";
import "./AttentionRequired.css";

export function AttentionRequired({ devices, navigate, openDevice }) {
  const attention = devices.flatMap((device) => {
    const sensorFault = device.sensors.find((sensor) =>
      ["critical", "offline"].includes(sensor.status),
    );
    if (device.online && !sensorFault) return [];
    const isOffline = !device.online || device.status === "offline" || device.status === "stale";
    const attentionStatus = isOffline
      ? "offline"
      : device.status === "warning"
      ? "warning"
      : "critical";
    const attentionMessage =
      device.lastSeen === "No telemetry yet"
        ? "Waiting for first telemetry message"
        : sensorFault
        ? `${sensorFault.name} disconnected`
        : isOffline
        ? "No live messages received · device offline"
        : device.status === "warning"
        ? "Device reported warning status"
        : "Device health critical";
    return [{ ...device, attentionStatus, attentionMessage }];
  });

  return (
    <Panel className="attention-panel">
      <PanelHeading
        title="Requires attention"
        meta={`${attention.length} ${
          attention.length === 1 ? "ACTION" : "ACTIONS"
        }`}
        action={
          <button
            className="text-action"
            onClick={() => navigate("Alert Center")}
          >
            View all <ArrowRight size={13} />
          </button>
        }
      />
      {attention.map((device) => (
        <button
          className="attention-row"
          key={device.id}
          onClick={() => openDevice(device.id, "Dashboard")}
        >
          <span className={`attention-symbol ${device.attentionStatus}`}>
            <AlertTriangle size={15} />
          </span>
          <span className="attention-copy">
            <strong>
              {device.id} · {device.asset}
            </strong>
            <small>{device.attentionMessage}</small>
            <span className="attention-meta">
              <span className={`attention-severity ${device.attentionStatus}`}>
                {device.attentionStatus === "offline"
                  ? "Offline"
                  : statusLabel(device.attentionStatus)}
              </span>
              <time>{device.lastSeen}</time>
            </span>
          </span>
          <ChevronRight size={15} className="row-chevron" />
        </button>
      ))}
    </Panel>
  );
}
