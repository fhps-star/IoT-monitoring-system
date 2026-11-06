import { Badge } from "../../components/index.jsx";
import { StatusSwitch } from "../../components/StatusSwitch/StatusSwitch.jsx";
import { Panel, TableToolbar, FilterSelect, LinkButton, EmptyState } from "../../components/shared.jsx";

import { AlertTriangle, CheckCheck, Eye } from "lucide-react";
import { useState } from "react";
import { alertAgeMinutes } from "../../utils/time.js";
import "./AlertCenter.css";

export function AlertCenter({
  alerts,
  devices,
  range,
  setAlertStatus,
  openDevice,
}) {
  const [severity, setSeverity] = useState("All severities"),
    [device, setDevice] = useState("All devices"),
    [state, setState] = useState("All states"),
    [sensorType, setSensorType] = useState("All sources");
  const maxAgeMinutes =
    range === "Last hour" ? 60 : range === "Last 24 hours" ? 1440 : 10080;
  const inRange = alerts.filter(
    (alert) => alertAgeMinutes(alert.age) <= maxAgeMinutes,
  );
  const countByStatus = (status) =>
    inRange.filter((alert) => alert.status === status).length;
  const rows = inRange.filter(
    (alert) =>
      (severity === "All severities" ||
        alert.severity.toLowerCase() === severity.toLowerCase()) &&
      (device === "All devices" || alert.device === device) &&
      (state === "All states" ||
        alert.status.toLowerCase() === state.toLowerCase()) &&
      (sensorType === "All sources" ||
        (sensorType === "Sensor alerts" ? Boolean(alert.sensor) : !alert.sensor)),
  );
  return (
    <div className="page-stack alert-center-page">
      <div className="alert-stats">
        <div className="alert-stat tone-active">
          <span className="alert-stat-icon">
            <AlertTriangle size={16} />
          </span>
          <div>
            <span>Active alerts</span>
            <strong>{countByStatus("Active").toLocaleString()}</strong>
            <small>Require operator review</small>
          </div>
        </div>
        <div className="alert-stat tone-acknowledged">
          <span className="alert-stat-icon">
            <Eye size={16} />
          </span>
          <div>
            <span>Acknowledged</span>
            <strong>{countByStatus("Acknowledged").toLocaleString()}</strong>
            <small>Being investigated</small>
          </div>
        </div>
        <div className="alert-stat tone-resolved">
          <span className="alert-stat-icon">
            <CheckCheck size={16} />
          </span>
          <div>
            <span>Resolved anomalies</span>
            <strong>{countByStatus("Resolved").toLocaleString()}</strong>
            <small>Marked as resolved</small>
          </div>
        </div>
      </div>
      <Panel>
        <TableToolbar
          title="Alert register"
          count={rows.length}
          controls={
            <>
              <FilterSelect
                value={severity}
                onChange={setSeverity}
                options={["All severities", "Warning", "High", "Critical"]}
              />
              <FilterSelect
                value={device}
                onChange={setDevice}
                options={["All devices", ...devices.map((d) => d.id)]}
              />
              <FilterSelect
                value={sensorType}
                onChange={setSensorType}
                options={["All sources", "Sensor alerts", "Network alerts"]}
              />
              <FilterSelect
                value={state}
                onChange={setState}
                options={["All states", "Active", "Acknowledged", "Resolved"]}
              />
            </>
          }
        />
        <div className="table-scroll">
          <table className="data-table alert-table">
            <thead>
              <tr>
                <th>Alert</th>
                <th>Source</th>
                <th>Severity</th>
                <th>Time</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((alert) => (
                <tr key={alert.id}>
                  <td>
                    <strong>{alert.title}</strong>
                    <small className="alert-meta">
                      <span className="alert-id">{alert.id}</span>
                      <span>{alert.description}</span>
                    </small>
                  </td>
                  <td>
                    <LinkButton
                      id={`${alert.device}${alert.sensor ? ` / ${alert.sensor}` : ""}`}
                      deviceId={alert.device}
                      openDevice={openDevice}
                    />
                  </td>
                  <td>
                    {alert.severity === "—" ? (
                      <span className="table-muted">Not rated</span>
                    ) : (
                      <Badge value={alert.severity} />
                    )}
                  </td>
                  <td title={alert.firstSeen}>{alert.age}</td>
                  <td>
                    <StatusSwitch
                      value={alert.status}
                      label={`Status of alert ${alert.id}`}
                      onChange={(next) => setAlertStatus(alert.id, next)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!rows.length && (
            <EmptyState
              title="No alerts match these filters"
              detail="Reset one or more filters to see more results."
            />
          )}
        </div>
      </Panel>
    </div>
  );
}
