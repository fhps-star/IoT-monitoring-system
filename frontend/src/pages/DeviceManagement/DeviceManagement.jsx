import { DeviceTable } from "../../components/index.jsx";
import { Panel } from "../../components/shared.jsx";
import { ChevronDown, Download, Filter, Search } from "lucide-react";
import { statusLabel } from "../../utils/status.js";
import "./DeviceManagement.css";

export function DeviceManagement({
  devices,
  query,
  setQuery,
  filter,
  setFilter,
  openDevice,
}) {
  return (
    <div className="page-stack device-management-page">
      <Panel>
        <div className="table-toolbar">
          <div className="table-count">
            <strong>{devices.length} {devices.length === 1 ? "device" : "devices"}</strong>
            <span className="device-online-count">
              · {devices.filter((device) => device.online).length} connected
            </span>
          </div>
          <div className="table-controls">
            <label className="search-field">
              <Search size={15} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search devices"
              />
              <kbd>⌘ K</kbd>
            </label>
            <label className="filter-control">
              <Filter size={14} />
              <select
                aria-label="Filter devices by status"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              >
                <option>All status</option>
                {["online", "warning", "critical", "stale", "offline"].map((s) => (
                  <option value={s} key={s}>
                    {statusLabel(s)}
                  </option>
                ))}
              </select>
              <ChevronDown size={13} />
            </label>
            <button
              className="icon-button"
              title="Export device list"
              onClick={() => window.print()}
            >
              <Download size={15} />
            </button>
          </div>
        </div>
        <DeviceTable devices={devices} onOpenDevice={openDevice} />
      </Panel>
      <div className="device-summary-strip">
        <span>Devices with stored MQTT telemetry: <strong>{devices.length}</strong></span>
        <span>Protocol: <strong>MQTT</strong></span>
        <span>Last telemetry: <strong>{devices.map((device) => device.lastSeen).find((value) => value !== "No telemetry yet") || "None received"}</strong></span>
      </div>
    </div>
  );
}
