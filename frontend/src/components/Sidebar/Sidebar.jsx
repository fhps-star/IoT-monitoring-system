import {
  Activity,
  Bell,
  Cpu,
  GitCompare,
  LayoutDashboard,
  Network,
  SlidersHorizontal,
  X,
} from "lucide-react";
import "./Sidebar.css";

const navGroups = [
  { label: null, items: [["Dashboard", LayoutDashboard]] },
  {
    label: "MONITORING",
    items: [
      ["Device Management", Cpu],
      ["Network Monitor", Network],
    ],
  },
  {
    label: "COMPARISONS",
    items: [
      ["AI Anomaly Detection", Activity],
      ["Protocol Comparison", GitCompare],
    ],
  },
  {
    label: "ALERTS",
    items: [
      ["Alert Center", Bell],
    ],
  },
  {
    label: "ANALYTICS",
    items: [
      ["Network Analytics", SlidersHorizontal],
    ],
  },
];

export function Sidebar({ page, activeAlerts, mobileNav, navigate, setMobileNav }) {
  return (
    <>
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark">
            <Activity size={19} strokeWidth={2.5} />
          </div>
          <div className="brand-name">
            INDUSTRIA<span>NETWORK INTELLIGENCE</span>
          </div>
          <button
            className="icon-button mobile-close"
            onClick={() => setMobileNav(false)}
            aria-label="Close navigation"
          >
            <X size={17} />
          </button>
        </div>
        <nav className="side-navigation" aria-label="Main navigation">
          {navGroups.map((group) => (
            <div className="nav-group" key={group.label ?? group.items[0][0]}>
              {group.label && <div className="nav-label">{group.label}</div>}
              {group.items.map(([name, Icon]) => (
                <button
                  key={name}
                  className={`nav-item ${
                    page === name ||
                    ((name === "Alert Center" || name === "Alerts") && (page === "Alert Center" || page === "Alerts")) ||
                    (name === "Device Management" && page === "Device Details")
                      ? "active"
                      : ""
                  }`}
                  onClick={() => navigate(name)}
                >
                  <Icon size={16} strokeWidth={1.8} />
                  <span>{name}</span>
                  {(name === "Alert Center" || name === "Alerts") && activeAlerts > 0 && (
                    <em>{activeAlerts}</em>
                  )}
                </button>
              ))}
            </div>
          ))}
        </nav>
      </aside>
      {mobileNav && (
        <button
          className="mobile-scrim"
          aria-label="Close menu"
          onClick={() => setMobileNav(false)}
        />
      )}
    </>
  );
}
