import { KPICard } from "../KPICard/KPICard.jsx";
import { Activity, AlertOctagon, Cable, Cpu, Wifi } from "lucide-react";
import "./KPICards.css";

export function KPICards({ devices, alerts }) {
  const online = devices.filter((device) => device.online).length;
  const unavailable = devices.filter((device) => !device.online).length;
  const activeAlerts = alerts.filter((alert) => alert.status === "Active");
  const activeSensorAlerts = activeAlerts.filter((alert) => alert.sensor);
  const metrics = [
    ["Configured devices", devices.length, "", Cpu, "cyan", "Devices reported by backend"],
    [
      "Online devices",
      online,
      "",
      Wifi,
      "green",
      devices.length
        ? `${Math.round((online / devices.length) * 100)}% currently online`
        : "No configured devices",
    ],
    ["Offline devices", unavailable, "", Cable, "slate", "Awaiting live MQTT messages"],
    [
      "Active anomalies",
      activeAlerts.length,
      "",
      Activity,
      "amber",
      "Persisted backend alerts",
    ],
    [
      "Sensor alerts",
      activeSensorAlerts.length,
      "",
      AlertOctagon,
      "red",
      "Sensor anomaly alerts",
    ],
  ];

  return (
    <section className="metrics-grid">
      {metrics.map(([label, value, suffix, icon, tone, detail], index) => (
        <KPICard
          key={label}
          label={label}
          value={value}
          suffix={suffix}
          icon={icon}
          tone={tone}
          detail={detail}
          delay={index * 35}
        />
      ))}
    </section>
  );
}
