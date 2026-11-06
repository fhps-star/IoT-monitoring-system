import { useEffect, useMemo, useState } from "react";
import { Check, X } from "lucide-react";
import { descriptions } from "./data/pageContent.js";
import { fetchDashboardData, fetchHistoricalData } from "./api/client.js";
import { mapDashboardData } from "./api/dashboard.js";
import { SectionTitle } from "./components/index.jsx";
import { RangeSelect } from "./components/shared.jsx";
import { applyAlertStatuses, useAlertStatusOverrides } from "./utils/alertStatus.js";
import { Header } from "./components/Header/Header.jsx";
import { Sidebar } from "./components/Sidebar/Sidebar.jsx";
import { DashboardSkeleton } from "./pages/Dashboard/Dashboard.jsx";
import { Dashboard } from "./pages/Dashboard/Dashboard.jsx";
import { DeviceManagement } from "./pages/DeviceManagement/DeviceManagement.jsx";
import { NetworkMonitor } from "./pages/NetworkMonitor/NetworkMonitor.jsx";
import { AlertCenter } from "./pages/AlertCenter/AlertCenter.jsx";
import { NetworkAnalytics } from "./pages/NetworkAnalytics/NetworkAnalytics.jsx";
import { ProtocolComparison } from "./pages/ProtocolComparison/ProtocolComparison.jsx";
import { DeviceDetails } from "./pages/DeviceDetails/DeviceDetails.jsx";


import { AIAnomalyDetection } from "./pages/AIAnomalyDetection/AIAnomalyDetection.jsx";
import "./App.css";

function App() {
  const [page, setPage] = useState("Dashboard"),
    [deviceId, setDeviceId] = useState(null),
    [returnPage, setReturnPage] = useState("Device Management");
  const [liveData, setLiveData] = useState({
    telemetry: [],
    predictions: [],
    xgboost_predictions: [],
    alerts: [],
    anomalies: [],
    deviceStatus: null,
    health: {
      components: { mqtt: "disconnected", firebase: "unavailable", model: "untrained" },
      mqtt_status: { subscription: "unknown", messages_received: 0 },
      data_state: "waiting",
    },
  });
  const [historyData, setHistoryData] = useState({
    history: [],
    historyPredictions: [],
    historyAlerts: [],
    historyAnomalies: [],
  });
  const [historyState, setHistoryState] = useState({ loading: true, error: null });
  const dashboardData = useMemo(
    () => mapDashboardData({ ...liveData, ...historyData }),
    [liveData, historyData],
  );
  const [connection, setConnection] = useState({
    api: "checking",
    mqtt: "unknown",
    firebase: "unknown",
    model: "untrained",
    xgboost: "unavailable",
    error: null,
    updatedAt: null,
  });
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("All status");
  const [range, setRange] = useState("Last 24 hours"),
    [booting, setBooting] = useState(true),
    [toast, setToast] = useState(""),
    [mobileNav, setMobileNav] = useState(false),
    [clock, setClock] = useState(new Date());
  const {
    devices,
    alerts: rawAlerts,

    telemetry,
    networkMetrics,
    dataState,
    hasHistoricalData,

    aiPredictions,
    aiAlerts: rawAiAlerts,
    xgboostStatus,
  } = dashboardData;
  const [alertStatusOverrides, setAlertStatus] = useAlertStatusOverrides();
  const alerts = useMemo(
    () => applyAlertStatuses(rawAlerts, alertStatusOverrides),
    [rawAlerts, alertStatusOverrides],
  );
  const aiAlerts = useMemo(
    () => applyAlertStatuses(rawAiAlerts, alertStatusOverrides),
    [rawAiAlerts, alertStatusOverrides],
  );

  useEffect(() => {
    const timer = window.setInterval(() => setClock(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let active = true;
    let timer;
    let controller;

    const refreshHistory = async () => {
      controller = new AbortController();
      try {
        const data = await fetchHistoricalData(controller.signal);
        if (!active) return;
        setHistoryData(data);
        setHistoryState({ loading: false, error: null });
      } catch (error) {
        if (!active || error.name === "AbortError") return;
        setHistoryState({ loading: false, error: error.message });
      } finally {
        if (active) timer = window.setTimeout(refreshHistory, 10000);
      }
    };

    refreshHistory();
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller?.abort();
    };
  }, []);

  useEffect(() => {
    let active = true;
    let timer;
    let controller;

    const refresh = async () => {
      controller = new AbortController();
      try {
        const data = await fetchDashboardData(controller.signal);
        if (!active) return;
        setLiveData(data);
        setConnection({
          api: "connected",
          mqtt: data.health.components?.mqtt ?? "disconnected",
          mqttSubscription: data.health.mqtt_status?.subscription ?? "unknown",
          mqttTopic: data.health.mqtt_status?.topic ?? "unknown",
          mqttSubscriptionError: data.health.mqtt_status?.subscription_error ?? null,
          firebase: data.health.components?.firebase ?? "unavailable",
          model: data.health.components?.model ?? "untrained",
          xgboost: data.health.components?.xgboost ?? "unavailable",
          dataState: data.health.data_state,
          lastMessageAt: data.health.mqtt_status?.last_message_at ?? null,
          error: null,
          updatedAt: new Date(),
        });
      } catch (error) {
        if (!active || error.name === "AbortError") return;
        setConnection({
          api: "disconnected",
          mqtt: "disconnected",
          mqttSubscription: "unknown",
          mqttTopic: null,
          mqttSubscriptionError: null,
          firebase: "unavailable",
          model: "untrained",
          xgboost: "unavailable",
          dataState: "mqtt_disconnected",
          error: error.message,
          updatedAt: null,
        });
      } finally {
        if (active) {
          setBooting(false);
          timer = window.setTimeout(refresh, 1000);
        }
      }
    };

    refresh();
    return () => {
      active = false;
      window.clearTimeout(timer);
      controller?.abort();
    };
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(""), 3000);
    return () => window.clearTimeout(timer);
  }, [toast]);
  const activeAlerts = [...alerts, ...aiAlerts].filter(
    (alert) => alert.status === "Active",
  ).length;

  const selectedDevice = devices.find((device) => device.id === deviceId);
  const filteredDevices = useMemo(
    () =>
      devices.filter(
        (device) =>
          `${device.id} ${device.name} ${device.asset} ${device.ip}`
            .toLowerCase()
            .includes(query.toLowerCase()) &&
          (filter === "All status" || device.status === filter.toLowerCase()),
      ),
    [devices, filter, query],
  );
  const openDevice = (id, origin = page) => {
    setReturnPage(origin);
    setDeviceId(id);
    setPage("Device Details");
    setMobileNav(false);
  };
  const navigate = (destination) => {
    setPage(destination);
    setDeviceId(null);
    setQuery("");
    setMobileNav(false);
  };
  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        activeAlerts={activeAlerts}
        mobileNav={mobileNav}
        navigate={navigate}
        setMobileNav={setMobileNav}
      />
      <div className="workspace">
        <Header
          clock={clock}
          activeAlerts={activeAlerts}
          navigate={navigate}
          setMobileNav={setMobileNav}
        />
        <main className="main-content">
          {booting && page === "Dashboard" ? (
            <DashboardSkeleton />
          ) : page === "Device Details" && selectedDevice ? (
            <DeviceDetails
              device={selectedDevice}
              onBack={() => navigate(returnPage)}
            />
          ) : (
            <>
              <SectionTitle title={page} subtitle={descriptions[page]}>

                {page === "Alert Center" && (
                  <RangeSelect range={range} setRange={setRange} />
                )}
              </SectionTitle>
              {page === "Dashboard" && (
                <Dashboard
                  devices={devices}

                  alerts={[...alerts, ...aiAlerts]}

                  networkMetrics={networkMetrics}
                  dataState={dataState}
                  hasHistoricalData={hasHistoricalData}
                  historyLoading={historyState.loading}
                  historyError={historyState.error}
                  firebaseReady={connection.firebase === "ready"}
                  navigate={navigate}
                  openDevice={openDevice}
                />
              )}
              {page === "Device Management" && (
                <DeviceManagement
                  devices={filteredDevices}
                  query={query}
                  setQuery={setQuery}
                  filter={filter}
                  setFilter={setFilter}
                  openDevice={(id) => openDevice(id, "Device Management")}
                />
              )}

              {page === "AI Anomaly Detection" && (
                <AIAnomalyDetection
                  predictions={aiPredictions}
                  modelStatus={xgboostStatus}
                />
              )}
              {page === "Network Monitor" && (
                <NetworkMonitor
                  devices={devices}
                  metrics={networkMetrics}
                  openDevice={(id) => openDevice(id, "Network Monitor")}
                />
              )}
              {page === "Alert Center" && (
                <AlertCenter
                  alerts={[...alerts, ...aiAlerts]}
                  devices={devices}
                  range={range}
                  setAlertStatus={setAlertStatus}
                  openDevice={(id) => openDevice(id, "Alert Center")}
                />
              )}
              {page === "Network Analytics" && (
                <NetworkAnalytics
                  range={range}
                  setRange={setRange}
                  devices={devices}
                  metrics={networkMetrics}
                  telemetry={telemetry}
                  now={clock.getTime()}
                  dataState={dataState}
                />
              )}
              {page === "Protocol Comparison" && (
                <ProtocolComparison />
              )}

            </>
          )}
        </main>
      </div>
      {toast && (
        <div className="toast">
          <span className="toast-icon">
            <Check size={15} />
          </span>
          {toast}
          <button
            onClick={() => setToast("")}
            aria-label="Dismiss notification"
          >
            <X size={14} />
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
