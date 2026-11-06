const apiBaseUrl = import.meta.env?.VITE_API_BASE_URL?.replace(/\/$/, "") ?? "";

async function getJson(path, signal) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: { Accept: "application/json" },
    signal,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function fetchDashboardData(signal) {
  const [data, devices] = await Promise.all([
    getJson("/api/v1/live?limit=500", signal),
    getJson("/api/v1/devices", signal),
  ]);
  return {
    ...data,
    devices,
    deviceStatus: data.device_status,
  };
}

export async function fetchHistoricalData(signal) {
  const [history, historyPredictions, historyAlerts, historyAnomalies] = await Promise.all([
    getJson("/api/v1/telemetry?limit=500", signal),
    getJson("/api/v1/predictions?limit=500", signal),
    getJson("/api/v1/alerts?limit=500", signal),
    getJson("/api/v1/anomalies?limit=500", signal),
  ]);
  return { history, historyPredictions, historyAlerts, historyAnomalies };
}

export async function fetchModelEvaluation(signal) {
  return getJson("/api/v1/model/evaluation", signal);
}

export async function fetchLiveComparison(signal) {
  return getJson("/api/v1/model/live-comparison", signal);
}

export async function fetchLiveMetrics(signal) {
  return getJson("/api/v1/model/live-metrics", signal);
}

export async function fetchProtocolComparison(signal) {
  return getJson("/api/v1/protocols/comparison", signal);
}

export async function fetchProtocolStatus(signal) {
  return getJson("/api/v1/protocols/status", signal);
}
