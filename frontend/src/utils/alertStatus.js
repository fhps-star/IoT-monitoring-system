import { useCallback, useEffect, useState } from "react";

export const ALERT_STATUSES = ["Active", "Acknowledged", "Resolved"];

const STORAGE_KEY = "iot.alertStatusOverrides";

function loadOverrides() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}");
    return stored && typeof stored === "object" ? stored : {};
  } catch {
    return {};
  }
}

/**
 * Operator-chosen alert statuses, kept in localStorage because the backend has
 * no endpoint for updating them. Held once in App so every page, the sidebar
 * badge and the header bell stay in sync.
 */
export function useAlertStatusOverrides() {
  const [overrides, setOverrides] = useState(loadOverrides);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    } catch {
      /* storage unavailable; changes stay in memory only */
    }
  }, [overrides]);

  const setAlertStatus = useCallback(
    (id, status) => setOverrides((current) => ({ ...current, [id]: status })),
    [],
  );

  return [overrides, setAlertStatus];
}

export function applyAlertStatuses(alerts = [], overrides = {}) {
  return alerts.map((alert) => {
    const status = overrides[alert.id];
    return ALERT_STATUSES.includes(status)
      ? { ...alert, status, state: status }
      : alert;
  });
}
