export function statusLabel(status) {
  return String(status ?? "unknown").charAt(0).toUpperCase()
    + String(status ?? "unknown").slice(1);
}

export function healthColor(status) {
  return status === "critical"
    ? "red"
    : status === "warning"
      ? "amber"
      : status === "offline"
        ? "slate"
        : "green";
}
