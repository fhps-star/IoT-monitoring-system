import assert from "node:assert/strict";
import test from "node:test";
import { fetchDashboardData } from "./client.js";

test("fetches live telemetry and backend device health together", async () => {
  const originalFetch = globalThis.fetch;
  const paths = [];
  globalThis.fetch = async (url) => {
    paths.push(url);
    const body = url.includes("/api/v1/devices")
      ? [{ device_id: "motor_01", status: "HEALTHY" }]
      : {
        telemetry: [],
        health: {
          components: {},
          mqtt_status: {},
          data_state: "waiting",
        },
      };
    return {
      ok: true,
      json: async () => body,
    };
  };

  try {
    const result = await fetchDashboardData();

    assert.deepEqual(paths.sort(), [
      "/api/v1/devices",
      "/api/v1/live?limit=500",
    ]);
    assert.equal(result.devices[0].device_id, "motor_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});
