import assert from "node:assert/strict";
import test from "node:test";
import { mapDashboardData } from "./dashboard.js";

const receivedAt = "2026-10-31T12:00:00.000Z";
const telemetry = {
  id: "firebase-telemetry-1",
  source: "mqtt",
  device_id: "motor-node-01",
  motor_id: "motor_01",
  received_at: receivedAt,
  raw_payload: {
    device_id: "motor-node-01",
    temperature_c: 41.25,
    estimated_vibration: 0.103,
    current_a: 0.42,
    motor_speed: 120,
  },
  sensor_values: { temperature: 41.25, vibration: 0.103, current: 0.42 },
  features: [41.25, 0.103, 0.42],
};

function dashboardData(overrides = {}) {
  return {
    telemetry: [],
    predictions: [],
    alerts: [],
    history: [],
    historyPredictions: [],
    historyAlerts: [],
    health: {
      components: { mqtt: "connected", firebase: "ready", model: "untrained" },
      mqtt_status: { subscription: "subscribed", messages_received: 1 },
      data_state: "waiting",
    },
    ...overrides,
  };
}

test("shows current firmware readings only with a traceable Firebase MQTT record", () => {
  const result = mapDashboardData(dashboardData({
    telemetry: [telemetry],
    health: {
      components: { mqtt: "connected", firebase: "ready", model: "untrained" },
      mqtt_status: {
        subscription: "subscribed",
        messages_received: 1,
        broker: "broker.hivemq.com",
        port: 1883,
        topic: "iiot/motor_01/telemetry",
        last_message_at: receivedAt,
      },
      data_state: "live",
    },
  }));
  const [device] = result.devices;

  assert.equal(device.status, "online");
  assert.equal(device.historical, false);
  assert.equal(device.telemetryRecordId, "firebase-telemetry-1");
  assert.deepEqual(
    device.sensors.map(({ name, value, unit }) => ({ name, value, unit })),
    [
      { name: "Temperature", value: "41.25", unit: "°C" },
      { name: "Vibration", value: "0.103", unit: "m/s²" },
      { name: "Motor current", value: "0.42", unit: "A" },
      { name: "Motor speed", value: "120", unit: "RPM" },
    ],
  );
  assert.equal(result.networkMetrics.last_message_at, receivedAt);
  assert.equal(result.networkMetrics.mqtt_broker, "broker.hivemq.com");
  assert.equal(result.networkMetrics.mqtt_port, 1883);
  assert.equal(result.networkMetrics.mqtt_topic, "iiot/motor_01/telemetry");
});

test("marks old Firebase telemetry historical when MQTT data is stale", () => {
  const result = mapDashboardData(dashboardData({
    history: [telemetry],
    health: {
      components: { mqtt: "connected", firebase: "ready", model: "untrained" },
      mqtt_status: {
        subscription: "subscribed",
        messages_received: 4,
        last_message_at: "2026-10-31T11:59:00.000Z",
      },
      data_state: "stale",
    },
  }));

  assert.equal(result.devices[0].status, "stale");
  assert.equal(result.devices[0].historical, true);
  assert.equal(result.devices[0].sensors[0].value, "41.25");
  assert.equal(result.events[0].title, "Historical MQTT telemetry");
});

test("does not turn legacy/sample-shaped records into sensor measurements", () => {
  const result = mapDashboardData(dashboardData({
    history: [
      { id: "temperature", type: "temperature", value: 25, unit: "°C" },
      { id: "untraceable", sensor_values: { temperature: 100 } },
    ],
  }));

  assert.deepEqual(result.devices, []);
  assert.deepEqual(result.telemetry, []);
  assert.deepEqual(result.events, []);
});

test("rejects telemetry records with non-numeric feature values", () => {
  const result = mapDashboardData(dashboardData({
    history: [{
      ...telemetry,
      features: [41.25, "0.103", 0.42],
    }],
  }));

  assert.deepEqual(result.devices, []);
  assert.deepEqual(result.telemetry, []);
});

test("shows only trained predictions linked to a known Firebase telemetry record", () => {
  const result = mapDashboardData(dashboardData({
    history: [telemetry],
    historyPredictions: [
      {
        id: "prediction-1",
        telemetry_record_id: "firebase-telemetry-1",
        prediction: "NORMAL",
        model_status: "TRAINED",
        model: "IsolationForest",
        predicted_at: receivedAt,
        anomaly_score: 0.123,
      },
      {
        id: "unlinked-prediction",
        telemetry_record_id: "not-a-stored-record",
        prediction: "ANOMALY",
        model_status: "TRAINED",
        predicted_at: receivedAt,
      },
      {
        id: "untrained-prediction",
        telemetry_record_id: "firebase-telemetry-1",
        prediction: "NORMAL",
        model_status: "UNTRAINED",
        predicted_at: receivedAt,
      },
    ],
  }));

  assert.equal(result.latestPrediction.id, "prediction-1");
  assert.equal(result.latestPrediction.telemetry_record_id, "firebase-telemetry-1");
  assert.equal(result.latestPrediction.historical, true);
  assert.equal(result.predictions.length, 1);
});

test("keeps valid partial sensor readings without inventing missing channels", () => {
  const partial = {
    ...telemetry,
    features: null,
    sensor_values: { temperature: 42, vibration: null, current: 0.7 },
    raw_payload: { temperature_c: 42, motor_current_a: 0.7 },
  };
  const result = mapDashboardData(dashboardData({
    telemetry: [partial],
    health: {
      components: { mqtt: "connected", firebase: "ready", model: "untrained", xgboost: "trained" },
      mqtt_status: { subscription: "subscribed", last_message_at: receivedAt },
      data_state: "live",
    },
  }));

  assert.equal(result.devices[0].sensors[0].value, "42.00");
  assert.equal(result.devices[0].sensors[1].value, "—");
  assert.equal(result.devices[0].sensors[2].value, "0.70");
});

test("maps linked XGBoost class predictions and AI alerts separately", () => {
  const prediction = {
    id: "xgb-1",
    telemetry_record_id: telemetry.id,
    predicted_class: "HIGH_VIBRATION",
    confidence: 0.94,
    model_name: "XGBoost",
    prediction_timestamp: receivedAt,
  };
  const anomaly = {
    id: "ai-alert-1",
    telemetry_record_id: telemetry.id,
    prediction_record_id: "xgb-1",
    device_id: telemetry.device_id,
    anomaly_type: "HIGH_VIBRATION",
    confidence: 0.94,
    model_name: "XGBoost",
    detected_timestamp: receivedAt,
    alert_status: "OPEN",
    related_sensor_values: { temperature_c: 41.25, motor_current_a: 0.42 },
  };
  const result = mapDashboardData(dashboardData({
    history: [telemetry],
    historyPredictions: [prediction],
    historyAnomalies: [anomaly],
  }));

  assert.equal(result.latestAIPrediction.predicted_class, "HIGH_VIBRATION");
  assert.equal(result.aiPredictions[0].confidence, 0.94);
  assert.equal(result.aiAlerts[0].source, "AI prediction");
  assert.equal(result.aiAlerts[0].status, "Active");
});

test("maps live backend network metrics without inventing health scores or sensors", () => {
  const result = mapDashboardData(dashboardData({
    devices: [{
      device_id: "motor_01",
      status: "HEALTHY",
      online: true,
      ip_address: "192.168.1.42",
      wifi_connected: true,
      mqtt_connected: true,
      latest_mqtt_rtt_ms: 24,
      average_mqtt_rtt_ms: 18,
      rtt_samples: 3,
      estimated_packet_loss_percent: 2.5,
      packets_lost: 1,
      sequence: 17,
      last_seen_at: receivedAt,
    }],
    health: {
      components: { mqtt: "connected", firebase: "ready", model: "untrained" },
      mqtt_status: { subscription: "subscribed", messages_received: 17 },
      data_state: "waiting",
    },
  }));
  const [device] = result.devices;

  assert.equal(device.status, "online");
  assert.equal(device.deviceHealth, "HEALTHY");
  assert.equal(device.ip, "192.168.1.42");
  assert.equal(device.wifiConnected, true);
  assert.equal(device.mqttConnected, true);
  assert.equal(device.latency, 24);
  assert.equal(device.averageLatency, 18);
  assert.equal(device.packetLoss, 2.5);
  assert.equal(device.health, null);
  assert.equal(device.sensorsCount, 0);
  assert.equal(result.networkMetrics.avg_latency_ms, 18);
});

test("maps network warning, stale, and offline health states separately", () => {
  for (const [status, expected] of [
    ["WARNING", "warning"],
    ["STALE", "stale"],
    ["OFFLINE", "offline"],
  ]) {
    const result = mapDashboardData(dashboardData({
      devices: [{ device_id: "motor_01", status, online: false }],
    }));
    assert.equal(result.devices[0].status, expected);
    assert.equal(result.devices[0].deviceHealth, status);
  }
});
