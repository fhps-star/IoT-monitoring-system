const hasValue = (value) =>
  value !== null
  && value !== undefined
  && value !== ""
  && typeof value !== "boolean"
  && Number.isFinite(Number(value));

const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const firstNumber = (...values) => {
  const value = values.find(hasValue);
  return value === undefined ? null : Number(value);
};

const reading = (value, digits) =>
  hasValue(value) ? Number(value).toFixed(digits) : "—";

const normalizeTimestamp = (value) => {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  const date = Number.isFinite(numeric)
    ? new Date(numeric < 1e12 ? numeric * 1000 : numeric)
    : new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
};

const isGenuineTelemetry = (record) =>
  record?.source === "mqtt"
  && typeof record.id === "string"
  && record.id.length > 0
  && record.raw_payload
  && typeof record.raw_payload === "object"
  && !Array.isArray(record.raw_payload)
  && (record.features == null || (
    Array.isArray(record.features)
    && record.features.every(isNumber)
  ))
  && normalizeTimestamp(record.received_at);

function normalizeTelemetry(record, historical = false) {
  const raw = record.raw_payload ?? {};
  const sensorValues = record.sensor_values ?? {};
  const accelerationX = firstNumber(record.acceleration_x, raw.acceleration_x);
  const accelerationY = firstNumber(record.acceleration_y, raw.acceleration_y);
  const accelerationZ = firstNumber(record.acceleration_z, raw.acceleration_z);
  const derivedMagnitude = [accelerationX, accelerationY, accelerationZ].every(isNumber)
    ? Math.hypot(accelerationX, accelerationY, accelerationZ)
    : null;
  return {
    ...record,
    firebase_record_id: record.id,
    device_id: record.device_id,
    motor_id: record.motor_id,
    temperature_c: firstNumber(record.temperature_c, sensorValues.temperature, raw.temperature_c),
    temperature_unit: raw.temperature_unit ?? "°C",
    vibration: firstNumber(record.estimated_vibration, sensorValues.vibration, raw.estimated_vibration),
    estimated_vibration: firstNumber(record.estimated_vibration, sensorValues.vibration, raw.estimated_vibration),
    vibration_unit: raw.vibration_unit ?? "m/s²",
    current_a: firstNumber(record.motor_current_a, sensorValues.current, raw.motor_current_a, raw.current_a),
    motor_current_a: firstNumber(record.motor_current_a, sensorValues.current, raw.motor_current_a, raw.current_a),
    current_unit: raw.current_unit ?? "A",
    acceleration_x: accelerationX,
    acceleration_y: accelerationY,
    acceleration_z: accelerationZ,
    gyro_x: firstNumber(record.gyro_x, raw.gyro_x),
    gyro_y: firstNumber(record.gyro_y, raw.gyro_y),
    gyro_z: firstNumber(record.gyro_z, raw.gyro_z),
    acceleration_magnitude: firstNumber(record.acceleration_magnitude, raw.acceleration_magnitude, derivedMagnitude),
    timestamp_ms: firstNumber(record.timestamp_ms, raw.timestamp_ms),
    sequence: record.sequence ?? raw.sequence ?? null,
    motor_steps: firstNumber(record.motor_steps, raw.motor_steps),
    motor_speed: firstNumber(record.motor_speed, raw.motor_speed),
    motor_status: record.motor_status ?? raw.motor_status ?? null,
    temperature_status: record.temperature_status ?? raw.temperature_status ?? null,
    vibration_status: record.vibration_status ?? raw.vibration_status ?? null,
    current_status: record.current_status ?? raw.current_status ?? null,
    wifi_connected: record.wifi_connected ?? raw.wifi_connected ?? null,
    received_at: normalizeTimestamp(record.received_at),
    historical,
  };
}

function apiStatusToUiStatus(status) {
  switch (String(status ?? "").toUpperCase()) {
    case "HEALTHY":
      return "online";
    case "WARNING":
      return "warning";
    case "STALE":
      return "stale";
    case "OFFLINE":
      return "offline";
    default:
      return null;
  }
}

function toDashboardDevice(record, status, history, network = {}) {
  const networkStatus = apiStatusToUiStatus(network.status) ?? status;
  const sensorStatus = record.historical ? "stale" : "online";
  const receivedAt = normalizeTimestamp(
    network.last_seen_at ?? network.last_telemetry_at ?? record.received_at,
  );
  const sensors = record.id ? [
    {
      name: "Temperature",
      kind: "temperature",
      model: "ESP32 MQTT",
      channel: "MQTT payload",
      value: reading(record.temperature_c, 2),
      unit: record.temperature_unit,
      status: sensorStatus,
    },
    {
      name: "Vibration",
      kind: "vibration",
      model: "ESP32 MQTT",
      channel: "MQTT payload",
      value: reading(record.vibration, 3),
      unit: record.vibration_unit,
      status: sensorStatus,
    },
    {
      name: "Motor current",
      kind: "current",
      model: "ESP32 MQTT",
      channel: "MQTT payload",
      value: reading(record.current_a, 2),
      unit: record.current_unit,
      status: sensorStatus,
    },
    ...(hasValue(record.motor_speed)
      ? [{
        name: "Motor speed",
        kind: "current",
        model: "ESP32 MQTT",
        channel: "MQTT payload",
        value: reading(record.motor_speed, 0),
        unit: "RPM",
        status: sensorStatus,
      }]
      : []),
  ] : [];

  return {
    id: record.device_id ?? network.device_id,
    name: record.motor_id ?? network.motor_id ?? record.device_id ?? network.device_id,
    asset: record.motor_id ?? network.motor_id ?? record.device_id ?? network.device_id,
    line: "—",
    status: networkStatus,
    online: typeof network.online === "boolean"
      ? network.online
      : networkStatus === "online" || networkStatus === "warning",
    historical: record.historical ?? !record.id,
    telemetryRecordId: record.id ?? null,
    sensorsCount: sensors.length,
    ip: network.ip_address ?? "—",
    protocol: "MQTT",
    latency: firstNumber(network.latest_mqtt_rtt_ms),
    averageLatency: firstNumber(network.average_mqtt_rtt_ms),
    packetLoss: firstNumber(network.estimated_packet_loss_percent),
    health: null,
    wifiConnected: typeof network.wifi_connected === "boolean"
      ? network.wifi_connected
      : null,
    mqttConnected: typeof network.mqtt_connected === "boolean"
      ? network.mqtt_connected
      : null,
    sequence: firstNumber(network.sequence),
    packetsLost: firstNumber(network.packets_lost),
    model: "ESP32",
    firmware: "—",
    mac: "—",
    lastSeen: receivedAt ? new Date(receivedAt).toLocaleString() : "No telemetry yet",
    installed: "—",
    deviceHealth: network.status ?? null,
    networkHealth: network.status ?? null,
    sensorHealth: null,
    motorHealth: null,
    sensors,
    history: history
      .filter((item) => hasValue(item.vibration))
      .map((item) => Number(item.vibration))
      .reverse(),
    telemetryHistory: history,
  };
}

function toAlert(record) {
  const timestamp = normalizeTimestamp(
    record.created_at ?? record.predicted_at ?? record.detected_timestamp ?? record.timestamp,
  );
  const state = String(record.status ?? record.alert_status ?? "OPEN").toUpperCase();
  const displayState = state === "RESOLVED"
    ? "Resolved"
    : state === "ACKNOWLEDGED"
      ? "Acknowledged"
      : "Active";
  const device = record.device_id ?? record.motor_id;
  let sensor = record.sensor ?? null;
  if (!sensor) {
    const anomalyType = String(record.anomaly_type ?? "");
    const desc = String(record.description ?? "").toLowerCase();
    const sVals = record.related_sensor_values ?? record.sensor_values ?? {};
    if (
      anomalyType.includes("TEMP")
      || desc.includes("temperature")
      || (hasValue(sVals.temperature ?? sVals.temperature_c) && Number(sVals.temperature ?? sVals.temperature_c) >= 48)
    ) {
      sensor = "Temperature Sensor";
    } else if (
      anomalyType.includes("VIB")
      || desc.includes("vibration")
      || (hasValue(sVals.vibration ?? sVals.vibration_rms ?? sVals.estimated_vibration) && Number(sVals.vibration ?? sVals.vibration_rms ?? sVals.estimated_vibration) >= 0.35)
    ) {
      sensor = "Vibration Sensor";
    } else if (
      anomalyType.includes("CURR")
      || desc.includes("current")
      || (hasValue(sVals.current ?? sVals.motor_current_a ?? sVals.current_a) && Number(sVals.current ?? sVals.motor_current_a ?? sVals.current_a) >= 1.2)
    ) {
      sensor = "Current Sensor";
    } else if (record.prediction === "ANOMALY") {
      sensor = "Motor Sensor";
    }
  }
  const score = hasValue(record.anomaly_score)
    ? ` (score ${Number(record.anomaly_score).toFixed(3)})`
    : "";
  return {
    id: record.id,
    title: record.description ?? `${record.prediction ?? record.anomaly_type ?? "ANOMALY"} · telemetry anomaly`,
    description: `${device} · ${record.model ?? record.model_name ?? "Threshold Monitor"}${score}`,
    device,
    sensor,
    type: record.prediction ?? record.anomaly_type ?? "ANOMALY",
    component: sensor ?? "Motor",
    severity: record.severity ?? "Warning",
    age: timestamp ? formatAge(timestamp) : "—",
    firstSeen: timestamp ? new Date(timestamp).toLocaleString() : "—",
    duration: "—",
    state: displayState,
    status: displayState,
  };
}

function toAIAlert(record) {
  const timestamp = normalizeTimestamp(record.detected_timestamp ?? record.prediction_timestamp);
  const state = String(record.alert_status ?? "OPEN").toUpperCase();
  let sensor = record.sensor ?? null;
  if (!sensor) {
    const anomalyType = String(record.anomaly_type ?? "");
    const desc = String(record.description ?? "").toLowerCase();
    const sVals = record.related_sensor_values ?? record.sensor_values ?? {};
    if (
      anomalyType.includes("TEMP")
      || desc.includes("temperature")
      || (hasValue(sVals.temperature ?? sVals.temperature_c) && Number(sVals.temperature ?? sVals.temperature_c) >= 48)
    ) {
      sensor = "Temperature Sensor";
    } else if (
      anomalyType.includes("VIB")
      || desc.includes("vibration")
      || (hasValue(sVals.vibration ?? sVals.vibration_rms ?? sVals.estimated_vibration) && Number(sVals.vibration ?? sVals.vibration_rms ?? sVals.estimated_vibration) >= 0.35)
    ) {
      sensor = "Vibration Sensor";
    } else if (
      anomalyType.includes("CURR")
      || desc.includes("current")
      || (hasValue(sVals.current ?? sVals.motor_current_a ?? sVals.current_a) && Number(sVals.current ?? sVals.motor_current_a ?? sVals.current_a) >= 1.2)
    ) {
      sensor = "Current Sensor";
    } else if (anomalyType && anomalyType !== "NORMAL") {
      sensor = "Motor Sensor";
    }
  }
  return {
    ...record,
    title: `XGBoost prediction · ${record.anomaly_type}`,
    description: "AI anomaly prediction; not a confirmed physical failure.",
    type: record.anomaly_type,
    device: record.device_id,
    source: "AI prediction",
    severity: record.severity ?? "Warning",
    status: state === "RESOLVED" ? "Resolved" : state === "ACKNOWLEDGED" ? "Acknowledged" : "Active",
    age: timestamp ? formatAge(timestamp) : "—",
    firstSeen: timestamp ? new Date(timestamp).toLocaleString() : "—",
    duration: "—",
    sensor,
    component: sensor ?? "Motor",
    timestamp,
  };
}

function formatAge(timestamp) {
  const timestampMs = Date.parse(normalizeTimestamp(timestamp) ?? "");
  if (!Number.isFinite(timestampMs)) return "—";
  const minutes = Math.max(0, Math.floor((Date.now() - timestampMs) / 60000));
  if (minutes < 60) return `${minutes} min ago`;
  if (minutes < 1440) return `${Math.floor(minutes / 60)} hr ago`;
  return `${Math.floor(minutes / 1440)} day ago`;
}

function toEvent(record) {
  const date = new Date(record.received_at);
  const historical = record.historical;
  return {
    id: record.id,
    kind: historical ? "offline" : "normal",
    title: historical ? "Historical MQTT telemetry" : "MQTT telemetry received",
    device: record.device_id,
    message: `Temperature ${reading(record.temperature_c, 2)} ${record.temperature_unit} · vibration ${reading(record.vibration, 3)} ${record.vibration_unit} · current ${reading(record.current_a, 2)} ${record.current_unit} · Firebase record ${record.id}`,
    time: date.toLocaleString(),
  };
}

const isXGBoostPrediction = (record) =>
  record?.model_name === "XGBoost"
  && ["NORMAL", "HIGH_VIBRATION", "HIGH_TEMPERATURE", "HIGH_CURRENT"].includes(record.predicted_class)
  && typeof record.id === "string"
  && normalizeTimestamp(record.prediction_timestamp)
  && typeof record.telemetry_record_id === "string";

export function mapDashboardData(data) {
  const history = (data.history ?? [])
    .filter(isGenuineTelemetry)
    .map((record) => normalizeTelemetry(record, true))
    .sort((left, right) => Date.parse(right.received_at) - Date.parse(left.received_at));
  const live = (data.telemetry ?? [])
    .filter(isGenuineTelemetry)
    .map((record) => normalizeTelemetry(record))
    .sort((left, right) => Date.parse(right.received_at) - Date.parse(left.received_at));
  const lastMessageAt = normalizeTimestamp(data.health.mqtt_status?.last_message_at);
  const currentRecord = data.health.data_state === "live"
    ? live.find((record) => record.received_at === lastMessageAt) ?? null
    : null;
  const displayRecord = currentRecord ?? history[0] ?? null;
  const deviceStatus = currentRecord
    ? "online"
    : data.health.data_state === "mqtt_disconnected"
      ? "offline"
      : "stale";
  const displayHistory = [
    ...live.map((record) => ({
      ...record,
      historical: !currentRecord || record.id !== currentRecord.id,
    })),
    ...history.filter((record) => !live.some((item) => item.id === record.id)),
  ].sort((left, right) => Date.parse(right.received_at) - Date.parse(left.received_at));
  const apiDevices = Array.isArray(data.devices) ? data.devices : [];
  const devices = apiDevices.length
    ? apiDevices.map((network) => {
      const deviceTelemetry = displayHistory.filter(
        (record) =>
          record.device_id === network.device_id
          || record.motor_id === network.device_id,
      );
      const latestTelemetry = deviceTelemetry[0] ?? {};
      const isCurrent = currentRecord
        && (
          currentRecord.device_id === network.device_id
          || currentRecord.motor_id === network.device_id
        );
      const fallbackStatus = isCurrent
        ? "online"
        : network.data_state === "mqtt_disconnected"
          ? "offline"
          : "stale";
      return toDashboardDevice(
        {
          ...latestTelemetry,
          device_id: network.device_id,
          motor_id: latestTelemetry.motor_id ?? network.motor_id,
          historical: !isCurrent,
        },
        fallbackStatus,
        deviceTelemetry,
        network,
      );
    })
    : displayRecord
      ? [toDashboardDevice(
        { ...displayRecord, historical: !currentRecord },
        currentRecord ? "online" : deviceStatus,
        displayHistory.filter((record) => record.device_id === displayRecord.device_id),
        data.health.network ?? {},
      )]
      : [];
  const validTelemetryIds = new Set(displayHistory.map((record) => record.id));
  const allPredictions = [
    ...(data.historyPredictions ?? []),
    ...(data.predictions ?? []),
    ...(data.xgboost_predictions ?? []),
  ];
  const predictionsById = new Map(allPredictions
    .filter((record) =>
      record.model_status === "TRAINED"
      && ["NORMAL", "ANOMALY"].includes(record.prediction)
      && typeof record.id === "string"
      && normalizeTimestamp(record.predicted_at)
      && validTelemetryIds.has(record.telemetry_record_id),
    )
    .map((record) => [record.id, record]));
  const predictions = [...predictionsById.values()].sort(
    (left, right) => Date.parse(right.predicted_at) - Date.parse(left.predicted_at),
  ).map((record) => ({
    ...record,
    historical: record.telemetry_record_id !== currentRecord?.id,
  }));
  const predictionIds = new Set(predictions.map((record) => record.id));
  const alertsById = new Map([
    ...(data.historyAlerts ?? []),
    ...(data.alerts ?? []),
    ...(data.historyAnomalies ?? []),
    ...(data.anomalies ?? []),
  ].filter((record) =>
      validTelemetryIds.has(record.telemetry_record_id)
      && (
        predictionIds.has(record.prediction_record_id)
        || String(record.prediction_record_id ?? "").startsWith("threshold-")
        || String(record.prediction_record_id ?? "").startsWith("xgb-")
        || typeof record.anomaly_type === "string"
      ),
    ).map((record) => [record.id, record]));
  const alerts = [...alertsById.values()].map(toAlert);
  const aiPredictionsById = new Map(allPredictions
    .filter((record) =>
      isXGBoostPrediction(record)
      && validTelemetryIds.has(record.telemetry_record_id),
    )
    .map((record) => [record.id, {
      ...record,
      historical: record.telemetry_record_id !== currentRecord?.id,
    }]));
  const aiPredictions = [...aiPredictionsById.values()].sort(
    (left, right) => Date.parse(right.prediction_timestamp) - Date.parse(left.prediction_timestamp),
  );
  const anomalyRecords = new Map([
    ...(data.historyAnomalies ?? []),
    ...(data.anomalies ?? []),
  ].filter((record) =>
    validTelemetryIds.has(record.telemetry_record_id)
    && typeof record.id === "string",
  ).map((record) => [record.id, record]));
  const aiAlerts = [...anomalyRecords.values()].map(toAIAlert).sort(
    (left, right) => Date.parse(right.timestamp ?? "") - Date.parse(left.timestamp ?? ""),
  );
  const telemetry = displayHistory;
  const onlineDevices = devices.filter((device) => device.online).length;
  const rttSamples = apiDevices.reduce(
    (total, device) => total + (Number(device.rtt_samples) || 0),
    0,
  );
  const averageLatency = rttSamples
    ? apiDevices.reduce(
      (total, device) =>
        total
        + (Number(device.average_mqtt_rtt_ms) || 0)
          * (Number(device.rtt_samples) || 0),
      0,
    ) / rttSamples
    : firstNumber(data.health.network?.average_mqtt_rtt_ms);
  const metrics = {
    broker_connected: data.health.components?.mqtt === "connected"
      && data.health.mqtt_status?.subscription === "subscribed",
    mqtt_broker: data.health.mqtt_status?.broker ?? null,
    mqtt_port: data.health.mqtt_status?.port ?? null,
    mqtt_topic: data.health.mqtt_status?.topic ?? null,
    avg_latency_ms: averageLatency,
    total_samples: history.length,
    total_messages: data.health.mqtt_status?.messages_received ?? 0,
    online_devices: onlineDevices,
    total_devices: devices.length,
    availability_pct: devices.length
      ? Math.round((onlineDevices / devices.length) * 100)
      : null,
    last_message_at: data.health.mqtt_status?.last_message_at
      ?? history[0]?.received_at
      ?? null,
  };

  return {
    devices,
    telemetry,
    predictions,
    alerts,
    events: telemetry.slice(0, 6).map(toEvent),
    networkMetrics: metrics,
    dataState: data.health.data_state,
    hasHistoricalData: history.length > 0,
    latestPrediction: predictions[0] ?? null,
    aiPredictions,
    latestAIPrediction: aiPredictions[0] ?? null,
    aiAlerts,
    modelStatus: data.health.components?.model ?? "untrained",
    xgboostStatus: data.health.components?.xgboost ?? "unavailable",
  };
}
