import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

from fastapi.testclient import TestClient

from backend.firebase_config import FirebaseStore
from backend.main import create_app
from backend.ai_model import IsolationForestModel
from backend.feature_engineering import FEATURE_NAMES
from backend.mqtt_handler import MQTTHandler
from backend.random_forest_predictor import RandomForestPredictor
from backend.xgboost_predictor import XGBoostPredictor


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        model_load_patcher = patch.object(
            IsolationForestModel, "load", return_value=False
        )
        model_load_patcher.start()
        self.addCleanup(model_load_patcher.stop)
        rf_load_patcher = patch.object(
            RandomForestPredictor, "load", return_value=False
        )
        rf_load_patcher.start()
        self.addCleanup(rf_load_patcher.stop)
        xgboost_load_patcher = patch.object(
            XGBoostPredictor, "load", return_value=False
        )
        xgboost_load_patcher.start()
        self.addCleanup(xgboost_load_patcher.stop)
        firebase_read_patcher = patch.object(FirebaseStore, "get_value", return_value=None)
        firebase_read_patcher.start()
        self.addCleanup(firebase_read_patcher.stop)

    def test_mqtt_network_flow_persists_metrics_separately_and_serves_device_apis(self) -> None:
        persisted_network: dict[str, dict] = {}
        persisted_telemetry: dict[str, dict] = {}

        def persist_network(path: str, record_id: str, value: dict) -> str:
            persisted_network[f"{path}/{record_id}"] = {**value, "id": record_id}
            return record_id

        def persist_telemetry(path: str, value: dict) -> str:
            record_id = f"telemetry-{len(persisted_telemetry) + 1}"
            persisted_telemetry[f"{path}/{record_id}"] = {**value, "id": record_id}
            return record_id

        payloads = [
            {
                "device_id": "motor_01",
                "temperature_c": 42.5,
                "estimated_vibration": 1.2,
                "motor_current_a": 0.8,
                "ip_address": "192.168.1.42",
                "wifi_connected": True,
                "mqtt_connected": True,
                "mqtt_rtt_ms": rtt,
                "sequence": sequence,
                "timestamp_ms": 5000 + sequence * 5000,
            }
            for sequence, rtt in ((5, 12), (6, 18), (7, 24))
        ]

        def receive_mqtt_messages(handler: MQTTHandler) -> None:
            for payload in payloads:
                handler._handle_message(
                    None,
                    None,
                    SimpleNamespace(
                        topic="iiot/motor_01/telemetry",
                        payload=json.dumps(payload).encode(),
                    ),
                )

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(FirebaseStore, "set_record", side_effect=persist_network),
            patch.object(FirebaseStore, "push", side_effect=persist_telemetry),
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=receive_mqtt_messages,
            ),
            patch.object(
                MQTTHandler,
                "connected",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                MQTTHandler,
                "subscription_status",
                new_callable=PropertyMock,
                return_value="subscribed",
            ),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            devices = client.get("/api/v1/devices")
            network = client.get("/api/v1/devices/motor_01/network")
            health = client.get("/api/v1/health")
            unknown_device = client.get("/api/v1/devices/other-device/network")

        self.assertIn("network_metrics/motor_01/latest", persisted_network)
        metrics = persisted_network["network_metrics/motor_01/latest"]
        self.assertEqual(metrics["ip_address"], "192.168.1.42")
        self.assertEqual(metrics["wifi_connected"], True)
        self.assertEqual(metrics["mqtt_connected"], True)
        self.assertEqual(metrics["latest_mqtt_rtt_ms"], 24.0)
        self.assertEqual(metrics["average_mqtt_rtt_ms"], 18.0)
        self.assertEqual(metrics["packets_expected"], 3)
        self.assertEqual(metrics["packets_lost"], 0)
        self.assertEqual(metrics["status"], "HEALTHY")

        self.assertEqual(len(persisted_telemetry), 3)
        for record in persisted_telemetry.values():
            self.assertNotIn("ip_address", record)
            self.assertNotIn("wifi_connected", record)
            self.assertNotIn("mqtt_connected", record)
            self.assertNotIn("mqtt_rtt_ms", record)
            self.assertNotIn("wifi_connected", record["raw_payload"])
            self.assertNotIn("mqtt_connected", record["raw_payload"])
            self.assertNotIn("mqtt_rtt_ms", record["raw_payload"])
            self.assertEqual(record["sensor_values"]["temperature"], 42.5)

        self.assertEqual(devices.status_code, 200)
        self.assertEqual(devices.json()[0]["status"], "HEALTHY")
        self.assertEqual(devices.json()[0]["ip_address"], "192.168.1.42")
        self.assertEqual(network.status_code, 200)
        self.assertEqual(network.json()["average_mqtt_rtt_ms"], 18.0)
        self.assertEqual(health.json()["device_health"], "HEALTHY")
        self.assertEqual(unknown_device.status_code, 404)

    def test_existing_firebase_telemetry_seeds_network_metrics_on_startup(self) -> None:
        received_at = datetime.now(timezone.utc).isoformat()
        existing_record = {
            "id": "existing-telemetry",
            "source": "mqtt",
            "received_at": received_at,
            "raw_payload": {
                "device_id": "motor_01",
                "temperature_c": 42.5,
                "estimated_vibration": 1.2,
                "motor_current_a": 0.8,
                "ip_address": "192.168.1.42",
                "wifi_connected": True,
                "mqtt_connected": True,
                "mqtt_rtt_ms": 24.0,
                "sequence": 12,
                "timestamp_ms": 60000,
            },
        }

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(
                FirebaseStore,
                "get_value",
                return_value=None,
            ),
            patch.object(
                FirebaseStore,
                "get_records",
                return_value=[existing_record],
            ),
            patch.object(FirebaseStore, "set_record", return_value="latest") as set_record,
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(MQTTHandler, "start"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            network = client.get("/api/v1/devices/motor_01/network")

        self.assertEqual(network.status_code, 200)
        self.assertEqual(network.json()["ip_address"], "192.168.1.42")
        self.assertEqual(network.json()["latest_mqtt_rtt_ms"], 24.0)
        self.assertEqual(network.json()["sequence"], 12)
        self.assertEqual(network.json()["status"], "HEALTHY")
        set_record.assert_called_once()
        self.assertEqual(
            set_record.call_args.args[0],
            "network_metrics/motor_01",
        )
        self.assertEqual(set_record.call_args.args[1], "latest")

    def test_health_and_firebase_unavailable_response(self) -> None:
        with (
            patch.object(
                FirebaseStore,
                "initialize",
                side_effect=FileNotFoundError("test credentials not configured"),
            ),
            patch.object(MQTTHandler, "start"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            health = client.get("/health")
            versioned_health = client.get("/api/v1/health")
            devices = client.get("/api/v1/devices")
            model_status = client.get("/api/v1/model/status")
            telemetry = client.get("/api/v1/telemetry")
            live = client.get("/api/v1/live")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(versioned_health.status_code, 200)
        self.assertEqual(devices.status_code, 200)
        self.assertEqual(devices.json()[0]["status"], "OFFLINE")
        self.assertEqual(model_status.status_code, 200)
        self.assertEqual(model_status.json()["model_name"], "XGBoost")
        self.assertEqual(model_status.json()["status"], "unavailable")
        self.assertEqual(health.json()["status"], "degraded")
        self.assertEqual(health.json()["components"]["firebase"], "unavailable")
        self.assertEqual(
            health.json()["mqtt_status"]["broker"],
            "broker.hivemq.com",
        )
        self.assertEqual(health.json()["mqtt_status"]["port"], 1883)
        self.assertEqual(
            health.json()["mqtt_status"]["topic"],
            "iiot/motor_01/telemetry",
        )
        self.assertEqual(telemetry.status_code, 503)
        self.assertIn("Firebase is unavailable", telemetry.json()["detail"])
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json()["telemetry"], [])
        self.assertIsNone(live.json()["device_status"])
        self.assertEqual(
            live.json()["health"]["mqtt_status"]["messages_received"],
            0,
        )

    def test_firebase_write_failure_never_exposes_unstored_telemetry(self) -> None:
        payload = {
            "device_id": "motor-node-01",
            "motor_id": "motor_01",
            "temperature_c": 42.5,
            "estimated_vibration": 1.2,
            "motor_current_a": 0.8,
        }

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                side_effect=FileNotFoundError("test credentials not configured"),
            ),
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=lambda handler: handler._on_telemetry(payload),
            ),
            patch.object(FirebaseStore, "push", return_value=None),
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")

        self.assertEqual(live.status_code, 200)
        data = live.json()
        self.assertEqual(data["telemetry"], [])
        self.assertIsNone(data["device_status"])
        self.assertEqual(data["health"]["components"]["firebase"], "unavailable")
        self.assertEqual(data["health"]["mqtt_status"]["messages_received"], 0)
        self.assertEqual(data["health"]["mqtt_status"]["payloads_processed"], 1)
        self.assertIsNotNone(data["health"]["mqtt_status"]["topic"])

    def test_partial_mqtt_payload_is_persisted_without_model_inference(self) -> None:
        payload = {
            "device_id": "motor-node-01",
            "temperature_c": 42.5,
            "estimated_vibration": True,
            "motor_current_a": 0.8,
        }

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(FirebaseStore, "push") as firebase_push,
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(IsolationForestModel, "predict") as model_predict,
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=lambda handler: handler._on_telemetry(payload),
            ),
            patch.object(
                MQTTHandler,
                "connected",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                MQTTHandler,
                "subscription_status",
                new_callable=PropertyMock,
                return_value="subscribed",
            ),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")

        self.assertEqual(live.status_code, 200)
        telemetry = live.json()["telemetry"]
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(telemetry[0]["sensor_values"]["temperature"], 42.5)
        self.assertIsNone(telemetry[0]["sensor_values"]["vibration"])
        self.assertEqual(live.json()["health"]["data_state"], "live")
        firebase_push.assert_called_once()
        model_predict.assert_not_called()

    def test_persisted_firmware_telemetry_is_live_but_untrained_model_predicts_nothing(self) -> None:
        payload = {
            "device_id": "motor-node-01",
            "timestamp_ms": 15000,
            "temperature_c": 42.5,
            "estimated_vibration": 1.2,
            "motor_current_a": 0.8,
            "acceleration_x": 0.1,
            "acceleration_y": 0.2,
            "acceleration_z": 9.81,
            "motor_speed": 30,
        }

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(FirebaseStore, "push", return_value="firebase-telemetry-id"),
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=lambda handler: handler._on_telemetry(payload),
            ),
            patch.object(MQTTHandler, "connected", new_callable=PropertyMock, return_value=True),
            patch.object(MQTTHandler, "subscription_status", new_callable=PropertyMock, return_value="subscribed"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")

        self.assertEqual(live.status_code, 200)
        data = live.json()
        self.assertEqual(data["health"]["data_state"], "live")
        self.assertEqual(data["telemetry"][0]["id"], "firebase-telemetry-id")
        self.assertEqual(data["telemetry"][0]["sensor_values"]["temperature"], 42.5)
        self.assertEqual(data["predictions"], [])
        self.assertEqual(data["health"]["components"]["model"], "untrained")

    def test_trained_prediction_is_written_with_telemetry_record_link(self) -> None:
        payload = {
            "device_id": "motor-node-01",
            "timestamp_ms": 15000,
            "temperature_c": 42.5,
            "motor_current_a": 0.8,
            "acceleration_x": 0.1,
            "acceleration_y": 0.2,
            "acceleration_z": 9.81,
            "motor_speed": 30,
        }

        def infer_after_telemetry_persisted(
            _features: list[float],
        ) -> tuple[str, float]:
            self.assertEqual(firebase_push.call_count, 3)
            self.assertEqual(len(_features), len(FEATURE_NAMES))
            return "NORMAL", 0.123

        def load_test_model(model: IsolationForestModel) -> bool:
            model._model = object()
            model._feature_names = FEATURE_NAMES
            return True

        def send_three_samples(handler: MQTTHandler) -> None:
            for sequence in range(3):
                handler._on_telemetry({**payload, "sequence": sequence})

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(
                FirebaseStore,
                "push",
                side_effect=[
                    "firebase-telemetry-1",
                    "firebase-telemetry-2",
                    "firebase-telemetry-3",
                    "firebase-prediction-id",
                ],
            ) as firebase_push,
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(
                IsolationForestModel,
                "load",
                new=load_test_model,
            ),
            patch.object(
                IsolationForestModel,
                "predict",
                side_effect=infer_after_telemetry_persisted,
            ) as model_predict,
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=send_three_samples,
            ),
            patch.object(MQTTHandler, "connected", new_callable=PropertyMock, return_value=True),
            patch.object(MQTTHandler, "subscription_status", new_callable=PropertyMock, return_value="subscribed"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")

        self.assertEqual(live.status_code, 200)
        prediction = live.json()["predictions"][0]
        self.assertEqual(prediction["id"], "firebase-prediction-id")
        self.assertEqual(prediction["telemetry_record_id"], "firebase-telemetry-3")
        self.assertEqual(prediction["prediction"], "NORMAL")
        self.assertEqual(
            firebase_push.call_args_list[0].args[0],
            "telemetry/motor_01",
        )
        self.assertEqual(
            firebase_push.call_args_list[0].args[1]["device_id"],
            "motor-node-01",
        )
        self.assertEqual(
            firebase_push.call_args_list[0].args[1]["motor_id"],
            "motor_01",
        )
        self.assertEqual(model_predict.call_count, 1)

    def test_failed_prediction_write_does_not_create_an_unlinked_alert(self) -> None:
        payload = {
            "device_id": "motor-node-01",
            "timestamp_ms": 15000,
            "temperature_c": 42.5,
            "motor_current_a": 0.8,
            "acceleration_x": 0.1,
            "acceleration_y": 0.2,
            "acceleration_z": 9.81,
            "motor_speed": 30,
        }

        def load_test_model(model: IsolationForestModel) -> bool:
            model._model = object()
            model._feature_names = FEATURE_NAMES
            return True

        def send_three_samples(handler: MQTTHandler) -> None:
            for sequence in range(3):
                handler._on_telemetry({**payload, "sequence": sequence})

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(FirebaseStore, "get_records", return_value=[]),
            patch.object(
                FirebaseStore,
                "push",
                side_effect=[
                    "firebase-telemetry-1",
                    "firebase-telemetry-2",
                    "firebase-telemetry-3",
                    None,
                ],
            ),
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(
                IsolationForestModel,
                "load",
                new=load_test_model,
            ),
            patch.object(
                IsolationForestModel,
                "predict",
                return_value=("ANOMALY", 0.987),
            ),
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=send_three_samples,
            ),
            patch.object(
                MQTTHandler,
                "connected",
                new_callable=PropertyMock,
                return_value=True,
            ),
            patch.object(
                MQTTHandler,
                "subscription_status",
                new_callable=PropertyMock,
                return_value="subscribed",
            ),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")

        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json()["predictions"], [])
        self.assertEqual(live.json()["alerts"], [])
        self.assertEqual(
            live.json()["telemetry"][0]["id"],
            "firebase-telemetry-3",
        )

    def test_xgboost_prediction_and_anomaly_are_stored_separately_by_telemetry_id(self) -> None:
        payload = {
            "device_id": "motor_01",
            "sequence": 45,
            "timestamp_ms": 1700000000000,
            "temperature_c": 81.2,
            "acceleration_x": 1.2,
            "acceleration_y": 1.1,
            "acceleration_z": 9.5,
            "gyro_x": 0.1,
            "gyro_y": 0.2,
            "gyro_z": 0.3,
            "motor_speed": 80,
            "motor_steps": 400,
            "motor_current_a": 1.4,
            "motor_status": "RUNNING",
            "simulation_mode": "HIGH_VIBRATION",
        }

        def load_xgboost(predictor: XGBoostPredictor) -> bool:
            predictor._feature_columns = (
                "temperature_c",
                "acceleration_x",
                "acceleration_y",
                "acceleration_z",
                "acceleration_magnitude",
                "motor_speed",
                "motor_current_a",
            )
            predictor._class_labels = (
                "HIGH_CURRENT",
                "HIGH_TEMPERATURE",
                "HIGH_VIBRATION",
                "NORMAL",
            )
            predictor._preprocessor = Mock()
            predictor._preprocessor.transform.side_effect = lambda values: values
            predictor._model = Mock()
            predictor._model.predict_proba.return_value = [[0.01, 0.02, 0.94, 0.03]]
            return True

        stored_records: dict[str, list[dict]] = {}

        def persist_idempotently(path: str, record_id: str, value: dict) -> str:
            stored_records[path] = [{**value, "id": record_id}]
            return record_id

        def get_stored_records(path: str, _limit: int) -> list[dict]:
            return stored_records.get(path, [])

        with (
            patch.object(
                FirebaseStore,
                "initialize",
                autospec=True,
                side_effect=lambda store: setattr(store, "_database", object()),
            ),
            patch.object(FirebaseStore, "get_records", side_effect=get_stored_records),
            patch.object(FirebaseStore, "push", return_value="telemetry-record-1") as firebase_push,
            patch.object(FirebaseStore, "set_record", side_effect=persist_idempotently) as firebase_set,
            patch.object(FirebaseStore, "update_value", return_value=True),
            patch.object(XGBoostPredictor, "load", new=load_xgboost),
            patch.object(
                MQTTHandler,
                "start",
                autospec=True,
                side_effect=lambda handler: handler._on_telemetry(payload),
            ),
            patch.object(MQTTHandler, "connected", new_callable=PropertyMock, return_value=True),
            patch.object(MQTTHandler, "subscription_status", new_callable=PropertyMock, return_value="subscribed"),
            patch.object(MQTTHandler, "stop"),
            TestClient(create_app()) as client,
        ):
            live = client.get("/api/v1/live")
            model_status = client.get("/api/v1/model/status")
            prediction_history = client.get("/api/v1/predictions")
            anomaly_history = client.get("/api/v1/anomalies")

        self.assertEqual(firebase_push.call_args.args[0], "telemetry/motor_01")
        self.assertEqual(firebase_set.call_args_list[0].args[0], "xgboost_predictions/motor_01")
        self.assertEqual(firebase_set.call_args_list[0].args[1], "telemetry-record-1")
        self.assertEqual(firebase_set.call_args_list[1].args[0], "anomaly_alerts/motor_01")
        self.assertEqual(firebase_set.call_args_list[1].args[1], "telemetry-record-1")
        telemetry = live.json()["telemetry"][0]
        prediction = live.json()["xgboost_predictions"][0]
        anomaly = live.json()["anomalies"][0]
        self.assertEqual(telemetry["motor_steps"], 400)
        self.assertEqual(telemetry["gyro_x"], 0.1)
        self.assertEqual(prediction["predicted_class"], "HIGH_VIBRATION")
        self.assertEqual(prediction["confidence"], 0.94)
        self.assertEqual(prediction["telemetry_record_id"], telemetry["id"])
        self.assertEqual(anomaly["anomaly_type"], "HIGH_VIBRATION")
        self.assertIn("not a confirmed physical failure", anomaly["description"])
        self.assertEqual(model_status.json()["status"], "ready")
        self.assertEqual(prediction_history.json()[0]["predicted_class"], "HIGH_VIBRATION")
        self.assertEqual(anomaly_history.json()[0]["telemetry_record_id"], telemetry["id"])
        self.assertEqual(
            model_status.json()["feature_columns"],
            [
                "temperature_c",
                "acceleration_x",
                "acceleration_y",
                "acceleration_z",
                "acceleration_magnitude",
                "motor_speed",
                "motor_current_a",
            ],
        )


if __name__ == "__main__":
    unittest.main()
