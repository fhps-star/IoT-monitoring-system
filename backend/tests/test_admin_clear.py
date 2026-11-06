"""Unit tests for POST /api/admin/clear-telemetry endpoint."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.firebase_config import FirebaseStore
from backend.main import create_app
from backend.settings import Settings


class TestAdminClearTelemetry(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_cleanup = {
            "status": "success",
            "deleted_counts": {
                "telemetry": 1,
                "predictions": 0,
                "alerts": 0,
                "random_forest_predictions": 0,
                "xgboost_predictions": 0,
                "anomaly_alerts": 0,
                "model_comparisons": 0,
                "protocol_comparison": 0,
                "protocol_status": 0,
                "firestore_documents": 0,
                "total": 1,
            },
            "preserved": [
                "device_status",
                "network_metrics",
                "device_configurations",
                "application_settings",
                "credentials",
            ],
            "device_id": "motor_01",
        }

        self.patch_push = patch.object(FirebaseStore, "push", return_value="test-telemetry-id")
        self.patch_push.start()
        self.addCleanup(self.patch_push.stop)

        self.patch_set = patch.object(FirebaseStore, "set_record", return_value="test-record-id")
        self.patch_set.start()
        self.addCleanup(self.patch_set.stop)

        self.patch_update = patch.object(FirebaseStore, "update_value", return_value=True)
        self.patch_update.start()
        self.addCleanup(self.patch_update.stop)

        self.patch_clear = patch.object(
            FirebaseStore, "clear_telemetry_records", return_value=self.mock_cleanup
        )
        self.patch_clear.start()
        self.addCleanup(self.patch_clear.stop)

        self.app = create_app()
        self.client = TestClient(self.app)

    def test_unauthorized_missing_key(self) -> None:
        with self.client as client:
            resp = client.post("/api/admin/clear-telemetry", json={"confirm": True})
            self.assertEqual(resp.status_code, 401)
            self.assertIn("Unauthorized", resp.json()["detail"])

    def test_unauthorized_invalid_key(self) -> None:
        with self.client as client:
            resp = client.post(
                "/api/admin/clear-telemetry",
                headers={"X-Admin-Key": "wrong-key-123"},
                json={"confirm": True},
            )
            self.assertEqual(resp.status_code, 401)

    def test_missing_confirmation(self) -> None:
        with self.client as client:
            settings: Settings = self.app.state.settings
            resp = client.post(
                "/api/admin/clear-telemetry",
                headers={"X-Admin-Key": settings.admin_api_key},
                json={"confirm": False},
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Confirmation required", resp.json()["detail"])

    def test_successful_clear_and_post_telemetry(self) -> None:
        with self.client as client:
            settings: Settings = self.app.state.settings

            # 1. Add dummy telemetry via HTTP
            post_resp = client.post(
                "/api/v1/telemetry",
                json={
                    "device_id": "motor_01",
                    "sequence": 1,
                    "rtt_ms": 50.0,
                    "temperature": 40.0,
                    "vibration": 0.25,
                    "current": 0.50,
                },
            )
            self.assertEqual(post_resp.status_code, 200)

            # Check that live buffer has data
            snapshot = self.app.state.live_data.snapshot(10)
            self.assertGreaterEqual(len(snapshot["telemetry"]), 1)

            # 2. Execute admin clear
            clear_resp = client.post(
                "/api/admin/clear-telemetry",
                headers={"X-Admin-Key": settings.admin_api_key},
                json={"confirm": True, "device_id": "motor_01"},
            )
            self.assertEqual(clear_resp.status_code, 200)
            body = clear_resp.json()

            self.assertEqual(body["status"], "success")
            self.assertIn("deleted_counts", body)
            self.assertEqual(body["deleted_counts"]["total"], 1)
            self.assertIn("preserved", body)
            self.assertIn("device_status", body["preserved"])
            self.assertEqual(body["device_id"], "motor_01")

            # 3. Check that in-memory buffers are now cleared
            cleared_snapshot = self.app.state.live_data.snapshot(10)
            self.assertEqual(len(cleared_snapshot["telemetry"]), 0)
            self.assertEqual(cleared_snapshot["total_received"], 0)

            # 4. Verify that new ESP32 telemetry can immediately be stored normally
            new_telemetry_resp = client.post(
                "/api/v1/telemetry",
                json={
                    "device_id": "motor_01",
                    "sequence": 2,
                    "rtt_ms": 45.0,
                    "temperature": 39.5,
                    "vibration": 0.20,
                    "current": 0.48,
                },
            )
            self.assertEqual(new_telemetry_resp.status_code, 200)
            post_snapshot = self.app.state.live_data.snapshot(10)
            self.assertEqual(len(post_snapshot["telemetry"]), 1)
            self.assertEqual(post_snapshot["telemetry"][0]["sequence"], 2)

    def test_bearer_token_auth(self) -> None:
        with self.client as client:
            settings: Settings = self.app.state.settings
            resp = client.post(
                "/api/v1/admin/clear-telemetry",
                headers={"Authorization": f"Bearer {settings.admin_api_key}"},
                json={"confirm": True},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "success")

    def test_query_param_auth_and_confirm(self) -> None:
        with self.client as client:
            settings: Settings = self.app.state.settings
            resp = client.post(
                f"/api/admin/clear-telemetry?admin_key={settings.admin_api_key}&confirm=true"
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
