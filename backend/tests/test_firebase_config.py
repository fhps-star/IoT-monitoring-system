import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from backend.firebase_config import FirebaseStore
from backend.settings import Settings


class FirebaseStoreTests(unittest.TestCase):
    def test_push_writes_record_with_its_firebase_key(self) -> None:
        settings = Settings.from_env()
        store = FirebaseStore(settings)
        reference = Mock()
        reference.key = "firebase-key"
        root = Mock()
        root.child.return_value.push.return_value = reference
        store._database = root

        record_id = store.push("telemetry/motor_01", {"source": "mqtt"})

        self.assertEqual(record_id, "firebase-key")
        reference.set.assert_called_once_with({
            "source": "mqtt",
            "id": "firebase-key",
        })

    def test_get_records_uses_database_key_as_record_id(self) -> None:
        settings = Settings.from_env()
        store = FirebaseStore(settings)
        child = Mock()
        child.order_by_key.return_value.limit_to_last.return_value.get.return_value = {
            "firebase-key": {"id": "untrusted-payload-id", "source": "mqtt"}
        }
        root = Mock()
        root.child.return_value = child
        store._database = root

        records = store.get_records("telemetry/motor_01", 10)

        self.assertEqual(records[0]["id"], "firebase-key")

    def test_set_record_uses_telemetry_id_as_idempotent_child_key(self) -> None:
        settings = Settings.from_env()
        store = FirebaseStore(settings)
        record_reference = Mock()
        collection_reference = Mock()
        collection_reference.child.return_value = record_reference
        root = Mock()
        root.child.return_value = collection_reference
        store._database = root

        record_id = store.set_record(
            "xgboost_predictions/motor_01",
            "firebase-telemetry-key",
            {"predicted_class": "HIGH_VIBRATION"},
        )

        self.assertEqual(record_id, "firebase-telemetry-key")
        collection_reference.child.assert_called_once_with("firebase-telemetry-key")
        record_reference.set.assert_called_once_with(
            {
                "predicted_class": "HIGH_VIBRATION",
                "id": "firebase-telemetry-key",
            }
        )


if __name__ == "__main__":
    unittest.main()
