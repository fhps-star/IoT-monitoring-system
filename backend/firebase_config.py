"""Firebase Realtime Database initialization and data access."""

from collections.abc import Mapping
import logging
from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, db
from firebase_admin.exceptions import FirebaseError
from google.auth.exceptions import GoogleAuthError
from requests import RequestException

from backend.settings import Settings

logger = logging.getLogger(__name__)


class FirebaseStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._database: Any | None = None
        self.last_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.initialized and self.last_error is None

    @property
    def initialized(self) -> bool:
        return self._database is not None

    def initialize(self) -> None:
        logger.info(
            "Initializing Firebase Realtime Database",
            extra={"database_url": self._settings.firebase_database_url},
        )
        credential_path = self._settings.google_application_credentials
        if credential_path:
            path = Path(credential_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Firebase credentials file does not exist: {path}"
                )
            credential = credentials.Certificate(str(path))
        else:
            credential = credentials.ApplicationDefault()
        credential.get_credential()

        try:
            app = firebase_admin.get_app("motor-monitoring")
        except ValueError:
            app = firebase_admin.initialize_app(
                credential,
                {"databaseURL": self._settings.firebase_database_url},
                name="motor-monitoring",
            )
        self._database = db.reference("/", app=app)
        self.last_error = None

    def _reference(self, path: str) -> Any:
        if self._database is None:
            raise RuntimeError("Firebase is not initialized")
        return self._database.child(path)

    def push(self, path: str, value: dict[str, Any]) -> str | None:
        key: str | None = None
        try:
            reference = self._reference(path).push()
            key = str(reference.key) if reference.key is not None else None
            if key is None:
                raise RuntimeError(f"Firebase did not return a key for '{path}'")
            reference.set({**value, "id": key})
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            self.last_error = str(exc)
            self._log_write_failed(path, key, exc)
            return None
        self.last_error = None
        logger.info(
            "Firebase record stored",
            extra={"firebase_path": path, "firebase_key": key},
        )
        return key

    def set_record(
        self,
        path: str,
        record_id: str,
        value: dict[str, Any],
    ) -> str | None:
        try:
            self._reference(path).child(record_id).set({**value, "id": record_id})
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
            RuntimeError,
        ) as exc:
            self.last_error = str(exc)
            self._log_write_failed(path, record_id, exc)
            return None
        self.last_error = None
        logger.info(
            "Firebase record stored idempotently",
            extra={"firebase_path": path, "firebase_key": record_id},
        )
        return record_id

    def update_value(self, path: str, value: dict[str, Any]) -> bool:
        try:
            self._reference(path).update(value)
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
        ) as exc:
            self.last_error = str(exc)
            self._log_write_failed(path, None, exc)
            return False
        self.last_error = None
        logger.info("Firebase status updated", extra={"firebase_path": path})
        return True

    def get_value(self, path: str) -> Any:
        try:
            value = self._reference(path).get()
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
        ) as exc:
            self.last_error = str(exc)
            raise RuntimeError(f"Firebase read failed at '{path}': {exc}") from exc
        self.last_error = None
        return value

    def get_records(self, path: str, limit: int) -> list[dict[str, Any]]:
        try:
            value = self._reference(path).order_by_key().limit_to_last(limit).get()
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
        ) as exc:
            self.last_error = str(exc)
            raise RuntimeError(f"Firebase read failed at '{path}': {exc}") from exc
        self.last_error = None
        if not isinstance(value, Mapping):
            return []
        return [
            {**record, "id": str(key)}
            for key, record in value.items()
            if isinstance(record, Mapping)
        ]

    def delete_path(self, path: str) -> int:
        """Delete a path in Firebase Realtime Database and return count of deleted items."""
        count = 0
        try:
            current = self._reference(path).get()
            if current is not None:
                if isinstance(current, Mapping):
                    count = len(current)
                elif isinstance(current, list):
                    count = len([x for x in current if x is not None])
                else:
                    count = 1
                self._reference(path).delete()
        except (
            FirebaseError,
            GoogleAuthError,
            RequestException,
            OSError,
            ValueError,
        ) as exc:
            self.last_error = str(exc)
            logger.warning("Failed to delete path '%s' in Firebase: %s", path, exc)
            return 0
        self.last_error = None
        logger.info("Deleted %d records at Firebase path '%s'", count, path)
        return count

    def _delete_firestore_subcollection(
        self, fs_db: Any, coll_ref: Any, batch_size: int = 500
    ) -> int:
        """Batch delete Firestore collection and all recursive subcollections."""
        deleted = 0
        try:
            docs = coll_ref.limit(batch_size).stream()
            batch = fs_db.batch()
            count = 0
            for doc in docs:
                for subcoll in doc.reference.collections():
                    deleted += self._delete_firestore_subcollection(fs_db, subcoll, batch_size)
                batch.delete(doc.reference)
                count += 1
                deleted += 1
            if count > 0:
                batch.commit()
                if count >= batch_size:
                    deleted += self._delete_firestore_subcollection(fs_db, coll_ref, batch_size)
        except Exception as exc:
            logger.debug("Firestore subcollection deletion skipped: %s", exc)
        return deleted

    def clear_telemetry_records(
        self, device_id: str | None = None
    ) -> dict[str, Any]:
        """Delete all old telemetry, predictions, alerts, and comparisons, preserving settings."""
        collections_to_clear = [
            "telemetry",
            "predictions",
            "alerts",
            "random_forest_predictions",
            "xgboost_predictions",
            "anomaly_alerts",
            "model_comparisons",
            "protocol_comparison",
            "protocol_status",
        ]
        deleted_counts: dict[str, int] = {}
        total = 0

        for coll in collections_to_clear:
            target_path = f"{coll}/{device_id}" if device_id else coll
            c = self.delete_path(target_path)
            deleted_counts[coll] = c
            total += c

        firestore_deleted = 0
        try:
            from firebase_admin import firestore
            app = firebase_admin.get_app("motor-monitoring")
            fs_db = firestore.client(app=app)
            fs_collections = [
                "telemetry",
                "predictions",
                "alerts",
                "anomalies",
                "anomaly_alerts",
                "model_comparisons",
                "experiments",
            ]
            for fs_name in fs_collections:
                coll_ref = fs_db.collection(fs_name)
                if device_id:
                    # Query where device_id matches
                    docs = coll_ref.where("device_id", "==", device_id).limit(500).stream()
                    batch = fs_db.batch()
                    b_count = 0
                    for doc in docs:
                        for subcoll in doc.reference.collections():
                            firestore_deleted += self._delete_firestore_subcollection(fs_db, subcoll)
                        batch.delete(doc.reference)
                        b_count += 1
                        firestore_deleted += 1
                    if b_count > 0:
                        batch.commit()
                    # Also check if device_id is a document
                    doc_ref = coll_ref.document(device_id)
                    doc_snap = doc_ref.get()
                    if doc_snap.exists:
                        for subcoll in doc_ref.collections():
                            firestore_deleted += self._delete_firestore_subcollection(fs_db, subcoll)
                        doc_ref.delete()
                        firestore_deleted += 1
                else:
                    firestore_deleted += self._delete_firestore_subcollection(fs_db, coll_ref)
        except Exception as exc:
            logger.debug("Firestore cleanup not applicable or skipped: %s", exc)

        deleted_counts["firestore_documents"] = firestore_deleted
        deleted_counts["total"] = total + firestore_deleted

        return {
            "status": "success",
            "deleted_counts": deleted_counts,
            "preserved": [
                "device_status",
                "network_metrics",
                "device_configurations",
                "application_settings",
                "credentials",
            ],
            "device_id": device_id,
        }

    @staticmethod
    def _log_write_failed(
        path: str, record_id: str | None, error: Exception
    ) -> None:
        logger.error(
            "Firebase write failed",
            exc_info=(type(error), error, error.__traceback__),
            extra={"firebase_path": path, "firebase_key": record_id},
        )
