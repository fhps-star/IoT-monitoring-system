"""Load and run the saved multiclass Random Forest motor classifier."""

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from backend.settings import BACKEND_DIR

logger = logging.getLogger(__name__)
ARTIFACT_DIR = BACKEND_DIR / "models" / "random_forest"
EXPECTED_CLASSES = {
    "NORMAL",
    "HIGH_VIBRATION",
    "HIGH_TEMPERATURE",
    "HIGH_CURRENT",
}


class RandomForestPredictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        self._artifact_dir = artifact_dir
        self._model = None
        self._preprocessor = None
        self._feature_columns = ()
        self._class_labels = ()
        self._load_error = None

    @property
    def is_ready(self) -> bool:
        return bool(
            self._model is not None
            and self._preprocessor is not None
            and self._feature_columns
            and self._class_labels
        )

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def class_labels(self) -> tuple[str, ...]:
        return self._class_labels

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model_path(self) -> Path:
        return self._artifact_dir / "random_forest_model.pkl"

    def load(self) -> bool:
        model_path = self.model_path
        preprocessing_path = self._artifact_dir / "preprocessing_pipeline.pkl"
        features_path = self._artifact_dir / "feature_columns.json"
        required = (model_path, preprocessing_path, features_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            self._load_error = "Missing Random Forest artifact(s): " + ", ".join(missing)
            return False

        feature_metadata = json.loads(features_path.read_text(encoding="utf-8"))
        feature_columns = feature_metadata.get("feature_columns")
        if not isinstance(feature_columns, list) or not feature_columns or not all(
            isinstance(name, str) for name in feature_columns
        ):
            raise ValueError("Random Forest feature metadata is invalid")

        pipeline = pickle.loads(model_path.read_bytes())
        preprocessor_saved = pickle.loads(preprocessing_path.read_bytes())
        if hasattr(pipeline, "named_steps"):
            model = pipeline.named_steps.get("classifier", pipeline)
            embedded_preprocessor = pipeline.named_steps.get("preprocessor")
            if embedded_preprocessor is not None and isinstance(
                embedded_preprocessor, SimpleImputer
            ):
                preprocessor_saved = embedded_preprocessor
        else:
            model = pipeline

        if not isinstance(model, RandomForestClassifier):
            raise ValueError("Saved Random Forest model is not a RandomForestClassifier")
        if not isinstance(preprocessor_saved, SimpleImputer):
            raise ValueError("Saved Random Forest preprocessing is not a SimpleImputer")
        if int(preprocessor_saved.n_features_in_) != len(feature_columns):
            raise ValueError("Random Forest preprocessor feature width does not match metadata")
        if int(getattr(model, "n_features_in_", 0)) != len(feature_columns):
            raise ValueError("Random Forest model feature width does not match metadata")

        class_labels = tuple(str(value) for value in getattr(model, "classes_", ()))
        if not class_labels or set(class_labels) != EXPECTED_CLASSES:
            raise ValueError("Random Forest class-label mapping is invalid")

        self._model = model
        self._preprocessor = preprocessor_saved
        self._feature_columns = tuple(feature_columns)
        self._class_labels = class_labels
        self._load_error = None
        logger.info("Loaded trained Random Forest", extra={"model_path": str(model_path)})
        return True

    def predict(self, telemetry: dict[str, Any]) -> dict[str, Any] | None:
        if (
            self._model is None
            or self._preprocessor is None
            or not self._feature_columns
            or not self._class_labels
        ):
            return None
        sample = telemetry.get("model_sample")
        if not isinstance(sample, dict):
            return None
        source_fields = {
            "temperature_c": "temperature_c",
            "acceleration_x": "vibration_x",
            "acceleration_y": "vibration_y",
            "acceleration_z": "vibration_z",
            "motor_speed": "motor_speed",
            "motor_current_a": "current_a",
        }
        values: dict[str, float] = {}
        for feature, source in source_fields.items():
            value = sample.get(source)
            if isinstance(value, bool) or value is None:
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(numeric):
                return None
            values[feature] = numeric
        values["acceleration_magnitude"] = math.sqrt(
            values["acceleration_x"] ** 2
            + values["acceleration_y"] ** 2
            + values["acceleration_z"] ** 2
        )
        if any(feature not in values for feature in self._feature_columns):
            return None

        raw_row = np.asarray(
            [[values[feature] for feature in self._feature_columns]],
            dtype=np.float64,
        )
        prepared = np.asarray(self._preprocessor.transform(raw_row), dtype=np.float32)
        probabilities = np.asarray(self._model.predict_proba(prepared)[0], dtype=float)
        if (
            len(probabilities) != len(self._class_labels)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-5, abs_tol=1e-6)
        ):
            raise ValueError("Random Forest returned invalid class probabilities")

        class_index = int(probabilities.argmax())
        predicted_class = self._class_labels[class_index]
        raw_payload = telemetry.get("raw_payload")
        sequence = raw_payload.get("sequence") if isinstance(raw_payload, dict) else None
        prediction = {
            "device_id": telemetry["device_id"],
            "telemetry_record_id": telemetry["id"],
            "predicted_class": predicted_class,
            "confidence": float(probabilities[class_index]),
            "class_probabilities": {
                label: float(probabilities[index])
                for index, label in enumerate(self._class_labels)
            },
            "model_name": "Random Forest",
            "prediction_timestamp": telemetry["received_at"],
            "anomaly_status": "NORMAL" if predicted_class == "NORMAL" else "AI_ANOMALY_PREDICTION",
            "confidence_calibrated": False,
            "confidence_note": "Model probability; not calibrated for real-world confidence.",
            "sensor_values": {
                "temperature_c": values["temperature_c"],
                "acceleration_x": values["acceleration_x"],
                "acceleration_y": values["acceleration_y"],
                "acceleration_z": values["acceleration_z"],
                "acceleration_magnitude": values["acceleration_magnitude"],
                "motor_speed": values["motor_speed"],
                "motor_current_a": values["motor_current_a"],
            },
        }
        if isinstance(sequence, (int, float, str)) and not isinstance(sequence, bool):
            prediction["telemetry_sequence"] = sequence
        return prediction
