"""Isolation Forest model loading, training, persistence, and inference."""

from collections.abc import Sequence
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestModel:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model: IsolationForest | None = None
        self._feature_names: tuple[str, ...] = ()
        self._artifact_metadata: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def feature_count(self) -> int | None:
        return len(self._feature_names) if self._feature_names else None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def artifact_metadata(self) -> dict[str, Any]:
        return self._artifact_metadata

    def load(self) -> bool:
        if not self._model_path.is_file():
            return False
        loaded: Any = joblib.load(self._model_path)
        if isinstance(loaded, IsolationForest):
            model = loaded
            width = getattr(model, "n_features_in_", None)
            if not isinstance(width, int) or width < 1:
                raise ValueError("Saved model has an invalid feature count")
            feature_names = tuple(f"feature_{index + 1}" for index in range(width))
            metadata: dict[str, Any] = {"artifact_version": 1, "legacy": True}
        elif isinstance(loaded, dict):
            model = loaded.get("model")
            feature_names_value = loaded.get("feature_names")
            metadata_value = loaded.get("metadata", {})
            if not isinstance(model, IsolationForest):
                raise ValueError(f"Unexpected model type in {self._model_path}")
            if not isinstance(feature_names_value, list) or not all(
                isinstance(name, str) for name in feature_names_value
            ):
                raise ValueError("Saved model artifact has invalid feature names")
            if not isinstance(metadata_value, dict):
                raise ValueError("Saved model artifact has invalid metadata")
            feature_names = tuple(feature_names_value)
            width = getattr(model, "n_features_in_", None)
            if width != len(feature_names) or not feature_names:
                raise ValueError("Saved model feature names do not match model inputs")
            metadata = metadata_value
        else:
            raise ValueError(f"Unexpected model type in {self._model_path}")
        self._model = model
        self._feature_names = feature_names
        self._artifact_metadata = metadata
        logger.info("Loaded trained Isolation Forest", extra={"model_path": str(self._model_path)})
        return True

    def set_anomaly_score_threshold(self, threshold: float) -> None:
        if self._model is None:
            raise RuntimeError("Cannot calibrate an untrained Isolation Forest")
        if not math.isfinite(threshold):
            raise ValueError("Anomaly score threshold must be finite")
        self._artifact_metadata = {
            **self._artifact_metadata,
            "anomaly_score_threshold": threshold,
        }

    def fit(
        self,
        training_features: Sequence[Sequence[float]],
        minimum_samples: int,
        *,
        feature_names: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> None:
        rows = [self._validated_row(row) for row in training_features]
        if len(rows) < minimum_samples:
            raise ValueError(
                f"Need at least {minimum_samples} valid real telemetry records; "
                f"got {len(rows)}"
            )
        feature_count = len(rows[0])
        if any(len(row) != feature_count for row in rows):
            raise ValueError("All training rows must have the same feature count")
        names = tuple(
            feature_names
            if feature_names is not None
            else (f"feature_{index + 1}" for index in range(feature_count))
        )
        if len(names) != feature_count or not all(names):
            raise ValueError("Feature names must match the training feature count")
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )
        model.fit(rows)
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_metadata = dict(metadata or {})
        if score_threshold is not None:
            if not math.isfinite(score_threshold):
                raise ValueError("Anomaly score threshold must be finite")
            artifact_metadata["anomaly_score_threshold"] = score_threshold
        artifact = {
            "artifact_version": 2,
            "model": model,
            "feature_names": list(names),
            "metadata": artifact_metadata,
        }
        with tempfile.NamedTemporaryFile(
            dir=self._model_path.parent,
            prefix=f".{self._model_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            joblib.dump(artifact, temporary_path)
            os.replace(temporary_path, self._model_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._model = model
        self._feature_names = names
        self._artifact_metadata = artifact_metadata
        logger.info(
            "Trained Isolation Forest from telemetry records",
            extra={
                "training_samples": len(rows),
                "feature_count": feature_count,
                "model_path": str(self._model_path),
            },
        )

    def predict(self, features: Sequence[float]) -> tuple[str, float]:
        if self._model is None:
            raise RuntimeError("Isolation Forest has not been trained")
        row = self._validated_row(features)
        if len(row) != self._model.n_features_in_:
            raise ValueError(
                f"Model expects {self._model.n_features_in_} features; got {len(row)}"
            )
        anomaly_score = -float(self._model.score_samples([row])[0])
        threshold = self._artifact_metadata.get("anomaly_score_threshold")
        if isinstance(threshold, (int, float)) and math.isfinite(threshold):
            label = "ANOMALY" if anomaly_score > threshold else "NORMAL"
        else:
            label = "ANOMALY" if self._model.predict([row])[0] == -1 else "NORMAL"
        return label, anomaly_score

    @staticmethod
    def _validated_row(values: Sequence[float]) -> list[float]:
        if not values:
            raise ValueError("Isolation Forest requires at least one feature")
        row: list[float] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError("Feature values must be finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Feature values must be finite numbers")
            row.append(number)
        return row
