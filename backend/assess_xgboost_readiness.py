"""Assess saved XGBoost deployment artifacts without fitting a model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import statistics
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, log_loss
from xgboost import XGBClassifier, __version__ as xgboost_version

from backend.settings import BACKEND_DIR

RF_ARTIFACT_DIR = BACKEND_DIR / "models" / "random_forest"
XGB_ARTIFACT_DIR = BACKEND_DIR / "models" / "xgboost"
CLASS_LABELS = ("NORMAL", "HIGH_VIBRATION", "HIGH_TEMPERATURE", "HIGH_CURRENT")
LATENCY_REPEATS = 50
LATENCY_SAMPLES_PER_CLASS = 10
CALIBRATION_BINS = 10


def _brier_score(probabilities: np.ndarray, true_indices: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[true_indices]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    probabilities: np.ndarray,
    true_indices: np.ndarray,
    bins: int = CALIBRATION_BINS,
) -> tuple[float, list[dict[str, Any]]]:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    details: list[dict[str, Any]] = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        selected = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        count = int(selected.sum())
        if not count:
            continue
        mean_confidence = float(confidences[selected].mean())
        accuracy = float((predictions[selected] == true_indices[selected]).mean())
        ece += count / len(true_indices) * abs(accuracy - mean_confidence)
        details.append(
            {
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "records": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), details


def _load_test_rows(
    combined_path: Path,
    split_indices: dict[str, Any],
    feature_columns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_keys: set[tuple[str, int]] = set()
    test_keys: set[tuple[str, int]] = set()
    expected_counts: dict[str, int] = {}
    for label in CLASS_LABELS:
        train_items = split_indices[label]["training"]
        test_items = split_indices[label]["testing"]
        train_keys.update(
            (item["source_file"], int(item["source_row"])) for item in train_items
        )
        test_keys.update(
            (item["source_file"], int(item["source_row"])) for item in test_items
        )
        expected_counts[label] = len(test_items)
    if train_keys & test_keys:
        raise ValueError("Saved training and test indices overlap")

    test_key_labels = {
        (item["source_file"], int(item["source_row"])): label
        for label in CLASS_LABELS
        for item in split_indices[label]["testing"]
    }
    found: dict[tuple[str, int], dict[str, Any]] = {}
    with combined_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"source_file", "source_row", "label", *feature_columns}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Combined dataset is missing columns: {missing}")
        for line_number, row in enumerate(reader, start=2):
            key = (row["source_file"], int(row["source_row"]))
            if key not in test_key_labels:
                continue
            label = test_key_labels[key]
            if row["label"] != label:
                raise ValueError(f"Combined dataset label mismatch at line {line_number}")
            values = np.asarray([float(row[name]) for name in feature_columns])
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite test feature at line {line_number}")
            if key in found:
                raise ValueError(f"Duplicate source provenance at line {line_number}")
            found[key] = {
                "source_file": key[0],
                "source_row": key[1],
                "label": label,
                "features": values,
            }

    if set(found) != test_keys:
        raise ValueError(
            f"Saved test indices reference {len(test_keys)} rows; "
            f"found {len(found)} in the combined dataset"
        )
    rows = list(found.values())
    counts = {
        label: sum(row["label"] == label for row in rows)
        for label in CLASS_LABELS
    }
    if counts != expected_counts:
        raise ValueError("Held-out test counts differ from the saved split index")
    return rows, {
        "training_indices": len(train_keys),
        "testing_indices": len(test_keys),
        "train_test_overlap": 0,
        "testing_records_per_class": counts,
    }


def _measure_single_record_latency(
    model: XGBClassifier,
    preprocessor: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_class: dict[str, list[float]] = {label: [] for label in CLASS_LABELS}
    for label in CLASS_LABELS:
        samples = [row for row in rows if row["label"] == label][
            :LATENCY_SAMPLES_PER_CLASS
        ]
        if not samples:
            raise ValueError(f"No unseen records available for {label} latency")
        for row in samples:
            raw = row["features"].reshape(1, -1)
            prepared = np.asarray(preprocessor.transform(raw), dtype=np.float32)
            model.predict_proba(prepared)
            for _ in range(LATENCY_REPEATS):
                started = time.perf_counter_ns()
                prepared = np.asarray(preprocessor.transform(raw), dtype=np.float32)
                model.predict_proba(prepared)
                by_class[label].append((time.perf_counter_ns() - started) / 1e6)
    all_times = [value for values in by_class.values() for value in values]
    return {
        "scope": "single record; includes preprocessing.transform and predict_proba",
        "warmup_calls_per_sample": 1,
        "repeats_per_sample": LATENCY_REPEATS,
        "samples_per_class": {
            label: len(values) // LATENCY_REPEATS for label, values in by_class.items()
        },
        "overall_ms": {
            "median": float(np.percentile(all_times, 50)),
            "p95": float(np.percentile(all_times, 95)),
            "mean": float(statistics.mean(all_times)),
        },
        "per_class_ms": {
            label: {
                "median": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            for label, values in by_class.items()
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "xgboost": xgboost_version,
            "note": "Local CPU/process timing, not end-to-end MQTT/backend latency.",
        },
    }


def assess_readiness(
    rf_artifact_dir: Path = RF_ARTIFACT_DIR,
    xgb_artifact_dir: Path = XGB_ARTIFACT_DIR,
) -> dict[str, Any]:
    paths = {
        "xgb_metrics": xgb_artifact_dir / "xgboost_metrics.json",
        "rf_metrics": rf_artifact_dir / "training_metrics.json",
        "combined_dataset": rf_artifact_dir / "combined_training_dataset.csv",
        "rf_features": rf_artifact_dir / "feature_columns.json",
        "rf_preprocessor": rf_artifact_dir / "preprocessing_pipeline.pkl",
        "xgb_model": xgb_artifact_dir / "xgboost_model.json",
        "xgb_preprocessor": xgb_artifact_dir / "xgboost_preprocessing.pkl",
        "label_mapping": xgb_artifact_dir / "xgboost_class_labels.json",
        "split_indices": xgb_artifact_dir / "split_indices.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing deployment artifact(s): " + ", ".join(missing))

    metrics = json.loads(paths["xgb_metrics"].read_text(encoding="utf-8"))
    rf_metrics = json.loads(paths["rf_metrics"].read_text(encoding="utf-8"))
    xgb_metadata = json.loads(paths["label_mapping"].read_text(encoding="utf-8"))
    rf_feature_metadata = json.loads(paths["rf_features"].read_text(encoding="utf-8"))
    split_indices = json.loads(paths["split_indices"].read_text(encoding="utf-8"))
    features = xgb_metadata["feature_columns"]
    labels = xgb_metadata["class_labels_by_encoded_value"]
    if features != rf_feature_metadata["feature_columns"]:
        raise ValueError("XGBoost feature order differs from Random Forest")
    if len(labels) != len(CLASS_LABELS) or set(labels) != set(CLASS_LABELS):
        raise ValueError(f"Unexpected class-label mapping: {labels}")

    preprocessing = pickle.loads(paths["xgb_preprocessor"].read_bytes())
    rf_preprocessing = pickle.loads(paths["rf_preprocessor"].read_bytes())
    xgb_preprocessing_params = preprocessing.get_params()
    rf_preprocessing_params = rf_preprocessing.get_params()
    missing_value_params_match = (
        np.isnan(xgb_preprocessing_params.pop("missing_values"))
        and np.isnan(rf_preprocessing_params.pop("missing_values"))
    )
    if (
        xgb_preprocessing_params != rf_preprocessing_params
        or not missing_value_params_match
        or not np.allclose(
            preprocessing.statistics_,
            rf_preprocessing.statistics_,
            rtol=0,
            atol=0,
        )
    ):
        raise ValueError("XGBoost preprocessing differs from saved Random Forest preprocessing")
    if int(preprocessing.n_features_in_) != len(features):
        raise ValueError("Preprocessing input count differs from the saved feature list")

    model = XGBClassifier()
    model.load_model(paths["xgb_model"])
    class_indices = [int(value) for value in model.classes_]
    if class_indices != list(range(len(labels))):
        raise ValueError(f"Model output classes do not align with label mapping: {class_indices}")
    if int(model.get_booster().num_features()) != len(features):
        raise ValueError("Model input count differs from the saved feature list")

    combined_hash = hashlib.sha256(paths["combined_dataset"].read_bytes()).hexdigest()
    if combined_hash != metrics["dataset"]["sha256"]:
        raise ValueError("Combined dataset changed since XGBoost training")
    source_hash_checks: dict[str, bool] = {}
    for label, source in rf_metrics["dataset_quality"]["source_files"].items():
        source_hash_checks[label] = (
            hashlib.sha256(Path(source["file"]).read_bytes()).hexdigest()
            == source["sha256"]
        )
    if not all(source_hash_checks.values()):
        raise ValueError("One or more original source CSVs changed after training")

    test_rows, split_check = _load_test_rows(
        paths["combined_dataset"], split_indices, features
    )
    if len(test_rows) != metrics["split"]["test_records"]:
        raise ValueError("Held-out records do not match XGBoost metrics")
    raw_test = np.vstack([row["features"] for row in test_rows])
    prepared_test = np.asarray(preprocessing.transform(raw_test), dtype=np.float32)
    true_labels = [row["label"] for row in test_rows]
    true_indices = np.asarray([labels.index(label) for label in true_labels])
    probabilities = model.predict_proba(prepared_test)
    if probabilities.shape != (len(test_rows), len(labels)):
        raise ValueError(f"Unexpected predict_proba shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("predict_proba contains invalid probability values")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("predict_proba rows do not sum to one")

    encoded_predictions = probabilities.argmax(axis=1)
    predicted_labels = [labels[index] for index in encoded_predictions]
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(CLASS_LABELS),
        target_names=list(CLASS_LABELS),
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    if abs(accuracy - metrics["xgboost"]["metrics"]["accuracy"]) > 1e-12:
        raise ValueError("Held-out accuracy differs from the saved XGBoost report")
    ece, calibration_bins = _expected_calibration_error(probabilities, true_indices)
    samples_by_class: dict[str, dict[str, Any]] = {}
    for label in CLASS_LABELS:
        index = next(i for i, row in enumerate(test_rows) if row["label"] == label)
        samples_by_class[label] = {
            "source_file": test_rows[index]["source_file"],
            "source_row": test_rows[index]["source_row"],
            "actual_class": true_labels[index],
            "predicted_class": predicted_labels[index],
            "predicted_probability": float(probabilities[index].max()),
            "probability_sum": float(probabilities[index].sum()),
            "correct": predicted_labels[index] == label,
        }

    latency = _measure_single_record_latency(model, preprocessing, test_rows)
    return {
        "readiness_status": "CONDITIONAL_NOT_PRODUCTION_READY",
        "model": {
            "format": "XGBoost JSON",
            "path": str(paths["xgb_model"]),
            "xgboost_version": xgboost_version,
            "feature_columns_in_order": features,
            "class_labels_by_probability_column": labels,
        },
        "artifact_checks": {
            "all_required_artifacts_present": True,
            "preprocessing_available_and_matches_rf": True,
            "feature_order_matches_rf": True,
            "class_mapping_matches_model_outputs": True,
            "combined_dataset_hash_matches_training": True,
            "source_csv_hashes_unchanged": source_hash_checks,
        },
        "preprocessing": {
            "artifact": str(paths["xgb_preprocessor"]),
            "operation": "Saved median SimpleImputer from the Random Forest preprocessing pipeline",
            "feature_count": int(preprocessing.n_features_in_),
            "deployment_note": (
                "Derive acceleration_magnitude as sqrt(x^2+y^2+z^2), preserve source field mappings, "
                "and send values in feature_columns_in_order."
            ),
        },
        "held_out_test": {
            "records": len(test_rows),
            "records_per_class": split_check["testing_records_per_class"],
            "training_indices": split_check["training_indices"],
            "test_indices": split_check["testing_indices"],
            "train_test_index_overlap": split_check["train_test_overlap"],
            "accuracy": accuracy,
            "classification_report": report,
            "unseen_sample_predictions": samples_by_class,
        },
        "predict_proba": {
            "shape": list(probabilities.shape),
            "finite_nonnegative": True,
            "rows_sum_to_one": True,
            "row_sum_min": float(sums.min()),
            "row_sum_max": float(sums.max()),
            "multiclass_brier_score": _brier_score(probabilities, true_indices),
            "log_loss": float(
                log_loss(true_indices, probabilities, labels=list(range(len(labels))))
            ),
            "top_label_ece_10_bins": ece,
            "top_label_calibration_bins": calibration_bins,
            "confidence_warning": (
                "These are model probabilities, not established calibrated confidence. "
                "The held-out diagnostics are descriptive; no calibration transform was fit."
            ),
        },
        "single_record_latency": latency,
        "leakage_and_data_limitations": [
            "Test rows are the saved chronological tail of the same capture files, not independent runs.",
            "No session/run ID exists, so generalization to a new run is not established.",
            "Gyroscope axes and estimated_vibration were unavailable in training and are not model inputs.",
            "The deployment feature mapping and acceleration magnitude calculation must match training exactly.",
        ],
        "overfitting_check": {
            "train_accuracy": metrics["xgboost"]["training_accuracy"],
            "test_accuracy": accuracy,
            "accuracy_gap": metrics["xgboost"]["training_accuracy"] - accuracy,
            "interpretation": "Small observed same-capture accuracy gap; independent-run performance remains unknown.",
        },
        "deployment_gates": [
            "Collect independent runs for every class and evaluate on a held-out run.",
            "Assess and calibrate probabilities on a separate calibration dataset before using them as alert confidence.",
            "Verify real MQTT payload mapping and end-to-end backend latency before integration.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    held_out = report["held_out_test"]
    probability_report = report["predict_proba"]
    latency = report["single_record_latency"]
    lines = [
        "# XGBoost Deployment Readiness",
        "",
        f"Status: **{report['readiness_status']}**",
        "",
        "## Artifact Verification",
        "",
        f"- Model: `{report['model']['path']}`",
        f"- Ordered features: `{', '.join(report['model']['feature_columns_in_order'])}`",
        f"- Probability-column labels: `{', '.join(report['model']['class_labels_by_probability_column'])}`",
        "- Saved preprocessing is available and matches Random Forest preprocessing.",
        "",
        "## Held-Out Records",
        "",
        f"- Tested records: {held_out['records']}; per class: {held_out['records_per_class']}.",
        f"- Accuracy: {held_out['accuracy']:.4f}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in CLASS_LABELS:
        scores = held_out["classification_report"][label]
        lines.append(
            f"| {label} | {scores['precision']:.4f} | {scores['recall']:.4f} | "
            f"{scores['f1-score']:.4f} | {int(scores['support'])} |"
        )
    lines.extend(
        [
            "",
            "## Probabilities And Latency",
            "",
            f"- Probability rows sum to one: `{probability_report['rows_sum_to_one']}`.",
            f"- Multiclass Brier score: {probability_report['multiclass_brier_score']:.6f}",
            f"- Log loss: {probability_report['log_loss']:.6f}",
            f"- Top-label ECE (10 bins): {probability_report['top_label_ece_10_bins']:.6f}",
            f"- Calibration caveat: {probability_report['confidence_warning']}",
            f"- Single-record latency including preprocessing: median {latency['overall_ms']['median']:.4f} ms; "
            f"p95 {latency['overall_ms']['p95']:.4f} ms.",
            "",
            "## Risks And Deployment Gates",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["leakage_and_data_limitations"])
    lines.extend(f"- Gate: {item}" for item in report["deployment_gates"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess saved XGBoost deployment artifacts without retraining."
    )
    parser.add_argument("--rf-artifact-dir", type=Path, default=RF_ARTIFACT_DIR)
    parser.add_argument("--xgb-artifact-dir", type=Path, default=XGB_ARTIFACT_DIR)
    args = parser.parse_args()
    rf_dir = args.rf_artifact_dir.expanduser().resolve()
    xgb_dir = args.xgb_artifact_dir.expanduser().resolve()
    report = assess_readiness(rf_dir, xgb_dir)
    json_path = xgb_dir / "deployment_readiness_report.json"
    markdown_path = xgb_dir / "deployment_readiness_report.md"
    report["report_artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["readiness_status"],
                "test_accuracy": report["held_out_test"]["accuracy"],
                "brier_score": report["predict_proba"]["multiclass_brier_score"],
                "log_loss": report["predict_proba"]["log_loss"],
                "top_label_ece": report["predict_proba"]["top_label_ece_10_bins"],
                "single_record_latency_ms": report["single_record_latency"]["overall_ms"],
                "reports": report["report_artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()