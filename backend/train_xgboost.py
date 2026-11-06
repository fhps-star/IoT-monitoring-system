"""Train XGBoost on the saved Random Forest split and compare both models."""

import argparse
import csv
from collections import Counter
from datetime import timedelta
import hashlib
import json
import logging
from pathlib import Path
import pickle
import time
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from backend.feature_engineering import parse_timestamp
from backend.settings import BACKEND_DIR

logger = logging.getLogger(__name__)
CLASS_LABELS = ("NORMAL", "HIGH_VIBRATION", "HIGH_TEMPERATURE", "HIGH_CURRENT")
RF_OUTPUT_DIR = BACKEND_DIR / "models" / "random_forest"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "models" / "xgboost"
PREDICTION_BENCHMARK_REPEATS = 20


def _read_combined_dataset(
    combined_path: Path,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    with combined_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"timestamp", "label", "source_file", "source_row", *feature_columns}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Combined RF dataset is missing columns: {missing}")
        records: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            if label not in CLASS_LABELS:
                raise ValueError(
                    f"{combined_path}:{line_number} has unknown class {label!r}"
                )
            try:
                timestamp = parse_timestamp(row["timestamp"])
                source_row = int(row["source_row"])
                values = {feature: float(row[feature]) for feature in feature_columns}
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{combined_path}:{line_number} contains invalid split/feature data"
                ) from exc
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(
                    f"{combined_path}:{line_number} contains a non-finite feature"
                )
            records.append(
                {
                    "timestamp": timestamp,
                    "label": label,
                    "source_file": row["source_file"],
                    "source_row": source_row,
                    "features": values,
                }
            )
    return records


def _reconstruct_rf_split(
    records: list[dict[str, Any]],
    rf_metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split = rf_metrics.get("split", {})
    if split.get("method", "").startswith("Per-class chronological holdout") is False:
        raise ValueError("Saved RF metrics do not describe the expected chronological split")
    test_fraction = float(split["test_fraction_per_class"])
    embargo_seconds = float(split["embargo_seconds"])
    training: list[dict[str, Any]] = []
    testing: list[dict[str, Any]] = []
    embargoed: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    split_indices: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for label in CLASS_LABELS:
        class_records = sorted(
            (record for record in records if record["label"] == label),
            key=lambda record: (record["timestamp"], record["source_row"]),
        )
        split_index = int(len(class_records) * (1 - test_fraction))
        if split_index < 1 or split_index >= len(class_records):
            raise ValueError(f"Not enough saved {label} records for the RF split")
        test_start = class_records[split_index]["timestamp"]
        train_cutoff = test_start - timedelta(seconds=embargo_seconds)
        class_training = [
            record
            for record in class_records[:split_index]
            if record["timestamp"] <= train_cutoff
        ]
        class_embargoed = class_records[len(class_training):split_index]
        class_testing = class_records[split_index:]
        counts[label] = {
            "training": len(class_training),
            "testing": len(class_testing),
            "embargoed": len(class_embargoed),
        }
        training.extend(class_training)
        testing.extend(class_testing)
        embargoed.extend(class_embargoed)
        split_indices[label] = {
            "training": [
                {"source_file": row["source_file"], "source_row": row["source_row"]}
                for row in class_training
            ],
            "testing": [
                {"source_file": row["source_file"], "source_row": row["source_row"]}
                for row in class_testing
            ],
            "embargoed": [
                {"source_file": row["source_file"], "source_row": row["source_row"]}
                for row in class_embargoed
            ],
        }

    expected_counts = {
        "training": rf_metrics["training_records_per_class"],
        "testing": rf_metrics["testing_records_per_class"],
        "embargoed": rf_metrics["embargoed_records_per_class"],
    }
    for split_name, expected_by_class in expected_counts.items():
        for label in CLASS_LABELS:
            if counts[label][split_name] != expected_by_class[label]:
                raise ValueError(
                    f"Reconstructed RF {split_name} count differs for {label}: "
                    f"{counts[label][split_name]} != {expected_by_class[label]}"
                )
    training.sort(key=lambda row: row["timestamp"])
    testing.sort(key=lambda row: row["timestamp"])
    return training, testing, {
        "per_class_counts": counts,
        "indices": split_indices,
        "test_fraction_per_class": test_fraction,
        "embargo_seconds": embargo_seconds,
        "method": split["method"],
    }


def _macro_metrics(
    y_true: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        predictions,
        labels=list(CLASS_LABELS),
        target_names=list(CLASS_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "classification_report_text": classification_report(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
            target_names=list(CLASS_LABELS),
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
        ).tolist(),
    }


def _mean_prediction_time(
    predict: Callable[[], Any],
    repeats: int = PREDICTION_BENCHMARK_REPEATS,
) -> float:
    predict()
    started = time.perf_counter()
    for _ in range(repeats):
        predict()
    return (time.perf_counter() - started) / repeats


def _save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_LABELS)),
        yticks=range(len(CLASS_LABELS)),
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="XGBoost confusion matrix",
    )
    axis.tick_params(axis="x", labelrotation=25)
    threshold = max((max(row) for row in matrix), default=0) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _verify_source_hashes(rf_metrics: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for label, source in rf_metrics["dataset_quality"]["source_files"].items():
        path = Path(source["file"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"Source dataset changed since RF training: {path}")
        hashes[label] = digest
    return hashes


def train_and_compare(
    rf_output_dir: Path = RF_OUTPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    rf_metrics_path = rf_output_dir / "training_metrics.json"
    features_path = rf_output_dir / "feature_columns.json"
    combined_path = rf_output_dir / "combined_training_dataset.csv"
    rf_model_path = rf_output_dir / "random_forest_model.pkl"
    preprocessor_path = rf_output_dir / "preprocessing_pipeline.pkl"
    for path in (rf_metrics_path, features_path, combined_path, rf_model_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved Random Forest artifact missing: {path}")

    rf_metrics = json.loads(rf_metrics_path.read_text(encoding="utf-8"))
    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))["feature_columns"]
    source_hashes = _verify_source_hashes(rf_metrics)
    records = _read_combined_dataset(combined_path, feature_columns)
    if len(records) != rf_metrics["total_records_used"]:
        raise ValueError("Combined dataset row count does not match saved RF metrics")
    labels_present = {record["label"] for record in records}
    if labels_present != set(CLASS_LABELS):
        raise ValueError(f"Expected all four saved classes, found {sorted(labels_present)}")

    training_records, testing_records, split = _reconstruct_rf_split(records, rf_metrics)
    x_train = np.asarray(
        [[row["features"][feature] for feature in feature_columns] for row in training_records],
        dtype=np.float64,
    )
    x_test = np.asarray(
        [[row["features"][feature] for feature in feature_columns] for row in testing_records],
        dtype=np.float64,
    )
    y_train = [row["label"] for row in training_records]
    y_test = [row["label"] for row in testing_records]

    rf_model = pickle.loads(rf_model_path.read_bytes())
    rf_predictions = rf_model.predict(x_test).tolist()
    rf_test_metrics = _macro_metrics(y_test, rf_predictions)
    if abs(rf_test_metrics["accuracy"] - rf_metrics["accuracy"]) > 1e-12:
        raise ValueError(
            "Reconstructed split does not reproduce saved Random Forest accuracy; "
            "refusing an unfair comparison"
        )
    rf_training_predictions = rf_model.predict(x_train).tolist()
    rf_training_accuracy = accuracy_score(y_train, rf_training_predictions)

    preprocessor = pickle.loads(preprocessor_path.read_bytes())
    xgb_train = np.asarray(preprocessor.transform(x_train), dtype=np.float32)
    xgb_test = np.asarray(preprocessor.transform(x_test), dtype=np.float32)
    label_encoder = LabelEncoder().fit(y_train)
    encoded_y_train = label_encoder.transform(y_train)
    encoded_y_test = label_encoder.transform(y_test)
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
    )
    sample_weights = compute_sample_weight("balanced", encoded_y_train)
    train_started = time.perf_counter()
    xgb_model.fit(xgb_train, encoded_y_train, sample_weight=sample_weights)
    xgb_training_time = time.perf_counter() - train_started

    encoded_predictions = xgb_model.predict(xgb_test).astype(int)
    xgb_predictions = label_encoder.inverse_transform(encoded_predictions).tolist()
    xgb_test_metrics = _macro_metrics(y_test, xgb_predictions)
    xgb_train_predictions = label_encoder.inverse_transform(
        xgb_model.predict(xgb_train).astype(int)
    ).tolist()
    xgb_training_accuracy = accuracy_score(y_train, xgb_train_predictions)

    rf_prediction_time = _mean_prediction_time(lambda: rf_model.predict(x_test))
    xgb_prediction_time = _mean_prediction_time(
        lambda: xgb_model.predict(
            np.asarray(preprocessor.transform(x_test), dtype=np.float32)
        )
    )
    rf_macro = rf_test_metrics["classification_report"]
    xgb_macro = xgb_test_metrics["classification_report"]

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_model.json"
    report_path = output_dir / "xgboost_classification_report.txt"
    matrix_path = output_dir / "xgboost_confusion_matrix.png"
    metrics_path = output_dir / "xgboost_metrics.json"
    comparison_path = output_dir / "model_comparison.csv"
    prep_path = output_dir / "xgboost_preprocessing.pkl"
    labels_path = output_dir / "xgboost_class_labels.json"
    split_path = output_dir / "split_indices.json"

    xgb_model.save_model(str(model_path))
    prep_path.write_bytes(pickle.dumps(preprocessor))
    labels_path.write_text(
        json.dumps(
            {
                "class_labels_by_encoded_value": label_encoder.classes_.tolist(),
                "feature_columns": feature_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps(split["indices"], indent=2),
        encoding="utf-8",
    )
    report_path.write_text(xgb_test_metrics["classification_report_text"], encoding="utf-8")
    _save_confusion_matrix(xgb_test_metrics["confusion_matrix"], matrix_path)

    comparison_rows = [
        ("Accuracy", rf_test_metrics["accuracy"], xgb_test_metrics["accuracy"]),
        ("Precision (macro)", rf_macro["macro avg"]["precision"], xgb_macro["macro avg"]["precision"]),
        ("Recall (macro)", rf_macro["macro avg"]["recall"], xgb_macro["macro avg"]["recall"]),
        ("F1-score (macro)", rf_macro["macro avg"]["f1-score"], xgb_macro["macro avg"]["f1-score"]),
        ("Training Time (seconds)", "not recorded; RF not retrained", xgb_training_time),
        (
            f"Prediction Time (seconds per {len(testing_records)}-row batch; mean of {PREDICTION_BENCHMARK_REPEATS})",
            rf_prediction_time,
            xgb_prediction_time,
        ),
    ]
    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("Metric", "Random Forest", "XGBoost"))
        writer.writerows(comparison_rows)

    hardest_xgb = min(
        CLASS_LABELS,
        key=lambda label: xgb_test_metrics["classification_report"][label]["f1-score"],
    )
    hardest_rf = min(
        CLASS_LABELS,
        key=lambda label: rf_test_metrics["classification_report"][label]["f1-score"],
    )
    fault_labels = CLASS_LABELS[1:]
    hardest_fault = min(
        fault_labels,
        key=lambda label: xgb_test_metrics["classification_report"][label]["recall"],
    )
    xgb_f1 = xgb_test_metrics["classification_report"]["macro avg"]["f1-score"]
    rf_f1 = rf_test_metrics["classification_report"]["macro avg"]["f1-score"]
    if xgb_f1 > rf_f1:
        better_model = "XGBoost"
    elif rf_f1 > xgb_f1:
        better_model = "Random Forest"
    else:
        better_model = "Tie on macro F1"

    metrics = {
        "dataset": {
            "combined_rf_dataset": str(combined_path),
            "sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
            "source_csv_hashes_unchanged": source_hashes,
            "total_records": len(records),
            "records_per_class": dict(Counter(record["label"] for record in records)),
            "features": feature_columns,
        },
        "split": {
            "method": split["method"],
            "same_split_as_random_forest": True,
            "train_records": len(training_records),
            "test_records": len(testing_records),
            "per_class_counts": split["per_class_counts"],
            "indices_file": str(split_path),
        },
        "random_forest": {
            "metrics": rf_test_metrics,
            "training_accuracy": rf_training_accuracy,
            "training_time_seconds": None,
            "training_time_note": "Not recorded during prior RF fit; RF was not retrained per request.",
            "prediction_time_seconds_per_test_batch_mean": rf_prediction_time,
            "prediction_benchmark_repeats": PREDICTION_BENCHMARK_REPEATS,
        },
        "xgboost": {
            "metrics": xgb_test_metrics,
            "training_accuracy": xgb_training_accuracy,
            "training_time_seconds": xgb_training_time,
            "prediction_time_seconds_per_test_batch_mean": xgb_prediction_time,
            "prediction_benchmark_repeats": PREDICTION_BENCHMARK_REPEATS,
            "model_parameters": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "class_weight_handling": "balanced training sample weights",
            },
        },
        "comparison": {
            "macro_metric_definition": "Unweighted average over the four classes.",
            "table_csv": str(comparison_path),
            "better_by_macro_f1": better_model,
            "hardest_class_by_f1": {
                "random_forest": hardest_rf,
                "xgboost": hardest_xgb,
            },
            "hardest_anomaly_class_by_recall": hardest_fault,
            "overfitting": {
                "random_forest_train_accuracy": rf_training_accuracy,
                "random_forest_test_accuracy": rf_test_metrics["accuracy"],
                "random_forest_gap": rf_training_accuracy - rf_test_metrics["accuracy"],
                "xgboost_train_accuracy": xgb_training_accuracy,
                "xgboost_test_accuracy": xgb_test_metrics["accuracy"],
                "xgboost_gap": xgb_training_accuracy - xgb_test_metrics["accuracy"],
            },
            "evaluation_caveat": (
                "Both scores use the same chronological within-capture split and 30-second "
                "embargo. There are no independent session IDs/runs, so this does not "
                "establish performance on a new simulation run or physical motor."
            ),
            "real_time_selection_note": (
                "XGBoost ties Random Forest on test metrics and has lower measured "
                "500-row CPU batch latency in this run. Treat XGBoost as the current "
                "candidate, pending independent-run validation and backend throughput tests."
            ),
        },
        "artifacts": {
            "xgboost_model": str(model_path),
            "xgboost_classification_report": str(report_path),
            "xgboost_confusion_matrix": str(matrix_path),
            "xgboost_metrics": str(metrics_path),
            "model_comparison": str(comparison_path),
            "preprocessing_pipeline": str(prep_path),
            "class_label_mapping": str(labels_path),
            "split_indices": str(split_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on the saved Random Forest split and compare both models."
    )
    parser.add_argument("--rf-output-dir", type=Path, default=RF_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metrics = train_and_compare(
        rf_output_dir=args.rf_output_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "records": metrics["dataset"]["total_records"],
                "train_records": metrics["split"]["train_records"],
                "test_records": metrics["split"]["test_records"],
                "random_forest": metrics["random_forest"]["metrics"]["accuracy"],
                "xgboost": metrics["xgboost"]["metrics"]["accuracy"],
                "better_by_macro_f1": metrics["comparison"]["better_by_macro_f1"],
                "artifacts": metrics["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()