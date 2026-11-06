"""Train and evaluate a multiclass Random Forest from labelled Wokwi CSVs."""

import argparse
import csv
from collections import Counter
from datetime import timedelta, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import pickle
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from backend.feature_engineering import parse_timestamp
from backend.settings import BACKEND_DIR

logger = logging.getLogger(__name__)
CLASS_FILES = {
    "NORMAL": "wokwi_normal_motor_telemetry.csv",
    "HIGH_VIBRATION": "wokwi_high_vibration_telemetry.csv",
    "HIGH_TEMPERATURE": "wokwi_high_temperature_telemetry.csv",
    "HIGH_CURRENT": "wokwi_high_current_telemetry.csv",
}
REQUESTED_FEATURE_COLUMNS = (
    "temperature_c",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "acceleration_magnitude",
    "estimated_vibration",
    "motor_speed",
    "motor_current_a",
)
FEATURE_ALIASES = {
    "temperature_c": ("temperature_c",),
    "acceleration_x": ("acceleration_x", "vibration_x"),
    "acceleration_y": ("acceleration_y", "vibration_y"),
    "acceleration_z": ("acceleration_z", "vibration_z"),
    "gyro_x": ("gyro_x",),
    "gyro_y": ("gyro_y",),
    "gyro_z": ("gyro_z",),
    "estimated_vibration": ("estimated_vibration",),
    "motor_speed": ("motor_speed",),
    "motor_current_a": ("motor_current_a", "current_a"),
}
DEFAULT_DATA_DIR = BACKEND_DIR.parent / "data"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "models" / "random_forest"
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_EMBARGO_SECONDS = 30.0


def _detect_delimiter(path: Path) -> str:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        sample = file.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def _source_field(feature: str, headers: set[str]) -> str | None:
    if feature == "acceleration_magnitude":
        if feature in headers:
            return feature
        axes = (
            _source_field("acceleration_x", headers),
            _source_field("acceleration_y", headers),
            _source_field("acceleration_z", headers),
        )
        return "__derived_from_acceleration_axes__" if all(axes) else None
    return next(
        (name for name in FEATURE_ALIASES[feature] if name in headers),
        None,
    )


def _number(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _read_source(
    path: Path,
    expected_label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    delimiter = _detect_delimiter(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = [column.strip() for column in next(reader, [])]
        if not {"timestamp", "label"}.issubset(headers):
            raise ValueError(f"{path} must include timestamp and label columns")
        raw_rows: list[dict[str, str]] = []
        records: list[dict[str, Any]] = []
        seen_rows: set[tuple[str, ...]] = set()
        duplicate_count = 0
        label_counts: Counter[str] = Counter()
        for line_number, values in enumerate(reader, start=2):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) > len(headers):
                raise ValueError(f"{path}:{line_number} has more fields than its header")
            row = {
                header: values[index].strip() if index < len(values) else ""
                for index, header in enumerate(headers)
            }
            signature = tuple(row[header] for header in headers)
            if signature in seen_rows:
                duplicate_count += 1
                continue
            seen_rows.add(signature)
            label = row.get("label", "").strip().upper()
            if label != expected_label:
                raise ValueError(
                    f"{path}:{line_number} has label {label!r}; "
                    f"expected {expected_label!r} from its class dataset"
                )
            timestamp_text = row.get("timestamp", "")
            try:
                timestamp = parse_timestamp(timestamp_text).astimezone(timezone.utc)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            raw_rows.append(row)
            records.append(
                {
                    "timestamp": timestamp,
                    "label": label,
                    "source_file": path.name,
                    "source_row": line_number,
                    "raw": row,
                }
            )
            label_counts[label] += 1

    missing_values = {
        header: sum(not row.get(header, "") for row in raw_rows)
        for header in headers
    }
    inferred_types: dict[str, str] = {}
    numeric_ranges: dict[str, dict[str, float]] = {}
    for header in headers:
        values = [row[header] for row in raw_rows if row.get(header, "")]
        if not values:
            inferred_types[header] = "empty"
            continue
        try:
            numbers = [float(value) for value in values]
            inferred_types[header] = "numeric"
        except ValueError:
            inferred_types[header] = "text"
            continue
        if numbers and all(math.isfinite(value) for value in numbers):
            numeric_ranges[header] = {
                "min": min(numbers),
                "median": median(numbers),
                "max": max(numbers),
            }

    audit = {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "delimiter": "tab" if delimiter == "\t" else delimiter,
        "columns": headers,
        "inferred_types": inferred_types,
        "rows_read": len(raw_rows) + duplicate_count,
        "rows_used": len(records),
        "exact_duplicate_rows_removed": duplicate_count,
        "missing_values_by_column": missing_values,
        "label_counts": dict(label_counts),
        "label_mismatches": 0,
        "timestamp_start": min(
            (record["timestamp"] for record in records), default=None
        ),
        "timestamp_end": max(
            (record["timestamp"] for record in records), default=None
        ),
        "numeric_ranges": numeric_ranges,
    }
    for key in ("timestamp_start", "timestamp_end"):
        if audit[key] is not None:
            audit[key] = audit[key].isoformat()
    return records, audit


def inspect_and_combine(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing_files = [
        str(data_dir / filename)
        for filename in CLASS_FILES.values()
        if not (data_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing required class dataset(s): " + ", ".join(missing_files)
        )

    records: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}
    headers_by_label: dict[str, set[str]] = {}
    for label, filename in CLASS_FILES.items():
        path = data_dir / filename
        source_records, audit = _read_source(path, label)
        if not source_records:
            raise ValueError(f"{path} contains no usable records")
        records.extend(source_records)
        audits[label] = audit
        headers_by_label[label] = set(audit["columns"])

    field_sources: dict[str, dict[str, str]] = {}
    feature_columns: list[str] = []
    unavailable_features: list[str] = []
    for feature in REQUESTED_FEATURE_COLUMNS:
        sources = {
            label: _source_field(feature, headers)
            for label, headers in headers_by_label.items()
        }
        if all(sources.values()):
            field_sources[feature] = sources  # type: ignore[assignment]
            feature_columns.append(feature)
        else:
            unavailable_features.append(feature)

    if not feature_columns:
        raise ValueError("No requested numeric features are available in all datasets")

    for record in records:
        raw = record["raw"]
        label = record["label"]
        features: dict[str, float] = {}
        for feature in feature_columns:
            source = field_sources[feature][label]
            if source == "__derived_from_acceleration_axes__":
                axes = [
                    _number(raw[field_sources[axis][label]])
                    for axis in ("acceleration_x", "acceleration_y", "acceleration_z")
                ]
                value = (
                    math.sqrt(sum(axis * axis for axis in axes))
                    if all(math.isfinite(axis) for axis in axes)
                    else math.nan
                )
            else:
                value = _number(raw.get(source))
            features[feature] = value
        record["features"] = features

    empty_features = [
        feature
        for feature in feature_columns
        if not any(math.isfinite(record["features"][feature]) for record in records)
    ]
    for feature in empty_features:
        feature_columns.remove(feature)
        unavailable_features.append(feature)
    if not feature_columns:
        raise ValueError("All available requested features are empty or invalid")

    missing_feature_values = {
        feature: sum(
            not math.isfinite(record["features"][feature]) for record in records
        )
        for feature in feature_columns
    }
    source_mapping = {
        feature: {
            label: (
                "sqrt(acceleration_x^2 + acceleration_y^2 + acceleration_z^2)"
                if source == "__derived_from_acceleration_axes__"
                else source
            )
            for label, source in sources.items()
        }
        for feature, sources in field_sources.items()
        if feature in feature_columns
    }
    class_counts = Counter(record["label"] for record in records)
    report = {
        "source_files": audits,
        "total_records": len(records),
        "class_counts": dict(class_counts),
        "requested_feature_columns": list(REQUESTED_FEATURE_COLUMNS),
        "feature_columns": feature_columns,
        "unavailable_requested_features": unavailable_features,
        "feature_source_mapping": source_mapping,
        "missing_feature_values_before_imputation": missing_feature_values,
        "exact_duplicate_rows_removed": sum(
            audit["exact_duplicate_rows_removed"] for audit in audits.values()
        ),
    }
    return records, report


def chronological_split(
    records: list[dict[str, Any]],
    test_fraction: float = DEFAULT_TEST_FRACTION,
    embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, int]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if embargo_seconds < 0:
        raise ValueError("embargo_seconds cannot be negative")

    training: list[dict[str, Any]] = []
    testing: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for label in CLASS_FILES:
        class_records = sorted(
            (record for record in records if record["label"] == label),
            key=lambda record: (record["timestamp"], record["source_row"]),
        )
        split_index = int(len(class_records) * (1 - test_fraction))
        if split_index < 1 or split_index >= len(class_records):
            raise ValueError(f"Not enough {label} records for a train/test split")
        test_start = class_records[split_index]["timestamp"]
        train_cutoff = test_start - timedelta(seconds=embargo_seconds)
        class_training = [
            record
            for record in class_records[:split_index]
            if record["timestamp"] <= train_cutoff
        ]
        class_testing = class_records[split_index:]
        counts[label] = {
            "training": len(class_training),
            "testing": len(class_testing),
            "embargoed": split_index - len(class_training),
        }
        if not class_training or not class_testing:
            raise ValueError(f"Not enough {label} records after the time embargo")
        training.extend(class_training)
        testing.extend(class_testing)

    training.sort(key=lambda record: record["timestamp"])
    testing.sort(key=lambda record: record["timestamp"])
    return training, testing, counts


def _write_confusion_matrix(
    matrix: list[list[int]],
    labels: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Random Forest confusion matrix",
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
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train_and_evaluate(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    embargo_seconds: float = DEFAULT_EMBARGO_SECONDS,
) -> dict[str, Any]:
    records, dataset_report = inspect_and_combine(data_dir)
    training_records, testing_records, split_counts = chronological_split(
        records,
        test_fraction=test_fraction,
        embargo_seconds=embargo_seconds,
    )
    feature_columns = dataset_report["feature_columns"]
    model = Pipeline(
        steps=[
            ("preprocessor", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    x_train = [
        [record["features"][feature] for feature in feature_columns]
        for record in training_records
    ]
    y_train = [record["label"] for record in training_records]
    x_test = [
        [record["features"][feature] for feature in feature_columns]
        for record in testing_records
    ]
    y_test = [record["label"] for record in testing_records]
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).tolist()
    labels = tuple(CLASS_FILES)
    report_dict = classification_report(
        y_test,
        predictions,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        labels=list(labels),
        target_names=list(labels),
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=list(labels)).tolist()
    test_counts = Counter(y_test)
    train_counts = Counter(y_train)
    sample_predictions: dict[str, dict[str, Any]] = {}
    for expected_label in labels:
        index = next(
            index
            for index, record in enumerate(testing_records)
            if record["label"] == expected_label
        )
        probabilities = model.predict_proba([x_test[index]])[0]
        predicted_index = int(probabilities.argmax())
        sample_predictions[expected_label] = {
            "expected_label": expected_label,
            "sample_features": {
                feature: testing_records[index]["features"][feature]
                for feature in feature_columns
            },
            "predicted_class": model.classes_[predicted_index],
            "confidence": float(probabilities[predicted_index]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "random_forest_model.pkl"
    preprocessing_path = output_dir / "preprocessing_pipeline.pkl"
    features_path = output_dir / "feature_columns.json"
    report_path = output_dir / "classification_report.txt"
    matrix_path = output_dir / "confusion_matrix.png"
    metrics_path = output_dir / "training_metrics.json"
    combined_path = output_dir / "combined_training_dataset.csv"

    with model_path.open("wb") as file:
        pickle.dump(model, file)
    with preprocessing_path.open("wb") as file:
        pickle.dump(model.named_steps["preprocessor"], file)
    features_path.write_text(
        json.dumps(
            {
                "feature_columns": feature_columns,
                "requested_feature_columns": list(REQUESTED_FEATURE_COLUMNS),
                "unavailable_requested_features": dataset_report[
                    "unavailable_requested_features"
                ],
                "feature_source_mapping": dataset_report["feature_source_mapping"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path.write_text(report_text, encoding="utf-8")
    _write_confusion_matrix(matrix, labels, matrix_path)
    with combined_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["timestamp", *feature_columns, "label", "source_file", "source_row"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["timestamp"]):
            writer.writerow(
                {
                    "timestamp": record["timestamp"].isoformat(),
                    **{
                        feature: (
                            record["features"][feature]
                            if math.isfinite(record["features"][feature])
                            else ""
                        )
                        for feature in feature_columns
                    },
                    "label": record["label"],
                    "source_file": record["source_file"],
                    "source_row": record["source_row"],
                }
            )

    all_feature_ranges: dict[str, dict[str, dict[str, float]]] = {}
    for label in labels:
        all_feature_ranges[label] = {}
        class_records = [record for record in records if record["label"] == label]
        for feature in feature_columns:
            values = [
                record["features"][feature]
                for record in class_records
                if math.isfinite(record["features"][feature])
            ]
            all_feature_ranges[label][feature] = {
                "min": min(values),
                "median": median(values),
                "max": max(values),
            }

    metrics = {
        "total_records_used": len(records),
        "records_per_class": dict(Counter(record["label"] for record in records)),
        "training_records": len(training_records),
        "testing_records": len(testing_records),
        "training_records_per_class": dict(train_counts),
        "testing_records_per_class": dict(test_counts),
        "embargoed_records_per_class": {
            label: counts["embargoed"] for label, counts in split_counts.items()
        },
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": report_dict,
        "confusion_matrix": {"labels": list(labels), "counts": matrix},
        "sample_predictions": sample_predictions,
        "dataset_quality": dataset_report,
        "class_feature_ranges": all_feature_ranges,
        "preprocessing": {
            "missing_values": "Non-numeric, non-finite, and blank numeric values are represented as missing.",
            "imputation": "Median imputation is fit on training data only via the saved pipeline.",
            "duplicate_rows": "Exact duplicate rows are excluded and counted in the dataset report.",
            "derived_features": dataset_report["feature_source_mapping"],
            "source_csv_files_are_unchanged": True,
        },
        "split": {
            "method": "Per-class chronological holdout: earliest rows train, latest 20% test.",
            "test_fraction_per_class": test_fraction,
            "embargo_seconds": embargo_seconds,
            "session_limitation": (
                "No explicit session/run ID exists. Each class is stored in one file and "
                "has no independent second run; a whole-file holdout would remove that "
                "class from training. The chronological tail test with an embargo is "
                "therefore not an independent-session evaluation."
            ),
        },
        "model_parameters": {
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 200,
            "random_state": 42,
            "class_weight": "balanced",
        },
        "artifacts": {
            "random_forest_model": str(model_path),
            "preprocessing_pipeline": str(preprocessing_path),
            "feature_columns": str(features_path),
            "classification_report": str(report_path),
            "confusion_matrix": str(matrix_path),
            "training_metrics": str(metrics_path),
            "combined_training_dataset": str(combined_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a multiclass Random Forest from labelled telemetry CSVs."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument(
        "--embargo-seconds",
        type=float,
        default=DEFAULT_EMBARGO_SECONDS,
        help="Time gap excluded between training and test rows within each class file",
    )
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if args.embargo_seconds < 0:
        parser.error("--embargo-seconds cannot be negative")
    metrics = train_and_evaluate(
        data_dir=args.data_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        test_fraction=args.test_fraction,
        embargo_seconds=args.embargo_seconds,
    )
    print(
        json.dumps(
            {
                "total_records_used": metrics["total_records_used"],
                "records_per_class": metrics["records_per_class"],
                "training_records": metrics["training_records"],
                "testing_records": metrics["testing_records"],
                "accuracy": metrics["accuracy"],
                "artifacts": metrics["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()