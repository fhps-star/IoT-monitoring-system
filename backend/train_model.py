"""Audit, evaluate, and train an Isolation Forest from labelled telemetry CSV."""

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

from backend.ai_model import IsolationForestModel
from backend.dataset import inspect_and_clean_csv, engineer_dataset
from backend.feature_engineering import FEATURE_NAMES
from backend.settings import Settings
from backend.utils.logger import configure_logging

logger = logging.getLogger(__name__)


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _evaluate(
    model: IsolationForestModel,
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not validation_rows:
        return {
            "validation_rows": 0,
            "note": "No rows available in the chronological validation split.",
        }
    features = [row["features"] for row in validation_rows]
    results = [model.predict(row) for row in features]
    predictions = [label for label, _ in results]
    scores = [score for _, score in results]
    normal_rows = [
        (row, prediction)
        for row, prediction in zip(validation_rows, predictions, strict=True)
        if row["label"] == "NORMAL"
    ]
    anomaly_rows = [
        (row, prediction)
        for row, prediction in zip(validation_rows, predictions, strict=True)
        if row["label"] == "ANOMALY"
    ]
    report: dict[str, Any] = {
        "validation_rows": len(validation_rows),
        "validation_start": validation_rows[0]["timestamp"],
        "validation_end": validation_rows[-1]["timestamp"],
        "predicted_anomalies": predictions.count("ANOMALY"),
        "anomaly_score": {
            "min": min(scores),
            "p05": _quantile(scores, 0.05),
            "median": statistics.median(scores),
            "p95": _quantile(scores, 0.95),
            "max": max(scores),
            "meaning": "higher scores are more anomalous",
        },
        "normal_validation_rows": len(normal_rows),
        "normal_rows_flagged_anomaly": sum(
            prediction == "ANOMALY" for _, prediction in normal_rows
        ),
        "normal_validation_flagged_rate": (
            sum(prediction == "ANOMALY" for _, prediction in normal_rows)
            / len(normal_rows)
            if normal_rows
            else None
        ),
        "labelled_anomaly_rows": len(anomaly_rows),
    }
    if anomaly_rows:
        true_positive = sum(
            prediction == "ANOMALY" for _, prediction in anomaly_rows
        )
        false_negative = len(anomaly_rows) - true_positive
        false_positive = report["normal_rows_flagged_anomaly"]
        true_negative = len(normal_rows) - false_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = true_positive / (true_positive + false_negative)
        report["confusion_counts"] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        }
        report["precision"] = precision
        report["recall"] = recall
        report["accuracy"] = (true_positive + true_negative) / len(validation_rows)
        report["f1"] = (
            2 * precision * recall / (precision + recall)
            if precision is not None and precision + recall
            else 0.0
        )
    else:
        report["classification_metrics"] = (
            "Not calculated: validation split contains no labelled anomalies."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a motor telemetry CSV, create a cleaned copy, evaluate with "
            "a time-based split, and train an Isolation Forest."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input dataset; never modified")
    parser.add_argument("--cleaned-output", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--max-gap-seconds", type=float, default=2.5)
    parser.add_argument("--min-window-samples", type=int, default=3)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Most recent chronological fraction reserved for final evaluation",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.2,
        help="Chronological fraction before evaluation used to calibrate normal scores",
    )
    args = parser.parse_args()
    if (
        not 0 < args.validation_fraction < 1
        or not 0 < args.calibration_fraction < 1
        or args.validation_fraction + args.calibration_fraction >= 1
    ):
        parser.error(
            "validation and calibration fractions must be positive and sum to less than 1"
        )
    if args.window_seconds <= 0 or args.max_gap_seconds <= 0:
        parser.error("window and maximum gap must be positive")
    if args.min_window_samples < 2:
        parser.error("--min-window-samples must be at least 2")

    configure_logging()
    input_path = args.csv.expanduser().resolve()
    cleaned_path = (
        args.cleaned_output
        or input_path.with_name(f"{input_path.stem}_cleaned.csv")
    ).expanduser().resolve()
    audit_path = (
        args.audit_report
        or input_path.with_name(f"{input_path.stem}_audit.json")
    ).expanduser().resolve()
    training_report_path = (
        args.training_report
        or input_path.with_name(f"{input_path.stem}_training_report.json")
    ).expanduser().resolve()
    if input_path in {cleaned_path, audit_path, training_report_path}:
        parser.error("Output paths must not overwrite the original input CSV")

    settings = Settings.from_env()
    audit = inspect_and_clean_csv(input_path, cleaned_path, audit_path)
    engineered_rows, feature_report = engineer_dataset(
        audit.rows,
        window_seconds=args.window_seconds,
        max_gap_seconds=args.max_gap_seconds,
        min_window_samples=args.min_window_samples,
    )
    if len(engineered_rows) < 2:
        raise ValueError(
            "Not enough rows have complete causal window features for a "
            "time-based train/validation split"
        )

    train_end = int(
        len(engineered_rows)
        * (1 - args.validation_fraction - args.calibration_fraction)
    )
    calibration_end = int(
        len(engineered_rows) * (1 - args.validation_fraction)
    )
    train_end = min(max(train_end, 1), len(engineered_rows) - 2)
    calibration_end = min(
        max(calibration_end, train_end + 1),
        len(engineered_rows) - 1,
    )
    train_rows = [
        row
        for row in engineered_rows[:train_end]
        if row["label"] == "NORMAL"
    ]
    calibration_rows = [
        row
        for row in engineered_rows[train_end:calibration_end]
        if row["label"] == "NORMAL"
    ]
    validation_rows = engineered_rows[calibration_end:]
    if len(train_rows) < settings.minimum_training_samples:
        raise ValueError(
            f"Need at least {settings.minimum_training_samples} NORMAL training "
            f"rows before the time split; got {len(train_rows)}"
        )
    if not calibration_rows:
        raise ValueError("No NORMAL rows available for threshold calibration")

    with tempfile.TemporaryDirectory(prefix="iiot-model-evaluation-") as directory:
        evaluation_model = IsolationForestModel(Path(directory) / "evaluation.joblib")
        evaluation_model.fit(
            [row["features"] for row in train_rows],
            settings.minimum_training_samples,
            feature_names=FEATURE_NAMES,
        )
        calibration_scores = [
            evaluation_model.predict(row["features"])[1]
            for row in calibration_rows
        ]
        calibration_median = statistics.median(calibration_scores)
        calibration_mad = statistics.median(
            abs(score - calibration_median) for score in calibration_scores
        )
        score_threshold = calibration_median + 3 * 1.4826 * calibration_mad
        if calibration_mad == 0:
            score_threshold = math.nextafter(calibration_median, math.inf)
        evaluation_model.set_anomaly_score_threshold(score_threshold)
        evaluation = _evaluate(evaluation_model, validation_rows)

    trained_at = datetime.now(timezone.utc).isoformat()
    training_metadata = {
        "trained_at": trained_at,
        "training_dataset": {
            "path": str(input_path),
            "sha256": audit.report["input_sha256"],
            "source_rows": audit.report["rows_read"],
            "clean_rows": audit.report["valid_unique_rows"],
            "label_counts": audit.report["label_counts"],
            "training_normal_rows": len(train_rows),
            "feature_rows": len(engineered_rows),
            "cleaned_dataset_path": str(cleaned_path),
            "audit_report_path": str(audit_path),
            "sampling_interval_seconds": audit.report[
                "sampling_interval_seconds"
            ],
            "sampling_gaps_over_default_limit": audit.report[
                "sampling_gaps_over_default_limit"
            ],
            "numeric_distributions": audit.report["numeric_distributions"],
            "outlier_counts_by_column": audit.report[
                "outlier_counts_by_column"
            ],
        },
        "features": list(FEATURE_NAMES),
        "preprocessing": {
            **feature_report,
            "feature_definitions": {
                "acceleration_magnitude": "sqrt(vibration_x^2 + vibration_y^2 + vibration_z^2)",
                "vibration_rms": (
                    "RMS of per-sample 3-axis vectors after subtracting each "
                    "axis's trailing-window mean"
                ),
                "vibration_peak": (
                    "maximum norm of the same detrended 3-axis vectors in the window"
                ),
                "vibration_std": (
                    "population standard deviation of acceleration magnitude in the window"
                ),
                "temperature_rate": (
                    "current temperature minus previous temperature, divided by "
                    "actual elapsed seconds"
                ),
                "current_rate": (
                    "current draw minus previous current, divided by actual elapsed seconds"
                ),
            },
            "vibration_window_semantics": (
                "Trailing, time-based, detrended acceleration samples. These "
                "low-rate statistics are not frequency-domain vibration features."
            ),
            "causal": True,
            "temperature_and_current_rates": (
                "backward differences divided by actual elapsed seconds"
            ),
        },
        "model_parameters": {
            "class": "sklearn.ensemble.IsolationForest",
            "n_estimators": 200,
            "contamination": "auto",
            "random_state": 42,
        },
        "evaluation": evaluation,
        "evaluation_method": (
            "chronological holdout; no random shuffling; fit on NORMAL training rows"
        ),
        "chronological_split": {
            "validation_fraction": args.validation_fraction,
            "calibration_fraction": args.calibration_fraction,
            "training_start": train_rows[0]["timestamp"],
            "training_end": train_rows[-1]["timestamp"],
            "calibration_start": calibration_rows[0]["timestamp"],
            "calibration_end": calibration_rows[-1]["timestamp"],
            "validation_starts_at": validation_rows[0]["timestamp"],
        },
        "threshold_calibration": {
            "method": "calibration NORMAL median + 3 * 1.4826 * MAD",
            "normal_calibration_rows": len(calibration_rows),
            "score_median": calibration_median,
            "score_mad": calibration_mad,
            "anomaly_score_threshold": score_threshold,
            "contamination": "auto for estimator fit; data-calibrated score threshold at inference",
        },
    }

    model = IsolationForestModel(settings.model_path)
    model.fit(
        [row["features"] for row in train_rows],
        settings.minimum_training_samples,
        feature_names=FEATURE_NAMES,
        metadata=training_metadata,
        score_threshold=score_threshold,
    )
    training_report = {
        **training_metadata,
        "model_path": str(settings.model_path),
        "deployment_training_rows": sum(
            row["label"] == "NORMAL" for row in train_rows
        ),
        "contamination_note": (
            "IsolationForest contamination='auto'; with NORMAL-only data there "
            "are no labelled faults to tune a fault-detection threshold against. "
            "The held-out normal flagged rate is reported as a false-alarm proxy."
        ),
        "limitations": [
            "This model identifies unusual patterns, not specific fault types.",
            "Low-rate time-window statistics cannot detect vibration frequencies.",
            "Wokwi-sourced data describes the simulator, not a physical motor.",
        ],
    }
    training_report_path.parent.mkdir(parents=True, exist_ok=True)
    training_report_path.write_text(
        json.dumps(training_report, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Model training complete",
        extra={
            "training_rows": training_report["deployment_training_rows"],
            "validation_rows": evaluation["validation_rows"],
            "feature_count": len(FEATURE_NAMES),
            "model_path": str(settings.model_path),
            "training_report": str(training_report_path),
        },
    )


if __name__ == "__main__":
    main()
