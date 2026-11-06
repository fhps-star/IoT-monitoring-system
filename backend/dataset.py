"""Audit, clean, and engineer time-series motor telemetry datasets."""

import csv
from dataclasses import dataclass
from datetime import timezone
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from backend.feature_engineering import (
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MIN_WINDOW_SAMPLES,
    DEFAULT_WINDOW_SECONDS,
    FEATURE_NAMES,
    FeatureEngineer,
    parse_timestamp,
)

logger = logging.getLogger(__name__)
CSV_COLUMNS = (
    "timestamp",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "temperature_c",
    "current_a",
    "motor_speed",
    "label",
)
NUMERIC_COLUMNS = CSV_COLUMNS[1:7]


@dataclass(frozen=True)
class DatasetAudit:
    rows: list[dict[str, Any]]
    report: dict[str, Any]


def _delimiter(path: Path) -> str:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        sample = file.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def _iqr_outliers(rows: list[dict[str, Any]]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for column in NUMERIC_COLUMNS:
        ordered = sorted(row[column] for row in rows)
        if len(ordered) < 4:
            continue

        def quantile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight

        q1 = quantile(0.25)
        q3 = quantile(0.75)
        spread = q3 - q1
        if spread == 0:
            median = ordered[len(ordered) // 2]
            flagged = {
                index for index, row in enumerate(rows) if row[column] != median
            }
        else:
            lower_bound = q1 - 1.5 * spread
            upper_bound = q3 + 1.5 * spread
            flagged = {
                index
                for index, row in enumerate(rows)
                if row[column] < lower_bound or row[column] > upper_bound
            }
        if flagged:
            result[column] = flagged
    return result


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def inspect_and_clean_csv(
    input_path: Path,
    cleaned_path: Path,
    report_path: Path,
) -> DatasetAudit:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    delimiter = _delimiter(input_path)
    invalid_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    duplicate_records = 0
    duplicate_timestamps = 0
    seen_signatures: set[tuple[Any, ...]] = set()
    seen_timestamps: set[str] = set()
    labels: dict[str, int] = {}

    with input_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        headers = tuple((name or "").strip() for name in (reader.fieldnames or ()))
        missing_columns = sorted(set(CSV_COLUMNS) - set(headers))
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        for line_number, source_row in enumerate(reader, start=2):
            row = {(key or "").strip(): (value or "").strip() for key, value in source_row.items()}
            problems: list[str] = []
            timestamp: str | None = None
            try:
                parsed = parse_timestamp(row.get("timestamp", ""))
                timestamp = parsed.astimezone(timezone.utc).isoformat()
            except ValueError as exc:
                problems.append(str(exc))

            numeric: dict[str, float] = {}
            for column in NUMERIC_COLUMNS:
                raw = row.get(column, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError, OverflowError):
                    problems.append(f"missing or invalid '{column}'")
                    continue
                if not math.isfinite(value):
                    problems.append(f"non-finite '{column}'")
                    continue
                numeric[column] = value
            label = row.get("label", "").upper()
            if label not in {"NORMAL", "ANOMALY"}:
                problems.append("label must be NORMAL or ANOMALY")
            if problems:
                invalid_rows.append({"line": line_number, "reasons": problems})
                continue
            if timestamp is None:
                invalid_rows.append(
                    {"line": line_number, "reasons": ["invalid timestamp"]}
                )
                continue
            signature = (timestamp, *(numeric[column] for column in NUMERIC_COLUMNS), label)
            if signature in seen_signatures:
                duplicate_records += 1
                continue
            seen_signatures.add(signature)
            if timestamp in seen_timestamps:
                duplicate_timestamps += 1
                continue
            seen_timestamps.add(timestamp)
            labels[label] = labels.get(label, 0) + 1
            clean_rows.append({"timestamp": timestamp, **numeric, "label": label})

    clean_rows.sort(key=lambda row: row["timestamp"])
    timestamps = [parse_timestamp(row["timestamp"]) for row in clean_rows]
    intervals = [
        (current - previous).total_seconds()
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    outliers_by_column = _iqr_outliers(clean_rows)
    outlier_fields: dict[int, list[str]] = {}
    for column, indexes in outliers_by_column.items():
        for index in indexes:
            outlier_fields.setdefault(index, []).append(column)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_columns = (*CSV_COLUMNS, "outlier_fields")
    with cleaned_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=cleaned_columns)
        writer.writeheader()
        for index, row in enumerate(clean_rows):
            writer.writerow(
                {
                    **row,
                    "outlier_fields": ";".join(outlier_fields.get(index, [])),
                }
            )

    report = {
        "input_file": str(input_path),
        "input_sha256": digest,
        "delimiter_detected": "tab" if delimiter == "\t" else delimiter,
        "rows_read": len(clean_rows) + len(invalid_rows) + duplicate_records + duplicate_timestamps,
        "valid_unique_rows": len(clean_rows),
        "invalid_rows_removed": len(invalid_rows),
        "duplicate_records_removed": duplicate_records,
        "duplicate_timestamps_removed": duplicate_timestamps,
        "invalid_row_details": invalid_rows,
        "label_counts": labels,
        "numeric_distributions": {
            column: _distribution([row[column] for row in clean_rows])
            for column in NUMERIC_COLUMNS
        },
        "sampling_interval_seconds": _distribution(intervals),
        "sampling_gaps_over_default_limit": sum(
            interval > DEFAULT_MAX_GAP_SECONDS for interval in intervals
        ),
        "outlier_method": "1.5 x IQR per numeric column; flags only, no rows removed",
        "physical_limits": (
            "No sensor-specific physical operating ranges were configured; "
            "statistical outliers are flagged, not treated as faults."
        ),
        "outlier_row_count": len(outlier_fields),
        "outlier_counts_by_column": {
            column: len(indexes) for column, indexes in outliers_by_column.items()
        },
        "cleaned_file": str(cleaned_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Dataset audit complete", extra=report)
    return DatasetAudit(clean_rows, report)


def engineer_dataset(
    rows: list[dict[str, Any]],
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    min_window_samples: int = DEFAULT_MIN_WINDOW_SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    engineer = FeatureEngineer(
        window_seconds=window_seconds,
        max_gap_seconds=max_gap_seconds,
        min_window_samples=min_window_samples,
    )
    transformed: list[dict[str, Any]] = []
    warmup_rows = 0
    invalid_rows = 0
    for row in rows:
        try:
            features = engineer.process(row, row["timestamp"])
        except ValueError as exc:
            invalid_rows += 1
            logger.warning(
                "Skipping row during feature engineering",
                extra={"timestamp": row.get("timestamp"), "reason": str(exc)},
            )
            engineer.reset()
            continue
        if features is None:
            warmup_rows += 1
            continue
        transformed.append(
            {
                "timestamp": row["timestamp"],
                "label": row["label"],
                "features": [features[name] for name in FEATURE_NAMES],
            }
        )
    return transformed, {
        "window_seconds": window_seconds,
        "max_gap_seconds": max_gap_seconds,
        "min_window_samples": min_window_samples,
        "rows_without_causal_window_features": warmup_rows,
        "invalid_feature_rows": invalid_rows,
        "rows_with_features": len(transformed),
        "feature_names": list(FEATURE_NAMES),
    }
