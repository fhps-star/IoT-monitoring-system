"""Environment-backed settings for the backend."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    mqtt_broker: str
    mqtt_port: int
    mqtt_topic: str
    device_id: str
    firebase_database_url: str
    google_application_credentials: str | None
    temperature_field: str | None
    vibration_field: str | None
    current_field: str | None
    telemetry_limit: int
    data_stale_timeout_seconds: int
    minimum_training_samples: int
    model_path: Path
    cors_origins: tuple[str, ...]
    admin_api_key: str = "admin-secret-key-iiot"

    @classmethod
    def from_env(cls) -> "Settings":
        credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        credential_path = Path(credentials).expanduser() if credentials else None
        if credential_path is not None and not credential_path.is_absolute():
            credential_path = BACKEND_DIR / credential_path
        fields = (
            os.getenv("SENSOR_TEMPERATURE_FIELD", "").strip(),
            os.getenv("SENSOR_VIBRATION_FIELD", "").strip(),
            os.getenv("SENSOR_CURRENT_FIELD", "").strip(),
        )
        origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        return cls(
            mqtt_broker=os.getenv("MQTT_BROKER", "broker.hivemq.com").strip(),
            mqtt_port=_positive_int("MQTT_PORT", 1883),
            mqtt_topic=os.getenv(
                "MQTT_TOPIC", "iiot/motor_01/telemetry"
            ).strip(),
            device_id=os.getenv("DEVICE_ID", "motor_01").strip(),
            firebase_database_url=os.getenv(
                "FIREBASE_DATABASE_URL",
                "https://iiot-912b2-default-rtdb.asia-southeast1.firebasedatabase.app",
            ).strip(),
            google_application_credentials=(
                str(credential_path) if credential_path is not None else None
            ),
            temperature_field=fields[0] or None,
            vibration_field=fields[1] or None,
            current_field=fields[2] or None,
            telemetry_limit=_positive_int("TELEMETRY_LIMIT", 500),
            data_stale_timeout_seconds=_positive_int(
                "DATA_STALE_TIMEOUT_SECONDS", 15
            ),
            minimum_training_samples=_positive_int(
                "MINIMUM_TRAINING_SAMPLES", 30
            ),
            model_path=BACKEND_DIR / "models" / "isolation_forest.joblib",
            cors_origins=tuple(
                origin.strip() for origin in origins.split(",") if origin.strip()
            ),
            admin_api_key=os.getenv("ADMIN_API_KEY", "admin-secret-key-iiot").strip() or "admin-secret-key-iiot",
        )
