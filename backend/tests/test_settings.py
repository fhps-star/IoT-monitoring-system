import os
import unittest
from unittest.mock import patch

from backend.settings import BACKEND_DIR, Settings


class SettingsTests(unittest.TestCase):
    def test_relative_firebase_credentials_resolve_from_backend_directory(self) -> None:
        with patch.dict(
            os.environ,
            {"GOOGLE_APPLICATION_CREDENTIALS": "./serviceAccountKey.json"},
        ):
            settings = Settings.from_env()

        self.assertEqual(
            settings.google_application_credentials,
            str(BACKEND_DIR / "serviceAccountKey.json"),
        )

    def test_absolute_firebase_credentials_are_preserved(self) -> None:
        credential_path = "/secure/firebase/serviceAccountKey.json"
        with patch.dict(
            os.environ,
            {"GOOGLE_APPLICATION_CREDENTIALS": credential_path},
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.google_application_credentials, credential_path)

    def test_stale_data_timeout_is_configurable(self) -> None:
        with patch.dict(os.environ, {"DATA_STALE_TIMEOUT_SECONDS": "25"}):
            settings = Settings.from_env()

        self.assertEqual(settings.data_stale_timeout_seconds, 25)

    def test_stale_data_timeout_must_be_positive(self) -> None:
        with patch.dict(os.environ, {"DATA_STALE_TIMEOUT_SECONDS": "0"}):
            with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
