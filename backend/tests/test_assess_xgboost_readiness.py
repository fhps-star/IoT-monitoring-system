import unittest

import numpy as np

from backend.assess_xgboost_readiness import _brier_score, _expected_calibration_error


class XGBoostReadinessTests(unittest.TestCase):
    def test_brier_score_is_zero_for_perfect_probabilities(self) -> None:
        probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        true_indices = np.asarray([0, 1])

        self.assertEqual(_brier_score(probabilities, true_indices), 0.0)

    def test_expected_calibration_error_uses_top_label_confidence(self) -> None:
        probabilities = np.asarray([[0.9, 0.1], [0.6, 0.4]])
        true_indices = np.asarray([0, 1])

        ece, bins = _expected_calibration_error(probabilities, true_indices, bins=2)

        self.assertAlmostEqual(ece, 0.25)
        self.assertEqual(sum(item["records"] for item in bins), 2)


if __name__ == "__main__":
    unittest.main()