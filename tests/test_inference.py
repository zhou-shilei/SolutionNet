"""Regression checks against the original training features and web predictions."""
import json
from pathlib import Path
import unittest

import torch

from src.featurizer import canonicalize_smiles
from src.inference_rf import RFInference
from src.inference_solutionnet import SolutionNetPredictor, smiles_to_train_data

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests" / "reference.json").read_text())


class InferenceRegressionTests(unittest.TestCase):
    def test_training_feature_encoding(self):
        for smiles, expected in REFERENCE["graphs"].items():
            with self.subTest(smiles=smiles):
                graph = smiles_to_train_data(canonicalize_smiles(smiles))
                for key, values in expected.items():
                    actual = getattr(graph, key)
                    target = torch.tensor(values, dtype=actual.dtype).reshape(actual.shape)
                    self.assertTrue(torch.equal(actual, target), key)

    def test_solutionnet_predictions_match_legacy(self):
        predictor = SolutionNetPredictor(ROOT)
        predictor.device = torch.device("cpu")
        for case in REFERENCE["cases"]:
            with self.subTest(smiles=case["smiles"], symmetric=case["features"]["symmetric"]):
                score, label, details = predictor.predict(*case["smiles"], case["features"])
                self.assertAlmostEqual(score, case["score"], delta=1e-5)
                self.assertEqual(label, case["prediction"])
                self.assertEqual(details["chosen_order"], case["chosen_order"])

    def test_rf_predictions_match_legacy(self):
        predictor = RFInference(ROOT / "models" / "rf_model.pkl")
        for case in REFERENCE["cases"]:
            with self.subTest(smiles=case["smiles"]):
                score, _ = predictor.predict(*case["smiles"])
                self.assertAlmostEqual(score, case["rf_score"], delta=1e-6)

    def test_invalid_smiles_rejected(self):
        for smiles in ["", "invalid-smiles"]:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError):
                    smiles_to_train_data(smiles)


if __name__ == "__main__":
    unittest.main()
