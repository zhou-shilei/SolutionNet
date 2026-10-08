from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Tuple

import numpy as np
import joblib


@lru_cache(maxsize=512)
def _pooled_features(smiles: str) -> np.ndarray:
    from .featurizer import smiles_to_data

    graph = smiles_to_data(smiles)
    if graph.x is None or graph.x.numel() == 0:
        return np.zeros(39, dtype=np.float32)
    features = graph.x.float().mean(dim=0).detach().cpu().numpy().astype(np.float32)
    if features.shape != (39,):
        raise ValueError(f"Expected 39 node features, got {features.shape}")
    return features


class RFInference:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self._model = None

    def load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Missing RF model file: {self.model_path}. "
                    f"Put your trained model at models/rf_model.pkl"
                )
            self._model = joblib.load(self.model_path)
        return self._model

    def featurize_triplet(self, smiles_a: str, smiles_b: str, smiles_c: str) -> np.ndarray:
        """
        Match baseline_compare_export.py:
          - Convert SMILES -> PyG Data (with node features x of dim 39)
          - mean pool nodes for each of A/B/C -> 39 dims each
          - concat -> 117 dims
        """
        feat = np.concatenate([_pooled_features(s) for s in (smiles_a, smiles_b, smiles_c)])
        return feat.reshape(1, -1)

    def predict(self, smiles_a: str, smiles_b: str, smiles_c: str) -> Tuple[float, bool]:
        m = self.load()
        X = self.featurize_triplet(smiles_a, smiles_b, smiles_c)

        if hasattr(m, "predict_proba"):
            score = float(m.predict_proba(X)[0][1])
        elif hasattr(m, "decision_function"):
            raw = float(m.decision_function(X)[0])
            score = float(1.0 / (1.0 + np.exp(-raw)))
        else:
            score = float(m.predict(X)[0])

        pred = score >= 0.5
        return score, pred
