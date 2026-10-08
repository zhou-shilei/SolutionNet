from __future__ import annotations

"""SolutionNet inference with local training-compatible features and cached graphs."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch_geometric.data import Data, Batch

from .solutionnet import SolutionNet
from .featurizer import canonicalize_smiles
from .gen_features import GenFeatures


@lru_cache(maxsize=512)
def smiles_to_train_data(smiles: str) -> Data:
    return GenFeatures()(Data(smiles=smiles))


DIST_KEYS = ["AB", "AC", "BC", "AA", "BB", "CC"]
DEFAULT_EMULATE_BS = 32


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip()
        return float(s) if s else float(default)
    except Exception:
        return float(default)


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "weights"):
            v = ckpt.get(key, None)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Unexpected checkpoint format.")


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _swap_bc_features(features: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(features)
    for suf in ["enit", "hnum", "havg", "hmin", "hmax"]:
        out[f"AB_{suf}"], out[f"AC_{suf}"] = out.get(f"AC_{suf}", 0.0), out.get(f"AB_{suf}", 0.0)
        out[f"BB_{suf}"], out[f"CC_{suf}"] = out.get(f"CC_{suf}", 0.0), out.get(f"BB_{suf}", 0.0)
    return out


def _features_to_dist_and_hab(features: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """flat features -> dist(6) + HAB(4) (checkpoint hbond_proj=4)"""
    missing: List[str] = []

    dist: Dict[str, float] = {}
    for k in DIST_KEYS:
        kk = f"{k}_enit"
        if kk not in features:
            missing.append(kk)
        dist[k] = _to_float(features.get(kk, 0.0), 0.0)

    def _get(sk: str) -> float:
        if sk not in features:
            missing.append(sk)
        return _to_float(features.get(sk, 0.0), 0.0)

    hab = {
        "HAB_H_num": _get("AB_hnum"),
        "HAB_H_aver": _get("AB_havg"),
        "HAB_H_min": _get("AB_hmin"),
        "HAB_H_max": _get("AB_hmax"),
    }
    return dist, hab, missing


class SolutionNetPredictor:
    def __init__(self, root: Path):
        self.root = Path(root)

        # config optional: models/solutionnet_config.json
        cfg_path = self.root / "models" / "solutionnet_config.json"
        self.cfg: Dict[str, Any] = {}
        if cfg_path.exists():
            try:
                self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                self.cfg = {}

        # weights
        weights_env = os.getenv("SOLUTIONNET_WEIGHTS", "").strip()
        if weights_env:
            self.weights_path = Path(weights_env).expanduser().resolve()
        else:
            self.weights_path = (self.root / self.cfg.get("weights", "models/solutionnet.pth")).resolve()

        if not self.weights_path.exists():
            raise FileNotFoundError(f"Missing weights: {self.weights_path}")

        # hyperparams (match eval defaults)
        self.in_channels = int(self.cfg.get("in_channels", 39))
        self.hidden_channels = int(self.cfg.get("hidden_channels", 144))
        self.edge_dim = int(self.cfg.get("edge_dim", 10))
        self.dropout = float(self.cfg.get("dropout", 0.2))
        self.threshold_default = float(self.cfg.get("threshold_default", 0.5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ModelCls = SolutionNet

        self._model = None
        self._missing_keys: List[str] = []
        self._unexpected_keys: List[str] = []

    def _load_model(self):
        if self._model is not None:
            return self._model

        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        sd = _strip_known_prefixes(_extract_state_dict(ckpt))

        model = self.ModelCls(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            edge_dim=self.edge_dim,
            dropout=self.dropout,
        )

        model.load_state_dict(sd, strict=True)

        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def _predict_once(
        self,
        a_smi: str,
        b_smi: str,
        c_smi: str,
        dist: Dict[str, float],
        hab: Dict[str, float],
        emulate_bs: int,
    ) -> float:
        model = self._load_model()

        a = canonicalize_smiles(a_smi) or (a_smi or "").strip()
        b = canonicalize_smiles(b_smi) or (b_smi or "").strip()
        c = canonicalize_smiles(c_smi) or (c_smi or "").strip()
        if not (a and b and c):
            raise ValueError("Invalid SMILES.")

        # ✅ 训练同款分子图
        dataA = smiles_to_train_data(a)
        dataB = smiles_to_train_data(b)
        dataC = smiles_to_train_data(c)

        n = int(emulate_bs) if emulate_bs and int(emulate_bs) > 0 else 1

        batch_A = Batch.from_data_list([dataA] * n).to(self.device)
        batch_B = Batch.from_data_list([dataB] * n).to(self.device)
        batch_C = Batch.from_data_list([dataC] * n).to(self.device)

        inter = Data(
            AB=float(dist.get("AB", 0.0)),
            AC=float(dist.get("AC", 0.0)),
            BC=float(dist.get("BC", 0.0)),
            AA=float(dist.get("AA", 0.0)),
            BB=float(dist.get("BB", 0.0)),
            CC=float(dist.get("CC", 0.0)),
        )
        batch_i = Batch.from_data_list([inter] * n).to(self.device)

        h = Data(
            HAB_H_num=float(hab.get("HAB_H_num", 0.0)),
            HAB_H_aver=float(hab.get("HAB_H_aver", 0.0)),
            HAB_H_min=float(hab.get("HAB_H_min", 0.0)),
            HAB_H_max=float(hab.get("HAB_H_max", 0.0)),
        )
        batch_h = Batch.from_data_list([h] * n).to(self.device)

        with torch.inference_mode():
            y = model(batch_A, batch_B, batch_C, batch_i, batch_h).view(-1)
            return float(y[0].item())

    def predict(self, smiles_a: str, smiles_b: str, smiles_c: str, features: Dict[str, Any]):
        thr = float(_to_float(features.get("threshold"), self.threshold_default))
        symmetric = bool(features.get("symmetric", True))
        emulate_bs = int(features.get("emulate_bs", DEFAULT_EMULATE_BS))

        dist1, hab1, miss1 = _features_to_dist_and_hab(features)
        s1 = self._predict_once(smiles_a, smiles_b, smiles_c, dist1, hab1, emulate_bs=emulate_bs)

        s2: Optional[float] = None
        miss2: List[str] = []
        chosen = "A-B-C"
        score = s1

        if symmetric:
            feats_sw = _swap_bc_features(features)
            dist2, hab2, miss2 = _features_to_dist_and_hab(feats_sw)
            s2 = self._predict_once(smiles_a, smiles_c, smiles_b, dist2, hab2, emulate_bs=emulate_bs)
            if s2 > s1:
                score = s2
                chosen = "A-C-B"

        pred = bool(score >= thr)
        extra = {
            "model_kind": "SolutionNet",
            "weights_path": str(self.weights_path),
            "threshold": thr,
            "symmetric": symmetric,
            "emulate_bs": int(emulate_bs),
            "score_BC": float(s1),
            "score_CB": float(s2) if s2 is not None else None,
            "chosen_order": chosen,
            "missing_keys": self._missing_keys,
            "unexpected_keys": self._unexpected_keys,
            "missing_or_invalid_features_filled_as_0": list(dict.fromkeys(miss1 + miss2)),
            "device": str(self.device),
        }
        return float(score), pred, extra


@lru_cache(maxsize=1)
def _get_predictor() -> SolutionNetPredictor:
    root = Path(__file__).resolve().parents[1]
    return SolutionNetPredictor(root)


def predict_solutionnet(smiles_a: str, smiles_b: str, smiles_c: str, features: Dict[str, Any]):
    return _get_predictor().predict(smiles_a, smiles_b, smiles_c, features or {})
