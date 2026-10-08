from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


def _norm(s: str) -> str:
    return str(s).strip()


def _pair_key(a: str, b: str) -> str:
    x, y = sorted([_norm(a), _norm(b)])
    return f"{x}||{y}"


class DataStore:
    """
    Data access layer for the demo.

    Required files (under data/):
      - solubility.xlsx  (Drug, Solvent_1, Solvent_2, co-solvency)
      - smiles.xlsx      (Drug, Canonical_smiles / SMILES)
    Optional but recommended for SolutionNet auto-fill:
      - hbonds.xlsx      (A, B, Hnum, average_hbond_length, min_hbond_length, max_hbond_length)
      - enit.xlsx        (A, B, Enit_cal)
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

        self.solubility_xlsx = self.data_dir / "solubility.xlsx"
        self.smiles_xlsx = self.data_dir / "smiles.xlsx"
        self.hbonds_xlsx = self.data_dir / "hbonds.xlsx"
        self.enit_xlsx = self.data_dir / "enit.xlsx"

    # -----------------------------
    # Base tables
    # -----------------------------
    @lru_cache(maxsize=1)
    def solubility_df(self) -> pd.DataFrame:
        if not self.solubility_xlsx.exists():
            raise FileNotFoundError(f"Missing data file: {self.solubility_xlsx}")
        df = pd.read_excel(self.solubility_xlsx)
        for c in ["Drug", "Solvent_1", "Solvent_2"]:
            if c in df.columns:
                df[c] = df[c].astype(str).map(_norm)
        return df

    @lru_cache(maxsize=1)
    def smiles_map(self) -> Dict[str, str]:
        if not self.smiles_xlsx.exists():
            raise FileNotFoundError(f"Missing data file: {self.smiles_xlsx}")
        df = pd.read_excel(self.smiles_xlsx)

        out: Dict[str, str] = {}
        for _, r in df.iterrows():
            name = _norm(r.get("Drug", ""))
            if not name:
                continue

            can = r.get("Canonical_smiles", None)
            smi = r.get("SMILES", None)

            v = None
            if pd.notna(can):
                v = _norm(can)
            elif pd.notna(smi):
                v = _norm(smi)

            if v:
                out[name] = v

        return out

    @lru_cache(maxsize=1)
    def solvent_pairs_df(self) -> pd.DataFrame:
        """
        Aggregate the solubility dataset into unique solvent pairs (order-invariant).

        Output columns:
          pair_key, solvent1, solvent2, solvent1_smiles, solvent2_smiles, n_total, n_pos, n_neg, pos_rate
        """
        df = self.solubility_df()
        smap = self.smiles_map()

        tmp = df.copy()
        tmp["pair_key"] = tmp.apply(lambda r: _pair_key(r["Solvent_1"], r["Solvent_2"]), axis=1)

        rows = []
        for key, g in tmp.groupby("pair_key"):
            s1, s2 = key.split("||", 1)
            n_total = int(len(g))
            n_pos = int((g["co-solvency"] == 1).sum())
            n_neg = int((g["co-solvency"] == 0).sum())
            pos_rate = float(n_pos / n_total) if n_total else 0.0

            rows.append(
                {
                    "pair_key": key,
                    "solvent1": s1,
                    "solvent2": s2,
                    "solvent1_smiles": smap.get(s1, ""),
                    "solvent2_smiles": smap.get(s2, ""),
                    "n_total": n_total,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "pos_rate": pos_rate,
                }
            )

        out = pd.DataFrame(rows)
        out = out.sort_values(["pos_rate", "n_total"], ascending=[False, False]).reset_index(drop=True)
        return out

    def pair_detail_df(self, solvent1: str, solvent2: str) -> pd.DataFrame:
        df = self.solubility_df()
        s1 = _norm(solvent1)
        s2 = _norm(solvent2)
        m = (
            ((df["Solvent_1"] == s1) & (df["Solvent_2"] == s2))
            | ((df["Solvent_1"] == s2) & (df["Solvent_2"] == s1))
        )
        g = df.loc[m, ["Drug", "Solvent_1", "Solvent_2", "co-solvency"]].copy()
        g = g.sort_values(["co-solvency", "Drug"], ascending=[False, True]).reset_index(drop=True)
        return g

    # -----------------------------
    # Interaction tables (SolutionNet auto-fill)
    # -----------------------------
    @lru_cache(maxsize=1)
    def hbonds_map(self) -> Dict[str, Dict[str, float]]:
        """
        Return dict: pair_key -> {hnum, havg, hmin, hmax}
        """
        if not self.hbonds_xlsx.exists():
            return {}

        df = pd.read_excel(self.hbonds_xlsx)

        # Normalize columns
        if "A" not in df.columns or "B" not in df.columns:
            return {}

        df["A"] = df["A"].astype(str).map(_norm)
        df["B"] = df["B"].astype(str).map(_norm)
        df["pair_key"] = df.apply(lambda r: _pair_key(r["A"], r["B"]), axis=1)

        # Map names to expected stats columns
        # (Your file uses: Hnum, average_hbond_length, min_hbond_length, max_hbond_length)
        col_map = {
            "hnum": "Hnum",
            "havg": "average_hbond_length",
            "hmin": "min_hbond_length",
            "hmax": "max_hbond_length",
        }
        for out_col, src_col in col_map.items():
            if src_col not in df.columns:
                df[out_col] = pd.NA
            else:
                df[out_col] = pd.to_numeric(df[src_col], errors="coerce")

        # Aggregate duplicates by mean
        agg = df.groupby("pair_key")[["hnum", "havg", "hmin", "hmax"]].mean(numeric_only=True)

        out: Dict[str, Dict[str, float]] = {}
        for key, r in agg.iterrows():
            d = {}
            for k in ["hnum", "havg", "hmin", "hmax"]:
                v = r.get(k, pd.NA)
                if pd.notna(v):
                    d[k] = float(v)
            out[str(key)] = d
        return out

    @lru_cache(maxsize=1)
    def enit_map(self) -> Dict[str, float]:
        """
        Return dict: pair_key -> enit (float)
        """
        if not self.enit_xlsx.exists():
            return {}

        df = pd.read_excel(self.enit_xlsx)
        if "A" not in df.columns or "B" not in df.columns:
            return {}

        df["A"] = df["A"].astype(str).map(_norm)
        df["B"] = df["B"].astype(str).map(_norm)
        df["pair_key"] = df.apply(lambda r: _pair_key(r["A"], r["B"]), axis=1)

        src_col = "Enit_cal" if "Enit_cal" in df.columns else None
        if src_col is None:
            return {}

        df["enit"] = pd.to_numeric(df[src_col], errors="coerce")
        agg = df.groupby("pair_key")["enit"].mean(numeric_only=True)

        out: Dict[str, float] = {}
        for key, v in agg.items():
            if pd.notna(v):
                out[str(key)] = float(v)
        return out

    def interaction_lookup(self, name1: str, name2: str) -> Optional[Dict[str, float]]:
        """
        Lookup pairwise interaction stats for (name1, name2). Order invariant.

        Returns keys among:
          - enit
          - hnum, havg, hmin, hmax
        """
        n1 = _norm(name1)
        n2 = _norm(name2)
        if not n1 or not n2:
            return None

        key = _pair_key(n1, n2)
        out: Dict[str, float] = {}

        en = self.enit_map().get(key)
        if en is not None:
            out["enit"] = float(en)

        hb = self.hbonds_map().get(key)
        if hb:
            out.update({k: float(v) for k, v in hb.items()})

        return out or None
