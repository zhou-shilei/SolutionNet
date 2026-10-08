# -*- coding: utf-8 -*-
"""
Cosolvency Predictor (Gradio Web App) — single-file app.py

Tabs (in this order)
- Prediction (ABC)
- A-only Screening (RF/SolutionNet)
- Solvent Pair Library

Key UX behavior
- Prediction Step 2 (solvent pair chooser) is ALWAYS visible as an Accordion (fold/unfold),
  and auto-folds when B/C are already valid; auto-expands when B/C missing/invalid.
- Step 2 selection is click-to-select: click ANY cell in the pairs table to pick that row and fill B/C.
- A-only Screening supports BOTH RF and SolutionNet:
  - Only input is A (API/solute). Solvents are taken from the library (auto).
  - Produces a colored matrix (solvent × solvent) where each cell is a pair; color follows probability.
  - Produces two ranked lists: Positive pairs and Negative pairs.
  - Export links are shown NEXT TO the outputs (no extra File component, no two-step export->download).
    * Matrix area: “导出 PNG” (data-uri download, avoids nginx /file 404)
    * Tables area: “导出 Excel” (data-uri download)

Defensive behavior
- If src.inference_solutionnet import fails, app still runs (RF-only).
- Gradio-version compatibility: filters unsupported kwargs for components.

Run
  python app.py --host 0.0.0.0 --port 7860

Reverse-proxy notes
- This app still sets root_path/allowed_paths for normal Gradio file serving,
  but A-only exports use data-URI download to bypass common nginx 404 issues.
"""

from __future__ import annotations

import argparse
import base64
import html
import inspect
import io
import json
import os
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from rdkit import Chem

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.db import DataStore
from src.inference_rf import RFInference
from src.molview import render_ball_stick_html


# ------------------------------
# Optional SolutionNet import (do not crash app if PyG missing)
# ------------------------------
SOLUTIONNET_IMPORT_ERROR: Optional[str] = None
predict_solutionnet = None
try:
    from src.inference_solutionnet import predict_solutionnet as _predict_solutionnet  # type: ignore
    predict_solutionnet = _predict_solutionnet
except Exception as e:  # noqa: BLE001
    SOLUTIONNET_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ==============================
# Paths / Globals
# ==============================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RF_MODEL_DEFAULT = MODELS_DIR / "rf_model.pkl"

APP_TITLE = "CosolvencyWeb: Cosolvency Prediction Platform"
APP_TITLE_ZH = "CosolvencyWeb: 潜溶预测平台"


UI_TEXT: Dict[str, Dict[str, str]] = {
    "zh": {
        "tab_prediction": "单次实验预测",
        "tab_screening": "溶剂虚拟筛选",
        "tab_library": "溶剂对库",
        "tab_deployment": "部署说明",
        "pair_chooser_title": "从数据库选择溶剂对 (B,C)（可选）",
        "settings_title": "<div class='h1' style='font-size:18px;font-weight:700'>设置</div>",
        "tools_title": "<div class='h1' style='font-size:18px;font-weight:700'>SMILES 便捷输入</div>",
        "model_label": "模型",
        "threshold_label": "预测阈值",
        "threshold_info": "概率 ≥ 阈值时判为潜溶。",
        "symmetric_info": "比较两个溶剂顺序，取较高概率。",
        "symmetric_label": "对称推理（交换溶剂 B/C）",
        "tool_target": "填入目标",
        "tool_text": "输入中文/英文名或 SMILES",
        "tool_text_placeholder": "例如：水 / C1CCOC1",
        "tool_parse": "解析 SMILES",
        "tool_name": "解析名称（可选）",
        "tool_smiles": "Canonical SMILES",
        "tool_fill": "填入目标并刷新",
        "main_title": f"<div class='h1'>{APP_TITLE_ZH}</div>",
        "step1": (
            "<div class='step-title'>"
            "<span class='step-badge'>1</span>"
            "<span class='step-text'>输入分子（名称或 SMILES）</span>"
            "</div>"
            "<div class='small'>"
            "输入框支持：中文别名 / 英文名 / SMILES（系统自动解析）。"
            "可在左侧 <b>SMILES 便捷输入</b> 解析并填入。"
            "</div>"
        ),
        "a_label": "A（溶质/API）",
        "b_label": "B（溶剂 1）",
        "c_label": "C（溶剂 2）",
        "summary_label": "SMILES",
        "view_a": "A 图示",
        "view_b": "B 图示",
        "view_c": "C 图示",
        "refresh": "刷新 SMILES 与结构图",
        "step2": (
            "<div class='small'>"
            "当你已经手动填入 B/C（并且都能解析到 SMILES）时，此步骤会自动折叠；"
            "当 B/C 为空或解析失败时会自动展开。<br/>"
            "<b>选择方式：</b>加载列表后，<b>直接点击表格任意单元格</b> 即可选中该行并填入 B/C。"
            "</div>"
        ),
        "pair_search": "搜索溶剂名称（可选）",
        "pair_min_pos": "最低 pos_rate",
        "pair_min_total": "最低 n_total",
        "pair_max_rows": "最大行数",
        "pair_reload": "加载溶剂对",
        "pair_table": "溶剂对（点击任意行使用）",
        "step3": (
            "<div class='step-title'>"
            "<span class='step-badge'>3</span>"
            "<span class='step-text'>填写 / 更新 interactions（SolutionNet 可选；默认 0 也可预测）</span>"
            "</div>"
            "<div class='small'>"
            "<b>pair 含义：</b>AA=A-A, AB=A-B, AC=A-C, BB=B-B, BC=B-C, CC=C-C<br/>"
            "<b>Update：</b>从数据库写入（仅覆盖当前为 0 的格子，不覆盖你手改的非 0）。"
            "</div>"
        ),
        "interactions_label": "Interactions 表（pair + enit + hbonds）",
        "reset": "重置为 0",
        "update_data": "从数据更新（enit/hbonds）",
        "update_meta": "更新报告 / Debug",
        "extra_json": "额外特征（JSON，可选）",
        "step4": (
            "<div class='step-title'>"
            "<span class='step-badge'>4</span>"
            "<span class='step-text'>预测</span>"
            "</div>"
            "<div class='small'>点击 Predict 后会使用当前 A/B/C（解析后的 SMILES）进行预测。</div>"
        ),
        "predict": "预测",
        "raw_result": "原始结果（JSON）",
        "screen_intro": (
            "只输入 A（溶质/API），系统会对溶剂库中的溶剂组合进行预测：\n"
            "- 输出一个 **溶剂×溶剂矩阵**（颜色随概率变化）\n"
            "- 输出两张表：**潜溶 pairs** 与 **非潜溶 pairs**（各自按 score 降序）\n\n"
            "导出方式：结果生成后，直接点击矩阵旁的“导出 PNG”或表格旁的“导出 Excel”。"
        ),
        "a2_label": "A（溶质/API）- 中文/英文名或 SMILES",
        "only_reported": "仅筛选文献中报道的溶剂组合（只显示/排序已出现的 pairs）",
        "max_solvents": "矩阵最大溶剂数量（高级）",
        "run_screen": "运行筛选",
        "screen_empty": "<div class='empty-state'><span class='empty-icon'>↗</span><strong>筛选结果将在这里展示</strong><p>输入溶质并运行筛选，查看溶剂组合矩阵和排序结果。</p></div>",
        "matrix_note": "<div class='small'>矩阵中的 NA 表示该溶剂对未计算（可能是未在文献中报道，或缺少 SMILES）。</div>",
        "pos_df": "潜溶（Positive）pairs - 按 score 降序",
        "neg_df": "非潜溶（Negative）pairs - 按 score 降序",
        "screen_meta": "元信息 / Debug",
        "search": "搜索溶剂名称",
        "reload": "重新加载 / 筛选",
        "pairs": "溶剂对",
        "library_hint": "<div class='small'>提示：在 Prediction 的 Step 2 里可直接点击表格选行填入 B/C。</div>",
        "deployment": (
            "### 启动\n"
            "```bash\npython app.py --host 0.0.0.0 --port 7860\n```\n\n"
            "### 子路径部署（可选）\n"
            "```bash\nROOT_PATH=/cosolvency python app.py --host 0.0.0.0 --port 7860\n```\n\n"
            "### 为什么 A-only 导出不再走 /file？\n"
            "很多 nginx 反代会漏配 `/file=` 相关路径，导致下载 404。"
            "本版本 A-only 的 PNG/Excel 导出使用 data-uri（浏览器直接下载），绕过该问题。\n\n"
            "### SolutionNet 可用性\n"
            "如果日志显示 `SolutionNet import: failed`，通常是 torch_geometric/torch-scatter 或 GLIBC 兼容问题；此时仍可使用 RF。"
        ),
    },
    "en": {
        "tab_prediction": "Single Experiment Prediction",
        "tab_screening": "Virtual Screening of Solvents",
        "tab_library": "Solvent Pair Library",
        "tab_deployment": "Deployment Notes",
        "pair_chooser_title": "Select solvent pair from database (B,C) (optional)",
        "settings_title": "<div class='h1' style='font-size:18px;font-weight:700'>Settings</div>",
        "tools_title": "<div class='h1' style='font-size:18px;font-weight:700'>Quick SMILES Input</div>",
        "model_label": "Model",
        "threshold_label": "Prediction threshold",
        "threshold_info": "Probability ≥ threshold means Positive.",
        "symmetric_info": "Compare both solvent orders and use the higher probability.",
        "symmetric_label": "Symmetric inference (swap B/C)",
        "tool_target": "Fill target",
        "tool_text": "name or SMILES",
        "tool_text_placeholder": "e.g., Water / C1CCOC1",
        "tool_parse": "Parse SMILES",
        "tool_name": "Resolved name (optional)",
        "tool_smiles": "Canonical SMILES",
        "tool_fill": "Fill target & refresh",
        "main_title": f"<div class='h1'>{APP_TITLE}</div>",
        "step1": (
            "<div class='step-title'>"
            "<span class='step-badge'>1</span>"
            "<span class='step-text'>Enter molecules (name or SMILES)</span>"
            "</div>"
            "<div class='small'>"
            "Inputs support Chinese aliases, English names, and SMILES strings. "
            "Use the <b>Quick SMILES Input</b> on the left to resolve and fill inputs."
            "</div>"
        ),
        "a_label": "A (Solute/API)",
        "b_label": "B (Solvent 1)",
        "c_label": "C (Solvent 2)",
        "summary_label": "SMILES",
        "view_a": "A view",
        "view_b": "B view",
        "view_c": "C view",
        "refresh": "Refresh SMILES & structures",
        "step2": (
            "<div class='small'>"
            "This step auto-collapses when B/C are already valid and auto-expands when B/C are empty or unresolved.<br/>"
            "<b>How to select:</b> load the table, then click any cell in a row to fill B/C."
            "</div>"
        ),
        "pair_search": "Search solvent name (optional)",
        "pair_min_pos": "Min pos_rate",
        "pair_min_total": "Min n_total",
        "pair_max_rows": "Max rows",
        "pair_reload": "Load pairs",
        "pair_table": "Pairs (click a row to use)",
        "step3": (
            "<div class='step-title'>"
            "<span class='step-badge'>3</span>"
            "<span class='step-text'>Fill / update interactions (optional for SolutionNet; zeros are allowed)</span>"
            "</div>"
            "<div class='small'>"
            "<b>Pair meaning:</b> AA=A-A, AB=A-B, AC=A-C, BB=B-B, BC=B-C, CC=C-C<br/>"
            "<b>Update:</b> fill from the database and only overwrite cells that are currently zero."
            "</div>"
        ),
        "interactions_label": "Interactions table (pair + enit + hbonds)",
        "reset": "Reset to 0",
        "update_data": "Update from data (enit/hbonds)",
        "update_meta": "Update report / Debug",
        "extra_json": "Extra features (JSON, optional)",
        "step4": (
            "<div class='step-title'>"
            "<span class='step-badge'>4</span>"
            "<span class='step-text'>Prediction</span>"
            "</div>"
            "<div class='small'>Click Predict to evaluate the currently resolved A/B/C SMILES.</div>"
        ),
        "predict": "Predict",
        "raw_result": "Raw result (JSON)",
        "screen_intro": (
            "Enter only A (solute/API), and the server will evaluate solvent combinations from the library:\n"
            "- Output a **solvent-by-solvent matrix** colored by prediction score\n"
            "- Output two ranked tables: **positive pairs** and **negative pairs**, sorted by score\n\n"
            "After screening, click “Export PNG” next to the matrix or “Export Excel” next to the tables."
        ),
        "a2_label": "A (Solute/API) - Chinese/English name or SMILES",
        "only_reported": "Only screen solvent pairs reported in the literature/database",
        "max_solvents": "Max solvents in matrix (advanced)",
        "run_screen": "Run screening",
        "screen_empty": "<div class='empty-state'><span class='empty-icon'>↗</span><strong>Your screening results will appear here</strong><p>Enter a solute and run screening to explore the solvent matrix and ranked pairs.</p></div>",
        "matrix_note": "<div class='small'>NA means the solvent pair was not evaluated, usually because it is unreported or missing SMILES.</div>",
        "pos_df": "Positive pairs - sorted by score descending",
        "neg_df": "Negative pairs - sorted by score descending",
        "screen_meta": "Meta / Debug",
        "search": "Search solvent name",
        "reload": "Reload / Filter",
        "pairs": "Pairs",
        "library_hint": "<div class='small'>Tip: in Prediction Step 2, click a row to fill B/C directly.</div>",
        "deployment": (
            "### Launch\n"
            "```bash\npython app.py --host 0.0.0.0 --port 7860\n```\n\n"
            "### Subpath deployment (optional)\n"
            "```bash\nROOT_PATH=/cosolvency python app.py --host 0.0.0.0 --port 7860\n```\n\n"
            "### Why do A-only exports avoid /file?\n"
            "Some nginx reverse proxies miss Gradio file routes and cause 404 errors. "
            "This version uses data-URI downloads for PNG/Excel exports.\n\n"
            "### SolutionNet availability\n"
            "If logs show `SolutionNet import: failed`, the usual cause is torch_geometric/torch-scatter or GLIBC compatibility. RF remains available."
        ),
    },
}


def _ui_lang(lang: str) -> str:
    s = str(lang or "").strip().lower()
    return "en" if s.startswith("en") or "english" in s else "zh"


def _ui_text(lang: str, key: str) -> str:
    return UI_TEXT[_ui_lang(lang)][key]


store = DataStore(DATA_DIR)
rf = RFInference(RF_MODEL_DEFAULT)


# 中文别名（按需继续补）
ALIASES: Dict[str, str] = {
    "水": "Water",
    "乙醇": "Ethanol",
    "酒精": "Ethanol",
    "四氢呋喃": "Tetrahydrofuran",
    "THF": "Tetrahydrofuran",
    "thf": "Tetrahydrofuran",
    "FDCA": "2,5-Furandicarboxylic acid",
    "fdca": "2,5-Furandicarboxylic acid",
    "2,5-呋喃二甲酸": "2,5-Furandicarboxylic acid",
    "2,5-呋喃二甲酸二甲酸": "2,5-Furandicarboxylic acid",
}


def _has_zh(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (s or ""))


EN2CN: Dict[str, str] = {}
for cn, en in ALIASES.items():
    if _has_zh(cn) and not _has_zh(en):
        EN2CN.setdefault(en, cn)


# ==============================
# Small helpers: kwargs filtering for older Gradio versions
# ==============================
def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(callable_obj)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        return kwargs


def DF(**kwargs):
    """gr.Dataframe wrapper with signature filtering (older gradio compatibility)."""
    return gr.Dataframe(**_filter_kwargs(gr.Dataframe.__init__, kwargs))


def ACCORDION(title: str, open_: bool = True, elem_id: Optional[str] = None):
    """Accordion wrapper (fallback to Group if Accordion missing)."""
    if hasattr(gr, "Accordion"):
        return gr.Accordion(title, open=open_, elem_id=elem_id)
    return gr.Group(elem_id=elem_id)


# ==============================
# Helpers: resolve name / SMILES
# ==============================
def _canon_smiles(smi: str) -> str:
    """Return RDKit canonical SMILES; empty if invalid."""
    try:
        m = Chem.MolFromSmiles((smi or "").strip())
        if m is None:
            return ""
        return Chem.MolToSmiles(m)
    except Exception:  # noqa: BLE001
        return ""


@lru_cache(maxsize=1)
def _smiles_reverse_map() -> Dict[str, str]:
    """canonical_smiles -> name (best-effort, first match)"""
    smap = store.smiles_map()
    rev: Dict[str, str] = {}
    for name, smi in smap.items():
        cs = _canon_smiles(smi)
        if cs and cs not in rev:
            rev[cs] = name
    return rev


def resolve_input(x: str) -> Tuple[str, str]:
    """
    Resolve user input into (name, smiles).

    - If x is a known name (or alias), return mapped name & smiles.
    - If x is a SMILES, canonicalize and best-effort reverse-map to a name.
    - Return ("","") if cannot resolve.
    """
    x = (x or "").strip()
    if not x:
        return "", ""

    if x in ALIASES:
        x = ALIASES[x]

    smap = store.smiles_map()

    if x in smap:
        return x, smap[x]

    for k, v in smap.items():
        if k.lower() == x.lower():
            return k, v

    cs = _canon_smiles(x)
    if cs:
        name = _smiles_reverse_map().get(cs, "")
        return name, cs

    hits = [(k, v) for k, v in smap.items() if x.lower() in k.lower()]
    if len(hits) == 1:
        return hits[0][0], hits[0][1]

    return "", ""


# ==============================
# 3D render cache
# ==============================
@lru_cache(maxsize=256)
def _render_cached_ball_stick(canon_smi: str, height: int = 280) -> str:
    return render_ball_stick_html(canon_smi, height=height)


# ==============================
# Interactions features (SolutionNet)
# ==============================
INTERACTION_ROWS = ["AA", "AB", "AC", "BB", "BC", "CC"]
INTERACTION_COLS = ["pair", "enit", "hnum", "havg", "hmin", "hmax"]

REQUIRED_KEYS_SIMPLIFIED = [
    "AA_enit", "AB_enit", "AC_enit", "BB_enit", "BC_enit", "CC_enit",
    "AA_hnum", "AA_havg", "AA_hmin", "AA_hmax",
    "AB_hnum", "AB_havg", "AB_hmin", "AB_hmax",
    "AC_hnum", "AC_havg", "AC_hmin", "AC_hmax",
    "BB_hnum", "BB_havg", "BB_hmin", "BB_hmax",
    "BC_hnum", "BC_havg", "BC_hmin", "BC_hmax",
    "CC_hnum", "CC_havg", "CC_hmin", "CC_hmax",
]


def make_interaction_template(default_zero: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({"pair": INTERACTION_ROWS})
    for c in ["enit", "hnum", "havg", "hmin", "hmax"]:
        df[c] = 0.0 if default_zero else None
    return df


def normalize_interaction_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return make_interaction_template(default_zero=True)

    df = df.copy()

    for c in INTERACTION_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c != "pair" else ""
    df = df[INTERACTION_COLS]

    pairs = [str(x).strip() for x in df["pair"].tolist()]
    if set(pairs) != set(INTERACTION_ROWS):
        if df.shape[0] == len(INTERACTION_ROWS):
            df["pair"] = INTERACTION_ROWS
        else:
            return make_interaction_template(default_zero=True)

    df["pair"] = df["pair"].astype(str).str.strip()
    df["_ord"] = df["pair"].apply(lambda x: INTERACTION_ROWS.index(x) if x in INTERACTION_ROWS else 999)
    df = df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    for c in ["enit", "hnum", "havg", "hmin", "hmax"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    return df


def interactions_to_features(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    df = normalize_interaction_df(df)
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        pair = str(row["pair"]).strip()
        if pair not in INTERACTION_ROWS:
            continue
        out[f"{pair}_enit"] = float(row["enit"])
        out[f"{pair}_hnum"] = float(row["hnum"])
        out[f"{pair}_havg"] = float(row["havg"])
        out[f"{pair}_hmin"] = float(row["hmin"])
        out[f"{pair}_hmax"] = float(row["hmax"])
    return out


def autofill_interactions(df: Optional[pd.DataFrame], name_a: str, name_b: str, name_c: str) -> pd.DataFrame:
    """Fill from db (only overwrite zeros)."""
    df = normalize_interaction_df(df).copy()

    def fill_pair(pair_key: str, n1: str, n2: str):
        if not n1 or not n2:
            return
        stats = store.interaction_lookup(n1, n2)
        if not stats:
            return

        idx = df.index[df["pair"] == pair_key]
        if len(idx) == 0:
            return
        i = int(idx[0])

        v_enit = stats.get("enit", None)
        if v_enit is not None and float(df.loc[i, "enit"]) == 0.0:
            df.loc[i, "enit"] = float(v_enit)

        for col in ["hnum", "havg", "hmin", "hmax"]:
            v = stats.get(col, None)
            if v is not None and float(df.loc[i, col]) == 0.0:
                df.loc[i, col] = float(v)

    fill_pair("AA", name_a, name_a)
    fill_pair("BB", name_b, name_b)
    fill_pair("CC", name_c, name_c)
    fill_pair("AB", name_a, name_b)
    fill_pair("AC", name_a, name_c)
    fill_pair("BC", name_b, name_c)

    return df


def build_solutionnet_features_from_db(
    name_a: str,
    name_b: str,
    name_c: str,
    threshold: float,
    symmetric: bool,
) -> Dict[str, Any]:
    """Build the SolutionNet feature dict from the interaction DB. Missing values are 0.0."""

    def get_stats(n1: str, n2: str) -> Dict[str, float]:
        if not n1 or not n2:
            return {}
        stats = store.interaction_lookup(n1, n2) or {}
        out: Dict[str, float] = {}
        for k in ["enit", "hnum", "havg", "hmin", "hmax"]:
            v = stats.get(k, None)
            if v is None:
                continue
            try:
                out[k] = float(v)
            except Exception:
                continue
        return out

    pairs = {
        "AA": (name_a, name_a),
        "AB": (name_a, name_b),
        "AC": (name_a, name_c),
        "BB": (name_b, name_b),
        "BC": (name_b, name_c),
        "CC": (name_c, name_c),
    }

    feats: Dict[str, Any] = {}
    for p, (n1, n2) in pairs.items():
        s = get_stats(n1, n2)
        feats[f"{p}_enit"] = float(s.get("enit", 0.0))
        feats[f"{p}_hnum"] = float(s.get("hnum", 0.0))
        feats[f"{p}_havg"] = float(s.get("havg", 0.0))
        feats[f"{p}_hmin"] = float(s.get("hmin", 0.0))
        feats[f"{p}_hmax"] = float(s.get("hmax", 0.0))

    feats["threshold"] = float(threshold)
    feats["symmetric"] = bool(symmetric)
    return feats


# ==============================
# UI callbacks: resolve/preview
# ==============================
def _make_summary(a_in: str, b_in: str, c_in: str, lang: str = "zh") -> Tuple[pd.DataFrame, Tuple[str, str, str, str, str, str]]:
    en_a, a = resolve_input(a_in)
    en_b, b = resolve_input(b_in)
    en_c, c = resolve_input(c_in)

    cn_a = EN2CN.get(en_a, "")
    cn_b = EN2CN.get(en_b, "")
    cn_c = EN2CN.get(en_c, "")

    is_en = _ui_lang(lang) == "en"
    names = (en_a, en_b, en_c) if is_en else (cn_a or en_a, cn_b or en_b, cn_c or en_c)
    field = "Field" if is_en else "字段"
    summary = pd.DataFrame([
        {field: "Name" if is_en else "名称", "A": names[0], "B": names[1], "C": names[2]},
        {field: "SMILES", "A": a or "", "B": b or "", "C": c or ""},
    ])
    return summary, (en_a, a, en_b, b, en_c, c)


def _summary_html(summary: pd.DataFrame, lang: str) -> str:
    boxes = []
    for field, key in [("A", "a_label"), ("B", "b_label"), ("C", "c_label")]:
        label = html.escape(_ui_text(lang, key))
        smiles = html.escape(str(summary.iloc[1][field] or ""))
        boxes.append(
            f"<label class='smiles-output'><span>SMILES</span>"
            f"<textarea readonly rows='1' spellcheck='false' aria-label='{label} SMILES'>{smiles}</textarea></label>"
        )
    return "<div class='smiles-output-grid'>" + "".join(boxes) + "</div>"


def ui_resolve_table_only(a_in: str, b_in: str, c_in: str, lang: str = "zh"):
    summary, _ = _make_summary(a_in, b_in, c_in, lang)
    return _summary_html(summary, lang)


def ui_refresh_summary(a_in: str, b_in: str, c_in: str, lang: str = "zh"):
    summary, (_en_a, a, _en_b, b, _en_c, c) = _make_summary(a_in, b_in, c_in, lang)

    def safe_html(smi: str):
        try:
            cs = _canon_smiles(smi)
            if not cs:
                return "<div class='muted'>Invalid SMILES</div>"
            return _render_cached_ball_stick(cs, height=280)
        except Exception as e:  # noqa: BLE001
            return f"<div style='color:#b91c1c'>Render failed: {type(e).__name__}: {e}</div>"

    view_a = safe_html(a) if a else "<div class='muted'>Unresolved</div>"
    view_b = safe_html(b) if b else "<div class='muted'>Unresolved</div>"
    view_c = safe_html(c) if c else "<div class='muted'>Unresolved</div>"

    return _summary_html(summary, lang), view_a, view_b, view_c


# ==============================
# Solvent pair library / selection
# ==============================
def ui_library_filter(search: str, min_pos_rate: float, min_total: int, max_rows: int):
    df = store.solvent_pairs_df().copy()
    q = (search or "").strip().lower()
    if q:
        df = df[df["solvent1"].str.lower().str.contains(q) | df["solvent2"].str.lower().str.contains(q)]
    df = df[(df["pos_rate"] >= float(min_pos_rate)) & (df["n_total"] >= int(min_total))].reset_index(drop=True)
    df = df.head(int(max_rows)).reset_index(drop=True)
    return df


def ui_use_pair_names(pair_key: str):
    if not pair_key:
        return "", ""
    s1, s2 = pair_key.split("||", 1)
    return s1, s2


def ui_use_pair_and_refresh(pair_key: str, a_in: str):
    b_name, c_name = ui_use_pair_names(pair_key)
    summary, view_a, view_b, view_c = ui_refresh_summary(a_in, b_name, c_name)
    return b_name, c_name, summary, view_a, view_b, view_c


# ==============================
# Literature-reported solvent pairs
# ==============================
@lru_cache(maxsize=1)
def _reported_pair_set() -> set[tuple[str, str]]:
    """Set of unordered solvent pairs that appear in the literature dataset."""
    try:
        df = store.solvent_pairs_df()
        if df is None or df.empty:
            return set()
        pairs = set()
        for _, r in df.iterrows():
            s1 = str(r.get("solvent1", "")).strip()
            s2 = str(r.get("solvent2", "")).strip()
            if not s1 or not s2:
                continue
            a, b = sorted([s1, s2])
            pairs.add((a, b))
        return pairs
    except Exception:
        return set()


def _is_reported_pair(s1: str, s2: str) -> bool:
    a, b = sorted([str(s1).strip(), str(s2).strip()])
    return (a, b) in _reported_pair_set()


# ==============================
# Step2 UX: auto fold/unfold + click-to-select
# ==============================
def ui_toggle_step2_open(b_in: str, c_in: str):
    """B/C 都可解析 => 折叠(open=False)，否则展开(open=True)"""
    _, b = resolve_input(b_in)
    _, c = resolve_input(c_in)
    should_open = not (b and c)
    if hasattr(gr, "Accordion"):
        return gr.update(open=should_open)
    return gr.update()


def ui_pick_pair_from_table(df: pd.DataFrame, evt: gr.SelectData, a_in: str, lang: str = "zh"):
    """Click any cell in the pair table -> choose that row's pair_key -> fill B/C and refresh previews."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return "", "", "<div class='err'>表格为空：先点 Load pairs 或输入关键词搜索。</div>", gr.update(), gr.update(), gr.update(), gr.update()

        row = int(evt.index[0])  # (row, col)
        if "pair_key" not in df.columns:
            return "", "", "<div class='err'>内部错误：表格缺少 pair_key 列。</div>", gr.update(), gr.update(), gr.update(), gr.update()

        pair_key = str(df.iloc[row]["pair_key"])
        b_name, c_name, summary, view_a, view_b, view_c = ui_use_pair_and_refresh(pair_key, a_in)
        summary = ui_resolve_table_only(a_in, b_name, c_name, lang)
        return b_name, c_name, "", summary, view_a, view_b, view_c
    except Exception as e:  # noqa: BLE001
        return "", "", f"<div class='err'>选择失败：{type(e).__name__}: {e}</div>", gr.update(), gr.update(), gr.update(), gr.update()


# ==============================
# Interactions table UI callbacks
# ==============================
def ui_reset_interactions():
    return make_interaction_template(default_zero=True), {"reset": True, "note": "Reset all interactions to 0.0"}


def ui_update_interactions(a_in: str, b_in: str, c_in: str, inter_df: pd.DataFrame):
    name_a, _ = resolve_input(a_in)
    name_b, _ = resolve_input(b_in)
    name_c, _ = resolve_input(c_in)

    before = normalize_interaction_df(inter_df)
    after = autofill_interactions(before, name_a, name_b, name_c)

    changed_cells: List[str] = []
    for r in INTERACTION_ROWS:
        rb = before[before["pair"] == r].iloc[0]
        ra = after[after["pair"] == r].iloc[0]
        for c in ["enit", "hnum", "havg", "hmin", "hmax"]:
            if float(rb[c]) != float(ra[c]):
                changed_cells.append(f"{r}_{c}: {float(rb[c])} -> {float(ra[c])}")

    feats = interactions_to_features(after)
    required_values = {k: float(feats.get(k, 0.0)) for k in REQUIRED_KEYS_SIMPLIFIED}

    meta = {
        "updated_from_data": True,
        "name_a": name_a,
        "name_b": name_b,
        "name_c": name_c,
        "changed_cells": changed_cells,
        "required_keys_for_model": REQUIRED_KEYS_SIMPLIFIED,
        "required_values_snapshot": required_values,
        "solutionnet_import_error": SOLUTIONNET_IMPORT_ERROR,
    }
    return after, meta


# ==============================
# Prediction UI callbacks
# ==============================
def _result_html(pred: bool, model: str, score: float, thr: float, name_a: str, name_b: str, name_c: str, extra_line: str = "", lang: str = "en", details: Optional[Dict[str, Any]] = None) -> str:
    is_en = _ui_lang(lang) == "en"
    title = ("Co-solvency likely" if pred else "Co-solvency unlikely") if is_en else ("更可能潜溶" if pred else "更可能不潜溶")
    probability_label = "Co-solvency probability" if is_en else "潜溶概率"
    threshold_label = "Threshold" if is_en else "判断阈值"
    details_label = "Solvent order comparison" if is_en else "溶剂顺序比较"
    selected_label = "Selected order" if is_en else "采用顺序"
    pct = max(0.0, min(1.0, float(score))) * 100.0
    names = [name_a, name_b, name_c] if is_en else [EN2CN.get(n, n) for n in (name_a, name_b, name_c)]
    molecules = "".join(
        f"<div><dt>{html.escape(_ui_text(lang, key))}</dt><dd>{html.escape(name or '—')}</dd></div>"
        for key, name in zip(("a_label", "b_label", "c_label"), names)
    )
    comparison = ""
    details = details or {}
    if details.get("score_BC") is not None and details.get("score_CB") is not None:
        comparison = (
            f"<details class='result-details'><summary>{details_label}</summary>"
            f"<div class='order-comparison'><span>A → B → C <b>{float(details['score_BC'])*100:.2f}%</b></span>"
            f"<span>A → C → B <b>{float(details['score_CB'])*100:.2f}%</b></span></div>"
            f"<p>{selected_label}: {html.escape(str(details.get('chosen_order', '—')))}</p></details>"
        )
    cls = "result-pos" if pred else "result-neg"
    return f"""<div class="result-card {cls}">
      <div class="result-header">
        <div><div class="result-eyebrow">{html.escape(model)}</div><div class="result-title">{title}</div></div>
        <div class="result-probability"><strong>{pct:.2f}%</strong><span>{probability_label}</span></div>
      </div>
      <div class="result-bar"><span style="width:{pct:.2f}%"></span></div>
      <div class="result-sub">{threshold_label}: {thr*100:.0f}%</div>
      <dl class="result-molecules">{molecules}</dl>
      {comparison}
    </div>"""


def ui_result_language(result: Optional[Dict[str, Any]], lang: str):
    if not result:
        return ""
    if result.get("error"):
        message = "Prediction unavailable. Check your inputs and model configuration." if _ui_lang(lang) == "en" else "无法完成预测，请检查输入与模型设置。"
        if result.get("missing"):
            message = ("Cannot resolve: " if _ui_lang(lang) == "en" else "无法解析：") + ", ".join(result["missing"])
        return f"<div class='err'>{html.escape(message)}</div>"
    details = result.get("extra") or {}
    return _result_html(
        bool(result["pred"]), "RandomForest" if result["model"] == "rf" else "SolutionNet",
        float(result["score"]), float(details.get("threshold", result.get("threshold", 0.5))),
        result.get("name_a", ""), result.get("name_b", ""), result.get("name_c", ""), lang=lang, details=details,
    )


def ui_predict(
    model_choice: str,
    a_in: str,
    b_in: str,
    c_in: str,
    inter_df: pd.DataFrame,
    extra_json: str,
    solutionnet_threshold: float,
    symmetric_bc: bool,
    lang: str = "en",
):
    def text(zh: str, en: str) -> str:
        return en if _ui_lang(lang) == "en" else zh

    name_a, a = resolve_input(a_in)
    name_b, b = resolve_input(b_in)
    name_c, c = resolve_input(c_in)

    if not (a and b and c):
        missing = []
        if not a:
            missing.append("A")
        if not b:
            missing.append("B")
        if not c:
            missing.append("C")
        tips = "；".join([
            "A/B/C 支持输入中文别名、英文名（需存在于 data/smiles.xlsx 或 ALIASES）",
            "或直接粘贴 SMILES（推荐）",
            "溶剂建议从 Step 2 直接点击表格选择",
        ])
        if _ui_lang(lang) == "en":
            tips = "Enter a known molecule name or a valid SMILES string. You can select solvents in step 2."
        error_title = text("无法解析：", "Cannot resolve: ")
        return (
            f"<div class='err'><b>{error_title}</b>{', '.join(missing)}<br/><span class='small'>{tips}</span></div>",
            {"error": "unresolved", "missing": missing, "name_a": name_a, "name_b": name_b, "name_c": name_c},
        )

    # RandomForest
    if "RandomForest" in (model_choice or ""):
        try:
            score, pred = rf.predict(a, b, c)
            html_out = _result_html(bool(pred), "RandomForest", float(score), 0.50, name_a, name_b, name_c, lang=lang)
            return html_out, {
                "model": "rf",
                "score": float(score),
                "threshold": 0.5,
                "pred": bool(pred),
                "name_a": name_a, "name_b": name_b, "name_c": name_c,
                "smiles_a": a, "smiles_b": b, "smiles_c": c,
            }
        except Exception as e:  # noqa: BLE001
            return "<div class='err'>" + text("RF 预测失败（模型文件缺失或特征维度不匹配）。", "RandomForest prediction failed. Check the model file and feature dimensions.") + "</div>", {"model": "rf", "error": f"{type(e).__name__}: {e}"}

    # SolutionNet
    if predict_solutionnet is None:
        return (
            "<div class='err'>" + text("SolutionNet 不可用：src.inference_solutionnet 导入失败（通常是 torch_geometric/torch-scatter 版本或 GLIBC 不匹配）。", "SolutionNet is unavailable. Check the model dependencies.") + "</div>",
            {"model": "solutionnet", "error": SOLUTIONNET_IMPORT_ERROR},
        )

    feats = interactions_to_features(inter_df if inter_df is not None else make_interaction_template(default_zero=True))
    feats["threshold"] = float(solutionnet_threshold)
    feats["symmetric"] = bool(symmetric_bc)

    extra: Dict[str, Any] = {}
    extra_json = (extra_json or "").strip()
    if extra_json:
        try:
            extra = json.loads(extra_json)
            if not isinstance(extra, dict):
                return "<div class='err'>" + text("Extra JSON 必须是 object/dict。", "Extra features must be a JSON object.") + "</div>", {"error": "extra_json_not_dict"}
        except Exception as e:  # noqa: BLE001
            return "<div class='err'>" + text("Extra JSON 格式错误。", "Invalid JSON format in extra features.") + "</div>", {"error": f"invalid_json: {type(e).__name__}: {e}"}
    feats.update(extra)

    try:
        score, pred, extra_out = predict_solutionnet(a, b, c, feats)  # type: ignore[misc]
        thr_used = float(extra_out.get("threshold", float(solutionnet_threshold)))

        html_out = _result_html(bool(pred), "SolutionNet", float(score), thr_used, name_a, name_b, name_c, lang=lang, details=extra_out)
        return html_out, {
            "model": "solutionnet",
            "threshold": thr_used,
            "score": float(score),
            "pred": bool(pred),
            "extra": extra_out,
            "name_a": name_a, "name_b": name_b, "name_c": name_c,
            "smiles_a": a, "smiles_b": b, "smiles_c": c,
        }
    except Exception as e:  # noqa: BLE001
        return "<div class='err'>" + text("SolutionNet 预测失败（检查权重路径/torch_geometric/RDKit 兼容/模型结构）。", "SolutionNet prediction failed. Check the model weights and dependencies.") + "</div>", {
            "model": "solutionnet",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=50),
        }


# ==============================
# Settings: SMILES tools
# ==============================
def ui_tool_parse_text(x: str):
    x = (x or "").strip()
    if not x:
        return "", "", "<div class='err'>请输入名称或 SMILES。</div>"
    name, smi = resolve_input(x)
    if not smi:
        return "", "", "<div class='err'>无法解析：请输入库中名称或合法 SMILES。</div>"
    cs = _canon_smiles(smi)
    if not cs:
        return name or "", "", "<div class='err'>SMILES 无法 canonicalize（可能不合法）。</div>"
    return name or "", cs, "<div class='ok'>解析成功：已得到 canonical SMILES。</div>"


def ui_tool_fill_target(target: str, smiles: str, a_in: str, b_in: str, c_in: str, lang: str = "zh"):
    smiles = (smiles or "").strip()
    cs = _canon_smiles(smiles)
    if not cs:
        summary = ui_resolve_table_only(a_in, b_in, c_in, lang)
        return (
            a_in, b_in, c_in,
            summary,
            "<div class='muted'>Unchanged</div>", "<div class='muted'>Unchanged</div>", "<div class='muted'>Unchanged</div>",
            "<div class='err'>Fill 失败：请先得到合法 SMILES。</div>",
        )

    a, b, c = a_in, b_in, c_in
    if target == "A":
        a = cs
    elif target == "B":
        b = cs
    elif target == "C":
        c = cs

    summary, view_a, view_b, view_c = ui_refresh_summary(a, b, c, lang)
    return a, b, c, summary, view_a, view_b, view_c, "<div class='ok'>已填入并刷新预览。</div>"


# ==============================
# A-only Screening: matrix + ranked lists + in-place export links
# ==============================
def _score_to_color(score: float) -> str:
    """Map score in [0,1] to a pleasant red->green background."""
    s = float(max(0.0, min(1.0, score)))
    hue = int(120 * s)  # 0=red, 120=green
    return f"hsl({hue}, 75%, 86%)"


def _build_matrix_html(solvents: List[str], grid: List[List[Optional[Dict[str, Any]]]]) -> str:
    """grid[i][j] is either None (diag / missing) or dict(score=..., pred=bool)"""
    esc = html.escape

    head_cells = ["<th class='corner'>Solvent</th>"]
    for s in solvents:
        head_cells.append(f"<th class='sticky'>{esc(s)}</th>")
    head_row = "<tr>" + "".join(head_cells) + "</tr>"

    rows_html: List[str] = [head_row]
    for i, si in enumerate(solvents):
        row_cells = [f"<th class='leftsticky'>{esc(si)}</th>"]
        for j, sj in enumerate(solvents):
            if i == j:
                row_cells.append("<td class='diag'>—</td>")
                continue
            cell = grid[i][j]
            if not cell:
                row_cells.append("<td class='na'>NA</td>")
                continue
            score = float(cell["score"])
            pred = bool(cell["pred"])
            bg = _score_to_color(score)
            icon = "✓" if pred else "×"
            row_cells.append(
                f"<td class='cell' style='background:{bg}' title='{esc(si)} + {esc(sj)} | score={score:.4f} | pred={pred}'>"
                f"{icon}<br/><span class='cell-score'>{score:.2f}</span>"
                "</td>"
            )
        rows_html.append("<tr>" + "".join(row_cells) + "</tr>")

    table = "<table class='matrix-table'>" + "".join(rows_html) + "</table>"
    return f"<div class='matrix-wrap'>{table}</div>"


def _solvent_universe(max_solvents: int = 60) -> List[str]:
    """Pick a solvent set from the library (most frequent solvents)."""
    df = store.solvent_pairs_df()
    if df is None or df.empty:
        return []
    s = pd.concat([df["solvent1"], df["solvent2"]], ignore_index=True)
    vc = s.value_counts()
    solvents = vc.index.tolist()
    return solvents[: int(max_solvents)]


def _grid_to_long_df(solvents: List[str], grid: List[List[Optional[Dict[str, Any]]]]) -> pd.DataFrame:
    rows = []
    n = len(solvents)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cell = grid[i][j]
            if not cell:
                continue
            rows.append({
                "solvent1": solvents[i],
                "solvent2": solvents[j],
                "score": float(cell["score"]),
                "pred": int(bool(cell["pred"])),
            })
    return pd.DataFrame(rows)


def _matrix_png_bytes(solvents: List[str], grid: List[List[Optional[Dict[str, Any]]]], title: str = "") -> bytes:
    """Generate a heatmap-like PNG in-memory and return bytes."""
    n = len(solvents)
    M = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cell = grid[i][j]
            if cell:
                M[i, j] = float(cell["score"])

    # Keep a reasonable size to avoid huge base64 payload
    fig = plt.figure(figsize=(max(6, n * 0.18), max(5, n * 0.18)), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="RdYlGn")

    ax.set_title(title or "A-only Screening Matrix", fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(solvents, rotation=90, fontsize=6)
    ax.set_yticklabels(solvents, fontsize=6)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Score", fontsize=8)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _screening_excel_bytes(
    a_name: str,
    a_smiles: str,
    meta: Dict[str, Any],
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    matrix_long_df: pd.DataFrame,
) -> bytes:
    """Generate XLSX in-memory and return bytes (no /file route needed)."""
    info_df = pd.DataFrame([{
        "A_name": a_name,
        "A_smiles": a_smiles,
        "model": meta.get("model"),
        "pairs_scored": meta.get("pairs_scored"),
        "positive": meta.get("positive"),
        "negative": meta.get("negative"),
        "only_reported_pairs": meta.get("only_reported_pairs"),
        "threshold_solutionnet": meta.get("threshold_solutionnet"),
        "symmetric_solutionnet": meta.get("symmetric_solutionnet"),
        "seconds": meta.get("seconds"),
    }])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        info_df.to_excel(writer, index=False, sheet_name="meta")
        (pos_df if pos_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="positive_pairs")
        (neg_df if neg_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="negative_pairs")
        (matrix_long_df if matrix_long_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="matrix_long")
    return buf.getvalue()


def _make_datauri_download_link(filename: str, mime: str, data_b64: str, label: str) -> str:
    """Return an <a download> link that downloads without calling Gradio /file (avoids nginx 404)."""
    safe_label = html.escape(label)
    safe_filename = html.escape(filename)
    return (
        f"<a class='dlbtn' download='{safe_filename}' "
        f"href='data:{mime};base64,{data_b64}'>{safe_label}</a>"
    )


def ui_a_only_screening(
    a_in: str,
    model_choice: str,
    solutionnet_threshold: float,
    solutionnet_symmetric: bool,
    max_solvents: int,
    only_reported_pairs: bool,
):
    """
    Return:
      - matrix_html
      - matrix_export_html (PNG link)
      - table_export_html (Excel link)
      - pos_df
      - neg_df
      - meta_json
    """
    t0 = time.time()

    name_a, a_smi = resolve_input(a_in)
    if not a_smi:
        return (
            "<div class='err'><b>A 无法解析</b>：请输入库中名称或合法 SMILES。</div>",
            "",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            {"error": "A unresolved"},
        )

    solvents = _solvent_universe(max_solvents=max_solvents)
    if len(solvents) < 2:
        return (
            "<div class='err'>溶剂库不足（<2 个溶剂）。</div>",
            "",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            {"error": "not enough solvents"},
        )

    smap = store.smiles_map()

    n = len(solvents)
    grid: List[List[Optional[Dict[str, Any]]]] = [[None for _ in range(n)] for _ in range(n)]

    pos_rows: List[Dict[str, Any]] = []
    neg_rows: List[Dict[str, Any]] = []

    # Pre-resolve solvent smiles
    solvent_smiles: Dict[str, str] = {}
    for s in solvents:
        _, smi = resolve_input(s)
        if not smi:
            smi = smap.get(s, "")
        solvent_smiles[s] = _canon_smiles(smi) if smi else ""

    if "SolutionNet" in (model_choice or "") and predict_solutionnet is None:
        return (
            "<div class='err'>SolutionNet 不可用：src.inference_solutionnet 导入失败。</div>",
            "",
            "",
            pd.DataFrame(),
            pd.DataFrame(),
            {"error": SOLUTIONNET_IMPORT_ERROR},
        )

    # Compute only i<j, then mirror
    for i in range(n):
        for j in range(i + 1, n):
            b_name = solvents[i]
            c_name = solvents[j]

            if bool(only_reported_pairs) and (not _is_reported_pair(b_name, c_name)):
                continue

            b_smi = solvent_smiles.get(b_name, "")
            c_smi = solvent_smiles.get(c_name, "")
            if not b_smi or not c_smi:
                continue

            try:
                if "RandomForest" in (model_choice or ""):
                    score, pred = rf.predict(a_smi, b_smi, c_smi)
                    score_f = float(score)
                    pred_b = bool(pred)
                else:
                    feats = build_solutionnet_features_from_db(
                        name_a=name_a,
                        name_b=b_name,
                        name_c=c_name,
                        threshold=float(solutionnet_threshold),
                        symmetric=bool(solutionnet_symmetric),
                    )
                    score, pred, _extra = predict_solutionnet(a_smi, b_smi, c_smi, feats)  # type: ignore[misc]
                    score_f = float(score)
                    pred_b = bool(pred)
            except Exception:
                continue

            cell = {"score": score_f, "pred": pred_b}
            grid[i][j] = cell
            grid[j][i] = cell

            row = {
                "solvent1": b_name,
                "solvent2": c_name,
                "pair_key": f"{b_name}||{c_name}",
                "score": score_f,
                "pred": int(pred_b),
            }
            if pred_b:
                pos_rows.append(row)
            else:
                neg_rows.append(row)

    pos_df = pd.DataFrame(pos_rows).sort_values("score", ascending=False).reset_index(drop=True) if pos_rows else pd.DataFrame(columns=["solvent1", "solvent2", "pair_key", "score", "pred"])
    neg_df = pd.DataFrame(neg_rows).sort_values("score", ascending=False).reset_index(drop=True) if neg_rows else pd.DataFrame(columns=["solvent1", "solvent2", "pair_key", "score", "pred"])

    matrix_html = _build_matrix_html(solvents, grid)
    matrix_long_df = _grid_to_long_df(solvents, grid)

    meta = {
        "model": "rf" if "RandomForest" in (model_choice or "") else "solutionnet",
        "name_a": name_a,
        "solvents": int(len(solvents)),
        "pairs_scored": int(len(pos_rows) + len(neg_rows)),
        "positive": int(len(pos_rows)),
        "negative": int(len(neg_rows)),
        "threshold_solutionnet": float(solutionnet_threshold),
        "symmetric_solutionnet": bool(solutionnet_symmetric),
        "only_reported_pairs": bool(only_reported_pairs),
        "reported_pairs_in_db": int(len(_reported_pair_set())),
        "seconds": round(time.time() - t0, 3),
        "solutionnet_import_error": SOLUTIONNET_IMPORT_ERROR,
    }

    # Build download links (data-uri)
    ts = int(time.time())
    title = f"A={name_a or 'A'} ({meta.get('model','')})"
    png_bytes = _matrix_png_bytes(solvents, grid, title=title)
    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    png_name = f"a_only_matrix_{ts}.png"
    png_link = _make_datauri_download_link(png_name, "image/png", png_b64, "导出 PNG")

    xlsx_bytes = _screening_excel_bytes(name_a, a_smi, meta, pos_df, neg_df, matrix_long_df)
    xlsx_b64 = base64.b64encode(xlsx_bytes).decode("ascii")
    xlsx_name = f"a_only_screening_{ts}.xlsx"
    xlsx_link = _make_datauri_download_link(
        xlsx_name,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        xlsx_b64,
        "导出 Excel",
    )

    matrix_export_html = f"<div class='dlrow'><span class='dlhint'>二维图：</span>{png_link}</div>"
    table_export_html = f"<div class='dlrow'><span class='dlhint'>表格：</span>{xlsx_link}</div>"

    return matrix_html, matrix_export_html, table_export_html, pos_df, neg_df, meta


# ==============================
# UI Styling
# ==============================
CSS = """
html, body { min-height: 100%; }
body { margin: 0; background: #f6f7fc; }
footer { display: none !important; }

.gradio-container {
  max-width: 1600px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 28px 32px 48px !important;
  color: #253858;
}
:root {
  --card-bg: #ffffff;
  --card-border: #e5e8f2;
  --text: #253858;
  --muted: #738099;
  --primary: #635bff;
  --ok: #47886a;
  --warn: #f59e0b;
  --danger: #a35268;
}
.h1 { font-size: 32px; font-weight: 700; margin: 0 0 6px; color: var(--text); letter-spacing: -0.6px; }
.sub { font-size: 14px; margin: 0; color: var(--muted); }
#page-header { align-items: center; margin-bottom: 26px; }
#platform-title { border: 0; padding: 0; background: transparent; }
#workspace-layout { gap: 24px; align-items: flex-start; }
#workspace-sidebar { flex: 0 0 300px !important; gap: 20px; }
#workspace-main { min-width: 0 !important; flex: 1 1 0 !important; }

.card, .sidebar-panel {
  background: #ffffff !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 16px !important;
  padding: 22px !important;
  box-shadow: 0 4px 18px rgba(35,49,91,0.035) !important;
}
.card > .styler, .card > .form, .card .form {
  border: 0 !important;
  border-radius: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
.card .block:not(.gradio-button) {
  border: 0;
  padding: 0;
  background: transparent;
  box-shadow: none;
  box-sizing: border-box;
}
 .card > .styler {
  display: flex;
  flex-direction: column;
  gap: 16px !important;
  --layout-gap: 16px !important;
  --form-gap-width: 16px !important;
}
.card .form { gap: 16px !important; }
.card [data-testid="block-info"] {
  color: #253858 !important;
  background: transparent !important;
  font-weight: 700 !important;
}
.molecule-inputs > .form { gap: 16px !important; }
.molecule-inputs > .form > .block { flex: 1 1 0; }
#model-selector .wrap { display: grid; grid-template-columns: 1fr; gap: 8px; }
#model-selector .wrap label {
  background: #fafbfe !important;
  border: 1px solid #e1e5f0 !important;
  border-radius: 9px !important;
  padding: 10px 12px;
  box-shadow: none !important;
}
#model-selector .wrap label:has(input:checked) { background: #eeedff !important; border-color: #c6c1ff !important; }
#model-selector .wrap label span { margin: 0; color: #4f5582 !important; }
.molecule-previews .label-container { position: static; margin-bottom: 8px; display: block; }
.molecule-previews [data-testid="block-label"] {
  position: static;
  padding: 0;
  border: 0;
  background: transparent;
  font-size: 13px;
  color: #62708a;
}
.molecule-previews [data-testid="block-label"] span { display: none; }
.molecule-previews iframe { border: 0 !important; border-radius: 8px !important; }

.card label > span, .card .block-label {
  background: transparent !important;
  color: #62708a !important;
  font-size: 13px !important;
  font-weight: 700 !important;
  padding: 0 !important;
  border-radius: 0 !important;
  margin-bottom: 8px;
}
.card textarea, .card input[type="text"], .card input[type="number"], .card .wrap.inner {
  border: 1px solid #e1e5f0 !important;
  border-radius: 10px !important;
  background: #fafbfe !important;
  box-shadow: none !important;
  box-sizing: border-box;
}
.card textarea { font-size: 14px; line-height: 1.6; }
.card textarea:focus, .card input:focus { border-color: #8c85ff !important; outline-color: #8c85ff; }
button.primary, button.secondary {
  min-height: 44px;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  line-height: 1.5;
  transition: background 150ms, box-shadow 150ms;
}
button.primary { background: #635bff !important; border: 1px solid #635bff !important; color: #ffffff !important; }
button.primary:hover { background: #5146df !important; box-shadow: 0 4px 12px rgba(99,91,255,0.18); }
button.secondary { background: #f2f3fd !important; border: 1px solid #e5e7f5 !important; color: #4f5582 !important; }
button:focus-visible, #language-toggle label:focus-within { outline: 2px solid #8c85ff; outline-offset: 3px; }
.sidebar-panel button.primary, .sidebar-panel button.secondary { width: 100%; }
#smiles-input textarea { font-family: "IBM Plex Mono", Consolas, monospace; }
.molecule-inputs, .molecule-previews { gap: 16px !important; }
.molecule-previews > .block { border: 1px solid #e5e8f2 !important; border-radius: 12px; padding: 12px !important; background: #fafbfe !important; }
.filter-row { align-items: flex-end; gap: 12px !important; }

#main-tabs { border: 0; padding: 0; }
#main-tabs > .tab-nav {
  display: flex;
  width: 100%;
  gap: 6px;
  padding: 6px;
  margin-bottom: 20px;
  box-sizing: border-box;
  background: #ffffff;
  border: 1px solid #e5e8f2;
  border-radius: 14px;
}
#main-tabs > .tab-nav > button {
  flex: 1 1 0;
  min-width: 0;
  margin: 0;
  padding: 12px 10px;
  font-size: 16px !important;
  font-weight: 600 !important;
  line-height: 1.5;
  white-space: normal;
  color: #738099 !important;
  background: transparent !important;
  border: 0 !important;
  border-radius: 9px;
}
#main-tabs > .tab-nav > button:hover { background: #f5f4ff !important; color: #635bff !important; }
#main-tabs > .tab-nav > button.selected,
#main-tabs > .tab-nav > button[aria-selected="true"] {
  background: #eeedff !important;
  color: #5146df !important;
  box-shadow: none;
}
#main-tabs > .tabitem { padding: 0 !important; border: 0 !important; background: transparent !important; }
#language-selector { margin-left: auto; max-width: 120px; min-width: 110px !important; }
#language-toggle { background: transparent; border: none; padding: 0; }
#language-selector .form, #language-toggle .wrap { gap: 2px; padding: 0; border: 0; background: transparent !important; box-shadow: none; }
#language-toggle label { border: 0; border-radius: 7px; padding: 6px 12px; cursor: pointer; background: transparent; color: #738099; }
#language-toggle input { position: absolute; opacity: 0; width: 0; }
#language-toggle label:has(input:checked) { background: transparent; color: #5146df; font-weight: 700; box-shadow: inset 0 -2px 0 #635bff; }
#solutionnet-threshold { width: 100%; box-sizing: border-box; }
#solutionnet-threshold .wrap { flex-wrap: wrap; justify-content: flex-start; gap: 8px; }
#solutionnet-threshold .wrap > label { flex-basis: 100%; }
#solutionnet-threshold .head { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; width: 100%; }
#solutionnet-threshold input[type="number"] { margin-left: 0 !important; text-align: left; width: 100%; padding: 10px 12px; }
#solutionnet-threshold input[type="range"] { margin-top: 12px; }

@media (max-width: 900px) {
  .gradio-container { padding: 20px 16px 32px !important; }
  #workspace-sidebar { flex: 1 1 100% !important; }
  #workspace-main { flex: 1 1 100% !important; }
  .card, .sidebar-panel { padding: 18px !important; }
  #main-tabs > .tab-nav > button { font-size: 14px !important; padding: 10px 6px; }
  .h1 { font-size: 26px; }
}

.empty-state { border: 1px dashed #d8dced; border-radius: 16px; padding: 44px 24px; text-align: center; color: #738099; }
.empty-state strong { display: block; font-size: 16px; font-weight: 600; color: #4f5582; margin: 12px 0 8px; }
.empty-state p { margin: 0; font-size: 14px; }
.empty-icon { display: inline-flex; align-items: center; justify-content: center; width: 42px; height: 42px; border-radius: 12px; background: #eeedff; color: #635bff; font-size: 24px; }
#pair-chooser {
  border: 1px solid #e5e8f2 !important;
  border-radius: 16px !important;
  background: #ffffff !important;
  box-shadow: 0 4px 18px rgba(35,49,91,0.035);
  overflow: hidden;
}
#pair-chooser > .label-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 22px;
  text-align: left;
  color: #253858;
  font-size: 16px;
  font-weight: 700;
}
#pair-chooser > .label-wrap::before {
  content: "2";
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: #eeedff;
  color: #5146df;
  font-size: 14px;
  font-weight: 600;
}
#pair-chooser > .label-wrap > span:first-child { flex: 1; font-size: 16px; font-weight: 700; color: #253858; }
#pair-chooser .card { border: 0 !important; border-radius: 0 !important; box-shadow: none !important; padding-top: 0 !important; }
#resolved-summary .summary-heading { font-size: 13px; font-weight: 700; color: #253858; margin-bottom: 8px; }
#resolved-summary .smiles-output-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
#resolved-summary .smiles-output { display: flex; flex-direction: column; min-width: 0; }
#resolved-summary .smiles-output > span { font-size: 13px; font-weight: 700; color: #253858; margin-bottom: 8px; }
#resolved-summary .smiles-output textarea {
  display: block;
  width: 100%;
  min-width: 0;
  min-height: 66px;
  margin: 0;
  padding: 20px 12px;
  text-align: center;
  font-family: "IBM Plex Mono", Consolas, monospace;
  font-size: 13px;
  line-height: 1.6;
  resize: vertical;
  color: #253858;
}
@media (max-width: 600px) {
  #resolved-summary .smiles-output-grid { grid-template-columns: 1fr; }
}

.card .label-wrap > p { font-weight: 700 !important; color: #253858 !important; }

.small { font-size: 13px; color: var(--muted); line-height: 1.35; }

.step-title{
  display:flex; align-items:center; gap:10px;
  margin:0 0 10px 0;
}
.step-badge{
  width:28px;height:28px;border-radius:999px;
  display:inline-flex;align-items:center;justify-content:center;
  background:#eeedff;color:#5146df;font-weight:600;
}
.step-text{font-size:16px;font-weight:700;color:var(--text);}

.muted{color:var(--muted);}

.ok{
  background: #eaf5ee;
  border: 1px solid #91cba9;
  color: #14532d;
  padding: 10px 12px;
  border-radius: 12px;
}
.err{
  background: #faeef1;
  border: 1px solid #e8b2be;
  color: #7f1d1d;
  padding: 10px 12px;
  border-radius: 12px;
}
.warn{
  border-left:4px solid var(--warn);
  background:#fffbeb;
  color:#92400e;
  padding:10px 12px;
  border-radius:12px;
}

/* Make dataframes scrollable (avoid Dataframe(height=...) which may be unsupported) */
.gr-dataframe { max-height: 420px; overflow: auto; }
.gr-dataframe table { font-size: 14px; }
.gr-dataframe thead th { background: #f1f5f9 !important; }

/* Prediction result */
.result-card { border: 1px solid #e1e5f0; border-radius: 14px; padding: 24px; background: #ffffff; }
 .result-pos {
  --result-accent: #23734e;
  --result-track: #cee9d9;
  --result-border: #b5d9c4;
  --result-muted: #4b715d;
  background: #eaf6ee;
  border-color: var(--result-border);
}
.result-neg {
  --result-accent: #a23f57;
  --result-track: #f1d4dc;
  --result-border: #e4b9c4;
  --result-muted: #865c68;
  background: #fceef1;
  border-color: var(--result-border);
}
.result-header { display: flex; align-items: center; justify-content: space-between; gap: 20px; }
.result-eyebrow { font-size: 13px; color: #738099; margin-bottom: 6px; }
.result-title { font-size: 22px; font-weight: 700; color: var(--result-accent); }
.result-probability { display: flex; flex-direction: column; text-align: right; }
.result-probability strong { font-size: 32px; font-weight: 700; color: var(--result-accent); line-height: 1.2; }
.result-probability span { font-size: 12px; color: #738099; margin-top: 4px; }
.result-bar { height: 8px; margin: 20px 0 8px; background: var(--result-track); border-radius: 8px; overflow: hidden; }
.result-bar > span { display: block; height: 100%; background: var(--result-accent); border-radius: inherit; }
.result-sub { font-size: 13px; color: #738099; }
.result-molecules { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 16px; margin: 20px 0 0 !important; padding-top: 16px; border-top: 1px solid #e5e8f2; }
.result-molecules dt { font-size: 12px; font-weight: 600; color: #738099; }
.result-molecules dd { font-size: 14px; color: #253858; margin: 6px 0 0; overflow-wrap: anywhere; }
.result-details { margin-top: 20px; padding-top: 16px; border-top: 1px solid #e5e8f2; font-size: 13px; color: #62708a; }
.result-details summary { font-weight: 600; cursor: pointer; }
.result-card .result-eyebrow, .result-card .result-sub,
.result-card .result-probability span, .result-card .result-molecules dt,
.result-card .result-details { color: var(--result-muted); }
.result-card .result-molecules, .result-card .result-details { border-color: var(--result-border); }
#settings-heading { padding-bottom: 14px !important; border-bottom: 1px solid #e5e8f2 !important; }
#settings-heading .h1 { margin-bottom: 0; }

.order-comparison { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; }
.order-comparison b { color: #253858; margin-left: 8px; }
@media (max-width: 600px) {
  .result-card { padding: 18px; }
  .result-header { align-items: flex-start; flex-direction: column; }
  .result-probability { text-align: left; }
  .result-molecules { grid-template-columns: 1fr; }
}

/* Matrix table */
.matrix-wrap{
  max-height: 640px;
  overflow: auto;
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  background: #ffffff;
}
.matrix-table{
  border-collapse: collapse;
  font-size: 13px;
  width: max-content;
}
.matrix-table th, .matrix-table td{
  border: 1px solid #e5e7eb;
  padding: 6px 8px;
  text-align: center;
  white-space: nowrap;
}
.matrix-table th.sticky{
  position: sticky;
  top: 0;
  background: #f1f5f9;
  z-index: 3;
}
.matrix-table th.leftsticky{
  position: sticky;
  left: 0;
  background: #f8fafc;
  z-index: 2;
  text-align: left;
}
.matrix-table th.corner{
  position: sticky;
  top: 0;
  left: 0;
  background: #f1f5f9;
  z-index: 4;
  text-align: left;
}
.matrix-table td.diag{ background:#f8fafc; color:#94a3b8; font-weight:700; }
.matrix-table td.na{ background:#ffffff; color:#cbd5e1; }
.matrix-table td.cell{ font-weight:600; }
.matrix-table .cell-score{ font-weight:700; color:#0f172a; opacity:0.9; font-size:11px; }

/* In-place download links */
.dlrow{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin: 6px 0 10px 0; }
.dlhint{ color: var(--muted); font-size: 13px; }
.dlbtn{
  display:inline-block;
  padding: 8px 12px;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
  color: #0f172a;
  font-weight: 800;
  text-decoration: none;
}
.dlbtn:hover{ background:#f1f5f9; }
"""


# ==============================
# Build UI
# ==============================
def build_demo(show_advanced: bool = False) -> gr.Blocks:
    language_bindings: List[Tuple[Any, Dict[str, Any]]] = []

    def bind(component: Any, **updates: Any) -> Any:
        language_bindings.append((component, updates))
        return component

    def apply_language(lang: str, a: str, b: str, c: str, screening_a: str, selected_tab: str):
        def display_name(value: str) -> str:
            value = value or ""
            is_name = value in ALIASES or any(value.casefold() == name.casefold() for name in store.smiles_map())
            if not is_name:
                return value
            name, smiles = resolve_input(value)
            if not smiles or not name:
                return value
            return name if _ui_lang(lang) == "en" else EN2CN.get(name, name)

        localized_inputs = [display_name(value) for value in (a, b, c, screening_a)]
        localized_components = dict(zip((a_in, b_in, c_in, a2), localized_inputs))
        out = []
        for _component, updates in language_bindings:
            kwargs: Dict[str, Any] = {}
            for attr, value in updates.items():
                kwargs[attr] = value(lang) if callable(value) else _ui_text(lang, str(value))
            if _component in localized_components:
                kwargs["value"] = localized_components[_component]
            out.append(gr.update(**kwargs))
        out.append(gr.update(label=_ui_text(lang, "summary_label"), value=ui_resolve_table_only(*localized_inputs[:3], lang=lang)))
        out.append(gr.update(selected=selected_tab))
        return tuple(out)

    theme = gr.themes.Soft(
        primary_hue="indigo",
        font=[gr.themes.GoogleFont("Noto Sans SC"), "PingFang SC", "Microsoft YaHei", "sans-serif"],
    ).set(
        body_background_fill="#f6f7fc",
        body_text_color="#253858",
        body_text_size="14px",
        block_padding="0px",
        block_border_width="0px",
        block_shadow="none",
        input_shadow="none",
        input_radius="10px",
        button_large_radius="10px",
        button_small_radius="8px",
        button_large_text_size="14px",
        button_large_text_weight="600",
        block_background_fill="#ffffff",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f6f7fc",
        border_color_primary="#e1e5f0",
        block_label_background_fill="transparent",
        block_label_text_color="#738099",
        block_label_text_size="13px",
        block_label_text_weight="700",
        input_background_fill="#fafbfe",
        input_border_color="#e1e5f0",
        button_primary_background_fill="#635bff",
        button_primary_background_fill_hover="#5146df",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#635bff",
        button_secondary_background_fill="#f0f3ff",
        button_secondary_text_color="#253858",
    )
    with gr.Blocks(title=APP_TITLE, css=CSS, fill_height=True, theme=theme) as demo:
        active_tab = gr.State("prediction")
        with gr.Row(elem_id="page-header"):
            with gr.Column(scale=4, min_width=320):
                main_title_md = gr.Markdown(_ui_text("en", "main_title"), elem_id="platform-title")
                bind(main_title_md, value="main_title")
            with gr.Column(scale=0, min_width=110, elem_id="language-selector"):
                language = gr.Radio(
                    ["中", "En"],
                    value="En",
                    label="Language / 语言",
                    show_label=False,
                    elem_id="language-toggle",
                    interactive=True,
                )

        with gr.Row(equal_height=False, elem_id="workspace-layout"):
            # ---- Sidebar (SMILES tools + Settings)
            with gr.Column(scale=1, min_width=280, elem_id="workspace-sidebar"):
                with gr.Group(elem_classes=["card", "sidebar-panel"]):
                    tools_title = gr.Markdown(_ui_text("en", "tools_title"))
                    bind(tools_title, value="tools_title")

                    tool_target = gr.Dropdown(["A", "B", "C"], value="A", label=_ui_text("en", "tool_target"))
                    bind(tool_target, label="tool_target")

                    tool_text = gr.Textbox(lines=2, label=_ui_text("en", "tool_text"), placeholder=_ui_text("en", "tool_text_placeholder"))
                    bind(tool_text, label="tool_text", placeholder="tool_text_placeholder")

                    tool_parse_btn = gr.Button(_ui_text("en", "tool_parse"), variant="secondary")
                    bind(tool_parse_btn, value="tool_parse")

                    tool_name = gr.Textbox(label=_ui_text("en", "tool_name"), interactive=False)
                    bind(tool_name, label="tool_name")

                    tool_smiles = gr.Textbox(label=_ui_text("en", "tool_smiles"), interactive=True, elem_id="smiles-input")
                    bind(tool_smiles, label="tool_smiles")
                    tool_msg = gr.HTML()

                    tool_fill_btn = gr.Button(_ui_text("en", "tool_fill"), variant="primary")
                    bind(tool_fill_btn, value="tool_fill")
                    tool_fill_msg = gr.HTML()

                with gr.Group(elem_classes=["card", "sidebar-panel"]):
                    settings_title = gr.Markdown(_ui_text("en", "settings_title"), elem_id="settings-heading")
                    bind(settings_title, value="settings_title")

                    model_choice = gr.Radio(
                        ["SolutionNet", "RandomForest"],
                        value="SolutionNet",
                        label=_ui_text("en", "model_label"),
                        elem_id="model-selector",
                    )
                    bind(model_choice, label="model_label")

                    solutionnet_threshold = gr.Slider(0, 1, value=0.5, step=0.01, label=_ui_text("en", "threshold_label"), elem_id="solutionnet-threshold", info=_ui_text("en", "threshold_info"))
                    bind(solutionnet_threshold, label="threshold_label", info="threshold_info")

                    solutionnet_symmetric = gr.Checkbox(value=True, label=_ui_text("en", "symmetric_label"), info=_ui_text("en", "symmetric_info"))
                    bind(solutionnet_symmetric, label="symmetric_label", info="symmetric_info")

            # ---- Main
            with gr.Column(scale=4, min_width=480, elem_id="workspace-main"):
                with gr.Tabs(elem_id="main-tabs", selected="prediction") as main_tabs:
                    # =========================
                    # Prediction (ABC)
                    # =========================
                    with gr.TabItem(_ui_text("en", "tab_prediction"), id="prediction") as prediction_tab:
                        bind(prediction_tab, label="tab_prediction")
                        with gr.Group(elem_classes=["card"]):
                            step1_md = gr.Markdown(_ui_text("en", "step1"))
                            bind(step1_md, value="step1")

                            with gr.Row(elem_classes=["molecule-inputs"]):
                                a_in = gr.Textbox(label=_ui_text("en", "a_label"), value="2,5-Furandicarboxylic acid")
                                bind(a_in, label="a_label")

                                b_in = gr.Textbox(label=_ui_text("en", "b_label"), value="Tetrahydrofuran")
                                bind(b_in, label="b_label")

                                c_in = gr.Textbox(label=_ui_text("en", "c_label"), value="Water")
                                bind(c_in, label="c_label")

                            summary_table = gr.HTML(
                                label=_ui_text("en", "summary_label"),
                                elem_id="resolved-summary",
                                show_label=False,
                                value=ui_resolve_table_only("", "", "", "en"),
                            )

                            with gr.Row(elem_classes=["molecule-previews"]):
                                view_a = gr.HTML(label=_ui_text("en", "view_a"), show_label=False)
                                bind(view_a, label="view_a")

                                view_b = gr.HTML(label=_ui_text("en", "view_b"), show_label=False)
                                bind(view_b, label="view_b")

                                view_c = gr.HTML(label=_ui_text("en", "view_c"), show_label=False)
                                bind(view_c, label="view_c")

                            refresh_btn = gr.Button(_ui_text("en", "refresh"), variant="secondary")
                            bind(refresh_btn, value="refresh")

                            a_in.change(ui_resolve_table_only, [a_in, b_in, c_in, language], [summary_table])
                            b_in.change(ui_resolve_table_only, [a_in, b_in, c_in, language], [summary_table])
                            c_in.change(ui_resolve_table_only, [a_in, b_in, c_in, language], [summary_table])

                            refresh_btn.click(ui_refresh_summary, [a_in, b_in, c_in, language], [summary_table, view_a, view_b, view_c])
                            demo.load(ui_refresh_summary, [a_in, b_in, c_in, language], [summary_table, view_a, view_b, view_c])

                        # Step 2 (Accordion, click-to-select)
                        with ACCORDION(_ui_text("en", "pair_chooser_title"), open_=True, elem_id="pair-chooser") as step2_acc:
                            if hasattr(gr, "Accordion"):
                                bind(step2_acc, label="pair_chooser_title")
                            with gr.Group(elem_classes=["card"]):
                                step2_md = gr.Markdown(_ui_text("en", "step2"))
                                bind(step2_md, value="step2")

                                with gr.Row(elem_classes=["filter-row"]):
                                    pair_search = gr.Textbox(label=_ui_text("en", "pair_search"), value="")
                                    bind(pair_search, label="pair_search")

                                    pair_min_pos = gr.Slider(0, 1, value=0.0, step=0.01, label=_ui_text("en", "pair_min_pos"))
                                    bind(pair_min_pos, label="pair_min_pos")

                                    pair_min_total = gr.Slider(0, 200, value=0, step=1, label=_ui_text("en", "pair_min_total"))
                                    bind(pair_min_total, label="pair_min_total")

                                    pair_max_rows = gr.Slider(50, 5000, value=800, step=50, label=_ui_text("en", "pair_max_rows"))
                                    bind(pair_max_rows, label="pair_max_rows")

                                    pair_reload = gr.Button(_ui_text("en", "pair_reload"), variant="primary")
                                    bind(pair_reload, value="pair_reload")

                                pair_table = DF(label=_ui_text("en", "pair_table"), interactive=False)
                                bind(pair_table, label="pair_table")
                                pick_msg = gr.HTML()

                                def _load_pairs(s, p, n, m):
                                    return ui_library_filter(s, p, n, m)

                                pair_reload.click(_load_pairs, [pair_search, pair_min_pos, pair_min_total, pair_max_rows], [pair_table])
                                demo.load(_load_pairs, [pair_search, pair_min_pos, pair_min_total, pair_max_rows], [pair_table])

                                if hasattr(pair_table, "select"):
                                    pair_table.select(
                                        ui_pick_pair_from_table,
                                        inputs=[pair_table, a_in, language],
                                        outputs=[b_in, c_in, pick_msg, summary_table, view_a, view_b, view_c],
                                    )
                                else:
                                    pick_msg.value = "<div class='warn'>当前 Gradio 版本不支持 Dataframe.select()，无法点击选行。</div>"

                        b_in.change(ui_toggle_step2_open, [b_in, c_in], [step2_acc])
                        c_in.change(ui_toggle_step2_open, [b_in, c_in], [step2_acc])
                        demo.load(ui_toggle_step2_open, [b_in, c_in], [step2_acc])

                        # Step 3 (SolutionNet interactions)
                        with gr.Group(elem_classes=["card"]) as step3_group:
                            step3_md = gr.Markdown(_ui_text("en", "step3"))
                            bind(step3_md, value="step3")

                            inter_df = DF(
                                label=_ui_text("en", "interactions_label"),
                                value=make_interaction_template(default_zero=True),
                                row_count=(6, "fixed"),
                                col_count=(6, "fixed"),
                                interactive=True,
                            )
                            bind(inter_df, label="interactions_label")

                            with gr.Row():
                                reset_btn = gr.Button(_ui_text("en", "reset"), variant="secondary")
                                bind(reset_btn, value="reset")

                                update_btn = gr.Button(_ui_text("en", "update_data"), variant="primary")
                                bind(update_btn, value="update_data")

                            update_meta = gr.JSON(label=_ui_text("en", "update_meta"), visible=show_advanced)
                            bind(update_meta, label="update_meta")

                            extra_json = gr.Textbox(label=_ui_text("en", "extra_json"), lines=3, value="{}", visible=show_advanced)
                            bind(extra_json, label="extra_json")

                            reset_btn.click(ui_reset_interactions, [], [inter_df, update_meta])
                            update_btn.click(ui_update_interactions, [a_in, b_in, c_in, inter_df], [inter_df, update_meta])

                        # Step 4
                        with gr.Group(elem_classes=["card"]):
                            step4_md = gr.Markdown(_ui_text("en", "step4"))
                            bind(step4_md, value="step4")

                            predict_btn = gr.Button(_ui_text("en", "predict"), variant="primary")
                            bind(predict_btn, value="predict")
                            result_html = gr.HTML()
                            result_json = gr.JSON(label=_ui_text("en", "raw_result"), visible=show_advanced)
                            bind(result_json, label="raw_result")

                            predict_btn.click(
                                ui_predict,
                                [model_choice, a_in, b_in, c_in, inter_df, extra_json, solutionnet_threshold, solutionnet_symmetric, language],
                                [result_html, result_json],
                            )

                        def _toggle_step3(model: str):
                            return gr.update(visible=("SolutionNet" in (model or "")))

                        model_choice.change(_toggle_step3, [model_choice], [step3_group])
                        demo.load(_toggle_step3, [model_choice], [step3_group])

                    # =========================
                    # A-only Screening
                    # =========================
                    with gr.TabItem(_ui_text("en", "tab_screening"), id="screening") as screening_tab:
                        bind(screening_tab, label="tab_screening")
                        screen_intro_md = gr.Markdown(_ui_text("en", "screen_intro"))
                        bind(screen_intro_md, value="screen_intro")

                        with gr.Group(elem_classes=["card"]):
                            a2 = gr.Textbox(label=_ui_text("en", "a2_label"), value="2,5-Furandicarboxylic acid")
                            bind(a2, label="a2_label")

                            only_reported = gr.Checkbox(value=True, label=_ui_text("en", "only_reported"))
                            bind(only_reported, label="only_reported")

                            max_solvents = gr.Slider(10, 120, value=30, step=1, label=_ui_text("en", "max_solvents"), visible=show_advanced)
                            bind(max_solvents, label="max_solvents")

                            run_screen = gr.Button(_ui_text("en", "run_screen"), variant="primary")
                            bind(run_screen, value="run_screen")

                        screen_empty = gr.HTML(_ui_text("en", "screen_empty"))
                        bind(screen_empty, value="screen_empty")

                        # Matrix area (download link sits above/next to matrix)
                        with gr.Group(elem_classes=["card"], visible=False) as matrix_group:
                            matrix_export_html = gr.HTML(value="")
                            matrix_html = gr.HTML()
                            matrix_note = gr.HTML(value=_ui_text("en", "matrix_note"))
                            bind(matrix_note, value="matrix_note")

                        # Tables area (download link sits above the dataframes)
                        with gr.Group(elem_classes=["card"], visible=False) as screen_tables_group:
                            table_export_html = gr.HTML(value="")
                            pos_df = DF(label=_ui_text("en", "pos_df"), interactive=False, value=pd.DataFrame())
                            bind(pos_df, label="pos_df")

                            neg_df = DF(label=_ui_text("en", "neg_df"), interactive=False, value=pd.DataFrame())
                            bind(neg_df, label="neg_df")

                        screen_meta = gr.JSON(label=_ui_text("en", "screen_meta"), visible=show_advanced)
                        bind(screen_meta, label="screen_meta")

                        screen_event = run_screen.click(
                            ui_a_only_screening,
                            inputs=[a2, model_choice, solutionnet_threshold, solutionnet_symmetric, max_solvents, only_reported],
                            outputs=[matrix_html, matrix_export_html, table_export_html, pos_df, neg_df, screen_meta],
                        )

                        def reveal_screen_results(meta):
                            succeeded = bool(meta) and not meta.get("error")
                            return gr.update(visible=True), gr.update(visible=succeeded), gr.update(visible=False)

                        screen_event.then(reveal_screen_results, [screen_meta], [matrix_group, screen_tables_group, screen_empty])

                    # =========================
                    # Solvent Pair Library
                    # =========================
                    with gr.TabItem(_ui_text("en", "tab_library"), id="library") as library_tab:
                        bind(library_tab, label="tab_library")

                        with gr.Group(elem_classes=["card"]):
                            with gr.Row(elem_classes=["filter-row"]):
                                search = gr.Textbox(label=_ui_text("en", "search"), value="")
                                bind(search, label="search")

                                min_pos_rate = gr.Slider(0, 1, value=0.0, step=0.01, label=_ui_text("en", "pair_min_pos"))
                                bind(min_pos_rate, label="pair_min_pos")

                                min_total = gr.Slider(0, 200, value=0, step=1, label=_ui_text("en", "pair_min_total"))
                                bind(min_total, label="pair_min_total")

                                max_rows = gr.Slider(50, 5000, value=800, step=50, label=_ui_text("en", "pair_max_rows"))
                                bind(max_rows, label="pair_max_rows")

                                reload_btn = gr.Button(_ui_text("en", "reload"), variant="primary")
                                bind(reload_btn, value="reload")

                            pairs_table = DF(label=_ui_text("en", "pairs"), interactive=False)
                            bind(pairs_table, label="pairs")
                            reload_meta = gr.HTML()

                            def _reload_lib(s, p, n, m):
                                df = ui_library_filter(s, p, n, m)
                                msg = f"<div class='small'>rows={len(df)}</div>"
                                return df, msg

                            reload_btn.click(_reload_lib, [search, min_pos_rate, min_total, max_rows], [pairs_table, reload_meta])
                            demo.load(_reload_lib, [search, min_pos_rate, min_total, max_rows], [pairs_table, reload_meta])

                        with gr.Group(elem_classes=["card"]):
                            library_hint_md = gr.Markdown(_ui_text("en", "library_hint"))
                            bind(library_hint_md, value="library_hint")

                # --------- Wire sidebar SMILES tools to Prediction inputs ---------
                tool_parse_btn.click(ui_tool_parse_text, [tool_text], [tool_name, tool_smiles, tool_msg])
                tool_fill_btn.click(
                    ui_tool_fill_target,
                    inputs=[tool_target, tool_smiles, a_in, b_in, c_in, language],
                    outputs=[a_in, b_in, c_in, summary_table, view_a, view_b, view_c, tool_fill_msg],
                )

                language.change(ui_result_language, [result_json, language], [result_html])

                prediction_tab.select(lambda: "prediction", [], [active_tab], show_progress="hidden")
                screening_tab.select(lambda: "screening", [], [active_tab], show_progress="hidden")
                library_tab.select(lambda: "library", [], [active_tab], show_progress="hidden")

                language_outputs = [component for component, _updates in language_bindings] + [summary_table, main_tabs]
                language.change(apply_language, [language, a_in, b_in, c_in, a2, active_tab], language_outputs)

    return demo


def _safe_launch(demo: gr.Blocks, host: str, port: int, share: bool):
    """Call demo.launch with only supported kwargs (gradio version differences)."""
    root_path = os.environ.get("ROOT_PATH", "")  # e.g. /cosolvency for subpath reverse-proxy
    out_dir = str((ROOT / "outputs").resolve())
    kwargs = {
        "server_name": host,
        "server_port": int(port),
        "share": bool(share),
        "show_error": True,
        "quiet": False,
        "debug": False,
        "prevent_thread_lock": False,
        "root_path": root_path,
        # still keep for other potential file ops (depends on gradio version)
        "allowed_paths": [out_dir],
    }
    sig = inspect.signature(demo.launch)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}

    print(f"[launch] gradio={gr.__version__} host={host} port={port} share={share} root_path={root_path!r}", flush=True)
    print(f"[launch] allowed_paths (if supported): {out_dir}", flush=True)
    print(f"[launch] if host=0.0.0.0, open via: http://<your_public_ip>:{port}", flush=True)

    return demo.launch(**filtered)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    p.add_argument("--share", action="store_true")
    p.add_argument("--advanced_ui", action="store_true", default=bool(int(os.environ.get("ADVANCED_UI", "0"))))
    args, _unknown = p.parse_known_args()

    try:
        demo = build_demo(show_advanced=args.advanced_ui)
    except Exception as e:  # noqa: BLE001
        print("[fatal] build_demo() failed:", flush=True)
        print(f"[fatal] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise

    # best-effort queue (optional)
    try:
        if hasattr(demo, "queue"):
            demo.queue(default_concurrency_limit=4)  # type: ignore[attr-defined]
    except Exception:
        pass

    _safe_launch(demo, args.host, args.port, args.share)


if __name__ == "__main__":
    main()
