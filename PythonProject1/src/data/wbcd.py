from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np

WDBC_COLS = [
    "ID","diagnosis",
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

def _find_candidate_csv(dirpath: Path):
    candidates = [
        "wbcd/wbcd.csv", "wbcd/WBCD.csv", "wbcd/wdbc.csv", "wbcd/wdbc.data",
        "WDBC.csv", "wbcd.csv", "wdbc.data", "wdbc.csv"
    ]
    for c in candidates:
        p = dirpath / c
        if p.exists():
            return p
    for p in dirpath.rglob("*"):
        name = p.name.lower()
        if p.is_file() and ("wdbc" in name or "wbcd" in name) and (name.endswith(".csv") or name.endswith(".data")):
            return p
    return None

def get_data_root() -> Path:
    env = os.getenv("DATA_ROOT")
    if env:
        return Path(env)
    return Path.cwd() / "data"

def load_wbcd(data_root: str | os.PathLike | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    root = Path(data_root) if data_root else get_data_root()
    cand = _find_candidate_csv(root)
    if cand is None:
        raise FileNotFoundError(f"Could not locate WBCD file under {root}. Expected something like 'wbcd/wbcd.csv' or 'wbcd/wdbc.data'.")
    if cand.suffix.lower() == ".data" and "wdbc" in cand.name.lower():
        df = pd.read_csv(cand, header=None)
        if df.shape[1] == 32:
            df.columns = WDBC_COLS
        else:
            cols = ["ID","diagnosis"] + [f"f{i}" for i in range(1, df.shape[1]-1)]
            df.columns = cols
    else:
        try:
            df = pd.read_csv(cand)
        except Exception:
            df = pd.read_csv(cand, header=None)
        if "diagnosis" not in df.columns.tolist():
            if df.shape[1] >= 32:
                df.columns = (WDBC_COLS + [f"extra_{i}" for i in range(df.shape[1]-len(WDBC_COLS))])[:df.shape[1]]
            else:
                df.columns = ["ID","diagnosis"] + [f"f{i}" for i in range(1, df.shape[1]-1)]
    if df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].str.upper().str[0].map({"M":1,"B":0})
    df["diagnosis"] = df["diagnosis"].astype(int)
    feature_cols = [c for c in df.columns if c not in ("ID","diagnosis","Unnamed: 32")]
    X = df[feature_cols].values.astype(float)
    y = df["diagnosis"].values.astype(int)
    return X, y, feature_cols
