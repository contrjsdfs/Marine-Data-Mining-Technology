"""sim_common.py — shared utilities for §3.5.2 verification simulations.

All experiments are ground-truth-known simulations (fixed seeds), so every
measured number can be verified against the injected truth.
"""
from __future__ import annotations
import json, os, csv
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def save_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def kn_to_ms(kn: float) -> float:
    return kn * 0.514444  # 1 knot = 0.514444 m/s