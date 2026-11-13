"""fig4_realdata.py — figure for the real-data two-layer validation."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ex4_realdata as e

OUT = os.path.join(HERE, "results", "fig4_realdata.png")


def main():
    real = e.load_real()
    stream = e.build_stream(real)
    flags = e.position_flags(stream)
    _, rule1, rule2 = e.registry_flags(stream)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

    # --- Panel 1: map of tracks ---
    ax = axes[0]
    a = stream[stream["fact"] == "A"]
    b = stream[stream["fact"] == "B"]
    ax.plot(a["Longitude"], a["Latitude"], ".", ms=2, color=(0.2, 0.45, 0.9), label="A: real track")
    ax.plot(b["Longitude"], b["Latitude"], ".", ms=2, color=(0.95, 0.6, 0.15), label="B: clone +0.6 deg lon")
    sh = stream[stream["event"].str.startswith("shared").fillna(False)]
    rw = stream[stream["event"].str.startswith("rewrite").fillna(False)]
    ax.plot(sh["Longitude"], sh["Latitude"], "s", ms=6, color="crimson", label="shared windows (A mmsi)")
    ax.plot(rw["Longitude"], rw["Latitude"], "^", ms=6, color="green", label="rewrite window (forged mmsi)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title(
        "Ex4 (real AIS 210253000, Kattegat): injected events on real positions")
    ax.legend(fontsize=8)

    # --- Panel 2: timeline of flags vs ground truth ---
    ax = axes[1]
    s = stream.copy()
    s["t_min"] = (e.epoch_s(s["Timestamp"]) - e.epoch_s(s["Timestamp"]).min()) / 60.0
    order = np.argsort(s["t_min"].values)
    t = s["t_min"].values[order]
    gt_share = s["event"].str.startswith("shared").fillna(False).values[order]
    gt_rew = s["event"].str.startswith("rewrite").fillna(False).values[order]
    y = np.zeros(len(t))
    ax.fill_between(t, y - 0.4, y + 2.4, where=gt_share, color="gray", alpha=0.45, label="GT shared")
    ax.fill_between(t, y - 0.4, y + 2.4, where=gt_rew, color="green", alpha=0.35, label="GT rewrite")
    # heuristic flag rows
    flagged = np.zeros(len(s), dtype=bool)
    for mmsi, tset in flags.items():
        tt = e.epoch_s(s[s["mmsi"] == mmsi]["Timestamp"])
        hit = np.isin(tt, np.array(sorted(tset)))
        flagged[s["mmsi"].values == mmsi] |= hit
    fl = flagged[order]
    ax.bar(t[fl], np.full(int(fl.sum()), 0.5), width=0.5, color="crimson", label="position heuristic flags")
    # registry flags (shared=rule1, rewrite=rule2) as scatter markers
    r1 = rule1[order]; r2 = rule2[order]
    ax.bar(t[r1], np.full(int(r1.sum()), 1.1), width=0.5, color="blue", alpha=0.7, label="registry: rule1 (shared)")
    ax.bar(t[r2], np.full(int(r2.sum()), 1.7), width=0.5, color="purple", alpha=0.7, label="registry: rule2 (rewrite)")
    ax.set_xlabel("minutes since 00:15"); ax.set_ylabel("flag level")
    ax.set_title("Timeline: ground truth vs position-heuristic vs registry (real data)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(t.min(), t.max())
    fig.tight_layout()
    fig.savefig(OUT)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()