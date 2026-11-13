"""fig6_mc_traj.py — Ex6 figure: real MarineCadastre trajectory slice."""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for n in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(f.name == n for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]; break
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fig6_mc_traj.png")

def main():
    j = json.load(open(os.path.join(HERE, "results", "ex6_mc_traj.json"), encoding="utf-8"))
    pos = j["position"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)

    ax = axes[0]
    mmsis = [str(m) for m in pos["flagged_mmsi"]]
    # per-mmsi approx: share flagged minutes equally across count (approx)
    n_min = pos["flagged_minutes"]
    vals = [n_min / max(1, len(mmsis))] * len(mmsis)
    ax.bar(mmsis, vals, color="#d98f3b", edgecolor="black", linewidth=0.4)
    ax.text(len(mmsis) - 1, vals[-1], f"合计 {n_min} 分钟位", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("flagged (approx. share of flagged minutes)")
    ax.set_title(f"Ex6a: position-conflict flagged vessels (5 / {j['n_vessels']:,} vessels, "
                 f"{j['n_rows']:,} real AIS rows)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    sp = np.array(json.load(open(os.path.join(HERE, "results", "_speed_hist.json"), "r"))["sp"])
    ax.hist(sp[sp < 200], bins=200, color="#7fb3a0", edgecolor="black", linewidth=0.2)
    for thr in (30, 50):
        ax.axvline(thr, color="crimson", ls="--", lw=1.2)
        ax.text(thr, ax.get_ylim()[1]*0.85, f"{thr} kn", color="crimson", fontsize=9)
    ax.set_xlabel("real AIS segment speed (kn, clipped at 200)")
    ax.set_ylabel("segments")
    ax.set_title("Ex6b: real speed profile (p99.9=32.7 kn; max≈179,000 kn = teleport)")
    fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
    print("saved", OUT)

if __name__ == "__main__":
    main()