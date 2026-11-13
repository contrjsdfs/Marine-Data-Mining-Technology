"""fig5_realmulti.py — figure for zero-injection real multi-vessel analysis."""
from __future__ import annotations
import os, sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for n in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(f.name == n for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "fig5_realmulti.png")

def main():
    j = json.load(open(os.path.join(HERE, "results", "ex5_realmulti.json"), encoding="utf-8"))
    iden, cov, pos = j["identity"], j["coverage"], j["position"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)

    ax = axes[0]
    cats = ["MMSI>1 IMO", "MMSI>1 呼号", "IMO→多MMSI\n(全部)", "IMO→多MMSI\n(真候选)", "呼号→多MMSI", "位置冲突flag", "精确重复"]
    vals = [iden["mmsi_multi_imo"], iden["mmsi_multi_callsign"],
            iden["imo_under_multiple_mmsi"], iden["imo_split_genuine_candidates"],
            iden["callsign_under_multiple_mmsi"], pos["flags_total"], j["duplicate"]["exact_duplicates"]]
    colors = ["#8fb0d9"] * 7
    colors[3] = "#d98f3b"
    ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("count  (zero-injection, real 5000-record national snapshot)")
    ax.set_title("Ex5a: identity conflicts & detector flags (real data)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    cc = ["imo", "call_sign", "vessel_name", "length", "width", "draft"]
    vv = [cov[c] * 100 for c in cc]
    ax.bar(cc, vv, color="#7fb3a0", edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vv):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("% records missing field")
    ax.set_title("Ex5b: static-field coverage (AIS blind-area baseline)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
    print("saved", OUT)

if __name__ == "__main__":
    main()