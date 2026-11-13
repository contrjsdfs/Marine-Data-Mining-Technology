"""plots.py — matplotlib figures (Chinese labels)."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Chinese fonts available on this machine
for n in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(f.name == n for f in font_manager.fontManager.ttflist):
        _CN = n
        break
else:
    _CN = None
if _CN:
    plt.rcParams["font.sans-serif"] = [_CN, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _c(color):
    """Accept 0-255 RGB tuples (PIL style) -> 0-1 floats for matplotlib."""
    if isinstance(color, (tuple, list)) and len(color) >= 3 and max(color) > 1:
        return tuple(v / 255.0 for v in color[:3])
    return color


def bar_chart(categories, series, out, title, ylabel="ratio", ymax=None):
    """series: list of (name, [values], color)"""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    n_cat, n_ser = len(categories), len(series)
    width = 0.8 / n_ser
    x = np.arange(n_cat)
    for si, (name, vals, color) in enumerate(series):
        ax.bar(x + (si - (n_ser - 1) / 2) * width, vals, width * 0.9,
               label=name, color=_c(color), edgecolor="black", linewidth=0.4)
        for xi, v in zip(x, vals):
            ax.text(xi + (si - (n_ser - 1) / 2) * width, v, f"{v:.4g}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def dual_panel_bars(panel_a, panel_b, out, title):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), dpi=150)
    for ax, p in zip(axes, (panel_a, panel_b)):
        cats, series = p["categories"], p["series"]
        n_cat, n_ser = len(cats), len(series)
        width = 0.8 / n_ser
        x = np.arange(n_cat)
        for si, (name, vals, color) in enumerate(series):
            ax.bar(x + (si - (n_ser - 1) / 2) * width, vals, width * 0.9,
                   label=name, color=_c(color), edgecolor="black", linewidth=0.4)
            for xi, v in zip(x, vals):
                ax.text(xi + (si - (n_ser - 1) / 2) * width, v,
                        f"{v:.4g}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(cats)
        ax.set_ylabel(p.get("ylabel", ""))
        ax.set_ylim(0, p.get("ymax") or (max(v for _, vs, _ in series for v in vs) * 1.2))
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def polyline_panels(ev, x, y, arr, out, title, n=140):
    """3 panels: arrival-ordered polyline (fold-back red), event-ordered polyline,
    and speed histogram of both orderings (substantiates the residual claim)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    speed_arr, speed_ev = [], []
    for order, lab in [("arrival", "到达序"), ("event", "事件序")]:
        idx = sorted(range(n), key=lambda i: arr[i] if order == "arrival" else ev[i])
        xe = x[idx]; ye = y[idx]; e = ev[idx]
        ax = axes[0] if order == "arrival" else axes[1]
        ax.plot(xe, ye, "-", color=(0.3, 0.58, 0.95), lw=1.2, alpha=0.9)
        fb = np.where(np.diff(e) < 0)[0]
        for i in fb:
            ax.plot([xe[i], xe[i + 1]], [ye[i], ye[i + 1]], "-", color="crimson", lw=2.2)
        ax.set_title(lab + ("（红色=折返段）" if order == "arrival" else ""))
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        seg = np.maximum(np.diff(e), 1e-9)
        sp = np.hypot(np.diff(xe), np.diff(ye)) / seg
        (speed_arr if order == "arrival" else speed_ev).extend(sp)
        ax.axhline(0, color="k", lw=0.4)
    # speed histogram with the 18 kn artifact threshold
    ax = axes[2]
    thr = 18 * 0.514444
    ax.hist(speed_arr, bins=60, alpha=0.55, label=f"到达序 (超速率 {np.mean(np.array(speed_arr) > thr):.2%})",
            color="crimson")
    ax.hist(speed_ev, bins=60, alpha=0.55, label=f"事件序 (超速率 {np.mean(np.array(speed_ev) > thr):.2%})",
            color=(0.3, 0.58, 0.95))
    ax.axvline(thr, color="k", ls="--", lw=1, label="18 kn 阈值")
    ax.set_xlabel("段速度 (m/s)"); ax.set_ylabel("段数"); ax.set_title("段速度分布")
    ax.legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def line_chart(xs, series, out, title, xlabel, ylabel):
    """series: list of (name, [values], color, marker)"""
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    for name, vals, color, marker in series:
        ax.plot(xs, vals, marker=marker, label=name, color=color, lw=1.8, ms=4)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for name, vals, _, marker in series:
        for xi, v in zip(xs, vals):
            ax.annotate(f"{v:.2%}" if max(map(max, [s[1] for s in series])) <= 1 else f"{v:.2f}",
                        (xi, v), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)