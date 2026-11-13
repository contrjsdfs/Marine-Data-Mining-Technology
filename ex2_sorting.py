"""ex2_sorting.py — Claim: '按到达顺序连线会画出折返与超速假象' (§3.5.2, 图3.10).

Factorial design to separate the two causal factors:
    sorting axis (arrival vs event time)  x  data quality (clean vs noisy)
Metrics per reconstruction:
    fold_back_frac : fraction of segments where event time decreases along polyline
    overspeed_frac : fraction of segments whose implied speed > 18 kn
Verification: clean + event-time sorting must give exactly 0 artifacts; the
arrival-sorted version must show artifacts whenever reordering is present.
The punch: after fixing the axis, residual artifacts from noise/teleports remain
> 0, so the textbook attribution (sorting alone causes the artifact) is incomplete.
"""
from __future__ import annotations
import numpy as np
import os

from sim_common import make_rng, save_csv, save_json, kn_to_ms

SPEED_LIMIT_MS = kn_to_ms(18.0)      # artifact threshold ("implausibly fast")
TRUE_VMAX_MS = kn_to_ms(10.0)        # simulator never exceeds this
DT_S = 60.0
N_PTS = 360                          # 6 h at 1 min


def make_track(rng, sigma_pos_m=0.0, n_tele=0):
    """Waypoint walk with bounded speed; adds Gaussian position noise and
    a few teleport outliers. Returns (event_times, x, y)."""
    ev = np.arange(N_PTS, dtype=float) * DT_S
    x = np.zeros(N_PTS); y = np.zeros(N_PTS)
    heading = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(0.5, 1.0) * TRUE_VMAX_MS
    for i in range(1, N_PTS):
        if rng.random() < 0.08:
            heading += rng.uniform(-1.2, 1.2)
            speed = rng.uniform(0.5, 1.0) * TRUE_VMAX_MS
        x[i] = x[i - 1] + speed * np.cos(heading) * DT_S
        y[i] = y[i - 1] + speed * np.sin(heading) * DT_S
    if sigma_pos_m > 0:
        x = x + rng.normal(0, sigma_pos_m, N_PTS)
        y = y + rng.normal(0, sigma_pos_m, N_PTS)
    if n_tele > 0:
        for _ in range(n_tele):
            k = int(rng.integers(2, N_PTS - 2))
            x[k] += rng.choice([-1, 1]) * rng.uniform(4_000, 9_000)
    return ev, x, y


def delivery_delays(rng, n):
    """Mixture: 80% fast terrestrial (seconds), 15% satellite (minutes),
    5% delayed satellite / backlog (tens of minutes)."""
    kind = rng.random(n)
    d = np.zeros(n)
    m = kind < 0.80
    d[m] = rng.normal(2.0, 0.7, int(m.sum()))
    m = (kind >= 0.80) & (kind < 0.95)
    d[m] = rng.uniform(60, 600, int(m.sum()))
    m = kind >= 0.95
    d[m] = rng.uniform(600, 3600, int(m.sum()))
    return d


def metrics(ev, x, y):
    """fold-back fraction and overspeed fraction along a polyline."""
    seg_ev = np.diff(ev)
    speed = np.hypot(np.diff(x), np.diff(y)) / np.maximum(seg_ev, 1e-9)
    fold = float(np.mean(seg_ev < 0))
    over = float(np.mean(speed > SPEED_LIMIT_MS))
    return round(fold, 6), round(over, 6)


def run(outdir: str, seed: int = 7):
    rng = make_rng(seed)
    out, rows = {}, []
    for name, kw in [("clean", dict(sigma_pos_m=0.0, n_tele=0)),
                     ("noisy", dict(sigma_pos_m=150.0, n_tele=2))]:
        ev, x, y = make_track(rng, **kw)
        arr = ev + delivery_delays(rng, N_PTS)
        row = {"condition": name}
        for order, lab in [("event", "event"), ("arrival", "arrival")]:
            idx = np.argsort(ev if order == "event" else arr, kind="stable")
            f, o = metrics(ev[idx], x[idx], y[idx])
            row[f"fold_{lab}"] = f
            row[f"over_{lab}"] = o
        rows.append(row)
        out[name] = row
        if name == "clean":
            assert row["fold_event"] == 0.0 and row["over_event"] == 0.0, \
                "clean + event sort must be artifact-free"
            assert row["fold_arrival"] > 0.0 or row["over_arrival"] > 0.0, \
                "arrival sort must show artifacts under reordering"
        if name == "noisy":
            assert row["over_event"] > 0.0, "residual artifact from noise/teleports expected"
    save_csv(os.path.join(outdir, "ex2_sorting.csv"), rows)
    save_json(os.path.join(outdir, "ex2_sorting.json"), out)
    return out