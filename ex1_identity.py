"""ex1_identity.py — Claim: 'MMSI 会被改写、共用、伪造' (§3.5.2) is unquantified.

Ground-truth-known simulation with three injected flaw mechanisms:
  * share   : two physical vessels broadcast under the SAME MMSI (detectable:
              same MMSI, far apart within a time window)
  * rewrite : one vessel broadcasts under two different MMSIs (split track)
  * forged  : a low-rate of reports carries a unique random MMSI (orphan
              records; position heuristics cannot catch them)
What we quantify:
  * true conflict rate (injected) vs detected conflict rate, precision/recall
  * downstream harm of trusting MMSI as the entity ID: fraction of physical
    vessels whose 'trajectory' gets merged (share) or split (rewrite)
  * the punch: on a CLEAN slice the measured rate ~ 0 -> the textbook's
    universal '必须身份解析' is conditional, not absolute.
"""
from __future__ import annotations
import numpy as np
import os

from sim_common import make_rng, save_csv, save_json, kn_to_ms

MAX_SPEED_MS = kn_to_ms(12.0)
CONFLICT_DIST_M = 30_000.0
WINDOW_S = 30 * 60
STEP_S = 5 * 60
N_STEPS = 288


def simulate(rng, n_ent=300, share_frac=0.05, rewrite_frac=0.03,
             forged_frac=0.01, report_prob=0.65, box_km=200.0, speed_kn=8.0):
    dt = STEP_S
    v = rng.normal(kn_to_ms(speed_kn), kn_to_ms(1.5), n_ent)
    ang = rng.uniform(-np.pi, np.pi, n_ent)
    pos = rng.uniform(0, box_km * 1000.0, (n_ent, 2))
    ents = list(range(n_ent))

    # --- injection: shared MMSI (pairs broadcast under one MMSI) ---
    pair_map = {}
    if share_frac > 0:
        rng.shuffle(ents)
        n_pairs = int(share_frac * n_ent / 2)
        for i in range(n_pairs):
            a, b = ents[2 * i], ents[2 * i + 1]
            shared_mmsi = a
            pair_map[a] = shared_mmsi
            pair_map[b] = shared_mmsi

    # --- injection: rewritten identity (alt MMSI for 50% of reports) ---
    rewritten = set()
    alt_mmsi = {}
    if rewrite_frac > 0:
        rng.shuffle(ents)
        rewritten = set(ents[: int(rewrite_frac * n_ent)])
        for e in rewritten:
            alt_mmsi[e] = int(rng.integers(1_000_000, 9_999_999))

    rows = []
    for e in ents:
        true_mmsi = e
        for t_step in range(N_STEPS):
            if rng.random() > report_prob:
                continue
            pos[e] = pos[e] + v[e] * dt * np.array([np.cos(ang[e]), np.sin(ang[e])])
            ang[e] += rng.normal(0, 0.12)
            tx = int(t_step * STEP_S)
            bm = pair_map.get(e, true_mmsi)
            if e in rewritten and rng.random() < 0.5:
                bm = alt_mmsi[e]
            elif rng.random() < forged_frac:
                # unique fake id from a huge space: creates orphan singletons
                bm = int(rng.integers(100_000_000, 999_999_999))
            rows.append((e, true_mmsi, bm, tx, float(pos[e][0]), float(pos[e][1])))
    return rows


def conflict_detector(rows):
    """Pure position heuristic: same MMSI, two reports > 30 km apart inside a
    30-min window => that MMSI cannot be a single physical vessel."""
    by_mmsi = {}
    for (e, tm, bm, t, x, y) in rows:
        by_mmsi.setdefault(bm, []).append((t, x, y))
    flagged = set()
    for bm, pts in by_mmsi.items():
        pts_sorted = sorted(pts)
        for i in range(len(pts_sorted)):
            t0, x0, y0 = pts_sorted[i]
            j = i + 1
            while j < len(pts_sorted) and pts_sorted[j][0] - t0 <= WINDOW_S:
                t1, x1, y1 = pts_sorted[j]
                if np.hypot(x1 - x0, y1 - y0) > CONFLICT_DIST_M:
                    flagged.add(bm)
                    break
                j += 1
            else:
                continue
            break
    return flagged


def run(outdir: str, seed: int = 42):
    rng = make_rng(seed)
    out = {}
    for name, kw in [("CLEAN", dict(share_frac=0.0, rewrite_frac=0.0, forged_frac=0.0)),
                     ("DIRTY", dict(share_frac=0.05, rewrite_frac=0.03, forged_frac=0.01))]:
        rows = simulate(rng, **kw)
        n_ent = len({r[0] for r in rows})
        n_rows = len(rows)

        mmsi_entities = {}
        for (e, tm, bm, t, x, y) in rows:
            mmsi_entities.setdefault(bm, set()).add(e)
        true_conflicted = {bm for bm, es in mmsi_entities.items() if len(es) > 1}
        true_merged = {e for es in mmsi_entities.values() if len(es) > 1 for e in es}
        # split = dominant broadcast MMSI accounts for < 90% of a vessel's
        # reports (rewrite hits this; 1% forged orphans do not)
        ent_mmsi_counts = {}
        for (e, tm, bm, t, x, y) in rows:
            ent_mmsi_counts.setdefault(e, {}).setdefault(bm, 0)
            ent_mmsi_counts[e][bm] += 1
        true_split = set()
        for e, cnt in ent_mmsi_counts.items():
            dom = max(cnt.values()) / sum(cnt.values())
            if dom < 0.90:
                true_split.add(e)

        detected = conflict_detector(rows)
        TP = len(detected & true_conflicted)
        FP = len(detected - true_conflicted)
        FN = len(true_conflicted - detected)
        prec = TP / (TP + FP) if (TP + FP) else float("nan")
        rec = TP / (TP + FN) if (TP + FN) else float("nan")

        n_corrupt = true_merged | true_split
        out[name] = dict(
            n_entities=n_ent, n_rows=n_rows,
            true_conflict_mmsi=len(true_conflicted),
            detected_conflict_mmsi=len(detected),
            true_merged_entities=len(true_merged),
            true_split_entities=len(true_split),
            corrupted_trajectories=len(n_corrupt),
            corrupt_frac=len(n_corrupt) / n_ent,
            precision=(float(prec) if prec == prec else None),
            recall=(float(rec) if rec == rec else None),
        )
    save_csv(os.path.join(outdir, "ex1_identity.csv"), [
        {"slice": k, **{kk: (round(vv, 6) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()}} for k, v in out.items()])
    save_json(os.path.join(outdir, "ex1_identity.json"), out)
    return out