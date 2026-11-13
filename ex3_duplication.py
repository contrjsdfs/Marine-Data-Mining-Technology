"""ex3_duplication.py — Claim: '同一报文经岸基与卫星双路到达、经重传到达
三次' (§3.5.2) asserts duplication happens, but never quantifies the RATE.

What we quantify (ground truth known):
    * duplicate ratio    = (delivered - unique facts) / delivered
    * out-of-order ratio = fraction of copies whose arrival order disagrees
                           with event order (per vessel)
    * dedup verification: idempotency key (MMSI, event_time, source_seq)
                          recovers EXACTLY the unique-fact count (0 error)
    * topology dependence: duplicate ratio as a function of the mean number of
      delivery paths per message (p_dup), swept 0.0 -> 1.4
    * internal sanity check (one line, not a standalone claim): a key that
      pollutes itself with arrival time inflates the recovered count
The punch: the textbook says duplication happens; we supply the number the
textbook lacks, and show it is a property of the receiving network topology.
"""
from __future__ import annotations
import numpy as np
import os

from sim_common import make_rng, save_csv, save_json

N_VESSELS = 120
N_STEPS = 50      # unique facts per vessel
EV_STEP_S = 60.0
SWEEP = [round(x, 2) for x in np.arange(0.0, 1.41, 0.1)]   # p_dup grid


def simulate(rng, p_dup, sat_frac):
    """Deliver copies of unique facts over two paths (terrestrial / satellite)."""
    facts = [(v, s * EV_STEP_S, 0) for v in range(N_VESSELS) for s in range(N_STEPS)]
    delivered = []
    for (v, et, seq) in facts:
        n_copies = 1 + rng.poisson(p_dup)
        for _ in range(n_copies):
            path = "sat" if rng.random() < sat_frac else "ter"
            arrival = et + (rng.normal(2.0, 0.7) if path == "ter" else rng.uniform(60, 3600))
            delivered.append((v, et, seq, arrival, path))
    return facts, delivered


def out_of_order_ratio(delivered):
    by_v = {}
    for (v, et, seq, arr, path) in delivered:
        by_v.setdefault(v, []).append((arr, et))
    bad = tot = 0
    for v, lst in by_v.items():
        lst.sort()
        for i in range(1, len(lst)):
            tot += 1
            if lst[i][1] < lst[i - 1][1]:
                bad += 1
    return bad / max(1, tot)


def run(outdir: str, seed: int = 11):
    rng = make_rng(seed)
    out, rows = {}, []
    for p_dup in SWEEP:
        facts, delivered = simulate(rng, p_dup, sat_frac=0.5)
        unique = len(facts)
        seen = {(d[0], d[1], d[2]) for d in delivered}
        recovered = len(seen)
        assert recovered == unique, "idempotency key must recover exact unique count"
        seen2 = {(d[0], d[1], round(d[3] * 100)) for d in delivered}
        row = dict(p_dup=p_dup, unique=unique, delivered=len(delivered),
                   duplicate_ratio=round(1.0 - unique / len(delivered), 6),
                   out_of_order=round(out_of_order_ratio(delivered), 6),
                   recovered=recovered, recovery_error=0,
                   frag_ratio_polluted_key=round(len(seen2) / unique, 4))
        rows.append(row)
        out[str(p_dup)] = row
    save_csv(os.path.join(outdir, "ex3_duplication.csv"), rows)
    save_json(os.path.join(outdir, "ex3_duplication.json"), out)
    return out