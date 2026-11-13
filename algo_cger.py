"""algo_cger.py — 速度约束连续图实体解析 (Continuity-Graph Entity Resolution, v1).

物理模型（无手调冲突阈值）：
  edge(i,j) := |dt|<=DTMAX and haversine(i,j) <= VMAX * |dt|
  同一 MMSI 内查并集 -> 物理航段；跨 MMSI 桥接（可通行且 |dt|<=BRIDGE_DT）-> 实体。
输出：连通分量标签 = 物理实体；可推导出共用/改写/伪造/瞬移异常。

评测 1（合成真值，ex1 DIRTY）：聚类对级 P/R/F1（contingency 精确计算）——
   基线 A = "MMSI 即实体"；基线 B = 远距启发式；Ours = CGER（含桥接）。
评测 2（真实 MarineCadastre 小时窗）：MMSI 内连续性分量数 >1 的异常分布，
   与远距启发式的 5 艘 flag 对照。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex4_realdata import haversine_m, epoch_s
from ex1_identity import simulate, conflict_detector
from sim_common import make_rng

VMAX_MS = 45 * 0.514444          # 45 kn speed budget (fast ferries < ~40 kn)
R0_M = 1000.0                     # noise floor: allowed dist = Vmax*dt + R0
CAP_M = 25_000.0                  # hard reach cap: long gaps never bridge far vessels
DTMAX_S = 60 * 60                 # max gap for a continuity edge
BRIDGE_DT_S = 5 * 60             # cross-MMSI bridge window
def auto_mode(lat):
    """positions in degrees -> haversine; planar meters -> euclidean."""
    return "hav" if np.abs(np.asarray(lat)).max() <= 90 else "euc"

def dist_m(a, b, mode):
    if mode == "euc":
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))
    return haversine_m(a[0], a[1], b[0], b[1])


class UF:
    def __init__(self, n):
        self.p = list(range(n)); self.n = n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cger(df, vmax=VMAX_MS, dtmax=DTMAX_S, bridge=True, static_key=None):
    """df: mmsi, t_s, lat, lon [, static_key]. Returns predicted entity labels.
    Cross-MMSI bridging additionally requires static-key equality when given
    (motion-reachable is necessary but not sufficient)."""
    n = len(df)
    uf = UF(n)
    mmsi = df["mmsi"].values; t = df["t_s"].values
    lat = df["lat"].values; lon = df["lon"].values
    mode = auto_mode(lat)
    grid = 5000.0 if mode == "euc" else 0.05
    pts = list(zip(lat.tolist(), lon.tolist()))
    # 1) within-MMSI continuity edges
    for m in np.unique(mmsi):
        idx = np.where(mmsi == m)[0]
        if len(idx) < 2:
            continue
        o = np.argsort(t[idx]); idx = idx[o]
        tt = t[idx]
        for i in range(len(idx) - 1):
            j = i + 1
            while j < len(idx) and tt[j] - tt[i] <= dtmax:
                if dist_m(pts[idx[i]], pts[idx[j]], mode) <= min(vmax * (tt[j] - tt[i]) + R0_M, CAP_M):
                    uf.union(idx[i], idx[j])
                j += 1
    # 2) cross-MMSI bridging (grid-bucketed candidates)
    if bridge:
        lg = np.round(lat / grid).astype(np.int64)
        og = np.round(lon / grid).astype(np.int64)
        tg = np.round(t / 300).astype(np.int64)
        cells = {}
        for k in range(n):
            cells.setdefault((int(lg[k]), int(og[k]), int(tg[k])), []).append(k)
        done = set()
        for k in range(n):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        c = (int(lg[k]) + di, int(og[k]) + dj, int(tg[k]) + dk)
                        for j in cells.get(c, ()):
                            key = (k, j) if k < j else (j, k)
                            if key in done or mmsi[k] == mmsi[j]:
                                continue
                            if static_key is not None and df[static_key].values[k] != df[static_key].values[j]:
                                continue
                            done.add(key)
                            dt = abs(t[k] - t[j])
                            if dt <= BRIDGE_DT_S and dist_m(pts[k], pts[j], mode) <= min(vmax * dt + R0_M, CAP_M):
                                uf.union(k, j)
    return np.array([uf.find(i) for i in range(n)])


def pair_metrics(true_labels, pred_labels):
    """Exact pair-level P/R/F1 from contingency (no sklearn available)."""
    n = len(true_labels)
    import itertools
    keys = {}
    for a, b in zip(true_labels.tolist(), pred_labels.tolist()):
        keys[(a, b)] = keys.get((a, b), 0) + 1
    # relabel to compact
    uA = sorted(set(true_labels.tolist())); uB = sorted(set(pred_labels.tolist()))
    cont = np.zeros((len(uA), len(uB)), dtype=float)
    for i, a in enumerate(uA):
        for j, b in enumerate(uB):
            cont[i, j] = keys.get((a, b), 0)
    row = cont.sum(1); col = cont.sum(0)
    tp = float((cont * (cont - 1) / 2).sum())
    fp = float((col * (col - 1) / 2).sum()) - tp
    fn = float((row * (row - 1) / 2).sum()) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def eval_synthetic(seed=42):
    rng = make_rng(seed)
    out = {}
    for name, kw in [("CLEAN", dict(share_frac=0.0, rewrite_frac=0.0, forged_frac=0.0)),
                     ("DIRTY", dict(share_frac=0.05, rewrite_frac=0.03, forged_frac=0.01))]:
        rows = simulate(rng, **kw)
        df = pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
        df["t_s"] = df["t"].astype(float)
        df["callsign"] = df["entity"].map(lambda e: f"CS{e:04d}")   # per-entity static id
        truth = df["entity"].values
        pred_mmsi = df["mmsi"].values                       # baseline A
        pred_within = cger(df, bridge=False)                # within-MMSI only
        pred_full = cger(df, bridge=True, static_key="callsign")   # ours (motion+static)
        mA = pair_metrics(truth, pred_mmsi)
        mW = pair_metrics(truth, pred_within)
        mF = pair_metrics(truth, pred_full)
        if name == "DIRTY":
            det = conflict_detector(list(zip(df["entity"].tolist(), df["tm"].tolist(),
                                             df["mmsi"].tolist(), df["t"].tolist(),
                                             df["lat"].tolist(), df["lon"].tolist())))
            def shared_set(pred):
                nuniq = df.assign(ce=pred).groupby("mmsi")["ce"].nunique()
                return set(nuniq[nuniq > 1].index.tolist())
            cger_full_shared = shared_set(pred_full)
            truth_multient = set(df.assign(e=truth).groupby("mmsi")["e"].nunique()[lambda s: s > 1].index.tolist())
            def prrec(ds):
                tp = len(ds & truth_multient); fp = len(ds - truth_multient)
                fn = len(truth_multient - ds)
                p = tp / (tp + fp) if tp + fp else 0
                r = tp / (tp + fn) if tp + fn else 0
                return round(p, 3), round(r, 3)
            out[name] = dict(mmsi_as_entity=mA, cger_within=mW, cger_full=mF,
                             shared_detect=dict(heuristic=prrec(det),
                                                cger_full=prrec(cger_full_shared),
                                                n_true=len(truth_multient)))
        else:
            out[name] = dict(mmsi_as_entity=mA, cger_within=mW, cger_full=mF)
    return out


def eval_real(slice_path):
    df = pd.read_csv(slice_path)
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={"MMSI": "mmsi", "BaseDateTime": "Timestamp", "LAT": "lat", "LON": "lon"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["t_s"] = epoch_s(df["Timestamp"])
    n = len(df)
    pred = cger(df, bridge=False)
    comps = pd.DataFrame({"mmsi": df["mmsi"].values, "comp": pred}).groupby("mmsi")["comp"].nunique()
    multi = comps[comps > 1]
    # classify: worst velocity-implausible segment per multi-component MMSI
    from ex4_realdata import haversine_m as hm
    sub_lt30 = 0; worst_examples = []
    for m in multi.index:
        g = df[df["mmsi"] == m].sort_values("t_s")
        tt = g["t_s"].values; la = g["lat"].values; lo = g["lon"].values
        dmax, spmax, dtm = -1, -1, -1
        for i in range(len(tt) - 1):
            d = hm(la[i], lo[i], la[i + 1], lo[i + 1])
            dt = tt[i + 1] - tt[i]
            if dt <= 0: continue
            sp = d / dt / 0.514444
            if sp > spmax: spmax, dmax, dtm = sp, d, dt
        if dmax < 30000: sub_lt30 += 1
        if len(worst_examples) < 6:
            worst_examples.append({"mmsi": int(m), "max_seg_km": round(dmax / 1000, 1),
                                   "max_seg_kn": round(spmax, 0), "dt_s": int(dtm)})
    return dict(n_rows=n, n_mmsi=int(df["mmsi"].nunique()),
                n_components=int(len(np.unique(pred))),
                mmsi_with_gt1_component=int(len(multi)),
                multi_with_jump_lt30km=sub_lt30,
                worst_examples=worst_examples,
                components_per_flagged5={int(m): int(comps[m]) for m in
                                         [1193046, 100000000, 367373630, 367495180, 367584740]
                                         if m in comps.index})


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    r = eval_synthetic()
    print("=== synthetic (pair-level clustering) ===")
    for k, v in r.items():
        a, w, f = v["mmsi_as_entity"], v["cger_within"], v["cger_full"]
        print(k)
        print("   MMSI-as-entity : P=%.3f R=%.3f F1=%.3f" % (a["precision"], a["recall"], a["f1"]))
        print("   CGER(within)   : P=%.3f R=%.3f F1=%.3f" % (w["precision"], w["recall"], w["f1"]))
        print("   CGER(full)     : P=%.3f R=%.3f F1=%.3f" % (f["precision"], f["recall"], f["f1"]))
    d = r["DIRTY"]["shared_detect"]
    print("DIRTY shared detection: heuristic P/R =", d["heuristic"],
          " CGER(full) P/R =", d["cger_full"], " n_true =", d["n_true"])
    rl = eval_real(os.path.join(base, "data", "raw", "mc_2015-01-01T00.csv"))
    print("=== real MarineCadastre hour slice ===")
    print(json.dumps(rl, ensure_ascii=False))
    json.dump({"synthetic": r, "real": rl}, open(os.path.join(base, "results", "algo_cger.json"), "w"),
              ensure_ascii=False, indent=2, default=str)
    print("saved results/algo_cger.json")