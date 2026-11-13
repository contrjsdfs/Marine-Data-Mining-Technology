"""model_link.py — 监督式实体链接预测（逻辑回归）替代手调阈值。

任务（同一 MMSI 内的两两链接）：
  同 MMSI 的两个报文属同一物理船？(1/0)
  训练标签来自注入真值（真实数据拿不到、这正是注入的价值：既能验证又能训练）。
特征（全部观测可得，无标签泄漏）：
  f1 = log10(Δt)          f2 = log10(距离+1m)
  f3 = 段速(kn, 截断)      f4 = 静态匹配(同呼号=1)
对照三轮：
  H 启发式（30km/30min）  C  CGER（运动连续分量）  L  学习模型（仅运动 / 运动+静态）
指标：PR-AUC、F1@p*、MMSI 级共享检测 P/R、校准 ECE。
评测：TRAIN=DIRTY(seed42)  TEST-A=DIRTY(seed7, 更脏)  TEST-B=CLEAN(误警检查)。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex1_identity import simulate, conflict_detector
from sim_common import make_rng
from algo_cger import cger

MAX_DT = 3600.0
SPEED_CAP_KN = 500.0


def build_pairs(rows, max_pairs_per_mmsi=30000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
    df["callsign"] = df["entity"].map(lambda e: f"CS{e:04d}")
    Xs, ys = [], []
    for m, g in df.groupby("mmsi"):
        g = g.sort_values("t")
        ent = g["entity"].values; t = g["t"].values.astype(float)
        x = g["lat"].values; y = g["lon"].values; cs = g["callsign"].values
        n = len(g)
        if n < 2:
            continue
        rows_pairs = []
        for i in range(n):
            j = i + 1
            while j < n and t[j] - t[i] <= MAX_DT:
                rows_pairs.append((i, j)); j += 1
        if not rows_pairs:
            continue
        ii = np.array([p[0] for p in rows_pairs]); jj = np.array([p[1] for p in rows_pairs])
        dt = (t[jj] - t[ii]).astype(float)
        dist = np.hypot(x[jj] - x[ii], y[jj] - y[ii]).astype(float)
        speed = dist / np.maximum(dt, 1.0) / 0.514444
        static = (cs[ii] == cs[jj]).astype(float)
        y = (ent[ii] == ent[jj]).astype(float)
        # balance / cap per mmsi
        if len(y) > max_pairs_per_mmsi:
            keep = rng.choice(len(y), max_pairs_per_mmsi, replace=False)
            dt, dist, speed, static, y = dt[keep], dist[keep], speed[keep], static[keep], y[keep]
        Xs.append(np.column_stack([np.log10(dt + 1e-6), np.log10(dist + 1.0),
                                  np.minimum(speed, SPEED_CAP_KN), static]))
        ys.append(y)
    X = np.vstack(Xs); Y = np.concatenate(ys)
    return X, Y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, lam=1e-2, iters=15):
    Xn = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / (X.std(0) + 1e-9)])
    w = np.zeros(Xn.shape[1])
    for _ in range(iters):
        p = sigmoid(Xn @ w)
        S = p * (1 - p)
        H = (Xn.T * S) @ Xn + lam * np.eye(Xn.shape[1])
        g = Xn.T @ (y - p) - lam * np.concatenate([[0.0], w[1:]])
        w = w + np.linalg.solve(H, g)
    return w


def predict(X, w, mean, std):
    Xn = np.column_stack([np.ones(len(X)), (X - mean) / (std + 1e-9)])
    return sigmoid(Xn @ w)


def pr_auc(y, s):
    order = np.argsort(-s)
    y = y[order]; s = s[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    p = tp / np.maximum(tp + fp, 1); r = tp / max(float(y.sum()), 1e-9)
    return float(np.trapezoid(p, r)) if len(p) > 1 else float("nan")


def ece(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for k in range(bins):
        m = (s > edges[k]) & (s <= edges[k + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(s)) * abs(y[m].mean() - s[m].mean())
    return float(e)


def shared_detect_mmsi(rows, method, model=None, meta=None, thr=0.5, min_neg_frac=0.05):
    """returns set of MMSI flagged as shared by given method."""
    df = pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
    df["callsign"] = df["entity"].map(lambda e: f"CS{e:04d}")
    if method == "heuristic":
        det = conflict_detector(list(zip(df["entity"].tolist(), df["tm"].tolist(),
                                         df["mmsi"].tolist(), df["t"].tolist(),
                                         df["lat"].tolist(), df["lon"].tolist())))
        return set(det)
    if method == "cger":
        d2 = df.rename(columns={"t": "t_s"})
        pred = cger(d2, bridge=False)
        ncomp = pd.DataFrame({"mmsi": d2["mmsi"].values, "c": pred}).groupby("mmsi")["c"].nunique()
        return set(ncomp[ncomp > 1].index.tolist())
    if method == "learn":
        X, Y = build_pairs(rows, seed=123)
        w, mean, std = model
        s = predict(X, w, mean, std)
        frac_neg = pd.Series(s < thr).groupby(pd.Series(pair_mmsi(rows))).mean()
        # recompute per-mmsi via pairs directly
        pair_m = pair_mmsi(rows)
        fr = {}
        for m in set(pair_m):
            sel = np.array([i for i, mm in enumerate(pair_m) if mm == m])
            if len(sel):
                fr[m] = float((s[sel] < thr).mean())
        return {m for m, f in fr.items() if f >= min_neg_frac}


def pair_mmsi(rows):
    df = pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
    mm = []
    for m, g in df.groupby("mmsi"):
        g = g.sort_values("t"); t = g["t"].values.astype(float); n = len(g)
        for i in range(n):
            j = i + 1
            while j < n and t[j] - t[i] <= MAX_DT:
                mm.append(m); j += 1
    return mm


def shared_detect_pp(rows, truth_shared):
    det = shared_detect_mmsi(rows, "heuristic")
    tp = len(det & truth_shared); fp = len(det - truth_shared); fn = len(truth_shared - det)
    return (round(tp / (tp + fp), 3) if tp + fp else 0, round(tp / (tp + fn), 3) if tp + fn else 0)

def shared_pr(rows, truth_shared):
    def pr(s):
        tp = len(s & truth_shared); fp = len(s - truth_shared); fn = len(truth_shared - s)
        return (round(tp / (tp + fp), 3) if tp + fp else 0,
                round(tp / (tp + fn), 3) if tp + fn else 0)
    return pr(shared_detect_mmsi(rows, "heuristic")), pr(shared_detect_mmsi(rows, "cger"))


def main():
    rng_tr = make_rng(42); rng_te = make_rng(7)
    rows_train = simulate(rng_tr, share_frac=0.05, rewrite_frac=0.03, forged_frac=0.01)
    rows_testA = simulate(rng_te, share_frac=0.08, rewrite_frac=0.05, forged_frac=0.02)
    rows_testB = simulate(rng_te, share_frac=0.0, rewrite_frac=0.0, forged_frac=0.0)

    Xtr, Ytr = build_pairs(rows_train, seed=1)
    print(f"train pairs {len(Ytr)} positive rate={Ytr.mean():.3f}")
    w_full = fit_logistic(Xtr, Ytr); mean = Xtr.mean(0); std = Xtr.std(0)
    # motion-only: features 0..2 (dt, dist, speed) — same input as CGER
    w_mot = fit_logistic(Xtr[:, :3], Ytr); mean_m = Xtr[:, :3].mean(0); std_m = Xtr[:, :3].std(0)
    results = {}
    for name, rows, is_clean in [("TEST-A(DIRTY harsher)", rows_testA, False),
                                 ("TEST-B(CLEAN)", rows_testB, True)]:
        Xte, Yte = build_pairs(rows, seed=2)
        pair_m = np.array(pair_mmsi(rows))
        truth_shared = set(pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
                           .groupby("mmsi")["entity"].nunique()[lambda s: s > 1].index.tolist())
        rows_out = {"true_shared": len(truth_shared),
                    "heur": shared_detect_mmsi(rows, "heuristic"),
                    "cger": shared_detect_mmsi(rows, "cger")}
        for tag, Xf, ww, mm, ss in [("motion", Xte[:, :3], w_mot, mean_m, std_m),
                                    ("motion+static", Xte, w_full, mean, std)]:
            s = predict(Xf, ww, mm, ss)
            auc = pr_auc(Yte, s); ec = ece(Yte, s)
            pred = s >= 0.5
            tp = int((pred & (Yte == 1)).sum()); fp = int((pred & (Yte == 0)).sum())
            fn = int((~pred & (Yte == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            # MMSI-level shared detect via learned model on these pairs
            fr = {}
            for m in set(pair_m.tolist()):
                sel = np.where(pair_m == m)[0]
                fr[m] = float((s[sel] < 0.5).mean())
            lrn = {m for m, f in fr.items() if f >= 0.05}
            tp_l = len(lrn & truth_shared); fp_l = len(lrn - truth_shared); fn_l = len(truth_shared - lrn)
            p_l = tp_l / (tp_l + fp_l) if tp_l + fp_l else 0
            r_l = tp_l / (tp_l + fn_l) if tp_l + fn_l else 0
            rows_out[tag] = dict(auc=round(auc, 3), pair_f1=round(f1, 3), pair_p=round(p, 3),
                                 pair_r=round(r, 3), ece=round(ec, 3),
                                 shared_p=round(p_l, 3), shared_r=round(r_l, 3))
        results[name] = rows_out
        print(f"\n=== {name} (true shared={len(truth_shared)}) ===")
        print(f"  heuristic shared P/R = {shared_detect_pp(rows, truth_shared)}")
        hpr, cpr = shared_pr(rows, truth_shared)
        print(f"  heuristic={hpr}  cger={cpr}")
        for tag in ("motion", "motion+static"):
            o = rows_out[tag]
            print(f"  learn[{tag}]  PR-AUC={o['auc']} pair-F1={o['pair_f1']} "
                  f"ECE={o['ece']}  shared P/R=({o['shared_p']},{o['shared_r']})")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "model_link.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=2, default=str)
    print("saved", out)


if __name__ == "__main__":
    main()