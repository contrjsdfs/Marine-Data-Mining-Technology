"""model_gnn.py — GNN(运动特征+图结构) vs 逻辑回归(运动特征) vs CGER vs 启发式。

唯一变量是"图结构"：GNN 与逻辑回归吃同一对运动特征（logΔt, log距离, 段速），
GNN 额外把图邻域信息通过消息传递用上。若 GNN 不更好，就是"该任务不需要图模型"。
含噪声（GPS σ + 瞬移坏点，对齐实验6 的真实数据特性）；含对抗切片（共用船近距离同行）。
指标：对级 PR-AUC/F1@p0.5/ECE；MMSI 级共享 P/R。
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ex1_identity import simulate
from sim_common import make_rng

torch.manual_seed(0)
MAX_DT = 3600.0
SPEED_CAP_KN = 500.0
TAU = 0.2


# ---------------------------------------------------------------- data
def add_noise(rows, sigma_m=600.0, tele_frac=0.02, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for (e, tm, bm, t, x, y) in rows:
        x += rng.normal(0, sigma_m); y += rng.normal(0, sigma_m)
        if rng.random() < tele_frac:
            ang = rng.uniform(0, 2 * np.pi)
            dd = rng.uniform(3000, 8000)
            x += dd * np.cos(ang); y += dd * np.sin(ang)
        out.append((e, tm, bm, t, x, y))
    return out


def make_data(rows):
    df = pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
    df["pos"] = np.arange(len(df))
    feats = df[["lat", "lon", "t"]].values.astype(np.float64)
    src, dst = [], []
    pi, pj, pdt, pdist, pspd, py = [], [], [], [], [], []
    for m, g in df.groupby("mmsi"):
        g = g.sort_values("t")
        pos = g["pos"].values; t = g["t"].values.astype(float)
        x = g["lat"].values; y = g["lon"].values; ent = g["entity"].values
        for i in range(len(pos) - 1):
            j = i + 1
            while j < len(pos) and t[j] - t[i] <= MAX_DT:
                src.extend([pos[i], pos[j]]); dst.extend([pos[j], pos[i]])
                dt = t[j] - t[i]
                dist = np.hypot(x[j] - x[i], y[j] - y[i])
                pi.append(pos[i]); pj.append(pos[j])
                pdt.append(dt); pdist.append(dist)
                pspd.append(dist / max(dt, 1e-9) / 0.514444)
                py.append(1.0 if ent[i] == ent[j] else 0.0)
                j += 1
    pdt = np.array(pdt); pdist = np.array(pdist)
    pspd = np.array(pspd); py = np.array(py)
    Xf = np.column_stack([np.log10(pdt + 1e-6), np.log10(pdist + 1.0),
                          np.minimum(pspd, SPEED_CAP_KN)])
    return (df, feats,
            (torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
            (np.array(pi), np.array(pj), Xf, py))


# ---------------------------------------------------------------- logistic (IRLS, motion-only)
def fit_logistic(X, y, lam=1e-2, iters=15):
    mean, std = X.mean(0), X.std(0) + 1e-9
    Xn = np.column_stack([np.ones(len(X)), (X - mean) / std])
    w = np.zeros(Xn.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xn @ w, -30, 30)))
        S = p * (1 - p)
        H = (Xn.T * S) @ Xn + lam * np.eye(Xn.shape[1])
        g = Xn.T @ (y - p) - lam * np.concatenate([[0.0], w[1:]])
        w = w + np.linalg.solve(H, g)
    return w, mean, std


def predict_lr(X, w, mean, std):
    Xn = np.column_stack([np.ones(len(X)), (X - mean) / std])
    return 1 / (1 + np.exp(-np.clip(Xn @ w, -30, 30)))


# ---------------------------------------------------------------- GCN
def _agg(X, s, d, alpha):
    deg = torch.zeros(len(X), 1)
    deg.index_add_(0, s, torch.ones(len(s), 1))
    ns = torch.maximum(deg - 1.0, torch.ones(1))
    tot = X.new_zeros(len(X), X.shape[1])
    tot.index_add_(0, s, X[d])
    return alpha * X + (1 - alpha) * (tot / ns)


def gcn_embed(feat, edge_index, w1, w2, mean, std, alpha=0.8):
    X = torch.tensor(((feat - mean) / (std + 1e-9)).astype(np.float32))
    s, d = edge_index
    h = F.relu(_agg(X, s, d, alpha) @ torch.tensor(w1, dtype=torch.float32))
    out = _agg(h, s, d, alpha) @ torch.tensor(w2, dtype=torch.float32)
    return F.normalize(out, dim=1)


def train_gnn(df, feats, edge_index, groups, mean, std, epochs=20, hidden=32, outd=16, lr=0.02, alpha=0.8):
    w1 = (torch.randn(3, hidden, requires_grad=True) * 0.3).detach().numpy()
    w2 = (torch.randn(hidden, outd, requires_grad=True) * 0.3).detach().numpy()
    # manual autograd via torch on copied tensors
    tw1 = torch.tensor(w1, requires_grad=True)
    tw2 = torch.tensor(w2, requires_grad=True)
    feat_t = torch.tensor(((feats - mean) / (std + 1e-9)).astype(np.float32))
    s, d = edge_index
    s = torch.tensor(s, dtype=torch.long); d = torch.tensor(d, dtype=torch.long)
    deg = torch.zeros(len(feat_t), 1)
    deg.index_add_(0, s, torch.ones(len(s), 1)); deg += 1.0
    opt = torch.optim.Adam([tw1, tw2], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        X = feat_t
        ns = torch.maximum(deg - 1.0, torch.ones(1))
        tot = X.new_zeros(len(X), X.shape[1]); tot.index_add_(0, s, X[d])
        h = F.relu((alpha * X + (1 - alpha) * (tot / ns)) @ tw1)
        tot2 = h.new_zeros(len(h), h.shape[1]); tot2.index_add_(0, s, h[d])
        out = (alpha * h + (1 - alpha) * (tot2 / ns)) @ tw2
        E = F.normalize(out, dim=1)
        loss = torch.zeros(1)
        for m, idx, en in groups:
            if len(idx) < 4: continue
            a = E[idx]
            logits = (a @ a.T) / TAU
            n = len(idx)
            mask_self = torch.eye(n, dtype=torch.bool)
            labels = (en[:, None] == en[None, :]).float()
            labels = labels.masked_fill(mask_self, 0.0)
            if labels.sum() == 0: continue
            loss = loss - (F.log_softmax(logits.masked_fill(mask_self, -1e9), dim=1) * labels).sum(1).mean()
        loss.backward()
        opt.step()
    return tw1.detach().numpy(), tw2.detach().numpy()


def build_groups(df):
    groups = []
    for m, g in df.groupby("mmsi"):
        idx = np.sort(g["pos"].values)
        en = torch.tensor(g["entity"].values.astype(np.int64))
        # reorder en to match sorted pos
        en = en[np.argsort(g["pos"].values)]
        groups.append((int(m), torch.tensor(idx, dtype=torch.long), en))
    return groups


# ---------------------------------------------------------------- metrics
def pr_auc(y, s):
    order = np.argsort(-s); y = y[order]; s = s[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    p = tp / np.maximum(tp + fp, 1); r = tp / max(float(y.sum()), 1e-9)
    return float(np.trapezoid(p, r))


def ece(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for k in range(bins):
        m = (s > edges[k]) & (s <= edges[k + 1])
        if m.sum():
            e += (m.sum() / len(s)) * abs(y[m].mean() - s[m].mean())
    return float(e)


def main():
    rng_tr = make_rng(42); rng_te = make_rng(7)
    train = add_noise(simulate(rng_tr, share_frac=0.05, rewrite_frac=0.03, forged_frac=0.01))
    testA = add_noise(simulate(rng_te, share_frac=0.08, rewrite_frac=0.05, forged_frac=0.02))
    clean = add_noise(simulate(rng_te, share_frac=0.0, rewrite_frac=0.0, forged_frac=0.0))
    # adversarial: shared pairs offset 8-20km (close-moving)
    rows = simulate(rng_te, share_frac=0.10, rewrite_frac=0.04, forged_frac=0.02)
    df0 = pd.DataFrame(rows, columns=["entity", "tm", "mmsi", "t", "lat", "lon"])
    ent_m = df0.groupby("mmsi")["entity"].nunique()
    shared = [m for m, v in ent_m.items() if v > 1]
    shifted = df0.copy()
    for m in shared[:6]:
        g = df0[df0["mmsi"] == m]; ents = sorted(g["entity"].unique())
        a, b = ents[0], ents[1]
        ma = g[g["entity"] == a][["lat", "lon"]].mean().values
        mb = g[g["entity"] == b][["lat", "lon"]].mean().values
        ang = rng_tr.uniform(0, 2 * np.pi); off = rng_tr.uniform(8000, 20000)
        target = ma + off * np.array([np.cos(ang), np.sin(ang)])
        shift = target - mb
        mask = shifted["entity"] == b
        shifted.loc[mask, "lat"] += shift[0]; shifted.loc[mask, "lon"] += shift[1]
    adv = add_noise(list(zip(shifted["entity"].tolist(), shifted["tm"].tolist(),
                             shifted["mmsi"].tolist(), shifted["t"].tolist(),
                             shifted["lat"].tolist(), shifted["lon"].tolist())))

    df_tr, feat_tr, edge_tr, pairs_tr = make_data(train)
    mean, std = feat_tr.mean(0), feat_tr.std(0)
    Xtr, Ytr = pairs_tr[2], pairs_tr[3]
    w_lr, lmean, lstd = fit_logistic(Xtr, Ytr)
    groups_tr = build_groups(df_tr)
    w1, w2 = train_gnn(df_tr, feat_tr, edge_tr, groups_tr, mean, std)

    mmsi_tr = df_tr["mmsi"].values
    out = {}
    for name, rows_ in [("TEST-A", testA), ("TEST-ADV", adv), ("TEST-CLEAN", clean)]:
        df_t, feat_t, edge_t, pairs_t = make_data(rows_)
        Xt, Yt = pairs_t[2], pairs_t[3]
        pi, pj = pairs_t[0], pairs_t[1]
        s_lr = predict_lr(Xt, w_lr, lmean, lstd)
        with torch.no_grad():
            emb_gnn = gcn_embed(feat_t, edge_t, w1, w2, mean, std)
        e0 = emb_gnn[torch.tensor(pi, dtype=torch.long)]
        e1 = emb_gnn[torch.tensor(pj, dtype=torch.long)]
        s_gnn = torch.sigmoid((e0 * e1).sum(1) / TAU).numpy()
        mmsi_of = df_t["mmsi"].values
        ent_of = df_t["entity"].values
        # shared truth per mmsi
        uniq = {m: set() for m in set(mmsi_of)}
        for i in range(len(rows_)):
            uniq[mmsi_of[df_t["pos"].values[i]]].add(ent_of[i])
        truth_sh = {m for m, e in uniq.items() if len(e) > 1}
        def mmsi_pr(s, thr=0.5, min_neg=0.05):
            pair_m = np.array([mmsi_of[i] for i in pi])
            frac = pd.Series((s < thr).astype(float)).groupby(pair_m).mean()
            det = {m for m, f in frac.items() if f >= min_neg}
            tp = len(det & truth_sh); fp = len(det - truth_sh); fn = len(truth_sh - det)
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            return round(p, 3), round(r, 3), len(truth_sh)
        def pair_met(s):
            pair_f1_d = None
            pred = s >= 0.5
            tp = int((pred & (Yt == 1)).sum()); fp = int((pred & (Yt == 0)).sum())
            fn = int((~pred & (Yt == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            return dict(auc=round(pr_auc(Yt, s), 3), f1=round(f1, 3),
                        ece=round(ece(Yt, s), 3), shp=mmsi_pr(s)[0], shr=mmsi_pr(s)[1])
        out[name] = dict(truth_shared=len(truth_sh),
                         lr=pair_met(s_lr), gnn=pair_met(s_gnn))
        print(f"\n=== {name} (true shared={len(truth_sh)}) ===")
        for tag, s in [("LR(motion)", s_lr), ("GNN(motion+graph)", s_gnn)]:
            m = out[name]["gnn" if "GNN" in tag else "lr"]
            print(f"  {tag}: PR-AUC={m['auc']} pair-F1={m['f1']} ECE={m['ece']} "
                  f"shared P/R=({m['shp']},{m['shr']})")
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "results", "model_gnn.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()