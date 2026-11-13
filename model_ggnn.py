"""model_ggnn.py — 边门控 GNN：先用逐对逻辑回归给边打分，剔除低分（跨实体）边，
再在"干净"邻域图上训练/评估 GNN（验证第9节负结果的机制解释）。
对照：LR(逐对) / 朴素GNN / GNN(门控) / 门控+UF连通分量(无GNN)。
τ 敏感性：门控阈值 {0.5, 0.8, 0.95}。
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch

sys_path_base = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path_base)
import model_gnn as G
from model_gnn import (add_noise, make_data, fit_logistic, predict_lr,
                       pr_auc, ece, train_gnn, gcn_embed, TAU, build_groups)
from ex1_identity import simulate
from sim_common import make_rng

torch.manual_seed(0)


def gate_edges(edge, pi, pj, scores, tau):
    score_of = {}
    for a, b, s in zip(pi.tolist(), pj.tolist(), scores.tolist()):
        key = (a, b) if a < b else (b, a)
        score_of[key] = s
    ks, kd = [], []
    src = edge[0].tolist(); dst = edge[1].tolist()
    for a, b in zip(src, dst):
        key = (a, b) if a < b else (b, a)
        if score_of.get(key, -1.0) >= tau:
            ks.append(a); kd.append(b)
    return torch.tensor(ks, dtype=torch.long), torch.tensor(kd, dtype=torch.long)


class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def components(n, src, dst):
    uf = UF(n)
    for a, b in zip(src.tolist(), dst.tolist()):
        uf.union(a, b)
    return np.array([uf.find(i) for i in range(n)])


def shared_pr_from_scores(pair_m, s, truth_sh, thr=0.5, min_neg=0.05):
    frac = pd.Series((s < thr).astype(float)).groupby(pair_m).mean()
    det = {m for m, f in frac.items() if f >= min_neg}
    tp = len(det & truth_sh); fp = len(det - truth_sh); fn = len(truth_sh - det)
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    return round(p, 3), round(r, 3)


def pair_metrics(Y, s=None, pred=None):
    if pred is None:
        pred = s >= 0.5
    tp = int((pred & (Y == 1)).sum()); fp = int((pred & (Y == 0)).sum())
    fn = int((~pred & (Y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    auc = pr_auc(Y, s) if s is not None else float("nan")
    ec = ece(Y, s) if s is not None else float("nan")
    return dict(auc=round(auc, 3), f1=round(f1, 3), ece=round(ec, 3))


def main():
    rng_tr = make_rng(42); rng_te = make_rng(7)
    train = add_noise(simulate(rng_tr, share_frac=0.05, rewrite_frac=0.03, forged_frac=0.01))
    testA = add_noise(simulate(rng_te, share_frac=0.08, rewrite_frac=0.05, forged_frac=0.02))
    clean = add_noise(simulate(rng_te, share_frac=0.0, rewrite_frac=0.0, forged_frac=0.0))
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
        sh = target - mb
        mask = shifted["entity"] == b
        shifted.loc[mask, "lat"] += sh[0]; shifted.loc[mask, "lon"] += sh[1]
    adv = add_noise(list(zip(shifted["entity"].tolist(), shifted["tm"].tolist(),
                             shifted["mmsi"].tolist(), shifted["t"].tolist(),
                             shifted["lat"].tolist(), shifted["lon"].tolist())))

    df_tr, feat_tr, edge_tr, pairs_tr = make_data(train)
    mean, std = feat_tr.mean(0), feat_tr.std(0)
    Xtr, Ytr = pairs_tr[2], pairs_tr[3]
    w_lr, lmean, lstd = fit_logistic(Xtr, Ytr)
    s_tr = predict_lr(Xtr, w_lr, lmean, lstd)
    edge_tr_g = gate_edges(edge_tr, pairs_tr[0], pairs_tr[1], s_tr, tau=0.5)
    groups_tr = build_groups(df_tr)
    w1, w2 = train_gnn(df_tr, feat_tr, edge_tr, groups_tr, mean, std)          # plain
    w1g, w2g = train_gnn(df_tr, feat_tr, edge_tr_g, groups_tr, mean, std)       # gated
    n_tr = len(df_tr)

    out = {}
    for name, rows_ in [("TEST-A", testA), ("TEST-ADV", adv), ("TEST-CLEAN", clean)]:
        df_t, feat_t, edge_t, pairs_t = make_data(rows_)
        Xt, Yt = pairs_t[2], pairs_t[3]
        pi, pj = pairs_t[0], pairs_t[1]
        pair_m = np.array([df_t["mmsi"].values[i] for i in pi])
        uniq = {}
        for i, m in enumerate(df_t["mmsi"].values):
            uniq.setdefault(int(m), set()).add(int(df_t["entity"].values[i]))
        truth_sh = {m for m, e in uniq.items() if len(e) > 1}

        r = {}
        # 1) LR pairwise
        s_lr = predict_lr(Xt, w_lr, lmean, lstd)
        r["LR"] = dict(pair_metrics(Yt, s_lr),
                       shared=shared_pr_from_scores(pair_m, s_lr, truth_sh))
        # 2) plain GNN
        with torch.no_grad():
            e0 = gcn_embed(feat_t, edge_t, w1, w2, mean, std)
        emb0 = e0[torch.tensor(pi, dtype=torch.long)]; emb1 = e0[torch.tensor(pj, dtype=torch.long)]
        s_gnn = torch.sigmoid((emb0 * emb1).sum(1) / TAU).numpy()
        r["GNN-plain"] = dict(pair_metrics(Yt, s_gnn),
                              shared=shared_pr_from_scores(pair_m, s_gnn, truth_sh))
        # 3) gated GNN + 4) gate+UF, with tau sweep
        s_gate = predict_lr(Xt, w_lr, lmean, lstd)
        for tau in (0.5, 0.8, 0.95):
            g_e = gate_edges(edge_t, pi, pj, s_gate, tau)
            tag = f"gate{tau}"
            with torch.no_grad():
                eg = gcn_embed(feat_t, g_e, w1g, w2g, mean, std)
            e0g = eg[torch.tensor(pi, dtype=torch.long)]; e1g = eg[torch.tensor(pj, dtype=torch.long)]
            s_gg = torch.sigmoid((e0g * e1g).sum(1) / TAU).numpy()
            r[f"GNN-gate{tau}"] = dict(pair_metrics(Yt, s_gg),
                                       shared=shared_pr_from_scores(pair_m, s_gg, truth_sh))
            # gate+UF (no GNN): components from gated edges
            comp = components(len(df_t), g_e[0], g_e[1])
            pred_comp = (comp[pi] == comp[pj])
            comp_truth = pd.DataFrame({"m": pair_m, "c": comp[pi]})
            ncomp = comp_truth.groupby("m")["c"].nunique()
            det_uf = {m for m, v in ncomp.items() if v > 1}
            tp = len(det_uf & truth_sh); fp = len(det_uf - truth_sh); fn = len(truth_sh - det_uf)
            sp = tp / (tp + fp) if tp + fp else 0; sr = tp / (tp + fn) if tp + fn else 0
            r[f"UF-gate{tau}"] = dict(pair_metrics(Yt, pred=pred_comp),
                                      shared=(round(sp, 3), round(sr, 3)))
        out[name] = dict(truth_shared=len(truth_sh), methods=r)
        print(f"\n=== {name} (true shared={len(truth_sh)}) ===")
        for k, v in r.items():
            up = f"auc={v['auc']} f1={v['f1']}"
            print(f"  {k:12s} {up} shared={v['shared']}")
    json.dump(out, open(os.path.join(sys_path_base, "results", "model_ggnn.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()