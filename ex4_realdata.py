"""ex4_realdata.py — 双层验证（方案A 半合成注入 + 方案B 登记册交叉）在真实 AIS 切片上。

数据：真实单船轨迹 mmsi_210253000（Kattegat/Skagerrak, 2026-09-26 00:15–12:54,
      1657 点；公开导出把 IMO/呼号/船名脱敏为 Unknown/<Null>）。
方案A（半合成真值注入）：
  * 克隆真实轨迹为第二艘物理船 B（经度 +0.6°，约 36 km，位置/时间/航速沿用真实值）；
  * 注入 3 个"共用"事件（02:15 / 06:15 / 10:15 各 25 分钟）：B 改用 A 的 MMSI 播发
    -> 同一 MMSI 同时出现在两个相距约 36 km 的真实位置（教材"共用"形态）；
  * 注入 1 个"改写"事件（04:15，25 分钟）：A 改用伪造 MMSI 播发（教材"改写"形态）。
  * 在合并流上跑与实验1相同的冲突检测器（同 MMSI 30 分钟窗内 >30 km 即冲突）。
方案B（登记册交叉，在真实导出上以模拟登记册快照实现——因公开导出把 IMO 脱敏）：
  * 为 A/B 各挂一个稳定 IMO 号；伪装报文携带发送端真实 IMO；
  * 规则1（时间局部化）：30 分钟内同 MMSI 出现 >1 个 IMO -> 共用；
  * 规则2（时间局部化）：30 分钟内同 IMO 出现在 >1 个 MMSI -> 改写/拆分。
验证目标：
  * 纯真实单船切片上检测器必须 0 flag（无假阳性基线）；
  * 位置启发式：共用召回≈1、改写召回=0（盲区）；
  * 登记册：两类召回都≈1，且把位置启发式的 flag 用独立证据复核。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from sim_common import save_csv, save_json

DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "raw", "extracted2", "AIS_Data-main", "mmsi_210253000.csv")

MMSI_A = 210253000
FALSE_MMSI = 999999999
IMO_A, IMO_B = 912345678, 923456789
OFFSET_LON = 0.6          # ~36 km at 57N
CONFLICT_DIST_M = 30_000.0
WINDOW_S = 30 * 60
INJECT_MIN = 25
EPOCH = pd.Timestamp("1970-01-01")


def epoch_s(series: pd.Series) -> np.ndarray:
    return ((series - EPOCH) / np.timedelta64(1, "s")).astype(float).to_numpy()


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_real():
    df = pd.read_csv(DATA_CSV)
    df = df.rename(columns=lambda c: c.strip().replace("# ", ""))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["Timestamp", "Latitude", "Longitude"]).reset_index(drop=True)
    df["mmsi"] = int(MMSI_A)
    return df[["Timestamp", "mmsi", "Latitude", "Longitude", "SOG", "COG", "Heading"]]


def build_stream(real):
    b = real.copy(); b["Longitude"] = b["Longitude"] + OFFSET_LON
    b["mmsi"] = 210253001; b["event"] = "none"; b["fact"] = "B"
    a = real.copy(); a["event"] = "none"; a["fact"] = "A"
    t0 = real["Timestamp"].min()
    shared_starts = [t0 + pd.Timedelta(hours=h) for h in (7, 9, 11)]
    rewrite_start = t0 + pd.Timedelta(hours=8)

    inj_b = []
    for i, st in enumerate(shared_starts):
        w = b[(b["Timestamp"] >= st) & (b["Timestamp"] < st + pd.Timedelta(minutes=INJECT_MIN))].copy()
        w["mmsi"] = MMSI_A; w["event"] = f"shared_{i + 1}"; w["fact"] = "B"
        inj_b.append(w)
    shared_all = pd.concat(inj_b, ignore_index=True)

    w_a = a[(a["Timestamp"] >= rewrite_start) &
            (a["Timestamp"] < rewrite_start + pd.Timedelta(minutes=INJECT_MIN))].copy()
    w_a["mmsi"] = FALSE_MMSI; w_a["event"] = "rewrite_1"; w_a["fact"] = "A"

    rw_mask = (a["Timestamp"] >= rewrite_start) & (a["Timestamp"] < rewrite_start + pd.Timedelta(minutes=INJECT_MIN))
    a_amb = a[~rw_mask]
    b_amb = b.iloc[[]]  # B's own mmsi reports only appear in injected windows
    return pd.concat([a_amb, b_amb, shared_all, w_a], ignore_index=True)


def position_flags(stream):
    """Minute-level flag: same MMSI, two reports >30 km apart within 30 min."""
    flags = {}
    for mmsi, g in stream.groupby("mmsi"):
        g = g.sort_values("Timestamp")
        tt = epoch_s(g["Timestamp"]); idx = np.argsort(tt); tt = tt[idx]
        lat = g["Latitude"].values[idx]; lon = g["Longitude"].values[idx]
        flagged = set()
        for i in range(len(tt)):
            j = i + 1
            while j < len(tt) and tt[j] - tt[i] <= WINDOW_S:
                if haversine_m(lat[i], lon[i], lat[j], lon[j]) > CONFLICT_DIST_M:
                    flagged.add(float(tt[i])); flagged.add(float(tt[j])); break
                j += 1
        flags[mmsi] = flagged
    return flags


def registry_flags(stream):
    """Time-localised registry cross-check (rule1 conflicts, rule2 rewrite)."""
    s = stream.copy()
    s["IMO"] = np.where(s["fact"] == "A", IMO_A, IMO_B)
    n = len(s)
    rule1 = np.zeros(n, dtype=bool); rule2 = np.zeros(n, dtype=bool)
    tt = epoch_s(s["Timestamp"])
    for i in range(n):
        dt = np.abs(tt - tt[i])
        w = (dt <= WINDOW_S) & (dt > 0)
        if np.any(w):
            same_mmsi = (s["mmsi"].values == s["mmsi"].iloc[i]) & w
            if np.any(s["IMO"].values[same_mmsi] != s["IMO"].iloc[i]):
                rule1[i] = True
            same_imo = (s["IMO"].values == s["IMO"].iloc[i]) & w
            if np.any(s["mmsi"].values[same_imo] != s["mmsi"].iloc[i]):
                rule2[i] = True
    return s, rule1, rule2


def run(outdir: str):
    if not os.path.exists(DATA_CSV):
        print("real data missing; run fetch + extract first. skip ex4.")
        return None
    real = load_real()
    stream = build_stream(real)
    gt_shared = stream["event"].str.startswith("shared").fillna(False).values
    gt_rewrite = stream["event"].str.startswith("rewrite").fillna(False).values
    n_shared = int(gt_shared.sum()); n_rewrite = int(gt_rewrite.sum())

    # --- A) real-only baseline: detector must be silent on a clean identity ---
    real_only = position_flags(real)
    real_only_flags = int(len(real_only.get(MMSI_A, set())))

    # --- A) position heuristic on combined stream ---
    flags = position_flags(stream)
    flagged = np.zeros(len(stream), dtype=bool)
    for mmsi, tset in flags.items():
        tt = epoch_s(stream[stream["mmsi"] == mmsi]["Timestamp"]).astype("float64")
        hit = np.isin(tt, np.array(sorted(tset)))
        flagged[stream["mmsi"].values == mmsi] |= hit
    tt_all = epoch_s(stream["Timestamp"])
    # grace: flags within WINDOW_S of any injected event are detector-window
    # neighbours, not false positives
    tt_events = tt_all[gt_shared | gt_rewrite]
    if len(tt_events):
        dist_to_event = np.min(np.abs(tt_all[:, None] - tt_events[None, :]), axis=1)
        in_grace = dist_to_event <= WINDOW_S + 60.0
    else:
        in_grace = np.zeros(len(stream), dtype=bool)
    tp_sh = int((flagged & gt_shared).sum()); fn_sh = int((~flagged & gt_shared).sum())
    tp_rw = int((flagged & gt_rewrite).sum())
    fp = int((flagged & ~in_grace).sum())
    h = dict(
        recall_shared=tp_sh / (tp_sh + fn_sh) if (tp_sh + fn_sh) else float("nan"),
        recall_rewrite=tp_rw / n_rewrite if n_rewrite else float("nan"),
        precision=tp_sh / (tp_sh + fp) if (tp_sh + fp) else float("nan"),
        tp_shared=tp_sh, fn_shared=fn_sh, fp=fp, n_shared=n_shared, n_rewrite=n_rewrite)

    # --- B) registry cross-check ---
    s2, rule1, rule2 = registry_flags(stream)
    r1_tp = int((rule1 & gt_shared).sum()); r1_fp = int((rule1 & ~in_grace).sum())
    r2_tp = int((rule2 & gt_rewrite).sum()); r2_fp = int((rule2 & ~in_grace).sum())
    reg = dict(
        recall_shared=r1_tp / (r1_tp + int((~rule1 & gt_shared).sum())) if n_shared else float("nan"),
        recall_rewrite=r2_tp / n_rewrite if n_rewrite else float("nan"),
        precision_shared=r1_tp / (r1_tp + r1_fp) if (r1_tp + r1_fp) else float("nan"),
        precision_rewrite=r2_tp / (r2_tp + r2_fp) if (r2_tp + r2_fp) else float("nan"))

    out = dict(
        data="mmsi_210253000 (Kattegat/Skagerrak) 2026-09-26 00:15-12:54 real AIS",
        n_real_points=len(real),
        real_only_flags=real_only_flags,
        n_shared_injected=3, n_rewrite_injected=1,
        heuristic=h, registry=reg)
    assert real_only_flags == 0, "clean real identity must not be flagged"
    save_csv(os.path.join(outdir, "ex4_realdata.csv"), [{"metric": k, "value": v}
                                                        for k, v in out.items() if not isinstance(v, dict)])
    save_json(os.path.join(outdir, "ex4_realdata.json"), out)
    return out


if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    r = run(outdir)
    if r:
        import json
        print(json.dumps(r, ensure_ascii=False, indent=2))