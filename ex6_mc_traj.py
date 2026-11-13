"""ex6_mc_traj.py — 真实多船轨迹切片（MarineCadastre 2015-01-01 00:00-01:00 UTC,
260,001 行 / 8,072 艘）上的零注入验证：
  * 位置冲突检测器（30km/30min）—— 自然异常数
  * 身份一致性（同 MMSI 多 IMO/呼号/船名；同 IMO/呼号 多 MMSI）
  * (MMSI, 事件时间) 精确重复率（原始产品层的重复）
  * 轨迹速度剖面（实际段速分位数 + 超阈值占比，对接实验2 的"残余异常"）
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from sim_common import save_csv, save_json
from ex4_realdata import position_flags, haversine_m, epoch_s

SLICE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "data", "raw", "mc_2015-01-01T00.csv")


def load():
    df = pd.read_csv(SLICE)
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={"MMSI": "mmsi", "BaseDateTime": "Timestamp", "LAT": "Latitude", "LON": "Longitude"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # normalize IMO (float w/ NaN -> int or None)
    def _imo(x):
        if pd.isna(x):
            return None
        try:
            v = int(float(x))
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None
    df["IMO"] = df["IMO"].map(_imo)
    return df


def speed_profile(df):
    g = df.sort_values(["mmsi", "Timestamp"])
    speeds = []
    for mmsi, sub in g.groupby("mmsi"):
        if len(sub) < 2:
            continue
        tt = epoch_s(sub["Timestamp"])
        dt = np.diff(tt)
        keep = dt > 0
        if not keep.any():
            continue
        d = np.array([haversine_m(sub["Latitude"].iloc[i], sub["Longitude"].iloc[i],
                                  sub["Latitude"].iloc[i + 1], sub["Longitude"].iloc[i + 1])
                      for i in range(len(sub) - 1)])[keep]
        sp = d / np.maximum(dt[keep], 1e-9) / 0.514444   # knots
        speeds.extend(sp.tolist())
    s = np.array(speeds)
    return s


def run(outdir: str):
    df = load()
    n_rows = len(df)
    n_vessels = int(df["mmsi"].nunique())
    span = [str(df["Timestamp"].min()), str(df["Timestamp"].max())]
    rows_per_vessel = df.groupby("mmsi").size()
    med_pts = float(rows_per_vessel.median())

    # --- position conflict detector ---
    flags = position_flags(df)
    flagged_mmsi = sorted(int(m) for m, s in flags.items() if s)
    flag_rows = int(sum(len(s) for s in flags.values()))

    # --- identity consistency (zero injection) ---
    g = df.groupby("mmsi")
    mmsi_multi_imo = int((g["IMO"].nunique(dropna=False) > 1).sum())
    # callsign: count mmsi with >1 distinct non-empty callsign
    cs = df[df["CallSign"].astype(str).str.strip().str.len() > 0]
    mmsi_multi_call = int((cs.groupby("mmsi")["CallSign"].nunique() > 1).sum())
    nm = df[df["VesselName"].astype(str).str.strip().str.len() > 0]
    mmsi_multi_name = int((nm.groupby("mmsi")["VesselName"].nunique() > 1).sum())

    imo_df = df[df["IMO"].notna()]
    imo_mmsi = imo_df.groupby("IMO")["mmsi"].nunique()
    imo_split = imo_mmsi[imo_mmsi > 1]
    n_imo_split = int(len(imo_split))
    n_imo_placeholder = int((imo_split.index < 1_000_000).sum())
    n_imo_genuine = n_imo_split - n_imo_placeholder

    cs_df = df[df["CallSign"].astype(str).str.strip().str.len() > 0]
    call_mmsi = cs_df.groupby("CallSign")["mmsi"].nunique()
    n_call_split = int((call_mmsi > 1).sum())

    # --- duplicates ---
    exact_dup = int(df.duplicated(subset=["mmsi", "Timestamp"]).sum())

    # --- speed profile ---
    sp = speed_profile(df)
    speed = dict(median_kn=float(np.median(sp)), p999_kn=float(np.percentile(sp, 99.9)),
                 max_kn=float(sp.max()),
                 over_30kn=float((sp > 30).mean()), over_50kn=float((sp > 50).mean()))

    out = dict(
        source="MarineCadastre AIS_2015_01_01 (USCG/BOEM/NOAA, public)",
        slice="2015-01-01 00:00-01:00 UTC (busiest hour of 6M-row scan)",
        n_rows=n_rows, n_vessels=n_vessels, span=span,
        median_points_per_vessel=med_pts,
        position=dict(flagged_minutes=flag_rows, flagged_mmsi=flagged_mmsi[:20],
                      n_flagged_mmsi=len(flagged_mmsi)),
        identity=dict(mmsi_multi_imo=mmsi_multi_imo, mmsi_multi_callsign=mmsi_multi_call,
                      mmsi_multi_name=mmsi_multi_name,
                      imo_under_multiple_mmsi=n_imo_split,
                      imo_split_genuine=n_imo_genuine, imo_split_placeholder=n_imo_placeholder,
                      callsign_under_multiple_mmsi=n_call_split),
        duplicate=dict(exact_duplicates=exact_dup,
                       dup_rate=exact_dup / n_rows if n_rows else 0.0),
        speed=speed,
    )
    save_json(os.path.join(outdir, "ex6_mc_traj.json"), out)
    save_csv(os.path.join(outdir, "ex6_mc_traj.csv"), [
        {"metric": k, "value": v} for k, v in
        {**{"n_rows": n_rows, "n_vessels": n_vessels,
            "flagged_minutes": out["position"]["flagged_minutes"],
            "n_flagged_mmsi": out["position"]["n_flagged_mmsi"],
            "mmsi_multi_imo": mmsi_multi_imo, "mmsi_multi_callsign": mmsi_multi_call,
            "imo_split_total": n_imo_split, "imo_split_genuine": n_imo_genuine,
            "callsign_split": n_call_split, "exact_duplicates": exact_dup,
            "speed_median_kn": speed["median_kn"], "speed_p999_kn": speed["p999_kn"],
            "speed_over_50kn_frac": speed["over_50kn"]},
         **out["speed"]}.items()])
    return out


if __name__ == "__main__":
    import json
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    r = run(outdir)
    print(json.dumps(r, ensure_ascii=False, indent=2))