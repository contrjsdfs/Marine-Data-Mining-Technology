"""ex5_realmulti.py — 零注入：真实多船国家快照上的身份一致性与位置检测能力。

数据：real-noaa.csv（USCG/BOEM/NOAA，CC0）
      = 2024-01-01 00:00:00–00:02:19 全美沿海 AIS 前 5000 条完整记录，
        4614 艘船；每船 1–2 条（快照型）；imo/call_sign/vessel_name 大多有值。
        （来源 blob noaaocm...ais-2024-01-01.csv.zst，取前 5000 条，见 provenance）
测量（全部零注入，真实观测）：
  * 位置冲突检测器（30km/30min）在每船≤2 条的快照上能发现什么（预期≈0）
  * 身份一致性：MMSI↔IMO / MMSI↔呼号 / MMSI↔船名 冲突数
  * 拆分信号：同一 IMO 出现在 >1 个 MMSI（教材"改写/拆分"形态的真实观测）
  * 静态覆盖：IMO 缺失率等（AIS 盲区的事实基线）
  * 精确重复率：(MMSI, 事件时间) 完全重复（对接实验3）
  * 第三方交叉参照：ais-quality 工具在该切片上的审计结果（1 条 position_jump）
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from sim_common import save_csv, save_json

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "raw", "extracted3", "ais-quality-main", "ais-quality", "examples")


def run(outdir: str):
    df = pd.read_csv(os.path.join(BASE, "real-noaa.csv"), encoding="utf-8")
    df["base_date_time"] = pd.to_datetime(df["base_date_time"])
    span = [str(df["base_date_time"].min()), str(df["base_date_time"].max())]

    # --- position conflict detector on >=2-row vessels ---
    multi = df[df.groupby("mmsi")["mmsi"].transform("size") >= 2]
    pos_flags = {}
    if len(multi):
        g = multi.rename(columns={"base_date_time": "Timestamp",
                                  "latitude": "Latitude", "longitude": "Longitude"})
        from ex4_realdata import position_flags as _pf
        pos_flags = _pf(g)
    pos_flagged_mmsi = [int(m) for m, s in pos_flags.items() if s]

    # --- identity consistency (zero injection) ---
    g_by_mmsi = df.groupby("mmsi")
    n_mmsi_multi_imo = int((g_by_mmsi["imo"].nunique(dropna=False) > 1).sum())
    n_mmsi_multi_call = int((g_by_mmsi["call_sign"].nunique(dropna=False) > 1).sum())
    n_mmsi_multi_name = int((g_by_mmsi["vessel_name"].nunique(dropna=False) > 1).sum())

    nonempty_imo = df[df["imo"].astype(str).str.strip().str.len() > 0]
    imo_mmsi = nonempty_imo.groupby("imo")["mmsi"].nunique()
    split_imos = imo_mmsi[imo_mmsi > 1].index.tolist()

    def _imo_digits(imo: str):
        return "".join(ch for ch in str(imo) if ch.isdigit())

    def _is_placeholder(imo: str) -> bool:
        d = _imo_digits(imo)
        try:
            return int(d) < 1_000_000 or d == ""
        except ValueError:
            return True

    split_examples = []
    n_placeholder, n_genuine = 0, 0
    for imo in split_imos:
        sub = nonempty_imo[nonempty_imo["imo"] == imo]
        mmsis = sorted(sub["mmsi"].unique())
        ph = _is_placeholder(imo)
        n_placeholder += ph
        n_genuine += (not ph)
        split_examples.append({"imo": imo, "placeholder": ph,
                               "mmsi_list": [int(m) for m in mmsis],
                               "vessel_name": sub["vessel_name"].dropna().iloc[0],
                               "vessel_type": sub["vessel_type"].dropna().iloc[0] if sub["vessel_type"].notna().any() else None})

    nonempty_call = df[df["call_sign"].astype(str).str.strip().str.len() > 0]
    call_mmsi = nonempty_call.groupby("call_sign")["mmsi"].nunique()
    split_calls = int((call_mmsi > 1).sum())

    # --- duplicates & coverage ---
    exact_dup = int(df.duplicated(subset=["mmsi", "base_date_time"]).sum())
    coverage = {c: float(df[c].isna().mean()) for c in
                ["imo", "call_sign", "vessel_name", "length", "width", "draft"]}

    # --- third-party auditor cross-reference ---
    third_party = None
    try:
        fr = pd.read_csv(os.path.join(BASE, "real-report", "flagged_records.csv"),
                         encoding="utf-8-sig")
        third_party = dict(flagged_rows=len(fr),
                           issues=fr["issue_codes"].dropna().unique().tolist() if len(fr) else [],
                           flagged_mmsi=fr["mmsi"].unique().tolist() if len(fr) else [])
    except Exception:
        pass

    out = dict(
        source="real-noaa.csv (USCG/BOEM/NOAA CC0; first 5000 records of 2024-01-01)",
        n_rows=len(df), n_vessels=int(df["mmsi"].nunique()), span=span,
        vessels_with_2_rows=int((df.groupby("mmsi").size() == 2).sum()),
        position=dict(flags_total=int(sum(len(s) for s in pos_flags.values())),
                      flagged_mmsi=pos_flagged_mmsi),
        identity=dict(mmsi_multi_imo=n_mmsi_multi_imo, mmsi_multi_callsign=n_mmsi_multi_call,
                      mmsi_multi_name=n_mmsi_multi_name,
                      imo_under_multiple_mmsi=len(split_imos),
                      imo_split_placeholder=n_placeholder,
                      imo_split_genuine_candidates=n_genuine,
                      split_examples=split_examples,
                      callsign_under_multiple_mmsi=split_calls),
        cross_ref=dict(position_flagged_mmsi=pos_flagged_mmsi,
                       third_party_flagged_mmsi=(third_party or {}).get("flagged_mmsi")),
        duplicate=dict(exact_duplicates=exact_dup),
        coverage=coverage,
        third_party_audit=third_party,
    )
    save_json(os.path.join(outdir, "ex5_realmulti.json"), out)
    rows = [{"metric": "n_rows", "value": out["n_rows"]},
            {"metric": "n_vessels", "value": out["n_vessels"]},
            {"metric": "span", "value": " -> ".join(out["span"])},
            {"metric": "position_flags_total", "value": out["position"]["flags_total"]},
            {"metric": "mmsi_multi_imo", "value": out["identity"]["mmsi_multi_imo"]},
            {"metric": "mmsi_multi_callsign", "value": out["identity"]["mmsi_multi_callsign"]},
            {"metric": "imo_under_multiple_mmsi", "value": out["identity"]["imo_under_multiple_mmsi"]},
            {"metric": "callsign_under_multiple_mmsi", "value": out["identity"]["callsign_under_multiple_mmsi"]},
            {"metric": "exact_duplicates", "value": out["duplicate"]["exact_duplicates"]},
            {"metric": "missing_imo_frac", "value": round(coverage["imo"], 4)},
            {"metric": "third_party_flagged_rows", "value": (third_party or {}).get("flagged_rows")}]
    save_csv(os.path.join(outdir, "ex5_realmulti.csv"), rows)
    return out


if __name__ == "__main__":
    import json
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    r = run(outdir)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:5]) for k, v in r.items()},
                     ensure_ascii=False, indent=2))