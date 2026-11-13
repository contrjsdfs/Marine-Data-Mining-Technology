"""adjudicate_registry.py — 真实登记册裁决（跨独立真实切片 + 在线登记册尝试）。

证据源A：real-noaa.csv（USCG/BOEM/NOAA，2024-01-01，4614船，含 IMO/呼号/船名）
证据源B：MarineCadastre 小时窗（2015-01-01，8072船，含 IMO/呼号/船名）
登记簿 = 两源合并生成的 (IMO/呼号 → {(MMSI, 船名, 来源, 年)})，跨年互查。
裁决对象 = 实验5的 3 例真 IMO 拆分候选 + 实验6的 4 例呼号拆分候选。
附带发现 = 跨年 MMSI 复用换名统计（零注入的"MMSI 会换主"直接证据）。
可选) 在线登记册（FleetMon）尝试，被墙/被拦则记录 blocked。
"""
from __future__ import annotations
import os, json, re, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")

A_CSV = os.path.join(RAW, "extracted3", "ais-quality-main", "ais-quality", "examples", "real-noaa.csv")
B_CSV = os.path.join(RAW, "mc_2015-01-01T00.csv")
OUT = os.path.join(HERE, "results", "adjudicate_registry.json")

CAND_IMO = {  # from Ex5: genuine (non-placeholder) IMO under 2+ MMSI
    "IMO7222152": [367501920, 367644590],
    "IMO8851273": [366896250, 367456860],
    "IMO9813644": [368006020, 368041830],
}
CAND_CALL = {  # from Ex6: callsign under 2+ MMSI
    "WCW6340": None, "WDF8768": None, "WSL5671": None, "WZX4855": None,
}



def norm_imo(x):
    if x is None or (isinstance(x, float) and x != x):
        return None
    s = str(x).strip().upper()
    if not s or s in ("NAN", "NONE", "NULL", "0"):
        return None
    if s.isdigit():
        return f"IMO{s}"
    return s if s.startswith("IMO") else f"IMO{s}"

def load_a():
    import pandas as pd
    df = pd.read_csv(A_CSV)
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={"call_sign": "CallSign"})
    if "imo" in df.columns:
        df["imo"] = df["imo"].map(norm_imo)
    df["src"], df["year"] = "noaa2024", 2024
    return df


def load_b():
    import pandas as pd
    df = pd.read_csv(B_CSV)
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns={"MMSI": "mmsi", "VesselName": "vessel_name",
                            "CallSign": "CallSign"})
    df["src"], df["year"] = "mc2015", 2015
    if "IMO" in df.columns:
        df["imo"] = df["IMO"].map(norm_imo)
    else:
        df["imo"] = None
    return df


def norm_name(s):
    if s is None: return None
    s = str(s).strip().upper()
    return s or None


def build_registry():
    import pandas as pd
    a, b = load_a(), load_b()
    cols_a = ["mmsi", "vessel_name", "imo", "CallSign", "src", "year"]
    cols_b = ["mmsi", "vessel_name", "imo", "CallSign", "src", "year"]
    a = a[cols_a].copy(); b = b[cols_b].copy()
    ab = pd.concat([a, b], ignore_index=True)
    ab["mmsi"] = ab["mmsi"].astype(int)
    imo_reg, call_reg, mmsi_reg = {}, {}, {}
    for _, r in ab.iterrows():
        name = norm_name(r["vessel_name"])
        rec = {"mmsi": int(r["mmsi"]), "name": name, "callsign": str(r["CallSign"]).strip() or None,
               "src": r["src"], "year": int(r["year"])}
        for key, reg in ((r["imo"], imo_reg), (r["CallSign"], call_reg)):
            if key and str(key).strip():
                reg.setdefault(str(key).strip(), []).append(rec)
        mmsi_reg.setdefault(int(r["mmsi"]), []).append(rec)
    return imo_reg, call_reg, mmsi_reg


def verdict_for(reg_rows):
    names = {r["name"] for r in reg_rows if r["name"]}
    mmsis = {r["mmsi"] for r in reg_rows}
    years = {r["year"] for r in reg_rows}
    if len(names) == 0:
        return "UNVERIFIABLE (no name)"
    if len(names) > 1:
        return "CONFLICT (不同船名共用同一标识)"
    if len(mmsis) > 1 and len(years) > 1:
        return "DUAL-MMSI 跨年一致（同船名多 MMSI，跨年成立）→ 合法双身份候选"
    if len(mmsis) > 1:
        return "DUAL-MMSI 同源一致（同船名多 MMSI，单年）→ 合法双身份候选"
    return "SINGLE (同 MMSI 同船名)"


def try_fleetmon(imo):
    """optional online check; returns name or None/blocked"""
    try:
        url = f"https://www.fleetmon.com/vessels/?imo={imo}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read(200_000).decode("utf-8", "ignore")
        m = re.search(r"<title>([^<]{3,80})</title>", html)
        if m and "fleetmon" not in m.group(1).lower():
            return m.group(1).strip()
        return None
    except Exception:
        return "BLOCKED"


def main():
    imo_reg, call_reg, mmsi_reg = build_registry()
    out = {"candidate_imo": [], "candidate_callsign": [], "mmsi_reuse_crosstime": None,
           "online": {}}

    for imo, mmsis in CAND_IMO.items():
        rows = imo_reg.get(imo, [])
        rows = [r for r in rows if r["mmsi"] in mmsis or True]
        verdict = verdict_for(rows) if rows else "UNVERIFIABLE (not in registry)"
        out["candidate_imo"].append({"imo": imo, "mmsi_candidates": mmsis,
                                     "records": rows, "verdict": verdict})
    for cs in CAND_CALL:
        rows = call_reg.get(cs, [])
        out["candidate_callsign"].append({"callsign": cs, "records": rows,
                                          "verdict": verdict_for(rows) if rows else "UNVERIFIABLE (not in registry)"})

    # cross-time MMSI reuse: same mmsi seen in both slices with different names
    reuse = []
    for mmsi, rows in mmsi_reg.items():
        years = {r["year"] for r in rows}
        if len(years) < 2:
            continue
        names = {r["name"] for r in rows if r["name"]}
        if len(names) > 0:
            chg = len(names) > 1
            reuse.append({"mmsi": mmsi, "names": sorted(names), "years": sorted(years),
                          "changed": chg})
    n_reuse = len(reuse)
    n_chg = sum(1 for r in reuse if r["changed"])
    out["mmsi_reuse_crosstime"] = {
        "mmsi_present_in_both_slices": n_reuse,
        "mmsi_with_changed_vessel_name": n_chg,
        "changed_frac": round(n_chg / n_reuse, 4) if n_reuse else None,
    }

    # online attempt (best-effort, one IMO)
    out["online"]["IMO7222152_fleetmon"] = try_fleetmon("7222152")

    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("== 候选裁决 ==")
    for c in out["candidate_imo"]:
        print(c["imo"], "->", c["verdict"])
        for r in c["records"]:
            print("   ", r["year"], r["src"], "mmsi", r["mmsi"], "name", r["name"], "cs", r["callsign"])
    for c in out["candidate_callsign"]:
        print(c["callsign"], "->", c["verdict"])
        for r in c["records"][:8]:
            print("   ", r["year"], r["src"], "mmsi", r["mmsi"], "name", r["name"])
    print("\n== 跨年 MMSI 复用 ==", out["mmsi_reuse_crosstime"])
    print("== 在线 ==", out["online"])
    print("saved", OUT)


if __name__ == "__main__":
    main()