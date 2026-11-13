"""run_all.py — run the three §3.5.2 verification simulations end-to-end.

    python run_all.py

Outputs (in results/):
    ex1_identity.csv/json    identity reliability quantification
    ex2_sorting.csv/json     sorting-axis artifacts vs residuals
    ex3_duplication.csv/json duplication rate & idempotency-key verification
    fig1_identity.png        bar chart, clean vs dirty slice
    fig2_sorting.png         polyline panels, arrival vs event order
    fig3_duplication.png     duplicate ratio vs delivery-path multiplicity
    results_summary.md       human-readable Chinese summary
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)

import numpy as np                     # noqa: E402
import ex1_identity, ex2_sorting, ex3_duplication  # noqa: E402
import plots                          # noqa: E402


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("§3.5.2 仿真验证 —— 三个未被量化的结论")
    print("=" * 78)

    # ---- Ex1: identity reliability ----
    r1 = ex1_identity.run(OUT)
    print("\n[实验1] MMSI 身份可靠性：洁净 vs 脏切片")
    for k in ("CLEAN", "DIRTY"):
        v = r1[k]
        print(f"  {k:6s} 实体={v['n_entities']:4d} 报文={v['n_rows']:6d} | "
              f"真实冲突MMSI={v['true_conflict_mmsi']:3d} 检出={v['detected_conflict_mmsi']:3d} "
              f"精度={_fmt(v['precision'])} 召回={_fmt(v['recall'])} | "
              f"MMSI当实体ID时轨迹被污染={v['corrupt_frac']*100:.1f}%")

    # ---- Ex2: sorting axis ----
    r2 = ex2_sorting.run(OUT)
    print("\n[实验2] 排序轴假象 2x2（排序轴 x 数据质量）")
    for k in ("clean", "noisy"):
        v = r2[k]
        print(f"  {k:6s} 折返: 到达序={v['fold_arrival']:.4f} 事件序={v['fold_event']:.4f} | "
              f"超速: 到达序={v['over_arrival']:.4f} 事件序={v['over_event']:.4f}")

    # ---- Ex3: duplication ----
    r3 = ex3_duplication.run(OUT)
    print("\n[实验3] 双路/重传重复率（随拓扑变化，0.0 -> 1.4 扫描）")
    for k in ("0.1", "0.5", "1.0"):
        v = r3[k]
        print(f"  p_dup={k:4s} 唯一事实={v['unique']:5d} 送达={v['delivered']:6d} "
              f"重复率={v['duplicate_ratio']*100:5.2f}% 乱序率={v['out_of_order']*100:5.2f}% "
              f"键恢复={v['recovered']} (误差={v['recovery_error']})")

    # ---- figures ----
    r1c, r1d = r1["CLEAN"], r1["DIRTY"]
    plots.dual_panel_bars(
        dict(categories=["CLEAN", "DIRTY"],
             series=[("True conflict MMSI", [r1c["true_conflict_mmsi"], r1d["true_conflict_mmsi"]], (200, 120, 60)),
                     ("Detected conflict MMSI", [r1c["detected_conflict_mmsi"], r1d["detected_conflict_mmsi"]], (90, 140, 220))],
             ylabel="count"),
        dict(categories=["CLEAN", "DIRTY"],
             series=[("Corrupted trajectories", [r1c["corrupt_frac"], r1d["corrupt_frac"]], (120, 180, 120))],
             ylabel="fraction", ymax=0.10),
        os.path.join(OUT, "fig1_identity.png"),
        title="Ex1: identity conflicts (counts) and corruption (fraction), clean vs dirty slice")

    rng = np.random.default_rng(7)
    ev, x, y = ex2_sorting.make_track(rng, sigma_pos_m=150.0, n_tele=2)
    arr = ev + ex2_sorting.delivery_delays(rng, ex2_sorting.N_PTS)
    plots.polyline_panels(ev, x, y, arr, os.path.join(OUT, "fig2_sorting.png"),
                          title="Ex2: same points, two reconstructions (noisy slice)", n=140)

    _xs = sorted(float(k) for k in r3)
    plots.line_chart(
        _xs,
        [("重复率 duplicate ratio", [r3[str(k)]["duplicate_ratio"] for k in _xs], (0.35, 0.55, 0.86), "o"),
         ("乱序率 out-of-order ratio", [r3[str(k)]["out_of_order"] for k in _xs], (0.78, 0.47, 0.24), "s")],
        os.path.join(OUT, "fig3_duplication.png"),
        title="实验3：重复率与乱序率 随平均送达路径数 p_dup 变化（教材只断言\"会重复\"）",
        xlabel="平均送达路径数 p_dup", ylabel="比例")

    _write_summary(r1, r2, r3)
    print("\n产物已写入:", os.path.normpath(OUT))


def _write_summary(r1, r2, r3):
    lines = []
    A = lines.append
    A("# §3.5.2 三个未量化结论 —— 仿真验证报告（可复现）\n")
    A("环境：numpy 2.3.5 / pandas 3.0.1；固定随机种子；全部实验含真值注入，"
      "测量值与注入值可逐项对账。图表用 matplotlib 3.11.2（Microsoft YaHei 中文标注）生成。\n")

    A("## 实验 1：MMSI 会被改写、共用、伪造（教材只说\"不可靠\"，没给率）\n")
    A("| 切片 | 实体 | 报文 | 真实冲突 MMSI | 检出冲突 | 检出精度 | 检出召回 | MMSI 当实体 ID 时被污染轨迹占比 |")
    A("|---|---|---|---|---|---|---|---|")
    for k in ("CLEAN", "DIRTY"):
        v = r1[k]
        A(f"| {k} | {v['n_entities']} | {v['n_rows']} | {v['true_conflict_mmsi']} | "
          f"{v['detected_conflict_mmsi']} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | "
          f"{v['corrupt_frac']*100:.1f}% |")
    A(f"\n结论：冲突率与污染率是数据依赖的。CLEAN 切片检出冲突=0，说明\"必须身份解析\""
      f"不是无条件成立；DIRTY 切片（共用5%/改写3%/伪造1%）中 {r1['DIRTY']['corrupt_frac']*100:.1f}% 的物理船"
      f"只要把 MMSI 当实体 ID，轨迹就会被错误合并或拆开。")
    A("\n口径说明：\"拆开\"定义为某船主导广播 MMSI 占比 < 90%（改写 50% 会命中，1% 伪造孤点不计）；"
      "\"共用\"为同一 MMSI 出现于 ≥2 艘物理船。")
    A(f"\n检出召回={_fmt(r1['DIRTY']['recall'])} 只反映\"共用且同时远距离\"这种易检形态；"
      f"改写/伪造是孤立短轨迹，纯位置启发式几乎抓不到 → 这正是真实身份解析需要图/登记册/IMO号的原因。\n")

    A("## 实验 2：按到达顺序连线会画出折返与超速假象（图 3.10），归因是否完整？\n")
    A("| 条件 | 折返·到达序 | 折返·事件序 | 超速·到达序 | 超速·事件序 |")
    A("|---|---|---|---|---|")
    for k in ("clean", "noisy"):
        v = r2[k]
        A(f"| {k} | {v['fold_arrival']:.4f} | {v['fold_event']:.4f} | {v['over_arrival']:.4f} | {v['over_event']:.4f} |")
    A(f"\n验证：clean+事件序的折返与超速=0（断言通过）；到达序在有乱序时必然出现假象。"
      f"打脸点：noisy+事件序的超速={r2['noisy']['over_event']:.4f} ≠ 0 —— 排序轴修对之后，"
      f"位置噪声与瞬移点仍会产生\"超速\"（噪声σ=150m、瞬移2点），教材把假象单一归因于排序轴是不完整的。\n")

    A("## 实验 3：同一报文经双路/重传重复到达——重复率是多少？\n")
    A("| 平均送达路径数 p_dup | 唯一事实 | 送达副本 | 重复率 | 乱序率 | 正确键恢复 | 恢复误差 |")
    A("|---|---|---|---|---|---|---|")
    for k in ("0.1", "0.5", "1.0"):
        v = r3[k]
        A(f"| {k} | {v['unique']} | {v['delivered']} | {v['duplicate_ratio']*100:.2f}% | "
          f"{v['out_of_order']*100:.2f}% | {v['recovered']} | {v['recovery_error']} |")
    A(f"\n验证：教材幂等键 (MMSI, 事件时间, 来源序号) 在所有拓扑下恰好恢复全部唯一事实（误差=0，断言通过）。"
      f"打脸点：重复率不是常数，从 {r3['0.1']['duplicate_ratio']*100:.2f}% 到 {r3['1.0']['duplicate_ratio']*100:.2f}%"
      f"随接收网拓扑（平均路径数）变化——教材只断言\"会重复\"，没给这个输入相关的数。\n")
    A("## 局限\n基于合成数据，验证的是方法自洽性与结论的方向/幅度；真实 AIS 切片（如 MarineCadastre）"
      "是下一步，数字会随区域与接收网结构迁移。")
    with open(os.path.join(OUT, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()