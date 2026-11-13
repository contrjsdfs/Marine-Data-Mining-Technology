# Marine Data Mining Technology

> 教材《海洋大数据》§3.5.2 三条未量化结论（MMSI 身份不可靠、排序轴造成折返/超速、双路/重传重复）的量化验证代码与实验结果。

- **GitHub**：https://github.com/contrjsdfs/Marine-Data-Mining-Technology
- **课程**：海洋数据挖掘 · 李静香（22634029）
- **AI 对话记录**：[`AI记录.md`](AI记录.md)（本次课题中用户与 AI 助手的完整对话）

## 一、内容概述

把 §3.5.2 的三条定性结论重构为**带成立条件的定量命题**，并以「真值注入 → 验证 → 训练 → 真实数据证据」四段式闭环给出可复现、可对账的量化证据链。方法沿「规则 → 算法（CGER）→ 学习模型 → 负结果边界」演进，并在三片真实 AIS 切片上交叉验证。

## 二、目录结构

```
.
├── run_all.py              # 实验1–3（合成）：身份可靠性/排序轴/重复率 + 出图
├── sim_common.py           # 公共工具
├── ex1_identity.py         # 实验1 MMSI 身份可靠性
├── ex2_sorting.py          # 实验2 排序轴 2×2 归因
├── ex3_duplication.py      # 实验3 重复率与幂等键
├── ex4_realdata.py         # 实验4 真实单船双层验证（半合成注入+登记册交叉）
├── ex5_realmulti.py        # 实验5 真实多船快照（NOAA 2024）
├── ex6_mc_traj.py          # 实验6 真实多船轨迹（MarineCadastre 2015 小时窗）
├── algo_cger.py            # CGER 速度约束连续图实体解析
├── model_link.py           # 学习型链接预测（注入即训练）
├── model_gnn.py            # GNN 负结果 + α 消融
├── model_ggnn.py           # 边门控 GNN（机制闭环）
├── adjudicate_registry.py  # 登记册交叉裁决（7/7 CONFLICT）
├── fetch_pubdata.py        # 公开数据检索/下载工具
├── fetch_mc.py             # MarineCadastre 下载+小时窗切片工具
├── fig4_realdata.py / fig5_realmulti.py / fig6_mc_traj.py   # 出图脚本
├── plots.py                # matplotlib 绘图工具
├── results/                # 全部数字与图（CSV/JSON/PNG/汇总）
├── data/raw/               # 三份真实数据切片（见下）
└── AI记录.md                # 用户与 AI 的完整对话记录
```

## 三、数据来源与许可

| 文件 | 来源 | 许可/说明 |
|---|---|---|
| `data/raw/mc_2015-01-01T00.csv` | MarineCadastre AIS_2015_01_01（2015-01-01 00:00–01:00 UTC，260,001 行 / 8,072 艘） | 美国沿海公开数据 |
| `data/raw/extracted2/AIS_Data-main/mmsi_210253000.csv` | GitHub 公开演示数据（单船，Kattegat，1657 点） | 身份字段已脱敏 |
| `data/raw/extracted3/.../real-noaa.csv` | USCG/BOEM/NOAA Nationwide AIS（2024-01-01 前 5000 条，4614 艘） | CC0（见 provenance.json） |

> 仓库未包含 241 MB 的 MarineCadastre 原始压缩包；可用 `python fetch_mc.py 2015 01` 重新生成小时窗切片。

## 四、快速开始

```bash
pip install numpy pandas matplotlib torch
python run_all.py              # 实验1–3（合成）+ 出图
python ex4_realdata.py         # 真实单船双层验证
python ex5_realmulti.py        # 真实多船快照
python ex6_mc_traj.py          # 真实多船轨迹
python algo_cger.py            # CGER 算法评测
python model_link.py           # 学习链接预测
python model_gnn.py            # GNN 负结果
python model_ggnn.py           # 边门控 GNN
python adjudicate_registry.py  # 登记册裁决
```

脚本使用相对路径（`data/raw/...`、`results/`），在仓库根目录直接运行即可；所有实验固定随机种子，关键数值内建 `assert` 自动复核。

## 五、核心结果（摘要）

| 实验 | 关键数字 |
|---|---|
| 实验1 身份可靠性 | CLEAN 0 冲突/0 污染；DIRTY 冲突 7/检出 7（P=R=1.00），污染轨迹 7.3% |
| 实验2 排序轴 | clean+事件序 = 0；noisy+事件序残余超速 16.7% |
| 实验3 重复率 | 8.95% → 50.56%（随拓扑）；乱序率 ≈31%；幂等键 6000/6000 零误差 |
| 实验4 真实单船 | 位置启发式抓共用 1.00、漏改写 0.00；登记册交叉两类 1.00/1.00 |
| 实验5 多船快照 | 与第三方审计收敛（MMSI 367373000）；IMO 缺失 57.3% |
| 实验6 多船轨迹 | 5/8072 瞬移坏点；非标准 MMSI 100000000；>50kn 占 0.043% |
| 算法对比 | CGER 实体 F1 0.961→0.991；LR 共享 P/R 1.00/1.00（ECE≈0）；GNN 负结果闭环 |
| 登记册裁决 | 7/7 CONFLICT；跨年 MMSI 换名率 11.06% |

详细数字与图见 [`results/`](results/)。

## 六、说明

- 数据仅用于学术复现；引用与许可见上表。
- AI 协作过程见 [`AI记录.md`](AI记录.md)。
