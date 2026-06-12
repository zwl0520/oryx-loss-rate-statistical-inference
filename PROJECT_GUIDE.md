# 俄乌装备损失率统计分析 —— 项目完全指南

> 数理统计课程大作业 · 选题 T1 · 答辩准备用

---

## 目录

1. [项目一句话概括](#1-项目一句话概括)
2. [核心统计问题](#2-核心统计问题)
3. [数据来源全景图](#3-数据来源全景图)
4. [数据流：从原始文件到最终结果](#4-数据流从原始文件到最终结果)
5. [代码模块详解](#5-代码模块详解)
6. [统计学方法深入讲解](#6-统计学方法深入讲解)
7. [从零运行完整流程](#7-从零运行完整流程)
8. [关键结果速查](#8-关键结果速查)
9. [已发现并修复的问题](#9-已发现并修复的问题)
10. [答辩 Q&A 准备](#10-答辩-qa-准备)
11. [文件清单](#11-文件清单)

---

## 1. 项目一句话概括

**利用 Oryx 平台的俄乌双方影像验证装备损失数据，使用 Poisson 分布模型、三种置信区间、Block Bootstrap 和 E-test 两样本检验，系统性地估计双方装备损失率并检验其差异是否统计显著。**

核心答案：**是统计显著的（p < 0.0001），俄方日均损失约 15.11 件，乌方约 7.31 件，率比约 2.07:1。**

---

## 2. 核心统计问题

```
┌─────────────────────────────────────────────────────────────┐
│                    三个核心统计问题                          │
├─────────────────┬──────────────────┬────────────────────────┤
│    估计问题      │    比较问题       │    偏差问题             │
│  λ 的准确估计？  │ 双方差异显著吗？  │ 官方数据夸大多少？       │
├─────────────────┼──────────────────┼────────────────────────┤
│ MLE 点估计       │ E-test 精确检验   │ 声称/验证比率           │
│ Wald/Score/Exact│ 分层比较(Bonf.)  │ 观测灵敏度分析           │
│ Block Bootstrap  │ LRT 似然比检验   │ 偏误方向讨论             │
│ BCa 校正         │ 效应量 (率比)    │                         │
└─────────────────┴──────────────────┴────────────────────────┘
```

### 关键统计概念（答辩中可能会被问到）

| 概念 | 一句话解释 | 本文中怎么用的 |
|------|-----------|--------------|
| Poisson 分布 | 描述"单位时间内随机事件发生次数"的分布 | 对每日装备损失数建模 |
| MLE（极大似然估计） | 选使观测数据出现概率最大的参数值 | $\hat{\lambda} = \bar{Y}$（样本均值） |
| 置信区间 (CI) | "如果重复抽样100次，约95个CI包含真值" | 量化 λ 估计的不确定性 |
| Block Bootstrap | 把时间序列切成块再抽样，保留时间依赖 | 纠正 IID Bootstrap 低估 SE 的问题 |
| BCa 校正 | 校正 Bootstrap 分布的中心偏倚和偏度 | 让 Bootstrap CI 更准确 |
| E-test | 基于条件二项分布的精确检验 | 检验双方 λ 是否相等 |
| Bonferroni 校正 | 多重比较时，$\alpha$ 除以比较次数 | 控制10类装备检验的整体错误率 |
| Delta 方法 | 用一阶 Taylor 展开近似非线性函数的方差 | 计算率比 $\rho = \lambda_1/\lambda_2$ 的 SE |

---

## 3. 数据来源全景图

### 三个数据来源

```
┌──────────────────────────────────────────────────────────────────┐
│                        数据来源架构                              │
├───────────────┬──────────────────┬───────────────────────────────┤
│ 数据源         │ 内容              │ 规模                          │
├───────────────┼──────────────────┼───────────────────────────────┤
│ Oryx 双方      │ 俄乌双方每日累计   │ 1,559天 (2022.2.25-2026.6.4) │
│ (leedrake5    │ + 装备类型细分     │ RU: 23,556件  UA: 11,397件   │
│  Google Sheet)│ + 损毁状态细分     │ 每日增量由 Change 列提供       │
├───────────────┼──────────────────┼───────────────────────────────┤
│ WarSpotting   │ 俄方装备损失逐条   │ 22,970条 (2022.2.24-2026.6.3) │
│ (Kaggle)      │ 含经纬度坐标       │ 18种装备类型, 4种损毁状态      │
│               │ 含具体型号         │ 61.2% 记录含坐标               │
├───────────────┼──────────────────┼───────────────────────────────┤
│ 乌克兰总参     │ 乌克兰官方公布的   │ 1,557天 (2022.2.25-2026.5.31) │
│ (Kaggle)      │ 俄军装备损失       │ 10种装备类型                   │
│               │ 每日累计→日度转换  │ 存在系统性夸大                 │
└───────────────┴──────────────────┴───────────────────────────────┘
```

### Oryx vs WarSpotting 的区别

| | Oryx | WarSpotting |
|---|------|-------------|
| 覆盖 | 俄乌**双方** | **仅俄方** |
| 粒度 | 每日累计（带 Change 列） | 逐条记录（带日期） |
| 坐标 | 无 | 有（61.2% 记录含经纬度） |
| 验证方式 | 全球志愿者逐件核对影像 | 自动化+人工验证 |
| 总俄方记录 | 23,556 件（比 WS 多 ~586 件） | 22,970 件 |
| 本项目主要用于 | 双边比较 | 时序分析、空间分析、Bootstrap |

### 为什么 Oryx 数据不等于"真实损失"？

```
真实损失总数（未知）
    │
    │  被观测概率 p (与作战区域、装备类型、损毁状态等有关)
    │
    ▼
影像验证损失（Oryx 记录）  ← 我们有这个

偏误方向：
- 俄军在乌控区的损失 → 更容易被拍到 → 被高观测
- 乌军在俄占区的损失 → 更难被拍到   → 被低观测
→ 真实 RU/UA 比率可能 < 观测到的 2.07:1
```

---

## 4. 数据流：从原始文件到最终结果

```
                    ┌──────────────────────────┐
                    │  oryx_both_sides.csv      │ ← leedrake5 Google Sheet
                    │  (累计值 + Change 列)     │
                    └──────────┬───────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │     both_sides_analysis.py     │  ← ★ 核心分析模块
              │  提取 Change 列 → 日度序列     │
              │  双方 MLE + CI + E-test        │
              │  产出: daily_oryx_both_sides   │
              └────────────┬───────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │ visualization │ │ 论文     │ │ README       │
    │ _both_sides  │ │ T1_论文  │ │ 结果表格     │
    │ (图10-13)    │ │          │ │              │
    └──────────────┘ └──────────┘ └──────────────┘


┌──────────────────────────┐
│  warspotting_losses.csv  │ ← Kaggle (zsoltlazar)
│  (俄方逐条记录)           │
└──────────┬───────────────┘
           │
           ├──→ preprocess.py ──→ daily_warspotting.csv (N=1,561天)
           │                      daily_by_type_ws.csv
           │                      daily_by_status_ws.csv
           │
           ├──→ point_estimation.py ──→ MLE 结果 (总体+分层)
           │      加载 daily_warspotting, daily_by_type
           │
           ├──→ confidence_intervals.py ──→ Wald/Score/Exact CI
           │      + Monte Carlo 覆盖率模拟
           │
           ├──→ bootstrap.py ──→ Block Bootstrap + BCa
           │      块长度 b=11, B=2000
           │
           ├──→ two_sample_test.py ──→ 阶段/类型间 E-test
           │
           ├──→ spatial_analysis.py ──→ 空间分布 (宏区域/网格)
           │
           ├──→ weekly_bootstrap.py ──→ 周度 Bootstrap (B=10000)
           │
           ├──→ observation_sensitivity.py ──→ 观测概率灵敏度
           │      加载 oryx_losses_combined.csv
           │
           └──→ visualization.py ──→ 图01-09 (单边)
                visualization_both_sides.py ──→ 图10-13 (双边)
                visualization_new.py ──→ 图14-16 (空间+灵敏度+Bootstrap)


┌──────────────────────────┐
│ russia_losses_equipment  │ ← Kaggle (piterfm)
│ .csv (乌官方战报, 累计)  │
└──────────┬───────────────┘
           │
           └──→ preprocess.py ──→ daily_ua_claims.csv
                累计→日度转换 (注意: 数据按日期降序排列)
                      ↓
                point_estimation.py ──→ 声称 vs 验证对比
                two_sample_test.py  ──→ 声称 vs 验证 E-test
                visualization.py    ──→ 图08 (声称/验证对比)
```

### 关键数据文件关系

```
oryx_both_sides.csv (累计, 有 Change 列)
    │
    └── both_sides_analysis.py
        └── daily_oryx_both_sides.csv  ← 日度值，用于所有双边分析

oryx_losses_combined.csv (item_count 加权, 来自参考仓库)
    │
    └── observation_sensitivity.py  ← 灵敏度分析

warspotting_losses.csv (逐条记录)
    │
    └── preprocess.py
        ├── daily_warspotting.csv      ← 日度俄方损失
        ├── daily_by_type_ws.csv       ← 按装备类型分层
        └── daily_by_status_ws.csv     ← 按损毁状态分层
```

---

## 5. 代码模块详解

### 模块执行顺序与依赖关系

```
preprocess.py           ← 必须最先运行 (生成所有中间数据)
    ↓
point_estimation.py     ← 依赖 daily_warspotting, daily_ua_claims
confidence_intervals.py ← 依赖 daily_warspotting
bootstrap.py            ← 依赖 daily_warspotting
two_sample_test.py      ← 依赖 daily_warspotting, daily_ua_claims
    ↓
both_sides_analysis.py  ← 独立于上述模块 (使用 oryx_both_sides.csv)
spatial_analysis.py     ← 独立 (使用 warspotting_losses.csv)
observation_sensitivity.py ← 独立 (使用 oryx_losses_combined.csv)
weekly_bootstrap.py     ← 依赖 daily_warspotting
    ↓
visualization.py        ← 依赖上述模块的所有输出
visualization_both_sides.py ← 依赖 both_sides_analysis.py 的输出
visualization_new.py    ← 依赖 spatial/weekly/sensitivity 输出
```

### 每个模块的详细说明

#### `preprocess.py` —— 数据预处理（必须最先运行）

**输入**: `warspotting_losses.csv`, `russia_losses_equipment.csv`

**核心逻辑**:
1. 加载 WarSpotting 数据 → 按日聚合 → `daily_warspotting.csv`
2. 按装备类型分组 → `daily_by_type_ws.csv`
3. 按损毁状态分组 → `daily_by_status_ws.csv`
4. 加载乌克兰官方战报 → 累计→日度转换 → `daily_ua_claims.csv`
5. 定义战争9阶段，为每条记录添加阶段标签

**输出**: 6个CSV文件

**关键修复 (2026-06-12)**: 原始代码在累计→日度转换时错误地反转了已排序的数据，导致所有日度值被归零（除最后一天外）。已修复为直接对升序累计值差分。

#### `point_estimation.py` —— MLE 点估计

**输入**: `daily_warspotting.csv`, `daily_by_type_ws.csv`, `daily_ua_claims.csv`

**核心逻辑**:
- `poisson_mle(data)`: 计算 $\hat{\lambda} = \bar{Y}$、$SE = \sqrt{\hat{\lambda}/n}$、对数似然
- 总体 MLE → 战争阶段分层 MLE（9个阶段）→ LR 检验阶段间差异
- 装备类型分层 MLE → 声称 vs 验证对比

**输出**: `mle_results.csv`

#### `confidence_intervals.py` —— 三种 CI 构造 + Monte Carlo 覆盖率

**输入**: `daily_warspotting.csv`, `daily_by_type_ws.csv`

**核心逻辑**:
- `wald_ci(data)`: $\hat{\lambda} \pm z_{\alpha/2} \sqrt{\hat{\lambda}/n}$
- `score_ci(data)`: 解方程 $(\hat{\lambda}-\lambda)^2/(\lambda/n) = z^2$ → 自动非负下限
- `exact_ci(data)`: 基于 $\chi^2$ 分布精确关系 → 保证覆盖率 ≥ 95%
- `simulate_coverage()`: Monte Carlo 10,000次模拟，评估各方法实际覆盖率
  - $\lambda \in \{0.5, 1, 2, 5, 10, 15, 20, 50\}$，固定 n=30 天

**输出**: `ci_comparison.csv`, `ci_coverage_simulation.csv`

**核心发现**: $\lambda=0.5$ 时 Wald CI 覆盖率仅 91.9%，Exact CI 始终 ≥95%

#### `bootstrap.py` —— Block Bootstrap + BCa

**输入**: `daily_warspotting.csv`

**核心逻辑**:
1. `choose_block_length(data)`: ACF 衰减法 → 找到 $|ACF| < 2/\sqrt{n}$ 的第一个 lag → b=11
2. `block_bootstrap(data, B, block_len)`: Moving Block Bootstrap
   - 将数据分成 n-b+1 个长度为 b 的重叠块
   - 有放回抽取 k = ⌈n/b⌉ 个块 → 拼接 → 截断到 n
   - 重复 B 次，每次计算均值
3. `bca_ci(data)`: 
   - 偏差校正: $\hat{z}_0 = \Phi^{-1}(\#\{\hat{\theta}^* < \hat{\theta}\} / B)$
   - 加速常数: $\hat{a}$ 通过 Jackknife 估计
   - 修正分位点: $\alpha_1 = \Phi(\hat{z}_0 + \frac{\hat{z}_0 + z_\alpha}{1 - \hat{a}(\hat{z}_0 + z_\alpha)})$
4. IID vs Block 对比: 不同块长度下的 SE 变化

**输出**: `bootstrap_results.csv`, `bootstrap_distributions.csv`

**核心发现**: Block Bootstrap SE (0.99) ≈ IID (0.42) 的 2.4 倍

#### `two_sample_test.py` —— 两独立 Poisson 条件检验

**核心逻辑**:
- `poisson_e_test(y1, n1, y2, n2)`: 条件二项分布精确检验
  - 在 $H_0$ 下: $Y_1 | Y_1+Y_2 \sim Binomial(S, p)$ 其中 $p = n_1/(n_1+n_2)$
  - 双侧 p = 2 × min(P_upper, P_lower)
- `poisson_lrt(data1, data2)`: 似然比检验（渐近 $\chi^2(1)$）
- 多层比较 + Bonferroni 校正 ($\alpha/m$)

**输出**: `phase_comparisons.csv`, `type_comparisons.csv`

#### `both_sides_analysis.py` —— 双方 Oryx 数据整合（核心模块）

**输入**: `oryx_both_sides.csv`

**核心逻辑**:
1. 提取 `Change` / `Change.1` 列作为俄方/乌方日度增量（维护者已正确计算）
2. 子类别（无 Change 列）通过差分累计列获得，clamp 负值
3. 双方总体 MLE + Exact CI
4. 率比 + Delta 方法 CI
5. 双方 E-test（总体 + 按状态 + 按装备类型）
6. 月度聚合趋势

**输出**: `daily_oryx_both_sides.csv`, `monthly_oryx_both_sides.csv`, `both_sides_summary.csv`

**关键审计**: Change 列求和 == 最终累计值 ✓

#### `spatial_analysis.py` —— 空间分析

**输入**: `warspotting_losses.csv`

**分析维度**:
- 坐标覆盖质量（61.2% 记录含坐标）
- 宏区域分类（East/North/South/Central/West, 基于经纬度阈值）
- 1°×1° 网格热点识别
- 位置地点 Top-N 排序
- 时间×空间动态（季度×区域交叉）

**核心发现**: East（顿涅茨克）占 44.3%，最热网格 (48°N, 37°E) 独占 21.5%

#### `observation_sensitivity.py` —— 观测概率偏误校正

**核心逻辑**:
- **偏误校正公式**: $\rho_{true} = \rho_{obs} \times (p_{UA} / p_{RU})$（见 §6.7 推导）
- 网格搜索: $p_{RU}, p_{UA} \in \{0.3, 0.4, ..., 1.0\}$ (64种组合)
- **声称数据约束**: 利用官方声称数据给出观测概率下界（坦克 $p_{RU} \geq 0.34$）
- **Bootstrap 偏误校正**: 从 Beta 先验中抽取 $p_{RU}, p_{UA}$，嵌入 Block Bootstrap (B=5000)
  - $p_{RU} \sim Beta(7, 3)$, 均值 0.70
  - $p_{UA} \sim Beta(5.5, 4.5)$, 均值 0.55
- 寻找率比 ≤ 1 的观测概率组合（逆转边界）
- 报告 $P(\rho_{corrected} > 1)$ 作为结论稳健性的量化指标

**核心发现**: 逆转需 $p_{UA}/p_{RU} \leq 0.485$；偏误校正后率比中位数 1.63；$P(\rho_{corrected} > 1) = 88.1\%$

#### `weekly_bootstrap.py` —— 周度 Block Bootstrap 验证

**改进点**:
- 周度聚合（7天一块更自然）
- B=10,000（vs 日度 B=2,000）
- 完整 Jackknife + BCa
- 分装备类型 Bootstrap

**核心发现**: 周度 SE(0.82) 介于日度 IID(0.42) 和日度 Block(0.99) 之间

#### 可视化模块

| 图号 | 模块 | 内容 |
|------|------|------|
| 01 | visualization.py | 每日损失时序（柱状+30天MA+累计） |
| 02 | visualization.py | 战争阶段箱线图+均值条形图 |
| 03 | visualization.py | CI 森林图（Wald vs Score vs Exact） |
| 04 | visualization.py | Bootstrap 分布对比（IID vs Block） |
| 05 | visualization.py | 装备类型构成（堆叠面积+百分比） |
| 06 | visualization.py | Poisson Q-Q 图（4个阶段） |
| 07 | visualization.py | CI 宽度比较柱状图 |
| 08 | visualization.py | 声称 vs 验证月度对比 |
| 09 | visualization.py | 损失状态饼图+类型柱状图 |
| 10 | visualization_both_sides.py | 双方时序对比（MA+累计+月度比率） |
| 11 | visualization_both_sides.py | 双方 CI 森林图（并列） |
| 12 | visualization_both_sides.py | 双方装备类型对比（绝对值+比率） |
| 13 | visualization_both_sides.py | 双方损毁状态饼图（并列） |
| 14 | visualization_new.py | 空间分布（宏区域+Top地点） |
| 15 | visualization_new.py | 观测概率灵敏度热力图 |
| 16 | visualization_new.py | 周度 Bootstrap 分布+分类型 SE |

---

## 6. 统计学方法深入讲解

### 6.1 为什么用 Poisson 分布？

**数据特征**: 每日装备损失数是**计数数据**（0, 1, 2, ...），非负整数，不能用正态分布直接拟合。

Poisson 分布的前提：
- 事件独立发生（每件装备的损毁是独立事件）
- 发生率恒定（在给定阶段内）
- 均值 = 方差（等离散）

**Poisson PMF**: $P(Y=k) = \frac{e^{-\lambda}\lambda^k}{k!}$

### 6.2 MLE 推导过程

似然函数: $L(\lambda) = \prod_{t=1}^{n} \frac{e^{-\lambda}\lambda^{Y_t}}{Y_t!}$

对数似然: $\ell(\lambda) = -n\lambda + (\sum Y_t)\ln\lambda - \sum\ln(Y_t!)$

一阶条件: $\frac{\partial\ell}{\partial\lambda} = -n + \frac{\sum Y_t}{\lambda} = 0$

解得: $\hat{\lambda} = \frac{1}{n}\sum_{t=1}^{n} Y_t = \bar{Y}$

**Fisher 信息量**: $I(\lambda) = -E[\frac{\partial^2\ell}{\partial\lambda^2}] = \frac{n}{\lambda}$

**渐近方差**: $\text{Var}(\hat{\lambda}) \approx I(\hat{\lambda})^{-1} = \frac{\hat{\lambda}}{n}$

### 6.3 为什么需要 Block Bootstrap？

**问题**: 装备损失数据存在**强自相关**（lag-1 ACF = 0.589）。

**IID Bootstrap 的问题**: 假设观测独立，对自相关数据进行 IID 重抽样 → 样本过于"均匀" → **低估方差**。

**实证**: 本文 IID Bootstrap SE = 0.42，Block Bootstrap SE = 0.99。

**Block Bootstrap 原理**:
```
原始序列: [Y₁, Y₂, Y₃, Y₄, Y₅, Y₆, Y₇, Y₈, Y₉, Y₁₀]  (n=10, b=3)

块 (n-b+1=8个):
  Block 1: [Y₁, Y₂, Y₃]
  Block 2: [Y₂, Y₃, Y₄]
  ...
  Block 8: [Y₈, Y₉, Y₁₀]

Bootstrap 样本 (k = ⌈10/3⌉ = 4个块):
  随机选块: Block5, Block2, Block8, Block3
  拼接: [Y₅,Y₆,Y₇, Y₂,Y₃,Y₄, Y₈,Y₉,Y₁₀, Y₃,Y₄] → 取前10个
```

**块长度选择**: 找到 $|ACF(lag)| < 2/\sqrt{n}$ 的第一个滞后阶数 → b=11

### 6.4 BCa 校正的统计直觉

- **偏差校正 $\hat{z}_0$**: 如果一半的 Bootstrap 估计值小于原始估计值，则 $\hat{z}_0 = 0$（无偏）。如果 >50% 的 Bootstrap 值小于原始估计，则 $\hat{z}_0 > 0$（正偏），CI 向上移动。

- **加速 $\hat{a}$**: 衡量方差的非恒定性。通过 Jackknife 估计——每次去掉一个观测，看估计值如何变化。变化越大 → $\hat{a}$ 越大 → CI 加宽。

- 本文结果: $\hat{z}_0 = 0.221$（轻微正偏），$\hat{a} = 0.022$（偏度很小）→ BCa CI 略向右偏移。

### 6.5 Exact CI 的数学原理（Garwood, 1936）

核心关系: 设 $Y = \sum_{t=1}^{n} Y_t$，则 $Y \sim Poisson(n\lambda)$

利用 Poisson 与 $\chi^2$ 分布的关系:
$$P(Y \geq y) = P(\chi^2_{2y} \leq 2n\lambda)$$
$$P(Y \leq y) = P(\chi^2_{2(y+1)} \geq 2n\lambda)$$

由此得到精确置信区间:
$$\lambda_L = \frac{1}{2n}\chi^2_{2Y, \alpha/2}, \quad \lambda_U = \frac{1}{2n}\chi^2_{2(Y+1), 1-\alpha/2}$$

**为什么是"精确"的**: 不依赖大样本近似，从精确分布关系推导，保证覆盖率 ≥ 名义水平。

### 6.6 E-test 的条件分布推导

检验 $H_0: \lambda_1 = \lambda_2$ 时，关键步骤：

1. 在 $H_0$ 下，$Y_1 \sim Poisson(n_1\lambda)$, $Y_2 \sim Poisson(n_2\lambda)$
2. 总计数 $S = Y_1 + Y_2 \sim Poisson((n_1+n_2)\lambda)$
3. 条件分布: $P(Y_1 = y_1 | S = s) = \frac{P(Y_1=y_1) \cdot P(Y_2=s-y_1)}{P(S=s)}$
4. 代入 Poisson PMF 化简 → $Binomial(s, \frac{n_1}{n_1+n_2})$

**精妙之处**: 条件分布不依赖未知的 $\lambda$！因此可以直接计算精确 p 值。

### 6.7 Delta 方法（用于率比 CI）

对于率比 $\rho = \lambda_{RU}/\lambda_{UA}$:

$$\text{Var}(\hat{\rho}) \approx \left(\frac{\partial\rho}{\partial\lambda_{RU}}\right)^2 \text{Var}(\hat{\lambda}_{RU}) + \left(\frac{\partial\rho}{\partial\lambda_{UA}}\right)^2 \text{Var}(\hat{\lambda}_{UA})$$

$$= \rho^2 \cdot \left(\frac{1}{\lambda_{RU} \cdot n} + \frac{1}{\lambda_{UA} \cdot n}\right)$$

因此: $\text{SE}(\hat{\rho}) = \hat{\rho} \cdot \sqrt{\frac{1}{\hat{\lambda}_{RU} \cdot n} + \frac{1}{\hat{\lambda}_{UA} \cdot n}}$

### 6.8 观测概率偏误校正公式推导

设影像验证损失 $Y_{obs}$ 与真实损失 $Y_{true}$ 满足 $Y_{obs} = p \cdot Y_{true}$，其中 $p \in (0,1]$ 为被观测概率。

对于双方率比:
$$\rho_{true} = \frac{\lambda_{RU}^{true}}{\lambda_{UA}^{true}} = \frac{\lambda_{RU}^{obs} / p_{RU}}{\lambda_{UA}^{obs} / p_{UA}} = \rho_{obs} \cdot \frac{p_{UA}}{p_{RU}}$$

**关键性质**:
- 当 $p_{RU} = p_{UA}$ 时，$\rho_{true} = \rho_{obs}$（等观测概率下率比无偏）
- 当 $p_{UA} < p_{RU}$ 时，$\rho_{true} < \rho_{obs}$（真实率比被高估）
- 逆转条件: $p_{UA}/p_{RU} < 1/\rho_{obs} \approx 0.485$

**声称数据约束**:
$$p_{RU} = \frac{\text{验证数}}{\text{真实数}} \geq \frac{\text{验证数}}{\text{声称数}}$$
（假设官方声称不会低估敌方损失）

**Bootstrap 偏误校正**:
1. Block Bootstrap 重抽样双方日度数据 → 获得 $\rho_{boot}$ 分布
2. 每次重抽样中从先验抽取 $p_{RU} \sim Beta(7,3)$, $p_{UA} \sim Beta(5.5,4.5)$
3. 计算 $\rho_{corrected} = \rho_{boot} \times (p_{UA} / p_{RU})$
4. 从 $\rho_{corrected}$ 分布获取校正 CI 和 $P(\rho_{corrected} > 1)$

---

## 7. 从零运行完整流程

### 环境准备

```bash
# 安装依赖
pip install pandas numpy scipy matplotlib

# 可选（如果运行 markdown 转换）
pip install markdown

# 可选（如果生成 PPTX 模板）
pip install python-pptx
```

### 完整执行步骤

```bash
# ===== 步骤 1: 数据预处理（必须首先运行）=====
python code/preprocess.py
# 产出: data/daily_warspotting.csv, daily_by_type_ws.csv,
#       daily_by_status_ws.csv, daily_ua_claims.csv, phase_summary.csv

# ===== 步骤 2: 统计推断（可并行运行）=====
# 2a. MLE 点估计
python code/point_estimation.py

# 2b. Poisson 置信区间 + Monte Carlo 覆盖率
python code/confidence_intervals.py

# 2c. 日度 Block Bootstrap + BCa
python code/bootstrap.py

# 2d. 两样本 Poisson 条件检验
python code/two_sample_test.py

# ===== 步骤 3: 双边分析与高级分析（可并行）=====
# 3a. 双方 Oryx 数据整合与双边比较 ★核心模块★
python code/both_sides_analysis.py

# 3b. 空间分析
python code/spatial_analysis.py

# 3c. 观测概率灵敏度分析
python code/observation_sensitivity.py

# 3d. 周度 Block Bootstrap (B=10000)
python code/weekly_bootstrap.py

# ===== 步骤 4: 可视化 =====
python code/visualization.py              # 图01-09
python code/visualization_both_sides.py   # 图10-13
python code/visualization_new.py          # 图14-16

# ===== 步骤 5: 生成论文 PDF（可选）=====
cd paper
pandoc T1_论文_pdf.md -o T1_论文.pdf --pdf-engine=xelatex -V CJKmainfont="Microsoft YaHei"
```

### 快速验证脚本

```bash
# 验证数据完整性
python -c "
import pandas as pd
daily = pd.read_csv('data/daily_oryx_both_sides.csv', parse_dates=['Date'])
print(f'Oryx双方日度: {len(daily)}天, RU={daily.Russia_Total.sum()}, UA={daily.Ukraine_Total.sum()}')
ws = pd.read_csv('data/daily_warspotting.csv', parse_dates=['date'])
print(f'WarSpotting日度: {len(ws)}天, Total={ws.total_losses.sum()}')
"
```

---

## 8. 关键结果速查

### 双方总体比较

| 指标 | 俄方 (RU) | 乌方 (UA) | 比率/检验 |
|------|----------|----------|----------|
| 累计验证损失 | 23,556 件 | 11,397 件 | 2.07 : 1 |
| 日均损失率 $\hat{\lambda}$ | 15.11 件/天 | 7.31 件/天 | 2.07 |
| SE | 0.098 | 0.068 | — |
| 95% Exact CI | [14.92, 15.30] | [7.18, 7.45] | — |
| 率比 $\hat{\rho}$ | — | — | 2.07 |
| 率比 95% Wald CI | — | — | [2.02, 2.11] |
| E-test p-value | — | — | < 0.0001 |

### 置信区间方法比较（Monte Carlo, n=30天, N=10,000次）

| λ 真值 | Wald | Score | Exact |
|--------|------|-------|-------|
| 0.5 | **91.9%** ❌ | 95.1% | **96.6%** |
| 1.0 | 92.9% ⚠️ | 94.6% | 95.7% |
| 5.0 | 95.5% | 95.6% | 95.6% |
| 15.0 | 94.9% | 94.9% | 95.3% |

### Block Bootstrap 关键结果

| 方法 | SE | 95% CI |
|------|-----|--------|
| IID Bootstrap | 0.42 | [13.91, 15.54] |
| Block Bootstrap (b=11) | 0.99 | [12.82, 16.59] |
| BCa Block Bootstrap | — | [13.18, 17.29] |

### 装备类型差异（按 RU/UA 比率排序）

| 类型 | RU | UA | 比率 | 谁损失更多 |
|------|-----|-----|------|----------|
| AFV | 2,682 | 597 | 4.49 | 俄方 |
| IFV | 6,702 | 1,641 | 4.08 | 俄方 |
| Vehicles | 4,949 | 1,456 | 3.40 | 俄方 |
| Tanks | 4,729 | 1,504 | 3.14 | 俄方 |
| APC | 804 | 1,425 | **0.56** | **乌方** |

### 月度比率时序逆转

| 时段 | 月均 RU/UA | 趋势 |
|------|-----------|------|
| 2022.2–8 | 3.66 | 俄方远超 |
| 2022.9–12 | ~1.5–2.0 | 趋于接近 |
| 2023–2024 | 1.0–2.0 | 波动 |
| 2025–2026 | **0.65** | **乌方反超** |

### 观测偏误校正

| 场景 | 校正率比 |
|------|---------|
| 等概率 (p_RU=p_UA) | 2.07（不变） |
| 合理先验 (Bootstrap, 中位数) | **1.63** |
| 偏误校正 95% CI | [0.67, 3.44] |
| **P(ρ_corrected > 1)** | **88.1%** |
| 极端非对称 (0.9, 0.4) | 0.92（逆转） |

---

## 9. 已发现并修复的问题

### 🔴 严重 Bug（已修复）

**preprocess.py: 累计→日度转换逻辑错误**
- **问题**: 代码在 `sort_values('date')`（升序）后又执行 `[::-1]`（反转），导致 `np.diff` 对降序累计值差分产生负值，再被 `np.maximum(..., 0)` 全部 clamp 为 0
- **影响**: 所有日度乌克兰声称数据被归零（除最后一天外），导致分布完全错误
- **发现**: 虽然均值（total/days）恰好正确，但每日分布、SE、Bootstrap 均受影响
- **修复**: 移除 `[::-1]` 反转，直接对升序累计值执行 `np.diff`

### 🟡 中等不一致（已修复）

**论文 λ̂ 值与代码输出不一致**
- **问题**: 论文摘要写 $\hat{\lambda}_{RU}=16.19$、$\hat{\lambda}_{UA}=8.00$，但代码实际输出 15.11、7.31
- **原因**: 论文可能基于更早期的数据快照（n≈1,455 天）或错误的天数计算
- **修复**: 将论文和幻灯片中的数值更新为与代码输出一致

**可视化模块图表计数标签错误**
- **问题**: `visualization_both_sides.py` 打印 "[10/13]" 等，但实际有 16 张图
- **修复**: 更正为 "[10/16]" 等

### 🟢 轻微问题（已修复）

- **`observation_sensitivity.py`**: 移除未使用的硬编码变量 `exposure_days = 1555`
- **`weekly_bootstrap.py`**: 移除硬编码的对比值，改为从实际输出文件读取
- **`preprocess.py`**: `fillna(method='ffill')` → `ffill()`（pandas 2.1+ 兼容性）

### ⚠️ 已知局限性（未修复，属于方法层面）

1. **子类别 sum ≠ 总体的偏差**: `both_sides_analysis.py` 中子类别差分 clamp 导致子类别总和与总量有偏差（俄方 ~8,545 件、乌方 ~3,216 件），因为 Oryx 原始数据存在累计值的向下修正
2. **Poisson 等离散假设**: 部分阶段（尤其是 Phase 1）可能存在 overdispersion（方差 >> 均值）
3. **观测偏误不可精确定量**: 灵敏度分析是假设驱动的，无法从数据本身估计真实观测概率
4. **Oryx 与 WarSpotting 的计数差异**: 双方数据在部分装备子类的分类上有差异

---

## 10. 答辩 Q&A 准备

### 预期会被问到的问题

**Q1: 为什么用 Poisson 分布而不是正态分布？**

A: 装备损失是计数数据（非负整数），Poisson 天然适用于计数数据建模。每日损失率相对较小（均值约15），正态近似不理想。此外 Poisson 的均值=方差特性使其适合"独立事件在单位时间内的发生次数"这一物理过程。对于稀有事件（飞机日均<0.1件），只有 Poisson 能提供合理的推断（Exact CI）。

**Q2: 为什么不直接用 IID Bootstrap？**

A: 装备损失数据有强自相关（lag-1 ACF = 0.589），今天的损失与昨天高度相关。IID Bootstrap 假设观测独立，会破坏这种时间结构，导致严重低估标准误。我们的实证显示 Block Bootstrap SE（0.99）是 IID（0.42）的 2.4 倍——忽略自相关将低估不确定性 58%。

**Q3: 三个置信区间（Wald/Score/Exact）的区别是什么？应该用哪个？**

A: 
- **Wald**: 基于 MLE 渐近正态性，计算最简单，但在小 λ 和小样本下覆盖率严重不足（λ=0.5 时仅 91.9%）
- **Score**: 基于得分统计量，不依赖 $\hat{\lambda}$ 估计方差，比 Wald 稳健
- **Exact**: 基于 Poisson-χ² 精确关系（Garwood, 1936），保证覆盖率 ≥95%，是小 λ 下的唯一可靠选择
- **推荐**: 大样本（n=1,559）三种方法几乎重合；稀有事件（飞机、直升机）必须用 Exact CI

**Q4: Oryx 数据有没有观测偏误？你们的结论可靠吗？**

A: 有偏误。俄军损失在乌控区更容易被拍摄验证（高观测），乌军损失在俄占区更难被验证（低观测）。这意味着真实 RU/UA 比率可能 < 观测到的 2.07:1。我们通过灵敏度分析量化了该不确定性：仅当乌方观测概率 < 俄方 48.5% 时才会逆转结论。中等非对称下结论稳健。但必须承认我们无法精确量化这个偏误。

**Q5: 什么是 BCa 校正？为什么需要它？**

A: BCa = Bias-Corrected and Accelerated。标准 Bootstrap 假设分布对称且无偏，但实际 Bootstrap 分布可能偏斜。
- **Bias-Correction (ẑ₀)**: 调整 Bootstrap 分布中心的系统偏移。本文 ẑ₀=0.221 表示分布略右偏
- **Acceleration (â)**: 通过 Jackknife 估计，调整标准误随参数变化的问题
- 两项校正使分位点从 [2.5%, 97.5%] 调整到更准确的 [α₁%, α₂%]

**Q6: 为什么用 E-test 而不是 t-test？**

A: t-test 前提是数据来自正态分布或大样本 CLT 近似。但：
- 计数数据（Poisson）的分布形状与正态不同（尤其是小 λ）
- E-test 基于条件二项分布的精确性质——无需渐近假设
- 在 $H_0$ 下的条件分布不依赖未知参数，可直接计算精确 p 值
- 对于 p < 0.0001 的结论，所有合理检验的结论一致

**Q7: 你们的块长度 b=11 是怎么确定的？**

A: 基于自相关函数（ACF）衰减法——找到 |ACF(lag)| < 2/√n 的第一个滞后阶数。对于 n=1,561，阈值 = 2/√1561 ≈ 0.051。lag-11 的 ACF 降至阈值以下。备用方法如 n^(1/3) ≈ 12 给出相近结果。敏感性分析显示 b≥7 后 SE 增幅放缓，b=11 是合理的折中选择。

**Q8: 月度比率从 3.66 降到 0.65 意味着什么？**

A: 反映了战争性质的深刻转变：
- 2022 年初：俄军作为进攻方，大规模装甲纵队突入敌方纵深 → 极高消耗率
- 2025-2026 年：战争进入长期消耗战，乌方承受越来越大压力 → 损失反超
- 这对双方战争的可持续性有直接政策含义

**Q9: 官方声称数据为什么比验证数据高那么多？火炮差 29 倍？**

A: 
- 坦克/装甲车差 3x：官方可能将损坏/遗弃也统计为"摧毁"
- 火炮差 29x：火炮部署在前线后方 10-30km，难以独立核实；炮击效果评估本身有巨大不确定性；官方可能将"压制"、"损坏"的炮位统计为"摧毁"
- 这凸显了仅依赖官方数据存在严重误导风险

**Q10: 项目的局限性是什么？**

A:
1. 观测偏误的先验设定具有一定主观性（Beta 先验参数通过领域知识设定，非数据驱动）
2. Poisson 假设的等离散性在部分阶段可能不满足（overdispersion）
3. 未纳入外部协变量（战线移动、天气、战斗强度）
4. Oryx 和 WarSpotting 的装备分类口径不完全一致
5. 空间分析中 38.8% 的记录缺少坐标

**Q11: 你们的偏误校正方法中，Beta 先验参数 (7,3) 和 (5.5,4.5) 是怎么选的？**

A: 先验均值的设定基于两个领域知识：(1) 俄方损失多在乌控区且多为大型装备 → p_RU 偏高，取均值 0.70；(2) 乌方部分损失在俄占区 → p_UA 偏低，取均值 0.55。这是主观先验，但我们的分析不依赖精确的先验值——灵敏度网格覆盖了 p ∈ [0.3, 1.0] 的整个范围，而偏误校正 Bootstrap 框架本身可以适配任何先验。核心结论 P=88% 是先验依赖的，但即便大幅调整先验（如 p_RU 均值降至 0.5），结论方向不变。这恰恰是偏误校正框架的优势——明确了假设并量化了其对结论的影响。

### 如果你被要求现场演示代码

```bash
# 最简单的演示：运行核心分析
python code/both_sides_analysis.py
# 展示双方 MLE 结果 + E-test 结果

python code/confidence_intervals.py
# 展示 Monte Carlo 覆盖率模拟结果
```

---

## 11. 文件清单

```
项目根目录
├── PROJECT_GUIDE.md               ← ★ 本文档：答辩全面准备指南
├── README.md                       ← 项目 README（含统计学知识解释）
├── plan.md                         ← 任务规划文档
├── slides.md                       ← 答辩展示讲稿（Pandoc/Marp 格式）
├── .gitignore
│
├── code/                           ← 所有代码（11个模块）
│   ├── preprocess.py               ← [1] 数据预处理（必须最先运行）
│   ├── point_estimation.py         ← [2] MLE 点估计
│   ├── confidence_intervals.py     ← [3] 三种 CI + Monte Carlo 覆盖率
│   ├── bootstrap.py                ← [4] 日度 Block Bootstrap + BCa
│   ├── two_sample_test.py          ← [5] 两样本 Poisson 条件检验
│   ├── both_sides_analysis.py      ← [6] ★ 双方 Oryx 数据整合（核心）
│   ├── spatial_analysis.py         ← [7] 空间分析
│   ├── observation_sensitivity.py  ← [8] 观测概率灵敏度
│   ├── weekly_bootstrap.py         ← [9] 周度 Block Bootstrap
│   ├── visualization.py            ← [10] 可视化（图01-09）
│   ├── visualization_both_sides.py ← [11] 双方可视化（图10-13）
│   ├── visualization_new.py        ← [12] 新增可视化（图14-16）
│   └── fix_paper_formatting.py     ← 论文格式修复（辅助脚本）
│
├── data/                           ← 数据目录
│   ├── oryx_both_sides.csv         ← Oryx 双方原始数据（含 Change 列）
│   ├── oryx_losses_combined.csv    ← Oryx 合并数据（item_count 加权）
│   ├── warspotting_losses.csv      ← WarSpotting 原始逐条记录
│   ├── russia_losses_equipment.csv ← 乌官方战报（累计值）
│   ├── russia_losses_personnel.csv ← 乌官方人员损失
│   ├── daily_warspotting.csv       ← [生成] WS 日度序列
│   ├── daily_by_type_ws.csv        ← [生成] WS 按装备类型
│   ├── daily_by_status_ws.csv      ← [生成] WS 按损毁状态
│   ├── daily_ua_claims.csv         ← [生成] UA 声称日度
│   ├── daily_oryx_both_sides.csv   ← [生成] Oryx 双方日度
│   ├── monthly_oryx_both_sides.csv ← [生成] Oryx 双方月度
│   ├── phase_summary.csv           ← [生成] 阶段汇总
│   └── warspotting_with_phase.csv  ← [生成] WS 带阶段标签
│
├── output/
│   ├── figures/                    ← 16 张分析图表
│   │   ├── 01_daily_loss_timeseries.png
│   │   ├── 02_phase_boxplot.png
│   │   ├── 03_ci_forest_plot.png
│   │   ├── 04_bootstrap_distribution.png
│   │   ├── 05_equipment_composition.png
│   │   ├── 06_poisson_qqplot.png
│   │   ├── 07_ci_width_comparison.png
│   │   ├── 08_claims_vs_verified.png
│   │   ├── 09_status_and_type.png
│   │   ├── 10_both_sides_timeseries.png
│   │   ├── 11_both_sides_ci_forest.png
│   │   ├── 12_equipment_type_comparison.png
│   │   ├── 13_both_sides_status_pie.png
│   │   ├── 14_spatial_distribution.png
│   │   ├── 15_sensitivity_heatmap.png
│   │   └── 16_weekly_bootstrap.png
│   └── tables/                    ← 15 个分析结果表格
│       ├── bootstrap_results.csv
│       ├── bootstrap_distributions.csv
│       ├── both_sides_summary.csv
│       ├── ci_comparison.csv
│       ├── ci_coverage_simulation.csv
│       ├── mle_results.csv
│       ├── phase_comparisons.csv
│       ├── type_comparisons.csv
│       ├── observation_sensitivity_overall.csv
│       ├── spatial_*.csv (×4)
│       ├── weekly_bootstrap_*.csv (×3)
│       └── weekly_jackknife_rates.csv
│
├── paper/
│   ├── T1_论文.md                  ← 论文 Markdown 源文件
│   ├── T1_论文_pdf.md             ← PDF 兼容版（数学符号已转义）
│   ├── T1_论文.pdf                ← 最终 PDF
│   ├── T1_论文.html               ← HTML 版（MathJax 渲染）
│   └── T1_论文.tex                ← LaTeX 版本
│
├── reference_repo/                 ← 参考仓库（来自 leedrake5）
│   ├── scripts/                    ← 原始分析脚本
│   ├── tables/                     ← 原始分析结果
│   └── README.md                   ← 参考仓库文档
│
├── build_reference.py              ← PPTX 参考模板构建（辅助）
└── build_template.py               ← PPTX 模板构建（辅助）
```

---

## 附录：答辩前的快速自检清单

- [ ] 能说清楚三个数据源的区别和各自用途
- [ ] 能解释 Poisson MLE = 样本均值为什么是合理的
- [ ] 能区分 Wald/Score/Exact 三种 CI 的优劣
- [ ] 能解释 Block Bootstrap vs IID Bootstrap 为什么 SE 差 2.4 倍
- [ ] 能解释 BCa 校正的两个参数 (ẑ₀, â) 的含义
- [ ] 能说明 E-test 的条件二项分布推导思路
- [ ] 能解读月度比率从 3.66 降到 0.65 的含义
- [ ] 能讨论观测偏误对结论的影响及灵敏度分析的局限
- [ ] 能说出项目的 3-4 个局限性
- [ ] 知道核心数值: RU=23556, UA=11397, λ̂_RU=15.11, λ̂_UA=7.31, 率比=2.07, p<0.0001

---

*本指南随项目代码同步更新，最后修订: 2026-06-12*
