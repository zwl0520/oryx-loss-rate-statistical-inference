# 俄乌装备损失率统计推断——数理统计课程大作业（选题 T1）

## 📋 项目概述

本项目为《数理统计》课程大作业，对应选题手册中的 **T1：俄乌装备损失率"公平比较"——从估计偏误到区间估计**。

基于开源情报平台 Oryx 提供的 **俄乌双方经影像验证的装备损失记录**（俄方 23,556 件，乌方 11,397 件，2022.2–2026.6），运用 Poisson 分布模型、三种置信区间构造方法（Wald/Score/Exact）、Block Bootstrap 偏差校正和两独立 Poisson 条件检验等方法，对装备损失率进行系统性的统计推断与双边比较分析。

## 📁 项目结构

```
├── plan.md                           # 任务规划文档
├── README.md                         # 本文档
├── data/                             # 数据目录
│   ├── warspotting_losses.csv        # WarSpotting 原始数据（22,970条俄方）
│   ├── oryx_losses_combined.csv   # ★ Oryx 直接抓取合并数据（33,517条，含item_count）
│   ├── daily_oryx_both_sides.csv   # 双方日度损失序列
│   ├── monthly_oryx_both_sides.csv # 双方月度汇总
│   ├── daily_warspotting.csv       # WarSpotting 每日时序
│   ├── daily_by_type_ws.csv        # 按装备类型分层
│   ├── daily_by_status_ws.csv      # 按损毁状态分层
│   ├── daily_ua_claims.csv         # 乌克兰声称日度数据
│   └── phase_summary.csv           # 战争阶段汇总
├── code/                           # 代码目录（11个模块）
│   ├── preprocess.py               # 数据预处理
│   ├── point_estimation.py         # MLE 点估计
│   ├── confidence_intervals.py     # Poisson 置信区间（含覆盖率模拟）
│   ├── bootstrap.py                # 日度 Block Bootstrap + BCa
│   ├── two_sample_test.py          # 两样本 Poisson 条件检验
│   ├── both_sides_analysis.py      # 双方 Oryx 数据整合与双边比较
│   ├── spatial_analysis.py         # ★ 空间分析（宏区域/网格/热点）
│   ├── observation_sensitivity.py  # ★ 观测概率灵敏度分析
│   ├── weekly_bootstrap.py         # ★ 周度Block Bootstrap (B=10000)
│   ├── visualization.py            # 单边可视化（9张）
│   ├── visualization_both_sides.py # 双方对比可视化（4张）
│   └── visualization_new.py        # ★ 新增可视化（3张）
├── output/                         # 输出目录
│   ├── figures/                    # 16 张分析图表 ★
│   └── tables/                     # 15 个分析结果表格 ★
└── paper/
    └── T1_论文.md                    # 完整学术论文
```

## 🔬 统计学知识详解

### 1. Poisson 分布与 MLE 点估计

**Poisson 分布**适用于描述单位时间内随机事件发生次数的概率。对于装备损失问题，假设每日损失数 $Y_t \sim Poisson(\lambda)$，其中 $\lambda$ 代表日均损失率。

**极大似然估计（MLE）** 的基本思想是：选择使观测数据出现概率最大的参数值。对于 Poisson 分布：

$$L(\lambda) = \prod_{t=1}^{n} \frac{e^{-\lambda} \lambda^{Y_t}}{Y_t!}$$

取对数后求导得零，得到直观的结果：$\hat{\lambda} = \bar{Y}$（即样本均值）。

**为什么用 Poisson 而非正态分布？** 装备损失是计数数据（非负整数），每日损失数相对较小（均值约 15），正态近似并不理想。Poisson 分布天然适合计数数据。此外，Poisson 分布的均值=方差特性使其特别适合个体独立、低概率发生的罕见事件计数。

### 2. 置信区间的三种构造方法

置信区间反映了参数估计的不确定性。**95% 置信区间的含义**：如果重复抽样 100 次，每次构建一个 CI，则约 95 个 CI 会包含真实参数值。

#### Wald CI（渐近正态）
- **原理**：利用 MLE 的渐近正态性 $\hat{\lambda} \stackrel{\cdot}{\sim} N(\lambda, \lambda/n)$
- **优点**：计算简单，大样本下表现良好
- **缺点**：小样本或小 λ 时覆盖率严重不足（模拟显示 λ=0.5 时仅 91.9%）；可能产生负下限
- **适用场景**：大样本（n > 100）且 λ 不太小

#### Score CI（Wilson型）
- **原理**：基于得分统计量 $S(\lambda) = (\hat{\lambda} - \lambda)/\sqrt{\lambda/n}$，不以 $\hat{\lambda}$ 估计方差
- **优点**：比 Wald 更稳健，自动保证非负下限
- **缺点**：计算略复杂（需解二次方程）
- **适用场景**：中小样本，推荐作为 Wald 的改进方案

#### Exact CI（Garwood 1936）
- **原理**：利用 Poisson 分布与 χ² 分布的精确关系：$2n\lambda \sim \chi^2$ 族
- **优点**：保证覆盖率 ≥ 名义水平（保守），是统计学上的"金标准"
- **缺点**：区间可能略宽（以保守性换取可靠性）
- **适用场景**：小样本、稀有事件（如本研究中的飞机、直升机损失日均仅 0.08–0.09 件/天）

#### 损失率比值的置信区间（Delta 方法）

对于双边比较，我们关心率比 $\rho = \lambda_{RU}/\lambda_{UA}$ 的置信区间。使用 Delta 方法：

$$SE(\hat{\rho}) = \hat{\rho} \cdot \sqrt{\frac{1}{\hat{\lambda}_{RU} \cdot n} + \frac{1}{\hat{\lambda}_{UA} \cdot n}}$$

### 3. Block Bootstrap 与 BCa 校正

**为什么需要 Block Bootstrap？**

标准的 IID Bootstrap 假设观测独立。但装备损失数据存在**强自相关**（lag-1 ACF = 0.589）——今天的高损失往往预示着明天的高损失。IID Bootstrap 在此条件下会**严重低估标准误**（本文实测低估约 2.4 倍）。

**Block Bootstrap 原理**：
1. 将时间序列切分为长度为 b 的块（保留块内的时间依赖结构）
2. 有放回地抽取块，拼接成与原序列等长的"伪样本"
3. 对每个伪样本计算统计量（如均值），重复 B 次得到 Bootstrap 分布

**块长度选择**：本文基于 ACF 衰减法——找到自相关衰减到 $2/\sqrt{n}$ 以下的第一个滞后阶数作为 b。对于 n = 1,561 的日损失数据，b = 11。

**BCa 校正**（Bias-Corrected and Accelerated）：
- **Bias-Correction (ẑ₀)**：校正 Bootstrap 分布中心的系统偏差（本文 ẑ₀ = 0.221，表明轻微正偏）
- **Acceleration (â)**：校正方差的非恒定性（偏度调整），通过 Jackknife 估计（本文 â = 0.022）
- 两项校正使 Bootstrap CI 的分位点从 [2.5%, 97.5%] 调整到更准确的 [α₁, α₂]

### 4. 两独立 Poisson 条件检验（E-test）

**核心统计原理**：给定两组 Poisson 样本的总和 $S = Y_1 + Y_2$，在 $H_0: \lambda_1 = \lambda_2$ 下：

$$Y_1 \mid S \sim Binomial(S, \frac{n_1}{n_1 + n_2})$$

这意味着我们可以在不依赖大样本渐近理论的情况下，精确计算在零假设为真时观察到当前（或更极端）数据的概率。这是 "exact test" 的精髓。

**本文中 E-test 的双层应用**：
1. **总体双边检验**：俄方 vs 乌方总体损失率差异（p < 0.0001）
2. **分层双边检验**：按装备类型（10类）和损毁状态（4类）分别检验双方差异

**Bonferroni 校正**：当进行 m 次假设检验时，至少一次 Type I 错误的概率为 $1 - (1-\alpha)^m$。Bonferroni 将显著性水平调整为 $\alpha/m$，虽然保守，但有效控制了整体错误率。

### 5. Poisson 拟合诊断

**Q-Q 图（Quantile-Quantile Plot）**：将经验分位数与理论 Poisson 分位数对比。如果数据符合 Poisson 分布，点应大致落在 45° 对角线上。本研究发现某些阶段（特别是 Phase 1，日均 53 件、方差极大）存在偏离，提示可能需要考虑过离散模型（如负二项分布）。

## 💻 代码实现详解

### 运行环境

```bash
pip install pandas numpy scipy matplotlib kagglehub
```

### 执行顺序

```bash
# Step 1: 数据预处理
python code/preprocess.py

# Step 2: MLE 点估计（单边 + 分层）
python code/point_estimation.py

# Step 3: Poisson 置信区间（三种方法 + 覆盖率模拟）
python code/confidence_intervals.py

# Step 4: Block Bootstrap + BCa
python code/bootstrap.py

# Step 5: 两样本 Poisson 检验（阶段/装备类型比较）
python code/two_sample_test.py

# Step 6: 可视化（单边 9 张）
python code/visualization.py

# ★ Step 7: 双方 Oryx 数据整合与双边比较
python code/both_sides_analysis.py

# ★ Step 8: 空间分析（坐标/区域/网格）
python code/spatial_analysis.py

# ★ Step 9: 观测概率灵敏度分析
python code/observation_sensitivity.py

# ★ Step 10: 周度 Block Bootstrap (B=10000)
python code/weekly_bootstrap.py

# Step 11: 可视化（全部16张）
python code/visualization.py
python code/visualization_both_sides.py
python code/visualization_new.py
```

### 核心代码片段解释

#### 双方 E-test（both_sides_analysis.py 核心逻辑）

```python
def e_test(y1, n1, y2, n2):
    """两独立 Poisson 条件精确检验（双边）
    H₀: λ₁ = λ₂,  条件分布: Y₁|S ~ Binomial(S, n₁/(n₁+n₂))
    """
    total = y1 + y2
    p_hat = n1 / (n1 + n2)                # 在 H₀ 下的成功概率
    p_upper = 1 - stats.binom.cdf(y1 - 1, total, p_hat)
    p_lower = stats.binom.cdf(y1, total, p_hat)
    return 2 * min(p_upper, p_lower)       # 双侧 p-value
```

#### 损失率比值的 CI（Delta 方法）

```python
ru_lam, ua_lam = np.mean(ru_data), np.mean(ua_data)
n = len(ru_data)
ratio_hat = ru_lam / ua_lam
ratio_se = ratio_hat * np.sqrt(1/(ru_lam * n) + 1/(ua_lam * n))
# 95% Wald CI for ratio
ci_lower = ratio_hat - 1.96 * ratio_se
ci_upper = ratio_hat + 1.96 * ratio_se
```

#### Block Bootstrap 核心（bootstrap.py）

```python
def block_bootstrap(data, B=2000, block_len=None):
    """Moving Block Bootstrap 重抽样"""
    n = len(data)
    k = int(np.ceil(n / block_len))
    bootstrap_means = np.zeros(B)
    for i in range(B):
        starts = np.random.randint(0, n - block_len + 1, k)
        bs_sample = np.concatenate([data[s:s+block_len] for s in starts])[:n]
        bootstrap_means[i] = np.mean(bs_sample)
    return bootstrap_means
```

#### Exact CI 实现（confidence_intervals.py）

```python
def exact_ci(data, alpha=0.05):
    """Garwood 精确 CI: 基于 Poisson 与 χ² 的关系"""
    n, Y = len(data), np.sum(data)
    lower = stats.chi2.ppf(alpha/2, 2*Y) / (2*n) if Y > 0 else 0
    upper = stats.chi2.ppf(1 - alpha/2, 2*(Y+1)) / (2*n)
    return lower, upper
```

## 📊 主要分析结果

### 俄乌双方 Oryx 影像验证损失比较 ⭐

| 指标 | 俄方 (RU) | 乌方 (UA) | 比率 (RU/UA) | p-value |
|------|----------|----------|-------------|---------|
| 累计验证损失 | 23,556 件 | 11,397 件 | **2.07** | — |
| 日均损失率 λ̂ | 15.11/天 | 7.31/天 | **2.07** | < 0.0001 |
| 95% Exact CI | [14.92, 15.30] | [7.18, 7.45] | — | — |
| 率比 95% Wald CI | — | — | **[2.02, 2.11]** | — |

注：以上 λ̂ 值和 CI 均基于日度 Oryx 双方数据（2022.2.25–2026.6.4, 1,559 天）的 Change 列直接求和计算。论文正文中的数值已据此更新。

### 双方装备类型差异（显著度排序）

| 装备类型 | 俄方 | 乌方 | 比率 | 谁损失更多 |
|---------|------|------|------|----------|
| 装甲车 (AFV) | 2,682 | 597 | **4.49** | 俄方 |
| 步战车 (IFV) | 6,702 | 1,641 | **4.08** | 俄方 |
| 车辆 | 4,949 | 1,456 | **3.40** | 俄方 |
| 坦克 | 4,729 | 1,504 | **3.14** | 俄方 |
| 后勤车辆 | 1,094 | 391 | **2.80** | 俄方 |
| 工程车辆 | 759 | 299 | **2.54** | 俄方 |
| 飞机 | 529 | 281 | 1.88 | 俄方 |
| 防空系统 | 603 | 394 | 1.53 | 俄方 |
| 火炮 | 1,762 | 1,202 | 1.47 | 俄方 |
| 装甲运兵车 (APC) | 804 | 1,425 | **0.56** | **乌方** |

### 双方损毁状态构成

| 状态 | 俄方 | 乌方 | 解读 |
|------|------|------|------|
| Destroyed | 78.8% | 77.8% | 双方接近 |
| Captured | **12.0%** | 10.4% | 俄方被缴获率高 → 多次撤退遗弃 |
| Abandoned | 5.1% | 5.9% | 乌方略高 |
| Damaged | 4.2% | 5.9% | 乌方略高 |

### 月度比率时序变化

| 时段 | 月均 RU/UA 比率 | 含义 |
|------|----------------|------|
| 2022.2–8（初期） | **3.66** | 俄方大规模进攻，损失远超乌方 |
| 2022.9–12（乌军反攻） | ~1.5–2.0 | 双方损失趋于接近 |
| 2025–2026（近期） | **0.65** | **乌方反超**——消耗战加重乌方负担 |

### CI 方法 Monte Carlo 覆盖率

| λ 真值 | Wald | Score | Exact |
|--------|------|-------|-------|
| 0.5 | 91.9% ❌ | 95.1% ✅ | 96.6% ✅ |
| 1.0 | 92.9% ⚠️ | 94.6% ✅ | 95.7% ✅ |
| 5.0 | 95.5% ✅ | 95.6% ✅ | 95.6% ✅ |
| 15.0 | 94.9% ✅ | 94.9% ✅ | 95.3% ✅ |

### Block Bootstrap 关键结果

| 指标 | 值 |
|------|---|
| 块长度 b | 11（基于 ACF 衰减法） |
| IID Bootstrap SE | 0.42 |
| Block Bootstrap SE | **0.99**（≈ IID 的 2.4 倍）|
| BCa 95% CI | [13.18, 17.29] |
| Phase 1 vs Phase 9 Bootstrap p | < 0.0001 |

### 声称 vs 验证偏差

| 装备 | 验证 λ̂ | 声称 λ̂ | 声称/验证比 |
|------|--------|--------|------------|
| 坦克 | 2.57/天 | 7.68/天 | **2.98x** |
| 装甲车 | 5.60/天 | 15.85/天 | **2.83x** |
| 火炮 | 0.94/天 | 27.62/天 | **29.44x** |

### 观测概率灵敏度分析 ⭐ NEW

| 假设 (p_RU, p_UA) | 修正率比 | 含义 |
|-------------------|---------|------|
| (0.8, 0.8) | 2.06 | 等概率 → 率比不变 |
| (0.8, 0.5) | 1.29 | 中等非对称 |
| (0.8, 0.4) | **1.03** | 几乎相等 |
| (0.9, 0.4) | **0.92** | **率比逆转!** |

### 空间分析 ⭐ NEW

| 宏区域 | 损失数 | 占比 |
|--------|--------|------|
| East (卢甘斯克-顿涅茨克) | 10,185 | 44.3% |
| North (哈尔科夫-基辅) | 2,276 | 9.9% |
| South (赫尔松-扎波罗热) | 1,517 | 6.6% |

最热网格 (48°N, 37°E) 单独占 21.5%（Pokrovsk 方向）

### 周度 Bootstrap ⭐ NEW

| 指标 | 日度 Block Bootstrap | 周度 Bootstrap |
|------|---------------------|---------------|
| Bootstrap SE | 0.99 | 0.82 |
| BCa 95% CI | [13.18, 17.29] | [13.31, 16.61] |
| vs IID SE (0.42) | 2.35× | 1.95× |

## 🎯 关键发现

1. **双方损失率存在极显著差异（p < 0.0001）**：Oryx 影像验证数据显示俄方日均损失约为乌方的 2 倍（率比 2.07:1），观测灵敏度分析表明该率比在中等非对称观测假设下具有稳健性
2. **时序逆转——从 3.66:1 到 0.65:1**：2022 年初期俄方损失远多于乌方，但 2025-2026 年乌方反超
3. **观测灵敏度量化了不确定性**：极端假设下（乌方观测率仅为俄方 40%）率比可降至 0.82——观测偏误方向对比较结论的稳健性至关重要 ⭐
4. **空间分布高度集中**：顿涅茨克州占俄方验证损失的 44%，最热网格 (48°N, 37°E) 单格 21.5%——揭示了战争的高度空间集中性 ⭐
5. **装备类型差异揭示作战方式**：俄方在 AFV/IFV/坦克上损失远超（3-4x），乌方在 APC/MRAP 上反超
6. **时间依赖性通过两种方法交叉验证**：日度 Block Bootstrap SE（0.99）和 周度 Bootstrap SE（0.82）均为 IID（0.42）的 2-2.4 倍 ⭐
7. **大样本下三种 CI 方法几乎等价**，但小样本/小 λ 时必须使用 Exact CI
8. **官方声称数据严重偏高**：声称/验证比从 3x（坦克）到 29x（火炮）
9. **俄方被缴获装备多于乌方**（12.0% vs 10.4%）：反映了多次战略撤退中的装备遗弃

## 📝 任务过程回顾

1. **选题理解**：仔细阅读选题手册中 T1 的要求，明确了核心统计问题
2. **数据收集**：
   - 从 Kaggle 下载 WarSpotting 数据（22,970 条俄方记录）
   - 从 Kaggle 下载乌克兰总参官方战报（1,557 天累计数据）
   - 从 leedrake5 Google Sheet 获取双方 Oryx 数据（1,559 天）
3. **数据预处理**：数据清洗、日期解析、累计→日度转换、分层构建
4. **统计方法实现**（6 大模块）：
   - MLE 点估计：总体 + 分层（装备类型/战争阶段）
   - CI 构造：Wald / Score / Exact + Monte Carlo 覆盖率验证
   - Block Bootstrap：块长度选择 + IID vs Block 对比 + BCa 校正
   - 两样本检验：E-test + LRT + Bonferroni 校正
   - **双边比较**：俄乌双方 Oryx 数据整合 + 总体检验 + 分层比较
   - 声称 vs 验证偏差分析
5. **可视化**：生成 13 张专业图表（9 张单边 + 4 张双边）
6. **论文撰写**：按照学术论文格式撰写完整报告（含双边比较章节）

## 📚 数据来源

| 数据源 | 链接 | 内容 |
|--------|------|------|
| WarSpotting | [Kaggle](https://www.kaggle.com/datasets/zsoltlazar/automated-warspotting-equipment-losses) | 22,970 条俄方装备损失逐条记录 |
| 乌克兰总参 | [Kaggle](https://www.kaggle.com/datasets/piterfm/2022-ukraine-russian-war) | 1,557 天官方累计损失 |
| **Oryx 双方数据** ⭐ | [Google Sheet](https://docs.google.com/spreadsheets/d/1bngHbR0YPS7XH1oSA1VxoL4R34z60SJcR3NxguZM9GI) | 俄乌双方每日累计（含细分） |
| Oryx 官网 | https://www.oryxspioenkop.com | 双方影像验证损失原始记录 |
| 对标论文 | arXiv:2509.07813 | Forecasting Russian Equipment Losses |

---

*本项目为教育目的，所有分析基于公开数据，不代表任何政治立场。*
