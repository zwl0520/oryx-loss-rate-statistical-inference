# 选题 T1：俄乌装备损失率"公平比较"——从估计偏误到区间估计

## 一、任务概述

基于 Oryx/WarSpotting 开源平台约 20,000+ 条经影像验证的装备损失记录，运用数理统计方法对俄乌双方装备损失率进行客观估计与比较。核心统计问题包括：
1. 如何用点估计和区间估计方法给出准确的装备损失率？
2. 双方损失率是否存在统计显著差异？

## 二、任务规划

### 阶段一：数据收集与预处理
- [x] 1.1 从 Kaggle 下载 Automated WarSpotting Equipment Losses 数据集
- [x] 1.2 数据探索：了解数据结构、字段含义、缺失情况
- [x] 1.3 数据清洗：处理缺失值、异常值、重复记录
- [x] 1.4 按照装备类型、时间、交战方进行数据分层

### 阶段二：统计方法实现
- [x] 2.1 MLE 点估计：基于 Poisson 分布的极大似然估计装备损失率 λ
- [x] 2.2 Poisson 置信区间构建：
  - Wald 置信区间
  - Score 置信区间
  - Exact（Garwood型）置信区间
- [x] 2.3 Block Bootstrap 方法实现：
  - 时间块 Bootstrap 重采样
  - BCa 偏差校正加速置信区间
- [x] 2.4 两独立 Poisson 条件检验：
  - 检验双方损失率是否统计显著差异
  - 计算 p-value 与效应量
- [x] 2.5 分层估计：
  - 按装备类型分层（坦克/装甲车/火炮/无人机等）
  - 按时间段分层（不同战争阶段）
- [x] 2.6 可视化：
  - 损失率时序图
  - 置信区间比较图（Wald vs Score vs Exact）
  - Bootstrap 分布对比图
  - 装备类型堆叠面积图

### 阶段三：论文撰写
- [x] 3.1 引言与背景
- [x] 3.2 数据描述
- [x] 3.3 统计方法论述
- [x] 3.4 实证结果与分析
- [x] 3.5 讨论与结论
- [x] 3.6 参考文献

### 阶段四：总结与文档
- [x] 4.1 撰写 README.md（任务过程总结、统计学知识解释、代码实现说明）

## 三、统计方法详细说明

### 3.1 模型设定
设每日装备损失数 $Y_t \sim Poisson(\lambda)$，其中 $\lambda$ 为日均损失率。

### 3.2 MLE 点估计
$$\hat{\lambda}_{MLE} = \frac{1}{n}\sum_{t=1}^{n} Y_t = \bar{Y}$$

方差估计：$\widehat{Var}(\hat{\lambda}) = \hat{\lambda}/n$

### 3.3 置信区间
- **Wald CI**: $\hat{\lambda} \pm z_{\alpha/2}\sqrt{\hat{\lambda}/n}$
- **Score CI**: 解方程 $\frac{(\hat{\lambda}-\lambda)^2}{\lambda/n} = z_{\alpha/2}^2$
- **Exact CI**: 基于 Poisson 分布的精确区间（Garwood, 1936）
- **BCa Bootstrap CI**: 偏差校正加速 Bootstrap

### 3.4 两 Poisson 条件检验
给定总损失 $S = Y_1 + Y_2$，检验 $H_0: \lambda_1 = \lambda_2$ vs $H_1: \lambda_1 \neq \lambda_2$

### 3.5 分层估计
按装备类型 k 和时间段 j 分别估计 $\lambda_{kj}$，分析异质性。

## 四、预期产出

1. `data/` — 原始数据与处理后的数据
2. `code/` — 完整的 Python 代码实现
3. `paper/T1_论文_俄乌装备损失率公平比较.pdf` — 最终论文
4. `README.md` — 项目总结与知识解释
