"""
数据预处理模块
=============
功能：
1. 加载 WarSpotting 影像验证损失数据（俄罗斯装备逐条记录）
2. 加载乌克兰总参公布的俄军装备损失数据（每日累计）
3. 生成每日时间序列、分层数据集
4. 数据清洗与异常值处理
"""

import pandas as pd
import numpy as np
import os
import sys

# 设置输出编码
sys.stdout.reconfigure(encoding='utf-8')

# ===========================
# 1. 加载 WarSpotting 数据
# ===========================
print("=" * 60)
print("加载 WarSpotting 影像验证数据...")
ws_path = os.path.join('data', 'warspotting_losses.csv')
df_ws = pd.read_csv(ws_path)
print(f"原始记录数: {len(df_ws)}")
print(f"列: {df_ws.columns.tolist()}")

# 解析日期
df_ws['date'] = pd.to_datetime(df_ws['date'])
print(f"日期范围: {df_ws['date'].min()} 至 {df_ws['date'].max()}")

# 状态统计
print("\n装备状态分布:")
print(df_ws['status'].value_counts())

# 装备类型统计
print("\n装备类型分布:")
print(df_ws['type'].value_counts())

# 创建"已损失"二值标记（Destroyed + Captured + Abandoned + Damaged 均为损失）
df_ws['is_loss'] = 1  # 所有记录均代表损失

# ===========================
# 2. 生成每日损失时间序列 (WarSpotting)
# ===========================
print("\n" + "=" * 60)
print("生成每日时间序列...")

# 按日聚合
daily_ws = df_ws.groupby('date').agg(
    total_losses=('id', 'count')
).reset_index()

# 确保日期连续
date_range = pd.date_range(start=daily_ws['date'].min(), end=daily_ws['date'].max(), freq='D')
daily_ws = daily_ws.set_index('date').reindex(date_range).fillna(0).astype({'total_losses': int})
daily_ws.index.name = 'date'
daily_ws = daily_ws.reset_index()

print(f"日均损失数 (WarSpotting 验证): {daily_ws['total_losses'].mean():.2f}")
print(f"总损失数: {daily_ws['total_losses'].sum()}")

# 按装备类型聚合
daily_by_type = df_ws.groupby(['date', 'type']).size().unstack(fill_value=0)
daily_by_type = daily_by_type.reindex(date_range).fillna(0).astype(int)
daily_by_type.index.name = 'date'

# 按装备状态聚合
daily_by_status = df_ws.groupby(['date', 'status']).size().unstack(fill_value=0)
daily_by_status = daily_by_status.reindex(date_range).fillna(0).astype(int)
daily_by_status.index.name = 'date'

# ===========================
# 3. 加载乌克兰总参数据（累计 → 日度转换）
# ===========================
print("\n" + "=" * 60)
print("加载乌克兰总参公布的俄军损失数据...")
ua_path = os.path.join('data', 'russia_losses_equipment.csv')
df_ua = pd.read_csv(ua_path)
df_ua['date'] = pd.to_datetime(df_ua['date'])
df_ua = df_ua.sort_values('date')

print(f"记录数: {len(df_ua)}")
print(f"日期范围: {df_ua['date'].min()} 至 {df_ua['date'].max()}")
print(f"列: {df_ua.columns.tolist()}")

# 关键装备列映射（仅选用数据较完整的列）
equipment_cols = {
    'tank': 'Tanks',
    'APC': 'Armoured Personnel Carriers',
    'field artillery': 'Field Artillery',
    'MRL': 'Multiple Rocket Launchers',
    'drone': 'Drones',
    'aircraft': 'Aircraft',
    'helicopter': 'Helicopters',
    'naval ship': 'Naval Ships',
    'anti-aircraft warfare': 'Anti-Aircraft Systems',
    'special equipment': 'Special Equipment',
}

# 将累计数据转为日度数据（处理 NaN 和缺失值）
df_ua_daily = pd.DataFrame()
df_ua_daily['date'] = df_ua['date']

for col, name in equipment_cols.items():
    if col in df_ua.columns:
        series = df_ua[col].copy()
        # 前向填充 NaN（新类别先出现NaN，后添加历史数据）
        series = series.fillna(method='ffill').fillna(0)
        cumulative = series.values
        # 日度差分（数据按日期降序排列，需反转）
        cumulative = cumulative[::-1]  # 反转为时间升序
        daily = np.diff(cumulative, prepend=0)
        daily = np.maximum(daily, 0)  # 修正可能的负值（数据修正）
        # 反转为与 df_ua_daily 一致的时间顺序
        daily = daily[::-1]
        df_ua_daily[name] = daily.astype(int)

print("\n乌克兰总参日均声称俄军损失:")
for name in df_ua_daily.columns:
    if name != 'date':
        mean_val = df_ua_daily[name].mean()
        print(f"  {name}: {mean_val:.1f}/天")

# ===========================
# 4. 对应装备类型对比（WarSpotting vs 乌克兰声称）
# ===========================
print("\n" + "=" * 60)
print("WarSpotting 验证 vs 乌克兰声称 对比...")

# WarSpotting 装备类型映射到乌克兰总参分类
ws_to_ua_mapping = {
    'Tanks': 'Tanks',
    'Infantry fighting vehicles': 'Armoured Personnel Carriers',
    'Self-propelled artillery': 'Field Artillery',
    'Towed artillery': 'Field Artillery',
    'Rocket and missile artillery': 'Multiple Rocket Launchers',
    'Drones': 'Drones',
    'Airplanes': 'Aircraft',
    'Helicopters': 'Helicopters',
    'Vessels': 'Naval Ships',
    'Anti-aircraft systems': 'Anti-Aircraft Systems',
}

# 聚合 WarSpotting 数据到 UA 分类
ws_daily_mapped_list = []
for ws_type, ua_name in ws_to_ua_mapping.items():
    if ws_type in daily_by_type.columns:
        col = daily_by_type[ws_type].reset_index()
        col['category'] = ua_name
        ws_daily_mapped_list.append(col)

ws_daily_mapped = pd.concat([x.set_index(['date', 'category']) for x in ws_daily_mapped_list])
ws_daily_mapped = ws_daily_mapped.groupby(level=[0, 1]).sum().unstack(fill_value=0)
ws_daily_mapped.columns = ws_daily_mapped.columns.droplevel(0)
ws_daily_mapped = ws_daily_mapped.reset_index()

# ===========================
# 5. 时间段分层
# ===========================
print("\n" + "=" * 60)
print("创建时间段分层...")

# 定义战争阶段
war_phases = {
    'Phase 1: Initial Invasion (2022.2-4)': ('2022-02-24', '2022-04-30'),
    'Phase 2: Donbas Offensive (2022.5-8)': ('2022-05-01', '2022-08-31'),
    'Phase 3: Counteroffensive (2022.9-12)': ('2022-09-01', '2022-12-31'),
    'Phase 4: Bakhmut/Winter (2023.1-5)': ('2023-01-01', '2023-05-31'),
    'Phase 5: Summer 2023 (2023.6-9)': ('2023-06-01', '2023-09-30'),
    'Phase 6: Avdiivka Campaign (2023.10-2024.2)': ('2023-10-01', '2024-02-29'),
    'Phase 7: Spring 2024 (2024.3-7)': ('2024-03-01', '2024-07-31'),
    'Phase 8: Kursk & Eastern Front (2024.8-12)': ('2024-08-01', '2024-12-31'),
    'Phase 9: Recent (2025.1-2026.6)': ('2025-01-01', '2026-06-30'),
}

# 为 WarSpotting 每日数据添加阶段标签
def assign_phase(d):
    for phase_name, (start, end) in war_phases.items():
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return phase_name
    return 'Other'

daily_ws['phase'] = daily_ws['date'].apply(assign_phase)

# 为逐条记录也添加阶段
df_ws['phase'] = df_ws['date'].apply(assign_phase)

# ===========================
# 6. 保存处理后的数据
# ===========================
print("\n" + "=" * 60)
print("保存处理后的数据...")

# WarSpotting 每日序列
daily_ws.to_csv(os.path.join('data', 'daily_warspotting.csv'), index=False)
print("✓ data/daily_warspotting.csv")

# 按类型分层
daily_by_type.to_csv(os.path.join('data', 'daily_by_type_ws.csv'))
print("✓ data/daily_by_type_ws.csv")

# 按状态分层
daily_by_status.to_csv(os.path.join('data', 'daily_by_status_ws.csv'))
print("✓ data/daily_by_status_ws.csv")

# 乌克兰声称日度数据
df_ua_daily.to_csv(os.path.join('data', 'daily_ua_claims.csv'), index=False)
print("✓ data/daily_ua_claims.csv")

# 原始数据带阶段标签
df_ws.to_csv(os.path.join('data', 'warspotting_with_phase.csv'), index=False)
print("✓ data/warspotting_with_phase.csv")

# 阶段汇总
phase_summary = daily_ws.groupby('phase').agg(
    days=('date', 'count'),
    total_losses=('total_losses', 'sum'),
    mean_daily_loss=('total_losses', 'mean'),
    std_daily_loss=('total_losses', 'std'),
).reset_index()
phase_summary.to_csv(os.path.join('data', 'phase_summary.csv'), index=False)
print("✓ data/phase_summary.csv")

# ===========================
# 7. 数据摘要输出
# ===========================
print("\n" + "=" * 60)
print("数据预处理完成！摘要：")
print(f"  WarSpotting 记录总数: {len(df_ws)}")
print(f"  日期跨度: {daily_ws['date'].min().date()} 至 {daily_ws['date'].max().date()}")
print(f"  总天数: {len(daily_ws)}")
print(f"  日均验证损失: {daily_ws['total_losses'].mean():.2f} 件/天")
print(f"  日均乌方声称损失: {df_ua_daily['Tanks'].mean():.1f} 坦克/天")
print(f"  装备类型数: {len(df_ws['type'].unique())}")
print(f"  战争阶段数: {len(war_phases)}")

print("\n阶段汇总:")
print(phase_summary.to_string(index=False))
print("\n数据预处理全部完成！")
