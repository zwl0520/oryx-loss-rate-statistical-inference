import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from utils import WAR_PHASES, assign_phase

print("=" * 60)
print("Loading WarSpotting image-verified loss data...")
ws_path = os.path.join('data', 'warspotting_losses.csv')
df_ws = pd.read_csv(ws_path)
print(f"Records: {len(df_ws)}")
print(f"Columns: {df_ws.columns.tolist()}")

df_ws['date'] = pd.to_datetime(df_ws['date'])
print(f"Date range: {df_ws['date'].min()} to {df_ws['date'].max()}")

print("\nStatus distribution:")
print(df_ws['status'].value_counts())
print("\nType distribution:")
print(df_ws['type'].value_counts())

df_ws['is_loss'] = 1

print("\n" + "=" * 60)
print("Generating daily time series...")

daily_ws = df_ws.groupby('date').agg(
    total_losses=('id', 'count')
).reset_index()

date_range = pd.date_range(start=daily_ws['date'].min(), end=daily_ws['date'].max(), freq='D')
daily_ws = daily_ws.set_index('date').reindex(date_range).fillna(0).astype({'total_losses': int})
daily_ws.index.name = 'date'
daily_ws = daily_ws.reset_index()

print(f"Mean daily losses (WarSpotting): {daily_ws['total_losses'].mean():.2f}")
print(f"Total losses: {daily_ws['total_losses'].sum()}")

daily_by_type = df_ws.groupby(['date', 'type']).size().unstack(fill_value=0)
daily_by_type = daily_by_type.reindex(date_range).fillna(0).astype(int)
daily_by_type.index.name = 'date'

daily_by_status = df_ws.groupby(['date', 'status']).size().unstack(fill_value=0)
daily_by_status = daily_by_status.reindex(date_range).fillna(0).astype(int)
daily_by_status.index.name = 'date'

print("\n" + "=" * 60)
print("Loading Ukraine General Staff claims data...")
ua_path = os.path.join('data', 'russia_losses_equipment.csv')
df_ua = pd.read_csv(ua_path)
df_ua['date'] = pd.to_datetime(df_ua['date'])
df_ua = df_ua.sort_values('date')

print(f"Records: {len(df_ua)}")
print(f"Date range: {df_ua['date'].min()} to {df_ua['date'].max()}")

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

df_ua_daily = pd.DataFrame()
df_ua_daily['date'] = df_ua['date']

for col, name in equipment_cols.items():
    if col in df_ua.columns:
        series = df_ua[col].copy()
        series = series.ffill().fillna(0)
        cumulative = series.values
        daily = np.diff(cumulative, prepend=0)
        daily = np.maximum(daily, 0)
        df_ua_daily[name] = daily.astype(int)

print("\nUkraine daily claims (mean):")
for name in df_ua_daily.columns:
    if name != 'date':
        print(f"  {name}: {df_ua_daily[name].mean():.1f}/day")

print("\n" + "=" * 60)
print("WarSpotting verified vs Ukraine claimed comparison...")

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

print("\n" + "=" * 60)
print("Creating phase stratification...")

daily_ws['phase'] = daily_ws['date'].apply(assign_phase)
df_ws['phase'] = df_ws['date'].apply(assign_phase)

print("\n" + "=" * 60)
print("Saving processed data...")

daily_ws.to_csv(os.path.join('data', 'daily_warspotting.csv'), index=False)
print("  data/daily_warspotting.csv")
daily_by_type.to_csv(os.path.join('data', 'daily_by_type_ws.csv'))
print("  data/daily_by_type_ws.csv")
daily_by_status.to_csv(os.path.join('data', 'daily_by_status_ws.csv'))
print("  data/daily_by_status_ws.csv")
df_ua_daily.to_csv(os.path.join('data', 'daily_ua_claims.csv'), index=False)
print("  data/daily_ua_claims.csv")
df_ws.to_csv(os.path.join('data', 'warspotting_with_phase.csv'), index=False)
print("  data/warspotting_with_phase.csv")

phase_summary = daily_ws.groupby('phase').agg(
    days=('date', 'count'),
    total_losses=('total_losses', 'sum'),
    mean_daily_loss=('total_losses', 'mean'),
    std_daily_loss=('total_losses', 'std'),
).reset_index()
phase_summary.to_csv(os.path.join('data', 'phase_summary.csv'), index=False)
print("  data/phase_summary.csv")

print("\n" + "=" * 60)
print("Preprocessing complete. Summary:")
print(f"  WarSpotting records: {len(df_ws)}")
print(f"  Date span: {daily_ws['date'].min().date()} to {daily_ws['date'].max().date()}")
print(f"  Total days: {len(daily_ws)}")
print(f"  Mean daily verified losses: {daily_ws['total_losses'].mean():.2f}/day")
print(f"  Mean daily UA tank claims: {df_ua_daily['Tanks'].mean():.1f}/day")
print(f"  Equipment types: {len(df_ws['type'].unique())}")
print(f"  War phases: {len(WAR_PHASES)}")
print("\nPhase summary:")
print(phase_summary.to_string(index=False))
