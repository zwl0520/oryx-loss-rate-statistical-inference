import pandas as pd
import numpy as np
import os, sys

sys.stdout.reconfigure(encoding='utf-8')
from utils import macro_region

print("=" * 70)
print("WarSpotting Spatial Analysis")
print("=" * 70)

df = pd.read_csv(os.path.join('data', 'warspotting_losses.csv'))
df['date'] = pd.to_datetime(df['date'])
total = len(df)

print("\n--- Coordinate Coverage ---")
has_coords = df['latitude'].notna() & df['longitude'].notna()
n_coords = has_coords.sum()
n_missing = total - n_coords
print(f"  Total records: {total}")
print(f"  With coordinates: {n_coords} ({n_coords/total*100:.1f}%)")
print(f"  Missing coordinates: {n_missing} ({n_missing/total*100:.1f}%)")
has_location = df['nearest_location'].notna()
n_location = has_location.sum()
print(f"  With location: {n_location} ({n_location/total*100:.1f}%)")

print("\n--- Macro Region Distribution ---")
df['macro_region'] = df.apply(lambda r: macro_region(r['latitude'], r['longitude']), axis=1)
region_counts = df['macro_region'].value_counts()
for region, count in region_counts.items():
    pct = count / total * 100
    bar = '#' * int(pct / 2)
    print(f"  {region:<40s}: {count:6d} ({pct:5.1f}%) {bar}")

print("\n--- Macro Region x Equipment Type Cross-tab ---")
top_types = df['type'].value_counts().head(8).index.tolist()
region_type = df[df['type'].isin(top_types)].groupby(['macro_region', 'type']).size().unstack(fill_value=0)
print(region_type.to_string())

print("\n--- Hotspot Grid (Top 20) ---")
def grid_1deg(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    return f"({int(np.floor(lat))}, {int(np.floor(lon))})"

df['grid_1deg'] = df.apply(lambda r: grid_1deg(r['latitude'], r['longitude']), axis=1)
grid_counts = df['grid_1deg'].value_counts()
print(f"  Total grids: {len(grid_counts)}")
print(f"  Top 20 hotspot grids:")
for i, (grid, count) in enumerate(grid_counts.head(20).items()):
    if grid is None:
        continue
    pct = count / n_coords * 100
    print(f"  {i+1:2d}. {grid}: {count:5d} ({pct:.1f}%)")

print("\n--- Top Grid Equipment Composition ---")
top_grid = grid_counts.index[0] if grid_counts.index[0] is not None else grid_counts.index[1]
top_grid_df = df[df['grid_1deg'] == top_grid]
grid_types = top_grid_df['type'].value_counts()
print(f"  Grid {top_grid} (total {len(top_grid_df)} items):")
for typ, count in grid_types.head(10).items():
    print(f"    {typ}: {count}")

print("\n--- Top 20 Loss Locations ---")
def extract_area(location_str):
    if pd.isna(location_str) or not location_str:
        return 'Unknown'
    parts = [p.strip() for p in str(location_str).split(',')]
    return parts[-1] if parts else 'Unknown'

df['location_area'] = df['nearest_location'].apply(extract_area)
area_counts = df['location_area'].value_counts()
for i, (area, count) in enumerate(area_counts.head(20).items()):
    if area == 'Unknown':
        continue
    pct = count / total * 100
    print(f"  {i+1:2d}. {area:<40s}: {count:5d} ({pct:.1f}%)")

print("\n--- Temporal-Spatial Dynamics ---")
df['year_quarter'] = df['date'].dt.to_period('Q')
region_quarter = df.groupby(['year_quarter', 'macro_region']).size().unstack(fill_value=0)
key_quarters = region_quarter.index[::6]
print(f"  Key quarters (every 6th):")
for q in key_quarters:
    if q in region_quarter.index:
        row = region_quarter.loc[q]
        total_q = row.sum()
        top_region = row.idxmax()
        print(f"  {q}: total={total_q:4d}, hottest region={top_region} ({row[top_region]})")

print("\n--- Saving Spatial Results ---")
os.makedirs(os.path.join('output', 'tables'), exist_ok=True)

quality_df = pd.DataFrame([{
    'total_records': total,
    'records_with_coordinates': n_coords,
    'records_missing_coordinates': n_missing,
    'coordinate_coverage': round(n_coords / total, 4),
    'records_with_nearest_location': n_location,
    'nearest_location_coverage': round(n_location / total, 4),
}])
quality_df.to_csv(os.path.join('output', 'tables', 'spatial_coordinate_quality.csv'), index=False)

region_df = pd.DataFrame({
    'macro_region': region_counts.index,
    'loss_count': region_counts.values,
    'share_all_records': [round(c/total, 4) for c in region_counts.values],
}).sort_values('loss_count', ascending=False)
region_df.to_csv(os.path.join('output', 'tables', 'spatial_macro_region_summary.csv'), index=False)

grid_df = pd.DataFrame({
    'grid_1deg': grid_counts.index,
    'loss_count': grid_counts.values,
}).head(50)
grid_df.to_csv(os.path.join('output', 'tables', 'spatial_grid_hotspots.csv'), index=False)

area_df = pd.DataFrame({
    'location_area': area_counts.index,
    'loss_count': area_counts.values,
    'share_all_records': [round(c/total, 4) for c in area_counts.values],
}).head(50)
area_df.to_csv(os.path.join('output', 'tables', 'spatial_location_summary.csv'), index=False)

print("  spatial_coordinate_quality.csv, spatial_macro_region_summary.csv")
print("  spatial_grid_hotspots.csv, spatial_location_summary.csv")
print("\nSpatial analysis complete.")
