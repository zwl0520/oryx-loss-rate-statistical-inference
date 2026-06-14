import os, sys, subprocess

sys.stdout.reconfigure(encoding='utf-8')
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
os.chdir(ROOT_DIR)
sys.path.insert(0, CODE_DIR)

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

STEPS = [
    ('preprocess.py', 'Data Preprocessing'),
    ('point_estimation.py', 'MLE Point Estimation'),
    ('confidence_intervals.py', 'Confidence Intervals'),
    ('bootstrap.py', 'Block Bootstrap'),
    ('two_sample_test.py', 'Two-Sample Tests'),
    ('both_sides_analysis.py', 'Both Sides Analysis'),
    ('spatial_analysis.py', 'Spatial Analysis'),
    ('observation_sensitivity.py', 'Observation Sensitivity'),
    ('weekly_bootstrap.py', 'Weekly Bootstrap'),
    ('visualization.py', 'Visualization (1-9)'),
    ('visualization_both_sides.py', 'Visualization (10-13)'),
    ('visualization_new.py', 'Visualization (14-16)'),
]

os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)

print("=" * 70)
print("Running full pipeline -> output/")
print("=" * 70)

failed = []
for script, desc in STEPS:
    script_path = os.path.join(CODE_DIR, script)
    print(f"\n{'#'*70}")
    print(f"# {desc} ({script})")
    print(f"{'#'*70}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=600,
            encoding='utf-8', errors='replace',
            cwd=ROOT_DIR, env={**os.environ, 'PYTHONPATH': CODE_DIR}
        )
        for line in result.stdout.split('\n'):
            print(line[:200])
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr[:2000]}")
            print(f"WARNING: {script} exited with code {result.returncode}")
            failed.append(script)
    except subprocess.TimeoutExpired:
        print(f"ERROR: {script} timed out")
        failed.append(script)
    except Exception as e:
        print(f"ERROR: {script} failed: {e}")
        failed.append(script)

print(f"\n{'='*70}")
if failed:
    print(f"Pipeline done with {len(failed)} failure(s): {failed}")
else:
    print("Pipeline complete — all 12 steps OK.")
