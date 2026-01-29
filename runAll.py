import subprocess
import sys

"""
End‑to‑end pipeline reproducing results reported in the paper.
Scripts must be run in this exact order.
"""

scripts = [
    "prepIOWADataframe.py",
    "prepSanDiegoDataframePD.py",
    "prepUNMDataframe.py",
    "cnnSlideAnalysisNestedCrossChan.py",
    "cnnSlideAnalysisNestedCrossChanVisual.py",
]

for script in scripts:
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script}")

print("\nPipeline completed successfully.")
