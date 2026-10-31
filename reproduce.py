"""Reproduce the synthetic uplift evaluation study in a new output directory."""
from pathlib import Path
import argparse
import subprocess
import sys

SCRIPTS = [
    'selection_revision.py', 'selection_audit.py', 'narrative_intervals.py',
    'narrative_figures.py', 'tuning_revision.py', 'tuning_figures.py',
    'unified_scaling.py', 'calibrated_scaling.py',
]

def run(output):
    source = Path(__file__).resolve().parent
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Output must be a new or empty directory.')
    destination.mkdir(parents=True, exist_ok=True)
    for folder in ['figures/narrative', 'figures/selection', 'figures/tuning']:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(source / script)], cwd=destination, check=True)
    from support_data import export
    export(destination)
    print(f'Results, figures and audit data: {destination}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='regenerated')
    args = parser.parse_args()
    run(args.output)
