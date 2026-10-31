# Uplift evaluation companion

Minimal reproduction code for **When Qini Rewards the Wrong Uplift Model**.
All data are simulated locally. No article drafts, saved experiment outputs, external datasets or credentials are included.

## Install

Use Python 3.10 (tested with 3.10.16):

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. XGBoost on macOS may require the OpenMP runtime (`brew install libomp`).

## Reproduce

Run from this directory:

```sh
python reproduce.py --output regenerated
```

The output directory must be new or empty. This is a substantial computation: one million simulated customers per baseline run, three assignment policies, three seeds, twelve learner specifications per run, 1,000 paired evaluation bootstrap resamples, 90 tuning candidates, and a fresh 300,000-customer tuning test. Models use four threads. Allow ample RAM and disk space; runtime depends on hardware. A full rerun is required before using the analysis notebook.

The command generates all article figures, result tables, bootstrap draws, calibration factors, tuning selections and formula-audit data under `regenerated/`. Then open `evaluation.ipynb` in a Jupyter-capable editor, select the installed Python environment and run all cells. Its default `RUN_ROOT` is `Path('regenerated')`; change that if a different output directory was used. The notebook verifies and displays the generated results without retraining.

To run only the standalone S-learner/IPW/DR example:

```sh
python article_example.py
```

## Scores and uncertainty

Raw Qini is divided by each evaluation sample's treatment share. Qini and IPW are then calibrated to DR on the true-CATE ranking in the high-selectivity, seed-42 baseline holdout. Those positive divisors stay fixed across learners, policies, seeds, bootstrap resamples and tuning. This simulation-only calibration does not correct confounding. The standalone example uses the original reference divisors; full reproduction estimates them from its generated reference run.

Article-scale tables are `regenerated/data/selection/comparison_calibrated.csv` and `regenerated/data/tuning/test_results_calibrated.csv`. Calibrated bootstrap tables/draws are under `regenerated/data/selection/intervals/`; exact divisors are in `regenerated/data/selection/calibration.json`. Raw and intermediate scores are retained in generated outputs for audit.

Intervals condition on fitted ranking/nuisance models and fixed calibration constants; they exclude retraining and calibration-estimation uncertainty. Across-policy whiskers are three-seed min–max ranges, not confidence intervals. Fixed seeds and dependency versions improve reproducibility; numerical results can vary across platforms or library builds.
