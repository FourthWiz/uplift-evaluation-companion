"""Single-panel comparison of calibrated evaluator errors; no model refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(data='data/selection/comparison.csv', output='figures/narrative/06_all_learner_evaluations.png'):
    df = pd.read_csv(data)
    names = ['S', 'T', 'X', 'DA', 'DR', 'DML']
    primary = df[(df.scenario == 'selection_high') & (df.seed == 42)].set_index('Model')
    if 'Qini_calibrated' not in primary:
        anchor = primary.loc['TrueCATE']
        primary['Qini_calibrated'] = primary.Qini_scaled / (anchor.Qini_scaled / anchor.DR_estimated)
        primary['IPW_calibrated'] = primary.IPW_estimated / (anchor.IPW_estimated / anchor.DR_estimated)
    rows = primary.loc[['xgb/' + name for name in names]]
    columns = ['Qini_calibrated', 'IPW_calibrated', 'DR_estimated']
    errors = rows[columns].subtract(rows.Oracle_obs, axis=0)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout='constrained')
    for j, (column, label, color, marker) in enumerate(zip(
        columns, ['Calibrated Qini', 'Calibrated IPW', 'DR'],
        ['#c87716', '#2474a6', '#29874b'], ['D', 'o', 's']
    )):
        ax.scatter(errors[column], np.arange(6) + (j - 1) * .18,
                   color=color, marker=marker, s=52, label=label, zorder=3)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(np.arange(6), names)
    ax.invert_yaxis()
    ax.set(xlabel='Calibrated evaluator score − known causal gain area',
           ylabel='XGBoost ranking learner',
           title='Evaluation error after calibration to the reference true-CATE score')
    ax.set_xticks(np.arange(0, .00081, .0001), ['0'] + [f'{v:.4f}' for v in np.arange(.0001, .00081, .0001)])
    ax.set_xlim(-.00003, .00083)
    ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    fig.supxlabel('Same scale for all evaluators. Positive = overstatement. Fixed reference calibration; no intervals shown.', fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches='tight')
    plt.close(fig)
    return errors


if __name__ == '__main__':
    render()
