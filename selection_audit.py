"""Additional review tables; no refitting or outcome-based policy tuning."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path('data/selection')
rows=[];health=[]
for sc in ['selection_low','selection_medium','selection_high']:
 for seed in [42,43,44]:
  p=ROOT/sc/f'seed_{seed}';d=pd.read_pickle(p/'obs.pkl')
  expected=pd.read_csv(p/'obs_metrics.csv')
  for _,r in expected.iterrows():
   rows.append(dict(scenario=sc,seed=seed,Model=r.Model,ExpectedQiniScaled=r.ExpectedObservational/d.Propensity.mean(),OracleGain=r.OracleGain))
  audit=json.loads((p/'evaluation_diagnostics.json').read_text())
  weights=pd.read_csv(p/'weight_diagnostics.csv');w=weights[weights.propensity=='estimated']
  health.append(dict(scenario=sc,seed=seed,**audit,ESS_control=w[w.arm==0].ESS.iloc[0],ESS_treated=w[w.arm==1].ESS.iloc[0]))
pd.DataFrame(rows).to_csv(ROOT/'expected_ranking_audit.csv',index=False)
pd.DataFrame(health).to_csv(ROOT/'evaluation_health.csv',index=False)
print(pd.DataFrame(rows).query("scenario=='selection_high' and Model in ['TrueCATE','linear/DR','xgb/DR','CATE+propensity']").to_string(index=False))
