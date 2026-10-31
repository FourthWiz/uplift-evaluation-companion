"""Paired held-out-row percentile intervals for the current main example."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from uplift_lib import bootstrap_qini, adjusted_bootstrap
ROOT=Path('data/selection');P=ROOT/'selection_high/seed_42';OUT=ROOT/'intervals';OUT.mkdir(exist_ok=True)
def main():
 start=time.time();d=pd.read_pickle(P/'obs.pkl');raw=dict(np.load(P/'obs_scores.npz'))
 scores={k:v for k,v in raw.items() if k=='TrueCATE' or k.startswith(('linear/','xgb/'))};zs=dict(np.load(P/'evaluation_scores.npz'))
 for method in ['Qini','IPW','DR']:
  if (OUT/(method+'_intervals.csv')).exists():continue
  if method=='Qini':ci,pairs,draws=bootstrap_qini(d,scores,iterations=1000,seed=1042)
  else:ci,pairs,draws=adjusted_bootstrap(scores,zs[method+'_estimated'],iterations=1000,seed=1042);ci=ci.rename(columns={'DRGain':method})
  ci.to_csv(OUT/(method+'_intervals.csv'),index=False);pairs.to_csv(OUT/(method+'_pairs.csv'),index=False);draws.to_csv(OUT/(method+'_draws.csv'),index=False)
  print('Completed',method,'elapsed',time.time()-start,flush=True)
 (OUT/'method.json').write_text(json.dumps({'iterations':1000,'seed':1042,'scenario':'selection_high','data_seed':42,'n_eval':len(d),'conditional_on':['ranking fits','cross-fitted evaluation nuisance fits'],'paired':True,'interval':'percentile .025/.975','includes_training_or_nuisance_refit':False,'seconds':time.time()-start},indent=2))
if __name__=='__main__':main()
