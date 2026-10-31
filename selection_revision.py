"""Prespecified selection gradient: bounded overlap, fixed population arm share."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from uplift_lib import *
from evaluation_models import evaluate_observational
ROOT=Path('data/selection')
SCENARIOS=list(SELECTION_STRENGTHS)

def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=[];summaries=[];start=time.time()
    for seed in [42,43,44]:
        for scenario in SCENARIOS:
            folder=ROOT/scenario/f'seed_{seed}'
            config=Config(n=1_000_000,seed=seed,assignment=scenario,include_forest=False)
            if (folder/'complete.json').exists():
                table=pd.read_csv(folder/'comparison.csv');summary=json.loads((folder/'selection_summary.json').read_text())
            else:
                r=fit_experiment(config,forest=False)
                for cohort in ['obs','rct']:
                    r[cohort+'_scores']['CATE+propensity']=r[cohort].TrueCATE.to_numpy()+r[cohort].Propensity.to_numpy()
                save_experiment(r,folder)
                table=evaluate_observational(r,folder,seed,threads=4,bootstrap=0)
                share=r['obs'].TreatmentGroup.mean()
                table['Qini_scaled']=table.NaiveQini/share
                table['Qini_error']=table.Qini_scaled-table.Oracle_obs
                table['IPW_error']=table.IPW_estimated-table.Oracle_obs
                table['DR_error']=table.DR_estimated-table.Oracle_obs
                table['OracleLoss']=table.loc[table.Model=='TrueCATE','Oracle_obs'].iloc[0]-table.Oracle_obs
                table['scenario']=scenario;table['seed']=seed
                table.to_csv(folder/'comparison.csv',index=False)
                summary=summarize(r['data']);summary.update(scenario=scenario,seed=seed,beta=SELECTION_STRENGTHS[scenario],max_propensity=float(r['data'].Propensity.max()),propensity_sd=float(r['data'].Propensity.std()))
                (folder/'selection_summary.json').write_text(json.dumps(summary,indent=2))
                (folder/'config.json').write_text(json.dumps(asdict(config),indent=2))
                (folder/'complete.json').write_text(json.dumps({'complete':True}))
                del r
            rows.append(table);summaries.append(summary)
            pd.concat(rows).to_csv(ROOT/'comparison.csv',index=False)
            pd.DataFrame(summaries).to_csv(ROOT/'summaries.csv',index=False)
            print('COMPLETE',scenario,seed,'elapsed',time.time()-start,flush=True)
    (ROOT/'execution.json').write_text(json.dumps({'seconds':time.time()-start,'n':1_000_000,'seeds':[42,43,44],'scenarios':SCENARIOS},indent=2))
if __name__=='__main__': run()
