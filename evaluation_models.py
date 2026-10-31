"""Run the new main scenario; reuse the shared DGP, factories and metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from uplift_lib import *


def evaluate_observational(result,folder,seed=42,threads=4,bootstrap=200):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    d=result['obs'];rct=result['rct']
    nuisance,audit=crossfit_evaluation_nuisance(d[FEATURES+['TreatmentGroup','MadeOrder']],seed+2000,threads)
    nuisance.to_csv(folder/'evaluation_nuisance.csv',index=False)
    (folder/'evaluation_crossfit_audit.json').write_text(json.dumps(audit,indent=2))
    y=d.MadeOrder.to_numpy();t=d.TreatmentGroup.to_numpy();e=d.Propensity.to_numpy()
    components={
        'IPW_estimated':adjusted_scores(y,t,nuisance.e_hat),
        'DR_estimated':adjusted_scores(y,t,nuisance.e_hat,nuisance.mu0_hat,nuisance.mu1_hat),
        'IPW_known':adjusted_scores(y,t,e),
        'DR_known':adjusted_scores(y,t,e,d.P0,d.P1),
        'Oracle_obs':d.TrueCATE.to_numpy()}
    np.savez_compressed(folder/'evaluation_scores.npz',**components)
    zrct=adjusted_scores(rct.MadeOrder.to_numpy(),rct.TreatmentGroup.to_numpy(),np.full(len(rct),.5))
    rows=[]
    for name,s in result['obs_scores'].items():
        sr=result['rct_scores'][name]
        rows.append(dict(Model=name,**cate_metrics(d.TrueCATE.to_numpy(),s),
            NaiveQini=qini_auc(y,t,s),
            NaiveGain=area(*gain_curve(s,adjusted_scores(y,t,np.full(len(t),t.mean())))),
            **{key:area(*gain_curve(s,z)) for key,z in components.items()},
            RandomizedGain=area(*gain_curve(sr,zrct)),Oracle_rct=area(*gain_curve(sr,rct.TrueCATE.to_numpy()))))
    table=pd.DataFrame(rows);table.to_csv(folder/'adjusted_metrics.csv',index=False)
    health=[]
    for label,ee in [('known',e),('estimated',nuisance.e_hat.to_numpy())]:
        for arm in (0,1):
            w=1/np.where(t==1,ee,1-ee)[t==arm]
            health.append(dict(propensity=label,arm=arm,n=len(w),ESS=w.sum()**2/(w@w),max_weight=w.max(),HT_mass=w.sum()/len(d)))
    pd.DataFrame(health).to_csv(folder/'weight_diagnostics.csv',index=False)
    diagnostics=dict(propensity_rmse=float(np.sqrt(np.mean((nuisance.e_raw-e)**2))),
        propensity_clip=.005,clipped_fraction=float(((nuisance.e_raw<.005)|(nuisance.e_raw>.995)).mean()),
        mu0_rmse=float(np.sqrt(np.mean((nuisance.mu0_hat-d.P0)**2))),
        mu1_rmse=float(np.sqrt(np.mean((nuisance.mu1_hat-d.P1)**2))),
        DR_ATE=float(components['DR_estimated'].mean()),IPW_ATE=float(components['IPW_estimated'].mean()),
        TrueATE=float(d.TrueCATE.mean()))
    (folder/'evaluation_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    sensitivity=[]
    for clip in [.001,.005,.02]:
        ec=np.clip(nuisance.e_raw,clip,1-clip)
        zz=adjusted_scores(y,t,ec,nuisance.mu0_hat,nuisance.mu1_hat)
        for name,s in result['obs_scores'].items():
            sensitivity.append(dict(clip=clip,Model=name,DRGain=area(*gain_curve(s,zz))))
    pd.DataFrame(sensitivity).to_csv(folder/'clipping_sensitivity.csv',index=False)
    if bootstrap:
        ci,pairs,draws=adjusted_bootstrap(result['obs_scores'],components['DR_estimated'],bootstrap)
        ci.to_csv(folder/'dr_bootstrap.csv',index=False);pairs.to_csv(folder/'dr_paired_differences.csv',index=False)
        draws.to_csv(folder/'dr_bootstrap_draws.csv',index=False)
    return table

