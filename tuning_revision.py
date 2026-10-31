"""Equal configuration-count tuning; validation selection precedes fresh test."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from uplift_lib import *
ROOT=Path('data/tuning');ROOT.mkdir(exist_ok=True)
def stamp():return datetime.now(timezone.utc).isoformat()
def candidates():
 rng=np.random.RandomState(271027);out=[dict(id=0,base={},final={})]
 while len(out)<15:
  b=dict(n_estimators=int(rng.choice([80,150,250])),max_depth=int(rng.choice([2,3,4,5,6])),learning_rate=float(rng.choice([.03,.06,.1])),min_child_weight=int(rng.choice([1,5,20])),reg_lambda=int(rng.choice([1,5,20])))
  f=dict(n_estimators=int(rng.choice([50,100,150])),max_depth=int(rng.choice([2,3,4])))
  if not any(o['base']==b for o in out):out.append(dict(id=len(out),base=b,final=f))
 return out

def run():
 start=time.time();configs=candidates()
 protocol=dict(created_at=stamp(),scenario='selection_high',data_n=1_000_000,data_seed=42,split_seed=2026,search_train_n=500000,validation_n=200000,refit_train_n=700000,test_n=300000,test_seed=9173,search_metric='cross-fitted estimated DR gain',trials_per_learner=15,learners=LEARNERS,configs=configs,truth_used_for_selection=False,nuisance_settings='fixed 200 trees depth4 lr.05 3fold, shared across all candidates',bootstrap=1000)
 protocol_path=ROOT/'protocol.json'
 if protocol_path.exists():
  old=json.loads(protocol_path.read_text());assert old['configs']==configs and old['test_seed']==9173
 else: protocol_path.write_text(json.dumps(protocol,indent=2))
 d=generate_dataset(1_000_000,42,assignment='selection_high');train,old_holdout=train_test_split(np.arange(len(d)),test_size=.3,random_state=42)
 search,validation=train_test_split(train,test_size=200000,random_state=2026)
 assert len(search)==500000 and len(validation)==200000 and not len(np.intersect1d(search,validation))
 assert not len(np.intersect1d(train,old_holdout))
 np.savez_compressed(ROOT/'split_indices.npz',refit_train=train,search_train=search,validation=validation,unused_old_holdout=old_holdout)
 obs=d.iloc[validation].reset_index(drop=True)
 if not (ROOT/'validation_scores.npz').exists():
  nu,audit=crossfit_evaluation_nuisance(obs[FEATURES+['TreatmentGroup','MadeOrder']],seed=3026,threads=4)
  z=adjusted_scores(obs.MadeOrder,obs.TreatmentGroup,nu.e_hat,nu.mu0_hat,nu.mu1_hat)
  np.savez_compressed(ROOT/'validation_scores.npz',DR=z);nu.to_csv(ROOT/'validation_nuisance.csv',index=False)
  (ROOT/'validation_crossfit.json').write_text(json.dumps(audit,indent=2))
 else:z=np.load(ROOT/'validation_scores.npz')['DR']
 X=d[FEATURES].to_numpy();y=d.MadeOrder.to_numpy();t=d.TreatmentGroup.to_numpy()
 history=pd.read_csv(ROOT/'search_history.csv').to_dict('records') if (ROOT/'search_history.csv').exists() else []
 for learner in LEARNERS:
  for cfg in configs:
   if any(r['Learner']==learner and r['config_id']==cfg['id'] for r in history):continue
   tick=time.time();est=model_factories(seed=42,threads=4,base_params=cfg['base'],final_params=cfg['final'])[learner]()
   est.fit(y[search],t[search],X=X[search]);pred=est.effect(X[validation]).ravel()
   score=area(*gain_curve(pred,z))
   history.append(dict(Learner=learner,config_id=cfg['id'],ValidationDR=score,seconds=time.time()-tick))
   pd.DataFrame(history).to_csv(ROOT/'search_history.csv',index=False)
   print('SEARCH',learner,cfg['id'],f'DR={score:.8f}',f'{time.time()-tick:.1f}s',flush=True);del est
 # Freeze choices solely from validation, before generating/evaluating final test.
 h=pd.DataFrame(history);assert len(h)==90
 winners={learner:int(h[h.Learner==learner].sort_values(['ValidationDR','config_id'],ascending=[False,True]).iloc[0].config_id) for learner in LEARNERS}
 frozen=dict(frozen_at=stamp(),selected=winners,search_history_sha256=hashlib.sha256((ROOT/'search_history.csv').read_bytes()).hexdigest(),configs={k:configs[i] for k,i in winners.items()})
 if (ROOT/'selection_frozen.json').exists():assert json.loads((ROOT/'selection_frozen.json').read_text())['selected']==winners
 else:(ROOT/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
 print('SELECTION FROZEN',winners,flush=True)
 test=generate_dataset(300000,9173,assignment='selection_high');test.to_pickle(ROOT/'test.pkl');Xt=test[FEATURES].to_numpy()
 (ROOT/'test_opened.json').write_text(json.dumps({'opened_at':stamp(),'selection_was_frozen':True,'seed':9173},indent=2))
 if not (ROOT/'test_scores_evaluation.npz').exists():
  nu,audit=crossfit_evaluation_nuisance(test[FEATURES+['TreatmentGroup','MadeOrder']],seed=11173,threads=4)
  ze=dict(IPW=adjusted_scores(test.MadeOrder,test.TreatmentGroup,nu.e_hat),DR=adjusted_scores(test.MadeOrder,test.TreatmentGroup,nu.e_hat,nu.mu0_hat,nu.mu1_hat))
  np.savez_compressed(ROOT/'test_scores_evaluation.npz',**ze);nu.to_csv(ROOT/'test_nuisance.csv',index=False)
  (ROOT/'test_crossfit.json').write_text(json.dumps(audit,indent=2))
 else:ze=dict(np.load(ROOT/'test_scores_evaluation.npz'))
 predictions=dict(np.load(ROOT/'test_predictions.npz')) if (ROOT/'test_predictions.npz').exists() else {}
 for learner in LEARNERS:
  for stage,cfg in [('Before',configs[0]),('After',configs[winners[learner]])]:
   key=learner+'/'+stage
   if key in predictions:continue
   if stage=='After' and winners[learner]==0:predictions[key]=predictions[learner+'/Before'].copy()
   else:
    tick=time.time();est=model_factories(seed=42,threads=4,base_params=cfg['base'],final_params=cfg['final'])[learner]()
    est.fit(y[train],t[train],X=X[train]);predictions[key]=est.effect(Xt).ravel();del est
    print('REFIT',key,round(time.time()-tick,1),flush=True)
   np.savez_compressed(ROOT/'test_predictions.npz',**predictions)
 rows=[]
 for key,pred in predictions.items():
  learner,stage=key.split('/')
  rows.append(dict(Learner=learner,Stage=stage,**cate_metrics(test.TrueCATE.to_numpy(),pred),RMSE=float(np.sqrt(np.mean((test.TrueCATE-pred)**2))),Qini=qini_auc(test.MadeOrder,test.TreatmentGroup,pred),IPW=area(*gain_curve(pred,ze['IPW'])),DR=area(*gain_curve(pred,ze['DR'])),Oracle=area(*gain_curve(pred,test.TrueCATE))))
 pd.DataFrame(rows).to_csv(ROOT/'test_results.csv',index=False)
 ci,pairs,draws=adjusted_bootstrap(predictions,ze['DR'],iterations=1000,seed=12173)
 ci.to_csv(ROOT/'dr_intervals.csv',index=False);pairs.to_csv(ROOT/'dr_pairs.csv',index=False);draws.to_csv(ROOT/'dr_draws.csv',index=False)
 changes=[]
 for learner in LEARNERS:
  delta=draws[learner+'/After']-draws[learner+'/Before'];lo,hi=np.quantile(delta,[.025,.975]);r=pd.DataFrame(rows).query('Learner==@learner').set_index('Stage')
  changes.append(dict(Learner=learner,DeltaDR=r.loc['After','DR']-r.loc['Before','DR'],CI_low=lo,CI_high=hi,RMSE_before=r.loc['Before','RMSE'],RMSE_after=r.loc['After','RMSE']))
 pd.DataFrame(changes).to_csv(ROOT/'before_after.csv',index=False)
 (ROOT/'execution.json').write_text(json.dumps({'seconds':time.time()-start,'completed_at':stamp(),'trials':len(history),'test_n':len(test)},indent=2))
 print('TUNING COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':run()
