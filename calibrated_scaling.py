"""Fixed, simulation-specific alignment to DR on the high-policy seed-42 oracle.
Positive constants are frozen across models, resamples, seeds, policies and tuning.
This changes presentation, not confounding, learner ordering or statistical evidence.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unified_scaling import curve
NAMES=['S','T','X','DA','DR','DML']
KEYS=['linear/DR']+['xgb/'+m for m in NAMES]
COLORS=['#c87716','#2474a6','#29874b']

def run(root='.',raw=None):
 root=Path(root);raw=Path(raw or root);p=root/'data/selection';ints=p/'intervals'
 tab=pd.read_csv(p/'comparison.csv');baseline=tab.query("scenario=='selection_high' and seed==42").set_index('Model')
 anchor=baseline.loc['TrueCATE'];cq=float(anchor.Qini_scaled/anchor.DR_estimated);ci=float(anchor.IPW_estimated/anchor.DR_estimated)
 calibration=dict(reference='TrueCATE, selection_high, seed_42',reference_DR=float(anchor.DR_estimated),Qini_divisor=cq,IPW_divisor=ci,DR_divisor=1.,frozen_across=['learners','bootstrap resamples','seeds','assignment policies','tuning test'],interpretation='Oracle-assisted display calibration; not confounding adjustment; not available without the reference true-CATE ranking',intervals='Conditional on fitted models and frozen calibration constants; calibration-estimation uncertainty is excluded')
 (p/'calibration.json').write_text(json.dumps(calibration,indent=2))
 tab['Qini_calibrated']=tab.Qini_scaled/cq;tab['IPW_calibrated']=tab.IPW_estimated/ci
 tab['Qini_calibrated_error']=tab.Qini_calibrated-tab.Oracle_obs;tab['IPW_calibrated_error']=tab.IPW_calibrated-tab.Oracle_obs
 tab.to_csv(p/'comparison_calibrated.csv',index=False)
 primary=tab.query("scenario=='selection_high' and seed==42").set_index('Model')
 np.testing.assert_allclose(primary.loc['TrueCATE',['Qini_calibrated','IPW_calibrated','DR_estimated']].to_numpy(float),anchor.DR_estimated,atol=1e-14)
 for source,target,divisor in [('Qini_scaled','Qini_calibrated',cq),('IPW','IPW_calibrated',ci)]:
  draws=pd.read_csv(ints/f'{source}_draws.csv')/divisor;draws.to_csv(ints/f'{target}_draws.csv',index=False)
  intervals=pd.read_csv(ints/f'{source}_intervals.csv');val=source
  intervals=intervals.rename(columns={val:target})
  intervals[[target,'CI_low','CI_high']]/=divisor;intervals.to_csv(ints/f'{target}_intervals.csv',index=False)
  pairs=pd.read_csv(ints/f'{source}_pairs.csv');pairs[['Delta','CI_low','CI_high']]/=divisor;pairs.to_csv(ints/f'{target}_pairs.csv',index=False)
  for _,r in intervals.iterrows():np.testing.assert_allclose(np.quantile(draws[r.Model],[.025,.975]),[r.CI_low,r.CI_high],atol=1e-12)
  for _,r in pairs.iterrows():np.testing.assert_allclose(np.quantile(draws[r.A]-draws[r.B],[.025,.975]),[r.CI_low,r.CI_high],atol=1e-12)
  # Positive scaling must preserve every pair's direction and zero inclusion.
  old=pd.read_csv(ints/f'{source}_pairs.csv');assert np.array_equal(np.sign(old.Delta),np.sign(pairs.Delta))
  assert np.array_equal((old.CI_low>0)|(old.CI_high<0),(pairs.CI_low>0)|(pairs.CI_high<0))
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 def save(fig,path):fig.savefig(root/path,dpi=170,bbox_inches='tight');plt.close(fig)
 d=pd.read_pickle(raw/'data/selection/selection_high/seed_42/obs.pkl');scores=dict(np.load(raw/'data/selection/selection_high/seed_42/obs_scores.npz'))
 fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
 for key,color,style,label in [('xgb/S','#2474a6','-','XGBoost S'),('TrueCATE','#333333','--','True-CATE ranking')]:
  f,q=curve(d,scores[key]);q=q/cq;ax.plot(f,q,color=color,ls=style,label=label)
  np.testing.assert_allclose(np.trapezoid(q-f*q[-1],f),primary.loc[key,'Qini_calibrated'],atol=1e-12)
 ax.plot([0,1],[0,q[-1]],':',color='gray',label='Random-ranking diagonal')
 ax.set(xlabel='Fraction targeted',ylabel='Calibrated Qini curve',title='Qini calibrated to DR at the reference true-CATE ranking');ax.legend()
 save(fig,'figures/narrative/04_qini_for_s.png')
 fig,axes=plt.subplots(1,2,figsize=(13,5.7),sharey=True,layout='constrained');export=[]
 for ax,metric,title in zip(axes,['Qini_calibrated','DR'],['Calibrated Qini','DR gain (reference scale)']):
  dd=pd.read_csv(ints/f'{metric}_draws.csv');point=primary['Qini_calibrated' if metric=='Qini_calibrated' else 'DR_estimated']
  for y,m in enumerate(KEYS):
   lo,hi=np.quantile(dd[m]-dd.TrueCATE,[.025,.975]);delta=point[m]-point['TrueCATE']
   ax.plot([lo,hi],[y,y],lw=2,color='#2474a6');ax.plot(delta,y,'o',color='#2474a6');export.append(dict(Metric=metric,Model=m,Delta=delta,CI_low=lo,CI_high=hi))
  ax.axvline(0,color='gray',ls='--');ax.grid(axis='x',alpha=.2);ax.set(title=title,xlabel='Model score minus true-CATE ranking score');ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
 axes[0].set_yticks(range(7),['Linear DR']+['XGBoost '+m for m in NAMES]);axes[0].invert_yaxis()
 fig.suptitle('Each ranking compared with true CATE — paired conditional 95% intervals')
 fig.supxlabel('1,000 paired row resamples; fits and calibration constants fixed. Different axis limits; unadjusted intervals.',fontsize=10)
 save(fig,'figures/narrative/05_oracle_paired_comparison.png');pd.DataFrame(export).to_csv(ints/'oracle_paired_comparison.csv',index=False)
 # Error for each boosted ranking after all evaluator scores share the same anchor.
 rows=primary.loc[['xgb/'+m for m in NAMES]];cols=['Qini_calibrated','IPW_calibrated','DR_estimated'];labels=['Calibrated Qini','Calibrated IPW','DR']
 errors=rows[cols].subtract(rows.Oracle_obs,axis=0);errors.to_csv(p/'calibrated_learner_errors.csv')
 from evaluation_comparison import render
 render(p/'comparison_calibrated.csv', root/'figures/narrative/06_all_learner_evaluations.png')
 fixed=tab[tab.Model=='TrueCATE'];sc=list(fixed.scenario.drop_duplicates());methods=['Qini_calibrated_error','IPW_calibrated_error','DR_error'];stats=fixed.groupby('scenario')[methods].agg(['mean','min','max']).loc[sc];stats.to_csv(p/'calibrated_policy_errors.csv')
 fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
 for ax,ms,title in [(axs[0],methods,'All evaluators, using fixed reference calibration'),(axs[1],methods[1:],'Expanded view: calibrated IPW and DR')]:
  for method in ms:
   j=methods.index(method);v=stats[method];xx=np.arange(3)+(j-1)*.06
   ax.errorbar(xx,v['mean'],yerr=[v['mean']-v['min'],v['max']-v['mean']],fmt='o-',markersize=3,capsize=6,elinewidth=1.5,capthick=1.5,color=COLORS[j],label=labels[j])
  ax.axhline(0,color='gray',ls='--',lw=.8);ax.set_xticks(range(3),['Lower','Middle','Higher']);ax.set(title=title,xlabel='Assignment selectivity',ylabel='Calibrated score − known causal area');ax.grid(alpha=.15);ax.legend(fontsize=9);ax.margins(x=.14,y=.15)
 fig.suptitle('Does calibration on one assignment policy carry over to another?')
 fig.supxlabel('Same true-CATE ranking within each seed. Fixed high-selectivity, seed-42 calibration. Dots: 3-seed means; whiskers: min–max.',fontsize=9)
 save(fig,'figures/selection/03_evaluation_bias.png')
 tuning=pd.read_csv(root/'data/tuning/test_results_scaled.csv');tuning['Qini_calibrated']=tuning.Qini_scaled/cq;tuning['IPW_calibrated']=tuning.IPW/ci;tuning.to_csv(root/'data/tuning/test_results_calibrated.csv',index=False);r=tuning.set_index(['Learner','Stage'])
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for ax,(metric,title) in zip(axs.flat,[('Qini_calibrated','Calibrated Qini (still confounded)'),('IPW_calibrated','Calibrated IPW gain'),('DR','DR gain (reference scale)'),('RMSE','CATE RMSE (synthetic diagnostic)')]):
  for stage,color,offset in [('Before','#3478b8',-.09),('After','#df8528',.09)]:ax.plot(np.arange(6)+offset,[r.loc[(m,stage),metric] for m in NAMES],'o',color=color,label=stage)
  for i,m in enumerate(NAMES):ax.plot([i-.09,i+.09],[r.loc[(m,'Before'),metric],r.loc[(m,'After'),metric]],color='.7',zorder=0)
  ax.set_xticks(range(6),NAMES);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 axs[0,0].legend();fig.suptitle('Before and after tuning — calibration constants remain fixed')
 save(fig,'figures/tuning/01_before_after.png')
 print(json.dumps(calibration,indent=2));print(primary.loc[['TrueCATE']+KEYS,cols+['Oracle_obs']].to_string());print(stats.to_string())
 print('Verified anchor equality, curve areas, interval quantiles and unchanged ranking/zero-exclusion decisions.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--raw-root');a=p.parse_args();run(a.root,a.raw_root)
