"""Figures in the original article's explanatory sequence."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uplift_lib import *
P=Path('data/selection/selection_high/seed_42');OUT=Path('figures/narrative');OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=100);plt.close(fig)
def main():
 d=pd.read_pickle(P/'obs.pkl');scores=dict(np.load(P/'obs_scores.npz'));pred=scores['xgb/S']
 bins=pd.qcut(d.TrueCATE,10,labels=False);group=d.assign(bucket=bins).groupby('bucket');m=group.mean();xx=np.arange(1,11)
 fig,axes=plt.subplots(1,3,figsize=(12,4.4))
 axes[0].plot(xx,m.MadeOrder,'o-',color='#2474a6');axes[0].set(title='Observed order rate',ylabel='Fraction making an order')
 axes[1].bar(xx,m.TreatmentGroup,color='#e18a27');axes[1].axhline(.5,color='gray',ls='--');axes[1].set(title='Who receives treatment',ylabel='Treated fraction',ylim=(0,1))
 axes[2].plot(xx,m.P1,'o-',label='If treated',color='#2474a6');axes[2].plot(xx,m.P0,'o-',label='If untreated',color='#e18a27');axes[2].set(title='Known potential-outcome means',ylabel='Mean purchase probability');axes[2].legend(fontsize=9)
 for ax in axes:ax.set_xlabel('True-CATE decile (low → high)');ax.set_xticks([1,3,5,7,10])
 save(fig,'01_meet_the_data')
 ix=np.random.RandomState(42).choice(len(d),30000,replace=False);metrics=cate_metrics(d.TrueCATE.to_numpy(),pred)
 fig,ax=plt.subplots(figsize=(12,5));ax.scatter(d.TrueCATE.iloc[ix],pred[ix],s=3,alpha=.15,color='#2474a6');ax.plot([-.6,.8],[-.6,.8],'--',color='#bd5f3f')
 ax.set(xlabel='True CATE',ylabel='S-learner prediction',title=f"XGBoost S-learner: MSE {metrics['MSE']:.6f}, correlation {metrics['Correlation']:.4f}")
 save(fig,'02_s_learner_predictions')
 b=pd.qcut(pred,10,labels=False);z=d.assign(bucket=b).groupby(['bucket','TreatmentGroup']).MadeOrder.mean().unstack();truth=d.assign(bucket=b).groupby('bucket').TrueCATE.mean()
 fig,axes=plt.subplots(1,2,figsize=(12,4.5));axes[0].bar(xx-.18,z[0],width=.36,label='Controls',color='#e18a27');axes[0].bar(xx+.18,z[1],width=.36,label='Treated',color='#2474a6');axes[0].set(title='The usual uplift barplot',ylabel='Observed order rate');axes[0].legend(fontsize=9)
 axes[1].bar(xx,z[1]-z[0],color='#2474a6',alpha=.7,label='Observed treated − control');axes[1].plot(xx,truth,'ko--',label='Known mean CATE');axes[1].axhline(0,color='gray',lw=1);axes[1].set(title='Does the observed difference equal the effect?',ylabel='Probability difference');axes[1].legend(fontsize=9)
 for ax in axes:ax.set_xlabel('Predicted-CATE decile (low → high)');ax.set_xticks(xx)
 save(fig,'03_uplift_bars')
 fig,ax=plt.subplots(figsize=(12,4.5))
 for name,color,label in [('xgb/S','#2474a6','XGBoost S'),('TrueCATE','#333333','True-CATE ranking')]:
  f,q=qini_curve(d.MadeOrder,d.TreatmentGroup,scores[name]);ax.plot(f,q,color=color,label=label)
 ax.plot([0,1],[0,q[-1]],'--',color='#999999',label='Random-ranking diagonal')
 ax.set(xlabel='Fraction targeted (highest predicted benefit first)',ylabel='Qini / evaluation sample size',title='Observational Qini for the S-learner');ax.legend(fontsize=10)
 save(fig,'04_qini_for_s')
 # The CI figure is generated only after all bootstrap draws are available.
 pairs=pd.read_csv('data/selection/intervals/DR_pairs.csv')
 fig,ax=plt.subplots(figsize=(12,5.5))
 rows=[]
 for other in ['xgb/T','xgb/X','xgb/DA','xgb/DR','xgb/DML']:
  r=pairs[(pairs.A=='xgb/S')&(pairs.B==other)].iloc[0];rows.append(r)
 v=pd.DataFrame(rows);yy=np.arange(5)
 ax.errorbar(v.Delta,yy,xerr=[v.Delta-v.CI_low,v.CI_high-v.Delta],fmt='o',capsize=5,color='#2474a6')
 ax.set_yticks(yy,['S minus T','S minus X','S minus DA','S minus DR','S minus DML']);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Difference in DR gain area',title='Paired conditional 95% intervals: is S better or worse?\n1,000 held-out-row resamples · ranking and nuisance fits held fixed');ax.grid(axis='x',alpha=.2)
 save(fig,'05_paired_confidence_intervals')
 from evaluation_comparison import render
 render()
 print('Created six narrative figures')
if __name__=='__main__':main()
