"""Plot the fixed-protocol tuning results."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('data/tuning');F=Path('figures/tuning');F.mkdir(exist_ok=True)
names=['S','T','X','DA','DR','DML']
r=pd.read_csv(P/'test_results.csv').set_index(['Learner','Stage']);d=pd.read_csv(P/'before_after.csv').set_index('Learner')
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(metric,title) in zip(axs.flat,[('Qini','Ordinary Qini area (confounded)'),('IPW','IPW-adjusted gain area'),('DR','DR-adjusted gain area'),('RMSE','CATE RMSE (synthetic diagnostic)')]):
 for stage,color,offset in [('Before','#3478b8',-.09),('After','#df8528',.09)]:
  vals=[r.loc[(m,stage),metric] for m in names];ax.plot(np.arange(6)+offset,vals,'o',color=color,label=stage)
 for i,m in enumerate(names):ax.plot([i-.09,i+.09],[r.loc[(m,'Before'),metric],r.loc[(m,'After'),metric]],color='.7',zorder=0)
 ax.set_xticks(range(6),names);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 ax.set_ylabel('Lower is better' if metric=='RMSE' else 'Area; higher score')
axs[0,0].legend();fig.suptitle('Before and after tuning: same fresh 300,000 customers',fontsize=15)
fig.savefig(F/'01_before_after.png',dpi=100);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
for i,m in enumerate(names):
 row=d.loc[m];ax.plot([row.CI_low,row.CI_high],[i,i],color='#3478b8',lw=2);ax.plot(row.DeltaDR,i,'o',color='#df8528')
ax.axvline(0,color='.4',ls='--');ax.set_yticks(range(6),names);ax.invert_yaxis();ax.set_xlabel('DR gain area: after minus before');ax.set_title('Tuning gains with paired conditional 95% intervals');ax.grid(axis='x',alpha=.2)
fig.savefig(F/'02_tuning_intervals.png',dpi=120);plt.close(fig)