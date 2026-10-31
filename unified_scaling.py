"""Use Qini / observed treatment share throughout the article; preserve raw audits.
Run with --raw-root pointing to full saved experiment data, or run in a full rerun.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['S','T','X','DA','DR','DML']
KEYS=['linear/DR']+['xgb/'+m for m in NAMES]
SCENARIOS=['selection_low','selection_mid','selection_high']

def curve(d,score):
    order=np.argsort(score,kind='mergesort')[::-1]
    end=np.r_[np.flatnonzero(np.diff(np.asarray(score)[order])),len(order)-1]
    t=d.TreatmentGroup.to_numpy()[order]; y=d.MadeOrder.to_numpy()[order]
    nt=np.cumsum(t)[end]; nc=np.cumsum(1-t)[end]
    rt=np.cumsum(t*y)[end]; rc=np.cumsum((1-t)*y)[end]
    q=rt-np.divide(rc*nt,nc,out=np.zeros_like(rt,dtype=float),where=nc>0)
    return np.r_[0,(end+1)/len(d)],np.r_[0,q/len(d)/t.mean()]

def run(root,raw):
    root=Path(root);raw=Path(raw); out=root/'data/selection/intervals'
    d=pd.read_pickle(raw/'data/selection/selection_high/seed_42/obs.pkl')
    t=d.TreatmentGroup.to_numpy();share=t.mean()
    tab=pd.read_csv(root/'data/selection/comparison.csv')
    primary=tab.query("scenario=='selection_high' and seed==42").set_index('Model')
    np.testing.assert_allclose(primary.NaiveQini/share,primary.Qini_scaled,atol=1e-12)
    meta=json.loads((out/'method.json').read_text())
    draws=pd.read_csv(out/'Qini_draws.csv');rng=np.random.RandomState(meta['seed'])
    shares=np.array([t[rng.randint(0,len(t),len(t))].mean() for _ in range(len(draws))])
    scaled=draws.div(shares,axis=0)
    scaled.to_csv(out/'Qini_scaled_draws.csv',index=False)
    pd.DataFrame({'treatment_share':shares}).to_csv(out/'Qini_bootstrap_shares.csv',index=False)
    cis=[];pairs=[]
    for model in scaled:
        lo,hi=np.quantile(scaled[model],[.025,.975])
        cis.append(dict(Model=model,Qini_scaled=primary.loc[model,'Qini_scaled'],CI_low=lo,CI_high=hi))
    for i,a in enumerate(scaled):
        for b in scaled.columns[i+1:]:
            lo,hi=np.quantile(scaled[a]-scaled[b],[.025,.975])
            pairs.append(dict(A=a,B=b,Delta=primary.loc[a,'Qini_scaled']-primary.loc[b,'Qini_scaled'],CI_low=lo,CI_high=hi))
    pd.DataFrame(cis).to_csv(out/'Qini_scaled_intervals.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'Qini_scaled_pairs.csv',index=False)
    (out/'scaling.json').write_text(json.dumps(dict(definition='raw Qini area divided by observed treatment share',observed_share=share,bootstrap='reconstructed identical RandomState row resamples; denominator recomputed within each resample',seed=meta['seed'],iterations=len(draws)),indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,path):
        fig.savefig(root/path,dpi=170,bbox_inches='tight');plt.close(fig)
    scores=dict(np.load(raw/'data/selection/selection_high/seed_42/obs_scores.npz'))
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    for key,color,style,label in [('xgb/S','#2474a6','-','XGBoost S'),('TrueCATE','#333333','--','True-CATE ranking')]:
        f,q=curve(d,scores[key]);ax.plot(f,q,color=color,ls=style,label=label)
        np.testing.assert_allclose(np.trapezoid(q-f*q[-1],f),primary.loc[key,'Qini_scaled'],atol=1e-12)
    ax.plot([0,1],[0,q[-1]],':',color='gray',label='Random-ranking diagonal')
    ax.set(xlabel='Fraction targeted',ylabel='Qini curve / treatment share',title='Same Qini scale throughout: Qini divided by treatment share');ax.legend()
    save(fig,'figures/narrative/04_qini_for_s.png')
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),sharey=True,layout='constrained')
    export=[]
    for ax,metric,title in zip(axes,['Qini_scaled','DR'],['Qini / treatment share','Doubly robust gain']):
        dd=pd.read_csv(out/f'{metric}_draws.csv')
        point=primary['Qini_scaled' if metric=='Qini_scaled' else 'DR_estimated']
        for y,m in enumerate(KEYS):
            lo,hi=np.quantile(dd[m]-dd.TrueCATE,[.025,.975]);delta=point[m]-point['TrueCATE']
            ax.plot([lo,hi],[y,y],lw=2,color='#2474a6');ax.plot(delta,y,'o',color='#2474a6')
            export.append(dict(Metric=metric,Model=m,Delta=delta,CI_low=lo,CI_high=hi))
        ax.axvline(0,color='gray',ls='--');ax.grid(axis='x',alpha=.2)
        ax.set(title=title,xlabel='Model score minus true-CATE ranking score')
        ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,-3))
    axes[0].set_yticks(range(7),['Linear DR']+['XGBoost '+m for m in NAMES]);axes[0].invert_yaxis()
    fig.suptitle('Each ranking compared with true CATE — paired conditional 95% intervals')
    fig.supxlabel('1,000 paired row resamples; fitted models fixed. Axis limits differ; intervals unadjusted for multiplicity.',fontsize=10)
    save(fig,'figures/narrative/05_oracle_paired_comparison.png')
    pd.DataFrame(export).to_csv(out/'oracle_paired_comparison.csv',index=False)
    fixed=tab[tab.Model=='TrueCATE'];SC=list(fixed.scenario.drop_duplicates())
    methods=['Qini_error','IPW_error','DR_error'];colors=['#c87716','#2474a6','#29874b']
    stats=fixed.groupby('scenario')[methods].agg(['mean','min','max']).loc[SC]
    stats.to_csv(root/'data/selection/evaluation_error_ranges.csv')
    fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained',gridspec_kw={'width_ratios':[1.4,1,1]})
    for ax,ms,title in [(axs[0],methods,'All evaluator errors'),(axs[1],['Qini_error'],'Qini: ranges shown numerically'),(axs[2],['IPW_error','DR_error'],'IPW / DR: expanded scale')]:
        for method in ms:
            j=methods.index(method);v=stats[method];xx=np.arange(3)+(j-1)*.055
            ax.errorbar(xx,v['mean'],yerr=[v['mean']-v['min'],v['max']-v['mean']],fmt='o-',markersize=3,capsize=7,elinewidth=1.7,capthick=1.7,color=colors[j],label=['Qini / share','IPW','DR'][j],zorder=3)
            if ax is axs[1]:
                for x,(_,row) in zip(xx,v.iterrows()):
                    ax.annotate(f"{row['min']:.5f}–{row['max']:.5f}",(x,row['mean']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
        ax.axhline(0,color='gray',ls='--',lw=.8);ax.set_xticks(range(3),['Lower','Middle','Higher']);ax.set_title(title,fontsize=11);ax.set_xlabel('Assignment selectivity');ax.grid(alpha=.15)
        ax.margins(x=.22,y=.2);ax.legend(fontsize=9)
    axs[0].set_ylabel('Evaluator score − known causal area\n(same true-CATE ranking on both sides)')
    fig.suptitle('A perfect ranking does not make its evaluator accurate')
    fig.supxlabel('Dots: mean over 3 seeds. Whiskers: minimum to maximum, not confidence intervals. Small ranges can be hidden by the full scale.',fontsize=10)
    save(fig,'figures/selection/03_evaluation_bias.png')
    test=pd.read_pickle(raw/'data/tuning/test.pkl');tshare=test.TreatmentGroup.mean()
    r=pd.read_csv(root/'data/tuning/test_results.csv');r['Qini_scaled']=r.Qini/tshare
    r.to_csv(root/'data/tuning/test_results_scaled.csv',index=False);r=r.set_index(['Learner','Stage'])
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(metric,title) in zip(axs.flat,[('Qini_scaled','Qini / treatment share (confounded)'),('IPW','IPW gain'),('DR','DR gain'),('RMSE','CATE RMSE (synthetic diagnostic)')]):
        for stage,color,offset in [('Before','#3478b8',-.09),('After','#df8528',.09)]:
            ax.plot(np.arange(6)+offset,[r.loc[(m,stage),metric] for m in NAMES],'o',color=color,label=stage)
        for i,m in enumerate(NAMES):ax.plot([i-.09,i+.09],[r.loc[(m,'Before'),metric],r.loc[(m,'After'),metric]],color='.7',zorder=0)
        ax.set_xticks(range(6),NAMES);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axs[0,0].legend();fig.suptitle('Before and after tuning: same fresh 300,000 customers')
    save(fig,'figures/tuning/01_before_after.png')
    print(pd.DataFrame(cis).set_index('Model').loc[['TrueCATE']+KEYS].to_string())
    print(stats.to_string())
    print('Verified scaled point estimates and curve integration; regenerated four figures.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--raw-root');a=p.parse_args();run(a.root,a.raw_root or a.root)
