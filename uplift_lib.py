"""Single source of truth for the corrected CATE experiment.

The published DGP is unchanged. Qini follows scikit-uplift's prefix-count
convention; reported area is baseline-subtracted and divided by N**2.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from econml.metalearners import SLearner, TLearner, XLearner, DomainAdaptationLearner
from econml.dr import DRLearner
from econml.dml import NonParamDML, CausalForestDML

FEATURES = ['Age', 'Income', 'MoneySpent', 'DaysLastActivity']
LEARNERS = ['S', 'T', 'X', 'DA', 'DR', 'DML']

@dataclass
class Config:
    n: int = 1_000_000
    seed: int = 42
    test_size: float = .3
    bootstrap: int = 200
    seeds: tuple = (42, 43, 44)
    sizes: tuple = (100_000, 1_000_000)
    threads: int = 4
    include_forest: bool = True
    propensity_clip: float = .001
    assignment: str = "published"


def outcome_probability(age, income, money, days, t):
    # Preserve expression order and legacy RandomState for published-seed replication.
    z = (-1 - .02*age + .00005*income + .00003*money + .01*days
         + .5*t - .011*t*days + .05*age*t - .000001*age*income
         - .0000005*money*days - .00004*days*age*t)
    return 1 / (1 + np.exp(-z))


SELECTION_STRENGTHS = {"selection_low": .5, "selection_medium": 1., "selection_high": 2.5}


def selection_propensity(age, days, strength):
    """Bounded selection with population treatment share 1/2.

    Calibrate on the exact discrete Age x Days population, without outcomes.
    The .1/.9 bounds hold for every profile, not only observed samples.
    """
    from scipy.special import expit
    grid = (np.arange(18,71)[:,None] * np.arange(366)[None,:]).ravel()
    mean, sd = grid.mean(), grid.std()
    zgrid = (grid-mean)/sd
    lo, hi = -20., 20.
    for _ in range(70):
        mid = (lo+hi)/2
        if expit(mid-strength*zgrid).mean() < .5: lo=mid
        else: hi=mid
    intercept=(lo+hi)/2
    return .1+.8*expit(intercept-strength*(np.asarray(age)*np.asarray(days)-mean)/sd)


def generate_dataset(n=1_000_000, seed=42, randomized=False, assignment="published"):
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 71, n)
    income = np.clip(rng.normal(50000, 15000, n), 0, None)
    money = np.clip(rng.normal(1000, 500, n), 0, None)
    days = rng.randint(0, 366, n)
    if assignment in SELECTION_STRENGTHS:
        e = selection_propensity(age, days, SELECTION_STRENGTHS[assignment])
    elif assignment in {"published", "moderate"}:
        slope = .01 if assignment == "published" else 1/4000
        e = 1 / (1 + np.exp(slope*(days*age - 7000)))
    else:
        raise ValueError("Unknown assignment scenario: " + str(assignment))
    assignment = np.full(n, .5) if randomized else e
    t = rng.binomial(1, assignment)
    p = outcome_probability(age, income, money, days, t)
    p1 = outcome_probability(age, income, money, days, 1)
    p0 = outcome_probability(age, income, money, days, 0)
    y = rng.binomial(1, p)
    return pd.DataFrame(dict(Age=age, Income=income, MoneySpent=money,
        DaysLastActivity=days, TreatmentGroup=t, MadeOrder=y, ProbOrder=p,
        P1=p1, P0=p0, TrueCATE=p1-p0, Propensity=e, AssignmentProb=assignment))


def summarize(d):
    t = d.TreatmentGroup.to_numpy().astype(bool)
    e = d.Propensity.to_numpy()
    return dict(n=len(d), treatment_rate=float(t.mean()),
        treatment_outcome=float(d.MadeOrder[t].mean()), control_outcome=float(d.MadeOrder[~t].mean()),
        naive_difference=float(d.MadeOrder[t].mean()-d.MadeOrder[~t].mean()),
        ATE=float(d.TrueCATE.mean()), ATT=float(d.TrueCATE[t].mean()), ATC=float(d.TrueCATE[~t].mean()),
        below_005=float((e<.05).mean()), above_095=float((e>.95).mean()),
        overlap=float(((e>=.05)&(e<=.95)).mean()), rounded_one=float((e==1).mean()),
        min_propensity=float(e.min()), corr_propensity_cate=float(d[['Propensity','TrueCATE']].corr().iloc[0,1]))


class ProbabilityXGB(XGBClassifier):
    def predict(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)[:, 1]


class ScaledLinear(RegressorMixin, BaseEstimator):
    """OLS with internal training-only scaling and sample_weight support."""
    def fit(self, X, y, sample_weight=None):
        self.scaler_ = StandardScaler().fit(X, sample_weight=sample_weight)
        self.model_ = LinearRegression().fit(self.scaler_.transform(X), y, sample_weight=sample_weight)
        self.n_features_in_ = X.shape[1]
        return self
    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X))


class ClippedLogistic(ClassifierMixin, BaseEstimator):
    """Training-only scaling; clipping explicitly changes the fitted propensity."""
    def __init__(self, clip=.001, seed=42):
        self.clip = clip
        self.seed = seed
    def fit(self, X, y, sample_weight=None):
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = LogisticRegression(max_iter=1000, random_state=self.seed).fit(
            self.scaler_.transform(X), y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self
    def predict_proba(self, X):
        p = np.clip(self.model_.predict_proba(self.scaler_.transform(X))[:, 1], self.clip, 1-self.clip)
        return np.column_stack([1-p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def model_factories(family='xgb', seed=42, threads=4, clip=.001, forest=False, base_params=None, final_params=None):
    """Explicit reproducible settings, not a claim to recover lost hyperparameters.

    X/DA use logistic propensity as in the former notebook, now scaled/clipped.
    DR/DML use family-matched propensity. No truth enters model training.
    """
    pars = dict(n_estimators=100, max_depth=6, learning_rate=.1,
                tree_method='hist', n_jobs=threads, random_state=seed, verbosity=0)
    if base_params is not None: pars.update(base_params)
    final_pars = {**pars, 'n_estimators':50, 'max_depth':4}
    if final_params is not None: final_pars.update(final_params)
    reg = lambda: XGBRegressor(**pars) if family=='xgb' else ScaledLinear()
    outcome = lambda: ProbabilityXGB(**pars, eval_metric='logloss') if family=='xgb' else ScaledLinear()
    final = lambda: XGBRegressor(**final_pars) if family=='xgb' else ScaledLinear()
    prop = lambda: XGBClassifier(**pars, eval_metric='logloss') if family=='xgb' else ClippedLogistic(clip,seed)
    cv = lambda: StratifiedKFold(3, shuffle=True, random_state=seed)
    models = {
        'S': lambda: SLearner(overall_model=outcome()),
        'T': lambda: TLearner(models=outcome()),
        'X': lambda: XLearner(models=outcome(), cate_models=reg(), propensity_model=ClippedLogistic(clip,seed)),
        'DA': lambda: DomainAdaptationLearner(models=outcome(), final_models=reg(), propensity_model=ClippedLogistic(clip,seed)),
        'DR': lambda: DRLearner(model_regression=reg(), model_propensity=prop(), model_final=final(),
                               min_propensity=clip, cv=cv(), random_state=seed),
        'DML': lambda: NonParamDML(model_y=reg(), model_t=prop(), model_final=final(),
                                 discrete_treatment=True, cv=cv(), random_state=seed),
    }
    if forest:
        models['CausalForest'] = lambda: CausalForestDML(model_y=reg(), model_t=prop(),
            discrete_treatment=True, cv=cv(), random_state=seed, n_estimators=100,
            max_samples=.1, min_samples_leaf=50, max_depth=12, n_jobs=threads,
            inference=False)
    return models


def cate_metrics(truth, score):
    if len(truth)==0:
        return dict(MSE=np.nan,MAE=np.nan,Correlation=np.nan)
    return dict(MSE=float(np.mean((truth-score)**2)), MAE=float(np.mean(abs(truth-score))),
                Correlation=float(np.corrcoef(truth,score)[0,1]) if np.std(score)>1e-14 else np.nan)


def ranking(score):
    order = np.argsort(score, kind='mergesort')[::-1]
    end = np.r_[np.flatnonzero(np.diff(np.asarray(score)[order])), len(order)-1]
    return order,end


def qini_curve(y, t, score, weights=None):
    """Q(k)=R_t(k)-R_c(k)*N_t(k)/N_c(k), at distinct score thresholds.

    Zero control prefixes use zero for the control contribution, matching sklift.
    Returns fraction targeted and Q/N; integrates to raw baseline-subtracted area/N².
    """
    order,end = ranking(score)
    return _qini_sorted(np.asarray(y)[order], np.asarray(t)[order], end,
                        None if weights is None else np.asarray(weights)[order])


def _qini_sorted(y,t,end,w=None):
    if w is None: w=np.ones(len(y))
    w=np.asarray(w,dtype=float)
    n=np.sum(w)
    nt=np.cumsum(w*t)[end]; nc=np.cumsum(w*(1-t))[end]
    rt=np.cumsum(w*t*y)[end]; rc=np.cumsum(w*(1-t)*y)[end]
    q=rt-np.divide(rc*nt,nc,out=np.zeros_like(rt),where=nc>0)
    return np.r_[0,(nt+nc)/n],np.r_[0,q/n]


def area(x,q):
    return float(np.trapezoid(q-x*q[-1],x))


def qini_auc(y,t,score,weights=None):
    return area(*qini_curve(y,t,score,weights))


def legacy_curve(y,t,score):
    """Old notebook formula with stable tie ordering, for diagnosis only."""
    order,_=ranking(score);y=np.asarray(y)[order];t=np.asarray(t)[order]
    f=np.arange(1,101)/100; k=(f*len(y)).astype(int)-1
    nt=np.cumsum(t)[k];nc=k+1-nt
    q=np.cumsum(y*t)[k]/t.sum()-np.cumsum(y*(1-t))[k]/(1-t).sum()
    q[(nt==0)|(nc==0)]=0
    return f,q


def legacy_ratio(y,t,score):
    f,q=legacy_curve(y,t,score)
    den=np.trapezoid(f*q[-1],f)
    return float(np.trapezoid(q,f)/den) if den else np.nan


def ipw_curve(y,t,score,e,clip=None):
    """HT gain: sum(TY/e - (1-T)Y/(1-e))/N, not a rescue for absent support.

    Unclipped uses only each observation's factual-arm denominator to avoid 0/0.
    Clipping and overlap restriction are separately labeled sensitivity analyses.
    """
    e=np.asarray(e);t=np.asarray(t);y=np.asarray(y)
    if clip is not None:e=np.clip(e,clip,1-clip)
    denom=np.where(t==1,e,1-e)
    if np.any(denom<=0):raise ValueError('Observed assignment has zero probability')
    z=np.where(t==1,y,-y)/denom
    order,end=ranking(score)
    return np.r_[0,(end+1)/len(y)],np.r_[0,np.cumsum(z[order])[end]/len(y)]


def expected_curve(d,score,kind='oracle'):
    order,end=ranking(score);n=len(d)
    e=d.Propensity.to_numpy()[order];p1=d.P1.to_numpy()[order];p0=d.P0.to_numpy()[order]
    if kind=='oracle':q=np.cumsum(p1-p0)
    elif kind=='legacy':q=np.cumsum(e*p1)/e.sum()-np.cumsum((1-e)*p0)/(1-e).sum();q=q*n
    else:
        et=np.cumsum(e);ec=np.cumsum(1-e)
        q=np.cumsum(e*p1)-np.divide(np.cumsum((1-e)*p0)*et,ec,out=np.zeros(n),where=ec>0)
    return np.r_[0,(end+1)/n],np.r_[0,q[end]/n]


def evaluate(d,scores):
    y=d.MadeOrder.to_numpy();t=d.TreatmentGroup.to_numpy();e=d.AssignmentProb.to_numpy()
    rows=[]
    for name,s in scores.items():
        rows.append(dict(Model=name,**cate_metrics(d.TrueCATE.to_numpy(),s),
            Qini=qini_auc(y,t,s), LegacyRatio=legacy_ratio(y,t,s),
            IPW=area(*ipw_curve(y,t,s,e)),IPW_clip_001=area(*ipw_curve(y,t,s,e,.01)),
            OracleGain=area(*expected_curve(d,s)),
            ExpectedObservational=area(*expected_curve(d,s,'observational')),
            ExpectedLegacy=area(*expected_curve(d,s,'legacy'))))
    return pd.DataFrame(rows)


def overlap_tables(d,scores):
    e=d.Propensity.to_numpy();rows=[]
    for region,mask in [('low',e<.05),('overlap',(e>=.05)&(e<=.95)),('high',e>.95)]:
        for name,s in scores.items():
            rows.append(dict(Region=region,Model=name,n=int(mask.sum()),
                treated=int(d.TreatmentGroup.to_numpy()[mask].sum()),**cate_metrics(d.TrueCATE.to_numpy()[mask],s[mask]),
                MeanEstimate=float(np.mean(s[mask])) if mask.any() else np.nan,
                MinEstimate=float(np.min(s[mask])) if mask.any() else np.nan,MaxEstimate=float(np.max(s[mask])) if mask.any() else np.nan))
    bins=pd.cut(d.TrueCATE, np.linspace(d.TrueCATE.min()-1e-9,d.TrueCATE.max()+1e-9,21))
    counts=pd.crosstab(bins,d.TreatmentGroup,dropna=False).rename(columns={0:'Control',1:'Treatment'})
    return pd.DataFrame(rows),counts


def bootstrap_qini(d,scores,iterations=200,seed=1042):
    """Paired ordinary iid evaluation bootstrap, conditional on fitted models.

    Sort once, use multinomial multiplicities (exact resampling, including ties).
    Identical row weights are used for every model; no model fitting occurs here.
    """
    y=d.MadeOrder.to_numpy();t=d.TreatmentGroup.to_numpy();n=len(d)
    prepared={name:(*ranking(s),) for name,s in scores.items()}
    rng=np.random.RandomState(seed);out=np.empty((iterations,len(scores)))
    for b in range(iterations):
        w=np.bincount(rng.randint(0,n,n),minlength=n)
        for j,(name,(order,end)) in enumerate(prepared.items()):
            out[b,j]=area(*_qini_sorted(y[order],t[order],end,w[order]))
    samples=pd.DataFrame(out,columns=scores)
    rows=[];pairs=[]
    for name in scores:
        lo,hi=np.quantile(samples[name],[.025,.975])
        rows.append(dict(Model=name,Qini=qini_auc(y,t,scores[name]),CI_low=lo,CI_high=hi))
    names=list(scores)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            delta=samples[a]-samples[b];lo,hi=np.quantile(delta,[.025,.975])
            pairs.append(dict(A=a,B=b,Delta=qini_auc(y,t,scores[a])-qini_auc(y,t,scores[b]),
                              CI_low=lo,CI_high=hi,excludes_zero=bool(lo>0 or hi<0)))
    return pd.DataFrame(rows),pd.DataFrame(pairs),samples


def fit_experiment(config, n=None, seed=None, families=('linear','xgb'), forest=None):
    n=config.n if n is None else n;seed=config.seed if seed is None else seed
    forest=config.include_forest if forest is None else forest
    d=generate_dataset(n,seed,assignment=config.assignment)
    train,test=train_test_split(np.arange(n),test_size=config.test_size,random_state=seed)
    obs=d.iloc[test].reset_index(drop=True)
    rct=generate_dataset(len(test),seed+10000,True,assignment=config.assignment)
    X=d[FEATURES].to_numpy();y=d.MadeOrder.to_numpy();t=d.TreatmentGroup.to_numpy()
    obs_scores={'TrueCATE':obs.TrueCATE.to_numpy()};rct_scores={'TrueCATE':rct.TrueCATE.to_numpy()}
    timings=[];audit={}
    for family in families:
        for name,make in model_factories(family,seed,config.threads,config.propensity_clip,forest and family=='xgb').items():
            key=f'{family}/{name}';start=time.perf_counter();est=make()
            print(f'n={n} seed={seed} fitting {key}',flush=True)
            if name=='DML':est.fit(y[train],t[train],X=X[train],cache_values=True)
            else:est.fit(y[train],t[train],X=X[train])
            obs_scores[key]=np.asarray(est.effect(X[test])).ravel()
            rct_scores[key]=np.asarray(est.effect(rct[FEATURES].to_numpy())).ravel()
            if np.std(obs_scores[key]) < 1e-12:
                obs_scores[key][:] = np.mean(obs_scores[key])
                rct_scores[key][:] = np.mean(rct_scores[key])
            assert np.isfinite(obs_scores[key]).all() and np.isfinite(rct_scores[key]).all(),key
            timings.append(dict(Model=key,seconds=time.perf_counter()-start))
            print(f'  {timings[-1]["seconds"]:.1f}s MSE={cate_metrics(obs.TrueCATE.to_numpy(),obs_scores[key])["MSE"]:.5f}',flush=True)
            if name=='DML':
                folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X[train],t[train]))
                yp=np.empty(len(train));tp=np.empty(len(train))
                for j,(tr,va) in enumerate(folds):
                    yp[va]=est.models_y[0][j].predict(X[train][va])
                    tp[va]=est.models_t[0][j].predict_proba(X[train][va])[:,1]
                ry,rt,*_=est.residuals_
                audit[family]=dict(folds=3,held_out_once=bool(np.all(np.bincount(np.concatenate([v for _,v in folds]))==1)),
                    disjoint=all(len(np.intersect1d(a,b))==0 for a,b in folds),
                    y_residual_max_error=float(np.max(abs(ry.ravel()-(y[train]-yp)))),
                    t_residual_max_error=float(np.max(abs(rt.ravel()-(t[train]-tp)))))
            del est
    # A diagnostic ranking based on assignment, not an additional trained learner.
    obs_scores['AssignmentRank']=obs.Propensity.to_numpy()
    rct_scores['AssignmentRank']=rct.Propensity.to_numpy()
    return dict(data=d,obs=obs,rct=rct,obs_scores=obs_scores,rct_scores=rct_scores,
                timings=pd.DataFrame(timings),audit=audit)


def save_experiment(result,folder='data'):
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    for split in ('obs','rct'):
        result[split].to_pickle(folder/f'{split}.pkl')
        np.savez_compressed(folder/f'{split}_scores.npz',**result[f'{split}_scores'])
        evaluate(result[split],result[f'{split}_scores']).to_csv(folder/f'{split}_metrics.csv',index=False)
    result['timings'].to_csv(folder/'fit_times.csv',index=False)
    (folder/'summary.json').write_text(json.dumps(summarize(result['data']),indent=2))
    (folder/'crossfit_audit.json').write_text(json.dumps(result['audit'],indent=2))
    regions,counts=overlap_tables(result['obs'],result['obs_scores'])
    regions.to_csv(folder/'overlap_metrics.csv',index=False);counts.to_csv(folder/'cate_bucket_counts.csv')


def load_experiment(folder='data'):
    folder=Path(folder)
    return {**{s:pd.read_pickle(folder/f'{s}.pkl') for s in ('obs','rct')},
            **{s+'_scores':dict(np.load(folder/f'{s}_scores.npz')) for s in ('obs','rct')}}


def diagnostic_experiments(r,config):
    """Known-propensity IPW support checks, plus source-only S and old X fits."""
    obs=r['obs'];e=obs.Propensity.to_numpy();t=obs.TreatmentGroup.to_numpy()
    weights=1/np.where(t==1,e,1-e)
    rows=[]
    for arm in (0,1):
        w=weights[t==arm]
        rows.append(dict(arm=arm,n=len(w),ESS=float(w.sum()**2/(w@w)),max_weight=float(w.max()),
                         ht_mass=float(w.sum()/len(obs))))
    pd.DataFrame(rows).to_csv('data/ipw_diagnostics.csv',index=False)
    mask=(e>=.05)&(e<=.95)
    evaluate(obs.loc[mask].reset_index(drop=True),{k:v[mask] for k,v in r['obs_scores'].items()}).to_csv('data/overlap_only_metrics.csv',index=False)
    d=r['data'];train,test=train_test_split(np.arange(len(d)),test_size=config.test_size,random_state=config.seed)
    X=d[FEATURES].to_numpy();y=d.MadeOrder.to_numpy();tt=d.TreatmentGroup.to_numpy()
    pars=dict(n_estimators=100,max_depth=6,learning_rate=.1,n_jobs=config.threads,random_state=config.seed,tree_method='hist')
    models={
        'printed/S_defaults':SLearner(overall_model=ProbabilityXGB(n_jobs=config.threads,random_state=config.seed)),
        'old_notebook/X':XLearner(models=ProbabilityXGB(**pars),cate_models=XGBRegressor(**pars)),
    }
    out=[]
    for name,est in models.items():
        print('Diagnostic fit',name,flush=True);est.fit(y[train],tt[train],X=X[train])
        for split in ('obs','rct'):
            dd=r[split];s=est.effect(dd[FEATURES].to_numpy()).ravel()
            tab=evaluate(dd,{name:s});tab['Split']=split;out.append(tab)
    pd.concat(out).to_csv('data/source_reconstruction.csv',index=False)
    return pd.DataFrame(rows)


def run_multiseed(config,primary):
    rows=[]
    for n in config.sizes:
        for seed in config.seeds:
            r=primary if n==config.n and seed==config.seed else fit_experiment(config,n,seed,forest=False)
            for split in ('obs','rct'):
                m=evaluate(r[split],r[f'{split}_scores']);m['n']=n;m['seed']=seed;m['Split']=split;rows.append(m)
            pd.concat(rows).to_csv('data/multiseed.csv',index=False)
    result=pd.concat(rows,ignore_index=True)
    result.groupby(['n','Split','Model'])[['Qini','LegacyRatio','MSE']].agg(['mean','std','min','max']).to_csv('data/multiseed_spread.csv')
    return result


def article_example(name='S', family='xgb', n=1_000_000, seed=42, assignment='published'):
    """Independent article example: no notebook globals or saved data required."""
    d=generate_dataset(n,seed,assignment=assignment);train,test=train_test_split(np.arange(n),test_size=.3,random_state=seed)
    X=d[FEATURES].to_numpy();y=d.MadeOrder.to_numpy();t=d.TreatmentGroup.to_numpy()
    est=model_factories(family,seed)[name]()
    est.fit(y[train],t[train],X=X[train])
    obs=d.iloc[test].reset_index(drop=True);rct=generate_dataset(len(test),seed+10000,True,assignment=assignment)
    scores={}
    for split,dd in [('obs',obs),('rct',rct)]:
        s=est.effect(dd[FEATURES].to_numpy()).ravel()
        if np.std(s)<1e-12:s[:]=s.mean()
        scores[split]={'TrueCATE':dd.TrueCATE.to_numpy(),f'{family}/{name}':s}
    return {'obs':obs,'rct':rct,'obs_scores':scores['obs'],'rct_scores':scores['rct']}


def confounding_counterexample(config):
    """Predefined oracle diagnostic: add assignment probability to CATE ranking.

    Not a deployable learner or tuned DGP. Holding outcomes fixed in expectation
    demonstrates that confounded evaluation can prefer a causally inferior rank.
    """
    rows=[]
    for n in config.sizes:
        for seed in config.seeds:
            d=generate_dataset(n,seed)
            _,test=train_test_split(np.arange(n),test_size=config.test_size,random_state=seed)
            for split,dd in [('obs',d.iloc[test].reset_index(drop=True)),
                             ('rct',generate_dataset(len(test),seed+10000,True))]:
                ss={'TrueCATE':dd.TrueCATE.to_numpy(),
                    'CATEplusPropensity':(dd.TrueCATE+dd.Propensity).to_numpy(),
                    'TreatedResponse':dd.P1.to_numpy()}
                tab=evaluate(dd,ss);tab['n']=n;tab['seed']=seed;tab['Split']=split;rows.append(tab)
                if n==config.n and seed==config.seed:
                    ci,pairs,_=bootstrap_qini(dd,ss,config.bootstrap,seed=1042)
                    ci.to_csv(f'data/counterexample_{split}_ci.csv',index=False)
                    pairs.to_csv(f'data/counterexample_{split}_pairs.csv',index=False)
    table=pd.concat(rows,ignore_index=True)
    table.to_csv('data/confounding_counterexample.csv',index=False)
    return table


# Observational evaluation: these nuisance models are separate from CATE learners.
def crossfit_evaluation_nuisance(observed, seed=2042, threads=4, clip=.005):
    """Accept ONLY X,T,Y. Out-of-fold evaluation scores use no synthetic truth.

    Ranking models must have been trained on a separate sample. Each evaluation
    row is excluded from its own propensity and outcome nuisance fits.
    """
    allowed=FEATURES+['TreatmentGroup','MadeOrder']
    if set(observed.columns)!=set(allowed):
        raise ValueError('Pass only the four features, TreatmentGroup and MadeOrder')
    X=observed[FEATURES].to_numpy();t=observed.TreatmentGroup.to_numpy();y=observed.MadeOrder.to_numpy()
    n=len(y);e=np.empty(n);mu0=np.empty(n);mu1=np.empty(n);fold=np.full(n,-1)
    cv=StratifiedKFold(3,shuffle=True,random_state=seed);audit=[]
    for j,(tr,va) in enumerate(cv.split(X,t)):
        pars=dict(n_estimators=200,max_depth=4,learning_rate=.05,tree_method='hist',
                  n_jobs=threads,random_state=seed+j,eval_metric='logloss')
        prop=XGBClassifier(**pars).fit(X[tr],t[tr]);e[va]=prop.predict_proba(X[va])[:,1]
        for arm,target in [(0,mu0),(1,mu1)]:
            idx=tr[t[tr]==arm]
            model=XGBClassifier(**pars).fit(X[idx],y[idx])
            target[va]=model.predict_proba(X[va])[:,1]
        fold[va]=j
        audit.append(dict(fold=j,train_n=len(tr),test_n=len(va),
                          disjoint=not bool(np.intersect1d(tr,va).size)))
    assert np.all(fold>=0) and all(a['disjoint'] for a in audit)
    return pd.DataFrame(dict(e_raw=e,e_hat=np.clip(e,clip,1-clip),mu0_hat=mu0,mu1_hat=mu1,fold=fold)),audit


def adjusted_scores(y,t,e,mu0=None,mu1=None):
    """HT (IPW) or AIPW (DR) pseudo-outcome. Supports factual exact e=0/1.

    Positivity is still required for causal interpretation. Only factual-arm
    denominators are divided, avoiding numerically undefined 0/0 branches.
    """
    y=np.asarray(y);t=np.asarray(t);e=np.asarray(e)
    denominator=np.where(t==1,e,1-e)
    if np.any(denominator<=0):raise ValueError('Zero factual assignment probability')
    if mu0 is None:
        return np.where(t==1,y,-y)/denominator
    mu0=np.asarray(mu0);mu1=np.asarray(mu1)
    residual=y-np.where(t==1,mu1,mu0)
    return mu1-mu0+np.where(t==1,residual,-residual)/denominator


def gain_curve(score,pseudo_outcome,weights=None):
    """Cumulative incremental outcomes / population, at complete score ties.

    Its baseline-subtracted area is on the same probability scale for IPW,
    DR, randomized transformed outcomes and synthetic oracle CATE.
    """
    order,end=ranking(score);z=np.asarray(pseudo_outcome)[order]
    w=np.ones(len(z)) if weights is None else np.asarray(weights,dtype=float)[order]
    return np.r_[0,np.cumsum(w)[end]/w.sum()],np.r_[0,np.cumsum(w*z)[end]/w.sum()]


def _gain_area_sorted(z,end,w):
    starts=np.r_[0,end[:-1]+1]
    groupw=np.add.reduceat(w,starts);groupz=np.add.reduceat(w*z,starts)
    n=w.sum();mid=(np.cumsum(groupw)-groupw/2)/n
    return float(np.sum(groupz*(.5-mid))/n)


def adjusted_bootstrap(scores,z,iterations=200,seed=1042):
    """Paired empirical bootstrap of fixed cross-fitted pseudo-outcomes.

    Conditions on ranking and evaluation nuisance fits; no full nuisance-refit
    uncertainty is claimed. Weighted ranks are recomputed on every resample.
    """
    n=len(z);rng=np.random.RandomState(seed)
    prepared={k:ranking(v) for k,v in scores.items()}
    draws=np.empty((iterations,len(scores)))
    for b in range(iterations):
        w=np.bincount(rng.randint(0,n,n),minlength=n).astype(float)
        for j,(name,(order,end)) in enumerate(prepared.items()):
            draws[b,j]=_gain_area_sorted(np.asarray(z)[order],end,w[order])
    samples=pd.DataFrame(draws,columns=scores)
    rows=[];pairs=[]
    for name,s in scores.items():
        lo,hi=np.quantile(samples[name],[.025,.975])
        rows.append(dict(Model=name,DRGain=area(*gain_curve(s,z)),CI_low=lo,CI_high=hi))
    keys=list(scores)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            delta=samples[a]-samples[b];lo,hi=np.quantile(delta,[.025,.975])
            pairs.append(dict(A=a,B=b,Delta=area(*gain_curve(scores[a],z))-area(*gain_curve(scores[b],z)),CI_low=lo,CI_high=hi))
    return pd.DataFrame(rows),pd.DataFrame(pairs),samples
