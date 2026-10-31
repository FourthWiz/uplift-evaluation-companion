import numpy as np
from sklearn.model_selection import train_test_split
from uplift_lib import (generate_dataset, FEATURES, model_factories,
    crossfit_evaluation_nuisance, adjusted_scores, gain_curve, qini_auc, area)

data = generate_dataset(1_000_000, seed=42, assignment="selection_high")
train, test = train_test_split(np.arange(len(data)), test_size=0.3, random_state=42)
X = data[FEATURES].to_numpy()
y = data.MadeOrder.to_numpy()
t = data.TreatmentGroup.to_numpy()
model = model_factories(family="xgb", seed=42, threads=4)["S"]()
model.fit(y[train], t[train], X=X[train])
ranking = model.effect(X[test]).ravel()
evaluation = data.iloc[test].reset_index(drop=True)
observed = evaluation[FEATURES + ["TreatmentGroup", "MadeOrder"]]
nu, audit = crossfit_evaluation_nuisance(observed, seed=2042)
ipw = adjusted_scores(evaluation.MadeOrder, evaluation.TreatmentGroup, nu.e_hat)
dr = adjusted_scores(evaluation.MadeOrder, evaluation.TreatmentGroup,
                     nu.e_hat, nu.mu0_hat, nu.mu1_hat)
c_q = 1.5445466434481507  # Frozen high-policy seed-42 reference.
c_ipw = 1.0005932054691182
print("Calibrated Qini", qini_auc(evaluation.MadeOrder, evaluation.TreatmentGroup, ranking) / evaluation.TreatmentGroup.mean() / c_q)
print("Calibrated IPW", area(*gain_curve(ranking, ipw)) / c_ipw)
print("DR", area(*gain_curve(ranking, dr)))

