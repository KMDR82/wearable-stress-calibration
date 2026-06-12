import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def gb_baseline(seed=0):
    return GradientBoostingClassifier(random_state=seed)


def logreg_baseline(seed=0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=seed))


def fit_predict_proba(model, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return p
