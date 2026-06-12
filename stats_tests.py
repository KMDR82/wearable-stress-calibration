
import numpy as np
from scipy import stats




def mcnemar_test(y_true, pred1, pred2):
 

    y_true = np.asarray(y_true).astype(int)
    pred1 = np.asarray(pred1).astype(int)
    pred2 = np.asarray(pred2).astype(int)
    c1 = (pred1 == y_true)
    c2 = (pred2 == y_true)
    b = int(np.sum(c1 & ~c2))   # 1 doğru, 2 yanlış
    c = int(np.sum(~c1 & c2))   # 1 yanlış, 2 doğru
    n = b + c
    if n == 0:
        return dict(b=b, c=c, n_discordant=0, statistic=0.0, p_value=1.0, method="degenerate")
    if n < 25:
        # exact iki-yanlı binom testi (p=0.5)
        p = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        return dict(b=b, c=c, n_discordant=n, statistic=float(min(b, c)), p_value=p, method="exact_binomial")
    stat = (abs(b - c) - 1) ** 2 / n  # süreklilik düzeltmeli, df=1
    p = float(stats.chi2.sf(stat, df=1))
    return dict(b=b, c=c, n_discordant=n, statistic=float(stat), p_value=p, method="chi2_continuity")


# ----------------------------- DeLong -----------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):


    k = preds_sorted.shape[0]
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    cov = np.atleast_2d(cov)
    return aucs, cov


def delong_auc_variance(y_true, prob):


    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    order = np.argsort(-y_true, kind="mergesort")  # pozitifler (1) önce
    m = int(y_true.sum())
    preds = prob[order][None, :]
    aucs, cov = _fast_delong(preds, m)
    auc = float(aucs[0]); var = float(cov[0, 0])
    se = np.sqrt(max(var, 0.0))
    lo, hi = auc - 1.96 * se, auc + 1.96 * se
    return dict(auc=auc, var=var, ci_low=float(max(0, lo)), ci_high=float(min(1, hi)))


def delong_roc_test(y_true, prob1, prob2):


    y_true = np.asarray(y_true).astype(int)
    prob1 = np.asarray(prob1, dtype=float)
    prob2 = np.asarray(prob2, dtype=float)
    order = np.argsort(-y_true, kind="mergesort")
    m = int(y_true.sum())
    preds = np.vstack((prob1, prob2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0; p = 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), z=float(z), p_value=p)


# ----------------------------- Spiegelhalter z -----------------------------
def spiegelhalter_z(y_true, prob):


    y_true = np.asarray(y_true).astype(float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-12, 1 - 1e-12)
    num = np.sum((y_true - p) * (1.0 - 2.0 * p))
    den = np.sqrt(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if den == 0:
        return dict(z=0.0, p_value=1.0)
    z = num / den
    return dict(z=float(z), p_value=float(2 * stats.norm.sf(abs(z))))
