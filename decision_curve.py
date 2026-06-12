
import numpy as np


def net_benefit_curve(y_true, prob, thresholds):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    n = len(y_true)
    prev = y_true.mean()
    rows = []
    for pt in thresholds:
        w = pt / (1 - pt)
        pred = (prob >= pt).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        nb_model = tp / n - (fp / n) * w
        nb_all = prev - (1 - prev) * w
        rows.append(dict(
            threshold=float(pt),
            net_benefit_model=float(nb_model),
            net_benefit_treat_all=float(nb_all),
            net_benefit_treat_none=0.0,
            tp=tp, fp=fp,
        ))
    return rows


def net_benefit_summary(rows):

    nb_m = np.array([r["net_benefit_model"] for r in rows])
    nb_a = np.array([r["net_benefit_treat_all"] for r in rows])
    nb_n = np.array([r["net_benefit_treat_none"] for r in rows])
    best_ref = np.maximum(nb_a, nb_n)
    superior = nb_m > best_ref
    return dict(
        frac_thresholds_superior=float(superior.mean()),
        mean_excess_net_benefit=float(np.mean(nb_m - best_ref)),
    )
