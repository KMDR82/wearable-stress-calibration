"""
chronological_fewshot.py
Prospective (chronological) evaluation of few-shot per-subject recalibration,
addressing the reviewer request to calibrate on early windows and evaluate on
later ones instead of sampling calibration windows at random.

The published few-shot result samples calibration windows at random from the
whole recording, temporally interspersed with the evaluation windows. This script
re-runs the recalibration chronologically and reports:

  1. time-to-both-classes: how many leading windows elapse before both stress and
     non-stress are observed (why strict first-k calibration is often infeasible);
  2. class-balanced chronological few-shot: calibrate on the earliest window block
     that contains both classes and at least k windows, evaluate on the remainder,
     with subject-level bootstrap 95% CIs.

Input : nurse_loso_pred.csv (fold, prob, y_true), rows in chronological order
        (windows are datetime-sorted upstream; equal to nurse_deep_pred[gb]).
Usage : python chronological_fewshot.py --csv_dir ./csv --B 5000 --seed 1337
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def ece(y, p, nb=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    b = np.linspace(0, 1, nb + 1); e = 0.0; n = len(p)
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pct_ci(v):
    v = np.asarray(v, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def time_to_both_classes(R):
    need = []
    for T in R.fold.unique():
        y = R[R.fold == T].y_true.values
        j = next((jj for jj in range(1, len(y) + 1) if len(np.unique(y[:jj])) == 2), len(y) + 1)
        need.append(j)
    need = np.array(need)
    print("time-to-both-classes (windows): "
          f"median={np.median(need):.0f}  IQR=[{np.percentile(need,25):.0f}, {np.percentile(need,75):.0f}]"
          f"  range={need.min()}-{need.max()}")
    return need


def balanced_chrono(R, ks=(5, 10, 20, 40), rng=None, B=5000):
    if rng is None:
        rng = np.random.default_rng(1337)
    print("\nclass-balanced chronological few-shot (calibrate on earliest both-class block >= k, evaluate on rest)")
    print(f"  {'k':>3} {'ECE':>7} {'95% CI':>18} {'folds':>6}")
    for k in ks:
        per = {}
        for T in R.fold.unique():
            te = R[R.fold == T]; y = te.y_true.values; p = te.prob.values; n = len(te)
            j = next((jj for jj in range(k, n) if len(np.unique(y[:jj])) == 2), None)
            if j is None or j >= n:
                continue
            yc, pc, ye, pe = y[:j], p[:j], y[j:], p[j:]
            if len(np.unique(ye)) < 2:
                continue
            pl = LogisticRegression(C=1e12, max_iter=1000).fit(logit(pc).reshape(-1, 1), yc)
            per[T] = ece(ye, pl.predict_proba(logit(pe).reshape(-1, 1))[:, 1])
        subs = list(per); vals = np.array([per[s] for s in subs])
        if len(vals) == 0:
            print(f"  {k:>3}  no evaluable folds"); continue
        bs = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(B)]
        lo, hi = pct_ci(bs)
        print(f"  {k:>3} {vals.mean():.3f}   [{lo:.3f}, {hi:.3f}]   {len(subs):>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="./csv")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    R = pd.read_csv(args.csv_dir.rstrip("/") + "/nurse_loso_pred.csv")
    rng = np.random.default_rng(args.seed)
    print("=" * 70)
    time_to_both_classes(R)
    balanced_chrono(R, rng=rng, B=args.B)


if __name__ == "__main__":
    main()
