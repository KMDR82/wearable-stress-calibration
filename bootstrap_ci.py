"""
bootstrap_ci.py
Subject-level (cluster) bootstrap 95% confidence intervals for the calibration
and few-shot recalibration results, addressing reviewer requests for uncertainty
quantification.

Resamples SUBJECTS (LOSO folds) with replacement, because windows within a
subject are correlated. Reproduces the published point estimates exactly and adds
percentile 95% CIs.

Usage:
    python bootstrap_ci.py --csv_dir ./csv --B 5000 --seed 1337

Sources (verified to reproduce the published numbers):
    calibration  : feat_loso_predictions.csv (WESAD),
                   nurse_deep_pred.csv [method=gb, label_frac=1.0] (Nurse),
                   exstress_loso_pred.csv (Exercise)
    few-shot     : nurse_loso_pred.csv, procedure identical to the paper
                   (per-subject Platt scaling on k random windows, ECE on the
                   rest, 30 draws; rng seed 0, published loop order).

    DCA          : nurse_deep_pred.csv [method=gb, label_frac=1.0]; the published
                   Nurse decision-curve procedure (k=20 few-shot Platt recal, 20
                   draws, rng seed 0, thresholds 0.05..0.95) is reproduced exactly
                   and bootstrapped by subject.
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


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pct_ci(v):
    v = np.asarray(v, float)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def boot_calibration(df, name, rng, B):
    """Subject-level bootstrap for per-subject mean ECE and pooled Brier."""
    folds = [f for f, g in df.groupby("fold") if g.y_true.nunique() > 1]
    gd = {f: df[df.fold == f] for f in folds}
    pt_ece = np.mean([ece(gd[f].y_true.values, gd[f].prob.values) for f in folds])
    pt_brier = brier(df.y_true.values, df.prob.values)
    be, bb = [], []
    for _ in range(B):
        samp = rng.choice(folds, len(folds), replace=True)
        be.append(np.mean([ece(gd[f].y_true.values, gd[f].prob.values) for f in samp]))
        cat = pd.concat([gd[f] for f in samp])
        bb.append(brier(cat.y_true.values, cat.prob.values))
    lo_e, hi_e = pct_ci(be); lo_b, hi_b = pct_ci(bb)
    print(f"  {name:9s} per-subject meanECE = {pt_ece:.3f}  95% CI [{lo_e:.3f}, {hi_e:.3f}]  (folds={len(folds)})")
    print(f"  {'':9s} pooled Brier        = {pt_brier:.3f}  95% CI [{lo_b:.3f}, {hi_b:.3f}]")


def fewshot_per_subject(R, ks=(0, 5, 10, 20, 40), draws=30, seed=0):
    """Per-subject few-shot ECE at each k, using the published loop order."""
    rng = np.random.default_rng(seed)
    psub = {k: {} for k in ks}
    for k in ks:
        for T in R.fold.unique():
            te = R[R.fold == T]; y = te.y_true.values; p = te.prob.values; n = len(te)
            if len(np.unique(y)) < 2 or n <= max(k, 5):
                continue
            if k == 0:
                psub[0][T] = ece(y, p); continue
            es = []
            for _ in range(draws):
                idx = rng.choice(n, k, replace=False)
                mask = np.ones(n, bool); mask[idx] = False
                if len(np.unique(y[idx])) < 2:
                    continue
                pl = LogisticRegression(C=1e12, max_iter=1000).fit(
                    logit(p[idx]).reshape(-1, 1), y[idx])
                es.append(ece(y[mask], pl.predict_proba(logit(p[mask]).reshape(-1, 1))[:, 1]))
            if es:
                psub[k][T] = float(np.mean(es))
    return psub


def boot_fewshot(psub, rng, B):
    print("  k    ECE      95% CI            folds")
    for k in sorted(psub):
        subs = list(psub[k]); vals = np.array([psub[k][s] for s in subs])
        pt = vals.mean()
        bs = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(B)]
        lo, hi = pct_ci(bs)
        print(f"  {k:<4} {pt:.3f}   [{lo:.3f}, {hi:.3f}]    {len(subs)}")


def dca_per_subject(G, k=20, draws=20, seed=0):
    """Per-subject pooled held-out arrays (raw + few-shot k=20 recal) across draws.
    Reproduces the published Nurse decision-curve procedure exactly."""
    rng = np.random.default_rng(seed)
    ysub, prsub, pcsub = {}, {}, {}
    for T in G.fold.unique():
        te = G[G.fold == T]; y = te.y_true.values; p = te.prob.values
        if len(np.unique(y)) < 2 or len(te) <= k:
            continue
        Y, PR, PC = [], [], []
        for _ in range(draws):
            idx = rng.choice(len(te), k, replace=False)
            mask = np.ones(len(te), bool); mask[idx] = False
            if len(np.unique(y[idx])) < 2:
                continue
            pl = LogisticRegression(C=1e12, max_iter=1000).fit(
                logit(p[idx]).reshape(-1, 1), y[idx])
            Y.append(y[mask]); PR.append(p[mask])
            PC.append(pl.predict_proba(logit(p[mask]).reshape(-1, 1))[:, 1])
        ysub[T] = np.concatenate(Y); prsub[T] = np.concatenate(PR); pcsub[T] = np.concatenate(PC)
    return ysub, prsub, pcsub


def _dca_frac_excess(sample, ysub, prsub, pcsub, thresholds):
    Y = np.concatenate([ysub[s] for s in sample])
    PR = np.concatenate([prsub[s] for s in sample])
    PC = np.concatenate([pcsub[s] for s in sample])
    n = len(Y); prev = Y.mean(); raw, rec, ref = [], [], []
    for pt in thresholds:
        w = pt / (1 - pt)
        raw.append((np.sum((PR >= pt) & (Y == 1)) - np.sum((PR >= pt) & (Y == 0)) * w) / n)
        rec.append((np.sum((PC >= pt) & (Y == 1)) - np.sum((PC >= pt) & (Y == 0)) * w) / n)
        ref.append(max(prev - (1 - prev) * w, 0.0))
    raw = np.array(raw); rec = np.array(rec); ref = np.array(ref)
    return (raw > ref).mean(), (rec > ref).mean(), np.mean(raw - ref), np.mean(rec - ref)


def boot_dca(G, rng, B):
    th = np.arange(0.05, 0.96, 0.05)
    ysub, prsub, pcsub = dca_per_subject(G)
    subs = list(ysub)
    fr_raw, fr_rec, ex_raw, ex_rec = _dca_frac_excess(subs, ysub, prsub, pcsub, th)
    A = np.array([_dca_frac_excess(list(rng.choice(subs, len(subs), replace=True)),
                                   ysub, prsub, pcsub, th) for _ in range(B)])

    def ci(v):
        return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))
    print(f"  superior-fraction raw   = {fr_raw:.3f}  95% CI [{ci(A[:,0])[0]:.3f}, {ci(A[:,0])[1]:.3f}]")
    print(f"  superior-fraction recal = {fr_rec:.3f}  95% CI [{ci(A[:,1])[0]:.3f}, {ci(A[:,1])[1]:.3f}]")
    print(f"  delta fraction          = {fr_rec-fr_raw:+.3f}  95% CI [{ci(A[:,1]-A[:,0])[0]:+.3f}, {ci(A[:,1]-A[:,0])[1]:+.3f}]")
    print(f"  mean excess NB raw      = {ex_raw:+.4f}  95% CI [{ci(A[:,2])[0]:+.4f}, {ci(A[:,2])[1]:+.4f}]")
    print(f"  mean excess NB recal    = {ex_rec:+.4f}  95% CI [{ci(A[:,3])[0]:+.4f}, {ci(A[:,3])[1]:+.4f}]")
    print(f"  delta excess NB         = {ex_rec-ex_raw:+.4f}  95% CI [{ci(A[:,3]-A[:,2])[0]:+.4f}, {ci(A[:,3]-A[:,2])[1]:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="./csv")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    d = args.csv_dir.rstrip("/") + "/"
    rng = np.random.default_rng(args.seed)

    wes = pd.read_csv(d + "feat_loso_predictions.csv")
    nd = pd.read_csv(d + "nurse_deep_pred.csv")
    ngb = nd[(nd.method == "gb") & (nd.label_frac == 1.0)].copy()
    exs = pd.read_csv(d + "exstress_loso_pred.csv")

    print("=" * 66)
    print("CALIBRATION — subject-level bootstrap 95% CI (per-subject meanECE, pooled Brier)")
    print("=" * 66)
    for nm, df in [("WESAD", wes), ("Nurse", ngb), ("Exercise", exs)]:
        boot_calibration(df, nm, rng, args.B)

    print("\n" + "=" * 66)
    print("FEW-SHOT ECE-vs-k (Nurse) — subject-level bootstrap 95% CI")
    print("=" * 66)
    R = pd.read_csv(d + "nurse_loso_pred.csv")
    psub = fewshot_per_subject(R)
    boot_fewshot(psub, rng, args.B)

    print("\n" + "=" * 66)
    print("DECISION CURVE (Nurse, raw vs few-shot k=20 recal) — subject-level bootstrap 95% CI")
    print("=" * 66)
    boot_dca(ngb, rng, args.B)


if __name__ == "__main__":
    main()
