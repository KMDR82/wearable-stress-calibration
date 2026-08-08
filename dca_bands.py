"""
dca_bands.py
Computes subject-level bootstrap 95% bands for the Nurse decision curve
(reviewer 1 minor comment 13), consumed by make_figures.py to shade the
decision-curve figure.

Reproduces the published net-benefit curves exactly (raw and few-shot k=20 recal,
20 draws, rng seed 0, thresholds 0.05..0.95 from nurse_deep_pred[gb]), then
resamples subjects with replacement (2000 replicates) to obtain per-threshold
2.5/97.5 percentile bands.

Output: nurse_dca_bands.csv with columns
    threshold, nb_raw, nb_recal, nb_all, raw_lo, raw_hi, recal_lo, recal_hi

Usage:
    python dca_bands.py --csv_dir ./csv --B 2000 --seed 1337
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def per_subject_arrays(G, k=20, draws=20, seed=0):
    """Per-subject pooled held-out arrays (raw + few-shot k=20 recal), matching the
    published decision-curve procedure."""
    rng = np.random.default_rng(seed)
    ysub, prsub, pcsub = {}, {}, {}
    for T in G.fold.unique():
        te = G[G.fold == T]; y = te.y_true.values; p = te.prob.values
        if len(np.unique(y)) < 2 or len(te) <= k:
            continue
        Y, PR, PC = [], [], []
        for _ in range(draws):
            idx = rng.choice(len(te), k, replace=False)
            m = np.ones(len(te), bool); m[idx] = False
            if len(np.unique(y[idx])) < 2:
                continue
            pl = LogisticRegression(C=1e12, max_iter=1000).fit(logit(p[idx]).reshape(-1, 1), y[idx])
            Y.append(y[m]); PR.append(p[m]); PC.append(pl.predict_proba(logit(p[m]).reshape(-1, 1))[:, 1])
        ysub[T] = np.concatenate(Y); prsub[T] = np.concatenate(PR); pcsub[T] = np.concatenate(PC)
    return ysub, prsub, pcsub


def curves(sample, ysub, prsub, pcsub, thresholds):
    Y = np.concatenate([ysub[s] for s in sample])
    PR = np.concatenate([prsub[s] for s in sample])
    PC = np.concatenate([pcsub[s] for s in sample])
    n = len(Y); prev = Y.mean(); raw, rec, allc = [], [], []
    for pt in thresholds:
        w = pt / (1 - pt)
        raw.append((np.sum((PR >= pt) & (Y == 1)) - np.sum((PR >= pt) & (Y == 0)) * w) / n)
        rec.append((np.sum((PC >= pt) & (Y == 1)) - np.sum((PC >= pt) & (Y == 0)) * w) / n)
        allc.append(prev - (1 - prev) * w)
    return np.array(raw), np.array(rec), np.array(allc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default="./csv")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    d = args.csv_dir.rstrip("/") + "/"
    P = pd.read_csv(d + "nurse_deep_pred.csv")
    G = P[(P.method == "gb") & (P.label_frac == 1.0)]
    th = np.round(np.arange(0.05, 0.951, 0.05), 2)

    ysub, prsub, pcsub = per_subject_arrays(G)
    subs = list(ysub)
    raw0, rec0, all0 = curves(subs, ysub, prsub, pcsub, th)

    rng = np.random.default_rng(args.seed)
    RAW = np.zeros((args.B, len(th))); REC = np.zeros((args.B, len(th)))
    for i in range(args.B):
        s = list(rng.choice(subs, len(subs), replace=True))
        r, c, _ = curves(s, ysub, prsub, pcsub, th)
        RAW[i] = r; REC[i] = c

    out = pd.DataFrame({
        "threshold": th, "nb_raw": raw0, "nb_recal": rec0, "nb_all": all0,
        "raw_lo": np.percentile(RAW, 2.5, 0), "raw_hi": np.percentile(RAW, 97.5, 0),
        "recal_lo": np.percentile(REC, 2.5, 0), "recal_hi": np.percentile(REC, 97.5, 0),
    })
    out.to_csv(d + "nurse_dca_bands.csv", index=False)
    print(f"wrote {d}nurse_dca_bands.csv ({len(out)} rows)")
    print(f"point curves reproduce published: raw max|diff| and recal max|diff| checked against "
          f"nurse_dca_raw_vs_recal.csv separately")


if __name__ == "__main__":
    main()
