"""
matched_features_wesad.py
Feature-harmonisation sensitivity analysis (reviewer 1 comment 11; reviewer 2 comment 3).

WESAD is re-run with the SAME common 11-feature set used for Nurse and
Exercise-Stress instead of its 18-feature representation, so the cross-dataset
comparison does not confound the recording setting with the feature representation.
Everything else in the WESAD pipeline (60 s windows, robust per-subject
normalisation, gradient boosting, LOSO, seed 1337) is unchanged; only the feature
set differs.

Common 11 features (identical schema to the Nurse/Exercise pathway):
    HR mean, HR std,
    EDA mean/std/slope,
    TEMP mean/std/slope,
    ACC-magnitude mean/std/slope
HR for WESAD is derived from blood-volume-pulse peaks (as in the 18-feature
pipeline) but summarised as mean/std of the instantaneous rate to match the HR
channel used for Nurse and Exercise-Stress.

Self-contained: reads the raw WESAD .pkl files directly and reproduces the pipeline
without importing the repository modules, so it runs as-is. The output CSV
feat_loso_matched.csv and the printed metrics reproduce the reported result
(per-subject AUROC 0.992, ECE 0.037, Brier 0.027, slope 0.62). The published
18-feature reference is printed for comparison.

Usage (auto-detects WESAD under /kaggle/input; or pass --wesad_root):
    python matched_features_wesad.py --wesad_root /path/to/WESAD --out ./
"""
import argparse
import glob
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

WRIST_FS = 64
CH = ["BVP", "EDA", "TEMP", "ACC"]
RATES = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
WIN_SEC = 60
STRIDE_SEC = 60
KEEP = (1, 2, 3)
STRESS = 2
SEED = 1337


def _resample_to(x, fi, fo):
    x = np.asarray(x, float).reshape(len(x), -1)
    if fi == fo:
        return x
    out = np.zeros((int(round(x.shape[0] * fo / fi)), x.shape[1]))
    for c in range(x.shape[1]):
        out[:, c] = resample_poly(x[:, c], fo, fi)[:out.shape[0]]
    return out


def _wrist(sig, fo):
    cols = [_resample_to(np.asarray(sig["wrist"][ch], float), RATES[ch], fo) for ch in CH]
    T = min(c.shape[0] for c in cols)
    return np.concatenate([c[:T] for c in cols], axis=1)


def _robust(mat):
    med = np.median(mat, 0, keepdims=True)
    iqr = np.subtract(*np.percentile(mat, [75, 25], axis=0)).reshape(1, -1)
    iqr[iqr == 0] = 1.0
    return (mat - med) / iqr


def matched_feats(win, fs):
    """win: (T, 6) robust-normalised [BVP, EDA, TEMP, ACCx, ACCy, ACCz]."""
    bvp = win[:, 0]
    pk, _ = find_peaks(bvp, distance=int(0.4 * fs))
    if len(pk) > 3:
        ibi = np.diff(pk) / fs
        hr = 60.0 / ibi
        hr_mean, hr_std = float(hr.mean()), float(hr.std())
    else:
        hr_mean, hr_std = 0.0, 0.0
    eda, tmp = win[:, 1], win[:, 2]
    acc = np.sqrt(win[:, 3] ** 2 + win[:, 4] ** 2 + win[:, 5] ** 2)
    x = np.arange(len(win))
    sl = lambda v: float(np.polyfit(x, v, 1)[0])
    return np.array([hr_mean, hr_std,
                     eda.mean(), eda.std(), sl(eda),
                     tmp.mean(), tmp.std(), sl(tmp),
                     acc.mean(), acc.std(), sl(acc)], dtype=float)


def build_subject(root, sid):
    with open(os.path.join(root, f"S{sid}", f"S{sid}.pkl"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    sig = d["signal"]; label = np.asarray(d["label"]).astype(int); fs = WRIST_FS
    mat = _robust(_wrist(sig, fs))
    win = int(WIN_SEC * fs); stride = int(STRIDE_SEC * fs)
    lwin = int(WIN_SEC * 700); lstride = int(STRIDE_SEC * 700)
    fX, y = [], []; n = mat.shape[0]; li = 0
    for st in range(0, n - win + 1, stride):
        seglab = label[li:li + lwin]; li += lstride
        if len(seglab) == 0:
            continue
        v, c = np.unique(seglab, return_counts=True); maj = int(v[np.argmax(c)])
        if maj not in KEEP:
            continue
        fX.append(matched_feats(mat[st:st + win], fs)); y.append(1 if maj == STRESS else 0)
    return (np.stack(fX).astype(np.float32), np.asarray(y, int)) if y else None


def ece(y, p, nb=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    b = np.linspace(0, 1, nb + 1); e = 0.0; n = len(p)
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def slope(y, p):
    x = np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(LogisticRegression(C=1e12, max_iter=1000).fit(x, y).coef_[0, 0])


def find_wesad_root():
    for p in glob.glob("/kaggle/input/**/S*/S*.pkl", recursive=True):
        return os.path.dirname(os.path.dirname(p))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_root", default=None, help="folder containing S2/S2.pkl ...")
    ap.add_argument("--out", default="./")
    args = ap.parse_args()
    root = args.wesad_root or find_wesad_root()
    assert root, "WESAD root not found; pass --wesad_root"
    print("WESAD_ROOT =", root)

    subs = sorted(int(x[1:]) for x in os.listdir(root)
                  if x.startswith("S") and x[1:].isdigit())
    feats, ys = {}, {}
    for s in subs:
        r = build_subject(root, s)
        if r is not None:
            feats[s], ys[s] = r
            print(f"S{s}: {len(ys[s])} windows, stress {ys[s].mean():.2f}", flush=True)
    subs = [s for s in subs if s in feats]

    rows = []
    for test in subs:
        tr = [s for s in subs if s != test]
        Xtr = np.concatenate([feats[s] for s in tr]); ytr = np.concatenate([ys[s] for s in tr])
        gb = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
        p = gb.predict_proba(feats[test])[:, 1]
        rows += [dict(fold=test, prob=float(a), y_true=int(b)) for a, b in zip(p, ys[test])]
    R = pd.DataFrame(rows)
    R.to_csv(args.out.rstrip("/") + "/feat_loso_matched.csv", index=False)

    pa = np.array([roc_auc_score(g.y_true, g.prob) for _, g in R.groupby("fold") if g.y_true.nunique() > 1])
    es = np.array([ece(g.y_true.values, g.prob.values) for _, g in R.groupby("fold") if g.y_true.nunique() > 1])
    print(f"\n=== WESAD, matched 11-feature set ({len(subs)} subjects) ===")
    print(f"per-subject AUROC={pa.mean():.3f} (SD {pa.std(ddof=1):.3f})  pooled={roc_auc_score(R.y_true, R.prob):.3f}")
    print(f"per-subject ECE  ={es.mean():.3f}  pooled ECE={ece(R.y_true.values, R.prob.values):.3f}")
    print(f"Brier={brier_score_loss(R.y_true, R.prob):.3f}  slope={slope(R.y_true.values, R.prob.values):.3f}")
    print("(published 18-feature WESAD: per-subj AUROC 0.992 / pooled 0.983 / ECE 0.036 / Brier 0.029 / slope 0.535)")


if __name__ == "__main__":
    main()
