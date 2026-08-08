"""
normalization_sensitivity.py
Causal-normalisation sensitivity analysis (reviewer 1 point 9).

The main pipeline normalises each subject's channels with the median and IQR
computed over the whole recording, which is not available prospectively. Here the
per-subject median and IQR are estimated from only the earliest fraction of each
subject's windows and applied causally to the whole recording, on WESAD (the only
pathway whose features are computed from the normalised signal; the Nurse and
Exercise-Stress gradient-boosting features use raw channel statistics and are
invariant to monotone rescaling).

Only the normalisation window changes; the 18-feature set, windowing, gradient
boosting, LOSO and seed are identical to the main pipeline. frac=1.0 reproduces
the published WESAD numbers (0.992 / 0.983 / 0.036 / 0.029), a built-in check.

Run on Kaggle where the raw WESAD .pkl files exist:
    python normalization_sensitivity.py --wesad_root /kaggle/input/.../WESAD --out ./
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, find_peaks
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

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


def early_norm(mat, frac):
    """Median/IQR from the first `frac` of the recording, applied to the whole recording.
    frac=1.0 is the published whole-recording normalisation."""
    k = max(10, int(round(frac * mat.shape[0])))
    head = mat[:k]
    med = np.median(head, 0, keepdims=True)
    iqr = np.subtract(*np.percentile(head, [75, 25], axis=0)).reshape(1, -1)
    iqr[iqr == 0] = 1.0
    return (mat - med) / iqr


def feats18(win, fs):
    pulse = win[:, 0]
    pk, _ = find_peaks(pulse, distance=int(0.4 * fs))
    if len(pk) > 3:
        ibi = np.diff(pk) / fs
        hr = 60.0 / np.mean(ibi); sdnn = np.std(ibi)
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 1 else 0.0
    else:
        hr, sdnn, rmssd = 0.0, 0.0, 0.0
    f = [hr, sdnn, rmssd]
    for c in range(1, win.shape[1]):
        s = win[:, c]
        f += [float(np.mean(s)), float(np.std(s)), float(np.polyfit(np.arange(len(s)), s, 1)[0])]
    return np.array(f, dtype=float)


def build(root, sid, frac):
    with open(os.path.join(root, f"S{sid}", f"S{sid}.pkl"), "rb") as fp:
        d = pickle.load(fp, encoding="latin1")
    sig = d["signal"]; label = np.asarray(d["label"]).astype(int); fs = WRIST_FS
    mat = early_norm(_wrist(sig, fs), frac)
    win = int(WIN_SEC * fs); stride = int(STRIDE_SEC * fs)
    lwin = int(WIN_SEC * 700); lstride = int(STRIDE_SEC * 700)
    fX, y = [], []; n = mat.shape[0]; li = 0
    for st in range(0, n - win + 1, stride):
        sl = label[li:li + lwin]; li += lstride
        if len(sl) == 0:
            continue
        v, c = np.unique(sl, return_counts=True); maj = int(v[np.argmax(c)])
        if maj not in KEEP:
            continue
        fX.append(feats18(mat[st:st + win], fs)); y.append(1 if maj == STRESS else 0)
    return (np.stack(fX).astype(np.float32), np.asarray(y, int)) if y else None


def ece(y, p, nb=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    b = np.linspace(0, 1, nb + 1); e = 0.0; n = len(p)
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_root", required=True)
    ap.add_argument("--out", default="./")
    ap.add_argument("--fracs", default="1.0,0.5,0.2,0.1")
    args = ap.parse_args()
    fracs = [float(x) for x in args.fracs.split(",")]
    subs = sorted(int(x[1:]) for x in os.listdir(args.wesad_root)
                  if x.startswith("S") and x[1:].isdigit())

    print(f"{'frac':>6} {'perAUROC':>9} {'poolAUROC':>10} {'perECE':>7} {'Brier':>7}")
    for frac in fracs:
        feats, ys = {}, {}
        for s in subs:
            r = build(args.wesad_root, s, frac)
            if r is not None:
                feats[s], ys[s] = r
        ss = [s for s in subs if s in feats]; rows = []
        for test in ss:
            tr = [s for s in ss if s != test]
            Xtr = np.concatenate([feats[s] for s in tr]); ytr = np.concatenate([ys[s] for s in tr])
            gb = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
            p = gb.predict_proba(feats[test])[:, 1]
            rows += [dict(fold=test, prob=float(a), y_true=int(b)) for a, b in zip(p, ys[test])]
        R = pd.DataFrame(rows)
        pa = np.mean([roc_auc_score(g.y_true, g.prob) for _, g in R.groupby("fold") if g.y_true.nunique() > 1])
        es = np.mean([ece(g.y_true.values, g.prob.values) for _, g in R.groupby("fold") if g.y_true.nunique() > 1])
        print(f"{frac:>6.2f} {pa:>9.3f} {roc_auc_score(R.y_true, R.prob):>10.3f} "
              f"{es:>7.3f} {brier_score_loss(R.y_true, R.prob):>7.3f}", flush=True)
        if abs(frac - 1.0) < 1e-9:
            R.to_csv(args.out.rstrip("/") + "/wesad_norm_full.csv", index=False)
    print("(frac=1.00 should match the published WESAD: 0.992 / 0.983 / 0.036 / 0.029)")


if __name__ == "__main__":
    main()
