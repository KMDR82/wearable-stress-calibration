"""
three_class_nurse.py
Three-class sensitivity analysis (reviewer 2 point 1).

The main analysis discards the intermediate Nurse self-report level and models
stress (level 2) versus baseline (level 0). Here the intermediate level (1) is
reinstated and a three-class gradient-boosting model is trained under the same
LOSO protocol, to check whether the intermediate level is separable.

Windowing, robust per-subject normalisation basis, the common 11-feature set,
gradient boosting and seed are identical to the binary pipeline; the only change
is that level-1 windows are kept and the label is 0/1/2 instead of 0/1.

Because three-class AUROC is not comparable to the binary 0-vs-2 AUROC, we report
one-vs-rest AUROC per class, macro-OVR AUROC, balanced accuracy, and the row-
normalised confusion matrix.

Input : nurse merged_data.csv (datetime, X, Y, Z, EDA, HR, TEMP, label, id)
Usage : python three_class_nurse.py --nurse_csv /path/to/merged_data.csv --out ./
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

SEED = 1337
CH = ["X", "Y", "Z", "EDA", "HR", "TEMP"]


def _feats(w):
    hr = w[:, 4]; eda = w[:, 3]; tmp = w[:, 5]
    acc = np.sqrt((w[:, :3] ** 2).sum(1))
    x = np.arange(len(w))
    sl = lambda v: float(linregress(x, v).slope) if v.std() > 0 else 0.0
    return [hr.mean(), hr.std(), eda.mean(), eda.std(), sl(eda),
            tmp.mean(), tmp.std(), sl(tmp), acc.mean(), acc.std(), sl(acc)]


def build_3class(csv_path, fs=32, win_sec=60, gap_s=1.0):
    """Identical windowing to the binary pipeline; only difference: level-1 kept."""
    df = pd.read_csv(csv_path, dtype={"id": str, **{c: "float32" for c in CH}},
                     parse_dates=["datetime"], low_memory=False)
    WIN = fs * win_sec
    feat, ys = {}, {}
    for nid, g in df.groupby("id"):
        g = g.sort_values("datetime").reset_index(drop=True)
        dt = g["datetime"].astype("int64").values
        sess = np.cumsum(np.diff(dt, prepend=dt[0]) > int(gap_s * 1e9))
        sr = g[CH].values.astype(np.float32); lab = g["label"].values.astype(int)
        fX, Y = [], []
        for s in np.unique(sess):
            m = sess == s; a, c = sr[m], lab[m]
            for st in range(0, len(a) - WIN + 1, WIN):
                lw = c[st:st + WIN]; maj = int(np.bincount(lw, minlength=3).argmax())
                if maj not in (0, 1, 2):
                    continue
                fX.append(_feats(a[st:st + WIN])); Y.append(maj)
        if fX:
            feat[nid] = np.array(fX, np.float32); ys[nid] = np.array(Y, int)
    return feat, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nurse_csv", required=True)
    ap.add_argument("--out", default="./")
    args = ap.parse_args()

    feat, ys = build_3class(args.nurse_csv)
    sids = list(feat.keys())

    print("subject   n    n0   n1   n2")
    for s in sids:
        y = ys[s]
        print(f"{s:>6} {len(y):>4} {int((y==0).sum()):>4} {int((y==1).sum()):>4} {int((y==2).sum()):>4}")

    rows = []
    for test in sids:
        tr = [s for s in sids if s != test]
        Xtr = np.concatenate([feat[s] for s in tr]); ytr = np.concatenate([ys[s] for s in tr])
        if len(np.unique(ytr)) < 2:
            continue
        gb = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
        P = gb.predict_proba(feat[test]); full = np.zeros((len(P), 3))
        for j, c in enumerate(gb.classes_):
            full[:, int(c)] = P[:, j]
        for pr, yt in zip(full, ys[test]):
            rows.append(dict(fold=test, y_true=int(yt),
                             p0=float(pr[0]), p1=float(pr[1]), p2=float(pr[2])))
    R = pd.DataFrame(rows)
    R.to_csv(args.out.rstrip("/") + "/nurse_3class_pred.csv", index=False)

    Y = R.y_true.values; PB = R[["p0", "p1", "p2"]].values; pred = PB.argmax(1)
    print("\n=== three-class (Nurse, GB LOSO) ===")
    print("pooled class counts:", {c: int((Y == c).sum()) for c in (0, 1, 2)})
    for c in (0, 1, 2):
        print(f"  OVR AUROC class {c}: {roc_auc_score((Y == c).astype(int), PB[:, c]):.3f}")
    print(f"  macro-OVR AUROC (pooled): {roc_auc_score(Y, PB, multi_class='ovr', average='macro'):.3f}")
    print(f"  balanced accuracy (pooled): {balanced_accuracy_score(Y, pred):.3f}")
    cm = confusion_matrix(Y, pred, labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
    print("  confusion (rows=true 0/1/2, row-normalised):")
    for row in cmn:
        print("   ", [f"{v:.2f}" for v in row])
    ps = [roc_auc_score(g.y_true.values, g[["p0", "p1", "p2"]].values, multi_class="ovr", average="macro")
          for _, g in R.groupby("fold") if g.y_true.nunique() == 3]
    if ps:
        print(f"  per-subject macro-OVR AUROC: mean={np.mean(ps):.3f} (n={len(ps)} three-class folds)")
    print("(reference: binary 0-vs-2 pooled AUROC 0.631 / per-subject 0.601)")


if __name__ == "__main__":
    main()
