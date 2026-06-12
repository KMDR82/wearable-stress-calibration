
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

V1_BLOCKS = ["Baseline", "Stroop", "First Rest", "TMCT", "Second Rest",
             "Real Opinion", "Opposite Opinion", "Subtract"]
STRESS = {"Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"}
WIN = 60.0


def _read_e4(path):
    raw = pd.read_csv(path, header=None)
    c0 = raw.iloc[0, 0]
    try:
        start = float(c0)
    except (ValueError, TypeError):
        start = pd.to_datetime(c0).value / 1e9
    rate = float(raw.iloc[1, 0])
    vals = raw.iloc[2:].to_numpy(dtype=float)
    return start, rate, vals


def _slice(start, rate, vals, w0, w1):
    t = start + np.arange(len(vals)) / rate
    m = (t >= w0) & (t < w1)
    return vals[m]


def build_v1(root):
    base = root + "/Wearable_Dataset/STRESS"
    si = pd.read_csv(root + "/subject-info.csv")
    si.columns = [c.strip() for c in si.columns]
    prot = dict(zip(si["Info"].astype(str).str.strip(), si["Protocol"]))
    feat, ys = {}, {}
    for sd in sorted(glob.glob(base + "/*")):
        sid = sd.split("/")[-1]
        if prot.get(sid) != "V1":
            continue
        tp = sd + "/tags.csv"
        if not os.path.exists(tp) or os.path.getsize(tp) == 0:
            continue
        tags = pd.to_datetime(pd.read_csv(tp, header=None)[0]).astype("int64").to_numpy() / 1e9
        if len(tags) < 9:
            continue
        hr = _read_e4(sd + "/HR.csv"); eda = _read_e4(sd + "/EDA.csv")
        tmp = _read_e4(sd + "/TEMP.csv"); acc = _read_e4(sd + "/ACC.csv")
        fX, Y = [], []
        w = tags[0]
        while w + WIN <= tags[8]:
            c = w + WIN / 2
            blk = next((V1_BLOCKS[i] for i in range(8) if tags[i] <= c < tags[i + 1]), None)
            if blk is None:
                w += WIN; continue
            h = _slice(*hr, w, w + WIN).ravel(); e = _slice(*eda, w, w + WIN).ravel()
            tt = _slice(*tmp, w, w + WIN).ravel(); a = _slice(*acc, w, w + WIN)
            if len(h) < 2 or len(e) < 2 or len(a) < 2:
                w += WIN; continue
            ae = np.sqrt((a ** 2).sum(1))
            sl = lambda v: float(linregress(np.arange(len(v)), v).slope) if np.std(v) > 0 else 0.0
            fX.append([h.mean(), h.std(), e.mean(), e.std(), sl(e),
                       tt.mean(), tt.std(), sl(tt), ae.mean(), ae.std(), sl(ae)])
            Y.append(1 if blk in STRESS else 0)
            w += WIN
        if fX:
            feat[sid] = np.array(fX, np.float32); ys[sid] = np.array(Y)
        print(sid, len(fX), "pencere, stres", round(float(np.mean(Y)), 2) if Y else None, flush=True)
    return feat, ys


def _ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0; n = len(p)
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def run(root, out_dir="/kaggle/working", seed=1337):
    feat, ys = build_v1(root)
    sids = [s for s in feat if len(np.unique(ys[s])) > 1]
    print("\ngeçerli LOSO fold:", len(sids), "| toplam pencere:", sum(len(ys[s]) for s in feat), flush=True)
    rows = []
    for test in sids:
        tr = [s for s in feat if s != test]
        Xtr = np.concatenate([feat[s] for s in tr]); ytr = np.concatenate([ys[s] for s in tr])
        gb = GradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        p = gb.predict_proba(feat[test])[:, 1]
        rows += [(test, float(a), int(b)) for a, b in zip(p, ys[test])]
        print(f"{test}: n={len(ys[test])} stres={ys[test].mean():.2f} AUC={roc_auc_score(ys[test],p):.3f}", flush=True)
    R = pd.DataFrame(rows, columns=["fold", "prob", "y_true"])
    R.to_csv(out_dir + "/exstress_loso_pred.csv", index=False)
    Y, P = R.y_true.values, R.prob.values
    print(f"\n[GB] HAVUZ AUC={roc_auc_score(Y,P):.3f} ECE={_ece(Y,P):.3f} Brier={brier_score_loss(Y,P):.3f} prevalans={Y.mean():.2f}", flush=True)

    # few-shot özne-uyarlamalı kalibrasyon (Nurse ile aynı)
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    rng = np.random.default_rng(0); curve = []
    for k in [0, 5, 10, 20, 40]:
        per = []
        for T in R.fold.unique():
            te = R[R.fold == T]; y = te.y_true.values; p = te.prob.values
            if len(np.unique(y)) < 2 or len(te) <= max(k, 5):
                continue
            if k == 0:
                per.append(_ece(y, p)); continue
            es = []
            for _ in range(30):
                idx = rng.choice(len(te), k, replace=False); mask = np.ones(len(te), bool); mask[idx] = False
                if len(np.unique(y[idx])) < 2:
                    continue
                pl = LogisticRegression(C=1e12, max_iter=1000).fit(lo(p[idx]).reshape(-1, 1), y[idx])
                es.append(_ece(y[mask], pl.predict_proba(lo(p[mask]).reshape(-1, 1))[:, 1]))
            if es:
                per.append(np.mean(es))
        curve.append(dict(k=k, ece_mean=round(float(np.mean(per)), 3), ece_std=round(float(np.std(per)), 3), n_fold=len(per)))
    C = pd.DataFrame(curve); print("\nfew-shot kalibrasyon eğrisi:\n", C.to_string(index=False), flush=True)
    C.to_csv(out_dir + "/exstress_fewshot_calib.csv", index=False)
    return R, C
