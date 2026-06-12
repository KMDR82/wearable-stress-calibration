

import numpy as np
import pandas as pd
from scipy.stats import linregress

CH = ["X", "Y", "Z", "EDA", "HR", "TEMP"]  


def _robust(a):
    med = np.median(a, 0, keepdims=True)
    iqr = (np.percentile(a, 75, 0) - np.percentile(a, 25, 0)).reshape(1, -1)
    iqr[iqr == 0] = 1.0
    return (a - med) / iqr


def _feats(w_raw):

    hr = w_raw[:, 4]; eda = w_raw[:, 3]; tmp = w_raw[:, 5]
    acc = np.sqrt((w_raw[:, :3] ** 2).sum(1))
    x = np.arange(len(w_raw))
    sl = lambda v: float(linregress(x, v).slope) if v.std() > 0 else 0.0
    return [hr.mean(), hr.std(), eda.mean(), eda.std(), sl(eda),
            tmp.mean(), tmp.std(), sl(tmp), acc.mean(), acc.std(), sl(acc)]


def build_nurse_windows(csv_path, fs=32, win_sec=60, gap_s=1.0, verbose=True):
    df = pd.read_csv(csv_path, dtype={"id": str, **{c: "float32" for c in CH}},
                     parse_dates=["datetime"], low_memory=False)
    WIN = fs * win_sec
    deep, feat, ys = {}, {}, {}
    for nid, g in df.groupby("id"):
        g = g.sort_values("datetime").reset_index(drop=True)
        dt = g["datetime"].astype("int64").values
        sess = np.cumsum(np.diff(dt, prepend=dt[0]) > int(gap_s * 1e9))  # >1s -> yeni vardiya
        sig_raw = g[CH].values.astype(np.float32)
        sig_n = _robust(sig_raw)
        lab = g["label"].values.astype(int)
        dX, fX, Y = [], [], []
        for s in np.unique(sess):
            m = sess == s
            sr, sn, sl_ = sig_raw[m], sig_n[m], lab[m]
            for st in range(0, len(sr) - WIN + 1, WIN):
                lw = sl_[st:st + WIN]
                maj = np.bincount(lw, minlength=3).argmax()
                if maj == 2:
                    y = 1
                elif maj == 0:
                    y = 0
                else:
                    continue
                dX.append(sn[st:st + WIN].T.astype(np.float32))  # (C, WIN)
                fX.append(_feats(sr[st:st + WIN]))
                Y.append(y)
        if dX:
            deep[nid] = np.stack(dX); feat[nid] = np.array(fX, np.float32); ys[nid] = np.array(Y)
        if verbose:
            print(nid, len(dX), "pencere, stres",
                  round(float(np.mean(Y)), 2) if Y else None, flush=True)
    return deep, feat, ys
