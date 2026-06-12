import os
import pickle
import numpy as np
from scipy.signal import resample_poly, find_peaks


# --------------------------------------------------------------------------
def _resample_to(x, fs_in, fs_out):
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    if fs_in == fs_out:
        return x
    out = np.zeros((int(round(x.shape[0] * fs_out / fs_in)), x.shape[1]))
    for c in range(x.shape[1]):
        out[: , c] = resample_poly(x[:, c], fs_out, fs_in)[: out.shape[0]]
    return out


def _robust_norm(x):
    med = np.median(x, axis=0, keepdims=True)
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0)).reshape(1, -1)
    iqr[iqr == 0] = 1.0
    return (x - med) / iqr


def load_wesad_subject(root, sid):
    path = os.path.join(root, f"S{sid}", f"S{sid}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _wrist_matrix(sig, fs_out, channels):

    rates = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
    cols = []
    for ch in channels:
        raw = np.asarray(sig["wrist"][ch], dtype=float)
        r = _resample_to(raw, rates[ch], fs_out)
        cols.append(r)
    T = min(c.shape[0] for c in cols)
    return np.concatenate([c[:T] for c in cols], axis=1)


def _common_features_from_window(win, fs):

    pulse = win[:, 0]
    peaks, _ = find_peaks(pulse, distance=int(0.4 * fs))
    if len(peaks) > 3:
        ibi = np.diff(peaks) / fs
        hr = 60.0 / np.mean(ibi)
        sdnn = np.std(ibi)
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) > 1 else 0.0
    else:
        hr, sdnn, rmssd = 0.0, 0.0, 0.0
    feats = [hr, sdnn, rmssd]

    for c in range(1, win.shape[1]):
        s = win[:, c]
        feats += [float(np.mean(s)), float(np.std(s)),
                  float(np.polyfit(np.arange(len(s)), s, 1)[0])]
    return np.array(feats, dtype=float)


def build_windows(data, cfg, site="wrist"):

    sig = data["signal"]
    label = np.asarray(data["label"]).astype(int)  # 700Hz
    fs_label = 700

    if site == "wrist":
        fs = cfg.wrist_fs
        mat = _wrist_matrix(sig, fs, cfg.wrist_channels)
    else:  
        fs = cfg.chest_fs
        ecg = _resample_to(sig["chest"]["ECG"], 700, fs)
        eda = _resample_to(sig["chest"]["EDA"], 700, fs)
        resp = _resample_to(sig["chest"]["Resp"], 700, fs)
        temp = _resample_to(sig["chest"]["Temp"], 700, fs)
        acc = _resample_to(sig["chest"]["ACC"], 700, fs)
        acc_e = np.sqrt((acc ** 2).sum(axis=1, keepdims=True))
        T = min(x.shape[0] for x in [ecg, eda, resp, temp, acc_e])
        mat = np.concatenate([ecg[:T], eda[:T], resp[:T], temp[:T], acc_e[:T]], axis=1)

    if cfg.normalization == "robust_per_subject":
        mat = _robust_norm(mat)

    win = int(cfg.window_sec * fs)
    stride = int(cfg.window_stride_sec * fs)
    lwin = int(cfg.window_sec * fs_label)
    lstride = int(cfg.window_stride_sec * fs_label)

    deep_X, feat_X, y = [], [], []
    n = mat.shape[0]
    li = 0
    for start in range(0, n - win + 1, stride):
        lab_seg = label[li:li + lwin]
        li += lstride
        if len(lab_seg) == 0:
            continue
        vals, counts = np.unique(lab_seg, return_counts=True)
        maj = int(vals[np.argmax(counts)])
        if maj not in cfg.keep_labels:
            continue
        seg = mat[start:start + win]  # (win, C)
        deep_X.append(seg.T)          # (C, win)
        feat_X.append(_common_features_from_window(seg, fs))
        y.append(1 if maj == cfg.stress_label else 0)

    if not y:
        return None
    return (np.stack(deep_X).astype(np.float32),
            np.stack(feat_X).astype(np.float32),
            np.asarray(y, dtype=np.int64))


def discover_subjects(root):
    if not os.path.isdir(root):
        return []
    sids = []
    for name in os.listdir(root):
        if name.startswith("S") and name[1:].isdigit():
            sids.append(int(name[1:]))
    return sorted(sids)



def synthetic_subject(cfg, sid, n_windows=60, n_channels=4, rng=None):
    """Plumbing testi için: deneğe-özgü kayma + stres sınıfı arası ayrılabilirlik."""
    rng = rng or np.random.default_rng(cfg.seed + sid)
    T = int(cfg.window_sec * cfg.wrist_fs)
    subj_shift = rng.normal(0, 0.5, size=(n_channels, 1))
    y = rng.integers(0, 2, size=n_windows)
    deep_X = []
    feat_X = []
    for i in range(n_windows):
        base = rng.normal(0, 1, size=(n_channels, T)) + subj_shift
        # stres sınıfı: ilk kanalda hafif frekans/genlik farkı
        if y[i] == 1:
            t = np.arange(T) / cfg.wrist_fs
            base[0] += 0.6 * np.sin(2 * np.pi * 1.4 * t)
        deep_X.append(base.astype(np.float32))
        feat_X.append(np.concatenate([base.mean(axis=1), base.std(axis=1)]).astype(np.float32))
    return np.stack(deep_X), np.stack(feat_X), y.astype(np.int64)
