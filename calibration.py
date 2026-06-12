
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression


def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def brier_score(y_true, prob):
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean((prob - y_true) ** 2))


def expected_calibration_error(y_true, prob, n_bins=15):
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_table(y_true, prob, n_bins=15):
    """Reliability diyagramı için bin-bin tablo (CSV'ye yazılır)."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    n = len(prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob > lo) & (prob <= hi) if i > 0 else (prob >= lo) & (prob <= hi)
        cnt = int(mask.sum())
        rows.append(dict(
            bin_low=float(lo), bin_high=float(hi),
            count=cnt,
            mean_pred=float(prob[mask].mean()) if cnt else np.nan,
            frac_pos=float(y_true[mask].mean()) if cnt else np.nan,
            weight=cnt / n if n else 0.0,
        ))
    return rows


def fit_temperature(logits_val, y_val):
    """Val seti üstünde tek skaler sıcaklık T'yi NLL minimize ederek bul (T>0)."""
    logits_val = np.asarray(logits_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)

    def nll(logT):
        T = np.exp(logT)  # T>0 garantisi
        p = 1.0 / (1.0 + np.exp(-logits_val / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits, T):
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits / T))


def calibration_slope_intercept(y_true, prob):
    """
    Cox kalibrasyon: outcome ~ logit(p).
    İdeal: slope=1, intercept=0. slope<1 -> aşırı-güven (overfit), intercept -> sistematik kayma.
    """
    y_true = np.asarray(y_true, dtype=int)
    x = _logit(np.asarray(prob, dtype=float)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return dict(slope=np.nan, intercept=np.nan)
    # C çok büyük -> pratikte cezasız; tüm sklearn sürümleriyle uyumlu (penalty=None deprecation'ından kaçınır)
    lr = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    lr.fit(x, y_true)
    return dict(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))


def all_calibration_metrics(y_true, prob, n_bins=15):
    out = dict(
        brier=brier_score(y_true, prob),
        ece=expected_calibration_error(y_true, prob, n_bins),
    )
    out.update(calibration_slope_intercept(y_true, prob))
    return out
