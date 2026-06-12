
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression

CSV_DIR = "/kaggle/input/datasets/aakkaya/csvfiles"
OUT_DIR = "/kaggle/working/figures"

CB = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
      "purple": "#CC79A7", "grey": "#666666", "black": "#000000", "yellow": "#E69F00"}
DSCOL = {"WESAD": CB["blue"], "Nurse": CB["orange"], "Exercise-Stress": CB["green"]}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color="#E8E8E8", linewidth=0.7, zorder=0)


def _barlabels(ax, bars, fmt="{:.2f}", dy=0.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=7.5, color=CB["black"])


def _f(n):
    p = os.path.join(CSV_DIR, n); return p if os.path.exists(p) else None


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0; n = len(p)
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum(): e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def per_fold_auc(df):
    return np.array([roc_auc_score(g.y_true, g.prob) for _, g in df.groupby("fold") if g.y_true.nunique() > 1])


def per_fold_auc_map(df):
    return {f: roc_auc_score(g.y_true, g.prob) for f, g in df.groupby("fold") if g.y_true.nunique() > 1}


def reliability(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1); mp, fp = [], []
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum(): mp.append(p[m].mean()); fp.append(y[m].mean())
    return np.array(mp), np.array(fp)


def calib_slope(y, p):
    lr = LogisticRegression(C=1e12, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def fewshot_per_fold(df, k=20, R=30, seed=0):
    rng = np.random.default_rng(seed); raw, rec = {}, {}
    for T, te in df.groupby("fold"):
        y = te.y_true.values; p = te.prob.values
        if len(np.unique(y)) < 2 or len(te) <= k: continue
        raw[T] = ece(y, p); es = []
        for _ in range(R):
            idx = rng.choice(len(te), k, replace=False); mask = np.ones(len(te), bool); mask[idx] = False
            if len(np.unique(y[idx])) < 2: continue
            pl = LogisticRegression(C=1e12, max_iter=1000).fit(_logit(p[idx]).reshape(-1, 1), y[idx])
            es.append(ece(y[mask], pl.predict_proba(_logit(p[mask]).reshape(-1, 1))[:, 1]))
        if es: rec[T] = np.mean(es)
    keys = [t for t in raw if t in rec]
    return np.array([raw[t] for t in keys]), np.array([rec[t] for t in keys])


def panel_labels(fig, axes, labels, y=0.015):
    fig.canvas.draw()
    for ax, lab in zip(axes, labels):
        bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, y, lab, ha="center", va="bottom", fontsize=11)


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig); print(f"[ok] {name}")


def _load():
    d = {}
    fw, fn, fe = _f("feat_loso_predictions.csv"), _f("nurse_deep_pred.csv"), _f("exstress_loso_pred.csv")
    if fw: d["wesad"] = pd.read_csv(fw)
    if fn:
        nu = pd.read_csv(fn); d["nurse_all"] = nu
        d["nurse"] = nu[(nu.method == "gb") & (nu.label_frac == 1.0)].copy()
    if fe: d["ex"] = pd.read_csv(fe)
    if _f("fewshot_calib_curve.csv"): d["fc_n"] = pd.read_csv(_f("fewshot_calib_curve.csv"))
    if _f("exstress_fewshot_calib.csv"): d["fc_e"] = pd.read_csv(_f("exstress_fewshot_calib.csv"))
    if _f("nurse_dca_raw_vs_recal.csv"): d["dca"] = pd.read_csv(_f("nurse_dca_raw_vs_recal.csv"))
    return d



def fig1(d):
    if not all(k in d for k in ("wesad", "nurse", "ex")): print("[skip] fig1"); return
    dss = [("WESAD", d["wesad"]), ("Nurse", d["nurse"]), ("Exercise-Stress", d["ex"])]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
    # (a) per-fold AUROC box+strip
    data = [per_fold_auc(df) for _, df in dss]
    bp = axs[0].boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=CB["black"], lw=1.2))
    for patch, (nm, _) in zip(bp["boxes"], dss):
        patch.set_facecolor(DSCOL[nm]); patch.set_alpha(0.3); patch.set_edgecolor(DSCOL[nm])
    rng = np.random.default_rng(0)
    for i, dd in enumerate(data):
        axs[0].scatter(np.full(len(dd), i + 1) + rng.uniform(-0.12, 0.12, len(dd)), dd, s=12,
                       color=CB["black"], alpha=0.55, zorder=3, linewidths=0)
    axs[0].axhline(0.5, ls="--", lw=0.8, color=CB["grey"])
    axs[0].set_xticks([1, 2, 3]); axs[0].set_xticklabels(["WESAD", "Nurse", "Exercise"], rotation=12)
    axs[0].set_ylabel("Leave-one-subject-out AUROC"); axs[0].set_ylim(0.2, 1.02)
    # (b) pooled ROC
    for nm, df in dss:
        fpr, tpr, _ = roc_curve(df.y_true, df.prob); a = roc_auc_score(df.y_true, df.prob)
        axs[1].plot(fpr, tpr, color=DSCOL[nm], lw=1.5, label=f"{nm} ({a:.2f})")
    axs[1].plot([0, 1], [0, 1], ls="--", lw=0.8, color=CB["grey"])
    axs[1].set_xlabel("False positive rate"); axs[1].set_ylabel("True positive rate")
    axs[1].legend(loc="lower right")
    # (c) per-subject prevalence
    for i, (nm, df) in enumerate(dss):
        prev = df.groupby("fold").y_true.mean().values
        axs[2].scatter(np.full(len(prev), i) + rng.uniform(-0.12, 0.12, len(prev)), prev, s=14,
                       color=DSCOL[nm], alpha=0.7, linewidths=0)
        axs[2].hlines(prev.mean(), i - 0.25, i + 0.25, color=CB["black"], lw=1.5)
    axs[2].set_xticks([0, 1, 2]); axs[2].set_xticklabels(["WESAD", "Nurse", "Exercise"], rotation=12)
    _grid(axs[2]); axs[2].set_ylabel("Per-subject stress prevalence"); axs[2].set_ylim(0, 1)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    panel_labels(fig, axs, ["(a)", "(b)", "(c)"])
    _save(fig, "figure1_discrimination")



def fig2(d):
    if "nurse_all" not in d: print("[skip] fig2"); return
    nu = d["nurse_all"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    methods = [("gb", CB["blue"], "o", "Feature + GB"),
               ("deep_ssl", CB["orange"], "s", "1D-CNN + SSL"),
               ("deep_scratch", CB["green"], "^", "1D-CNN scratch")]
    for m, c, mk, lab in methods:
        sub = nu[nu.method == m]; fr = sorted(sub.label_frac.unique())
        means, los, his = [], [], []
        for f in fr:
            af = per_fold_auc(sub[sub.label_frac == f])
            mu = af.mean(); se = af.std(ddof=1) / np.sqrt(len(af)) if len(af) > 1 else 0
            means.append(mu); los.append(mu - 1.96 * se); his.append(mu + 1.96 * se)
        x = [v * 100 for v in fr]
        axs[0].plot(x, means, marker=mk, color=c, label=lab, lw=1.4, ms=5)
        axs[0].fill_between(x, los, his, color=c, alpha=0.15, linewidth=0)
    axs[0].axhline(0.5, ls="--", lw=0.8, color=CB["grey"])
    _grid(axs[0]); axs[0].set_xscale("log"); axs[0].set_xticks([5, 10, 25, 100]); axs[0].set_xticklabels(["5", "10", "25", "100"])
    axs[0].set_xlabel("Labelled training data (%)"); axs[0].set_ylabel("Mean AUROC (95% CI)")
    axs[0].legend(loc="upper left")
    # (b) fold-wise GB vs SSL at full labels
    gb = per_fold_auc_map(nu[(nu.method == "gb") & (nu.label_frac == 1.0)])
    ss = per_fold_auc_map(nu[(nu.method == "deep_ssl") & (nu.label_frac == 1.0)])
    keys = [k for k in gb if k in ss]
    gx = np.array([gb[k] for k in keys]); sy = np.array([ss[k] for k in keys])
    axs[1].scatter(gx, sy, s=22, color=CB["purple"], alpha=0.8, linewidths=0, zorder=3)
    lim = [0.2, 1.0]; axs[1].plot(lim, lim, ls="--", lw=0.9, color=CB["grey"])
    axs[1].set_xlim(lim); axs[1].set_ylim(lim)
    axs[1].set_xlabel("AUROC, Feature + GB"); axs[1].set_ylabel("AUROC, 1D-CNN + SSL")
    axs[1].text(0.97, 0.06, "below line:\nGB better", transform=axs[1].transAxes,
                ha="right", va="bottom", fontsize=8, color=CB["grey"])
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    panel_labels(fig, axs, ["(a)", "(b)"])
    _save(fig, "figure2_label_efficiency")


# ---------------- FIGURE 3: calibration problem (3 panels) ----------------
def fig3(d):
    if not all(k in d for k in ("wesad", "nurse", "ex")): print("[skip] fig3"); return
    dss = [("WESAD", d["wesad"]), ("Nurse", d["nurse"]), ("Exercise-Stress", d["ex"])]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9))
    axs[0].plot([0, 1], [0, 1], ls="--", lw=0.9, color=CB["grey"])
    for nm, df in dss:
        mp, fp = reliability(df.y_true.values, df.prob.values, 10)
        axs[0].plot(mp, fp, marker="o", color=DSCOL[nm], label=nm, lw=1.3, ms=4)
    axs[0].set_xlabel("Mean predicted probability"); axs[0].set_ylabel("Observed frequency")
    axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1); axs[0].legend(loc="lower right")
    # (b) calibration slope
    names = [nm for nm, _ in dss]; slopes = [calib_slope(df.y_true.values, df.prob.values)[0] for _, df in dss]
    _grid(axs[1]); _b1=axs[1].bar(range(3), slopes, color=[DSCOL[n] for n in names], alpha=0.8, width=0.6, zorder=3); _barlabels(axs[1], _b1)
    axs[1].axhline(1.0, ls="--", lw=0.9, color=CB["black"])
    axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["WESAD", "Nurse", "Exercise"], rotation=12)
    axs[1].set_ylabel("Calibration slope"); axs[1].set_ylim(0, max(slopes + [1]) * 1.15)
    axs[1].text(0.02, 1.02, "ideal = 1", transform=axs[1].get_yaxis_transform(), fontsize=7.5, color=CB["grey"])
    # (c) ECE & Brier grouped
    eces = [ece(df.y_true.values, df.prob.values) for _, df in dss]
    briers = [brier_score_loss(df.y_true.values, df.prob.values) for _, df in dss]
    x = np.arange(3); w = 0.38
    _grid(axs[2]); _be=axs[2].bar(x - w / 2, eces, w, color=CB["orange"], alpha=0.85, label="ECE", zorder=3)
    _bb=axs[2].bar(x + w / 2, briers, w, color=CB["blue"], alpha=0.85, label="Brier", zorder=3)
    _barlabels(axs[2], _be); _barlabels(axs[2], _bb)
    axs[2].set_xticks(x); axs[2].set_xticklabels(["WESAD", "Nurse", "Exercise"], rotation=12)
    axs[2].set_ylabel("Error"); axs[2].legend(loc="upper left")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    panel_labels(fig, axs, ["(a)", "(b)", "(c)"])
    _save(fig, "figure3_calibration_problem")


# ---------------- FIGURE 4: few-shot remedy (2 panels) ----------------
def fig4(d):
    if not all(k in d for k in ("fc_n", "fc_e", "nurse")): print("[skip] fig4"); return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    cn = d["fc_n"].dropna(subset=["ece_mean"]); ce = d["fc_e"].dropna(subset=["ece_mean"])
    _grid(axs[0])
    axs[0].errorbar(cn.k, cn.ece_mean, yerr=cn.ece_std, marker="s", color=CB["orange"], label="Nurse", lw=1.4, ms=5, capsize=2.5)
    axs[0].errorbar(ce.k, ce.ece_mean, yerr=ce.ece_std, marker="^", color=CB["green"], label="Exercise-Stress", lw=1.4, ms=5, capsize=2.5)
    axs[0].set_xlabel("Subject-specific calibration labels (k)"); axs[0].set_ylabel("Expected calibration error")
    axs[0].set_ylim(0, max(cn.ece_mean.max(), ce.ece_mean.max()) * 1.35); axs[0].legend(loc="upper left")
    # (b) per-subject raw vs few-shot (k=20) on Nurse
    raw, rec = fewshot_per_fold(d["nurse"], k=20)
    axs[1].scatter(raw, rec, s=26, color=CB["orange"], alpha=0.8, linewidths=0, zorder=3)
    lim = [0, max(raw.max(), rec.max(), 0.5) * 1.05] if len(raw) else [0, 0.6]
    axs[1].plot(lim, lim, ls="--", lw=0.9, color=CB["grey"]); axs[1].set_xlim(lim); axs[1].set_ylim(lim)
    axs[1].set_xlabel("Per-subject ECE, global"); axs[1].set_ylabel("Per-subject ECE, few-shot (k=20)")
    axs[1].text(0.97, 0.06, "below line:\nimproved", transform=axs[1].transAxes, ha="right", va="bottom", fontsize=8, color=CB["grey"])
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    panel_labels(fig, axs, ["(a)", "(b)"])
    _save(fig, "figure4_fewshot_calibration")


# ---------------- FIGURE 5: decision curve (2 panels) ----------------
def fig5(d):
    if "dca" not in d: print("[skip] fig5"); return
    t = d["dca"]; th = t.threshold.values
    ref = np.maximum(t.nb_all.values, 0.0)
    sup_frac = float((t.nb_recal.values > ref).mean())
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a = axs[0]; _grid(a)
    # karar-faydalı bölge gölgesi (recal > en iyi referans)
    a.fill_between(th, ref, t.nb_recal.values, where=t.nb_recal.values > ref,
                   color=CB["orange"], alpha=0.14, linewidth=0, zorder=1)
    a.plot(th, t.nb_recal, color=CB["orange"], lw=2.2, label="Recalibrated", zorder=4)
    a.plot(th, t.nb_raw, color=CB["blue"], lw=1.6, label="Uncalibrated", zorder=3)
    a.plot(th, t.nb_all, color=CB["grey"], lw=1.2, ls="--", label="Treat all", zorder=2)
    a.axhline(0.0, color=CB["black"], lw=1.0, ls=":", label="Treat none", zorder=2)
    a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit")
    a.set_xlim(th.min(), th.max())
    ymax = max(t.nb_recal.max(), t.nb_raw.max())
    a.set_ylim(max(-0.3, t.nb_all.min()), ymax * 1.40)   # üstte lejant için boşluk
    a.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
             columnspacing=1.4, handlelength=1.7)
    _lab = "Decision-useful over %.0f%% of thresholds" % (sup_frac * 100)
    a.annotate(_lab,
               xy=(th[int(len(th)*0.7)], t.nb_recal.values[int(len(th)*0.7)]),
               xytext=(0.50, 0.42), textcoords="axes fraction", fontsize=8.5,
               color=CB["orange"], ha="left",
               arrowprops=dict(arrowstyle="->", color=CB["orange"], lw=1.0))
    # (b) artımsal net-fayda
    b = axs[1]; _grid(b)
    inc_r = t.nb_recal.values - ref; inc_u = t.nb_raw.values - ref
    b.fill_between(th, 0, inc_r, where=inc_r > 0, color=CB["orange"], alpha=0.18, linewidth=0, zorder=1)
    b.plot(th, inc_r, color=CB["orange"], lw=2.0, label="Recalibrated", zorder=3)
    b.plot(th, inc_u, color=CB["blue"], lw=1.6, label="Uncalibrated", zorder=3)
    b.axhline(0.0, color=CB["black"], lw=0.9, ls=":", zorder=2)
    b.set_xlabel("Threshold probability"); b.set_ylabel("Net benefit vs. best reference")
    b.set_xlim(th.min(), th.max()); b.legend(loc="upper left")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    panel_labels(fig, axs, ["(a)", "(b)"])
    _save(fig, "figure5_decision_curve")


def main():
    print("CSV_DIR:", sorted(os.listdir(CSV_DIR)) if os.path.isdir(CSV_DIR) else "YOK", flush=True)
    d = _load()
    fig1(d); fig2(d); fig3(d); fig4(d); fig5(d)
    print("Şekiller ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
