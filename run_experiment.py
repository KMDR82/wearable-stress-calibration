
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from config import CFG
import data_wesad as dw
import calibration as cal
import decision_curve as dca
import stats_tests as st
import baselines as bl

try:
    import models as M
    TORCH_OK = M.TORCH_OK
except Exception:
    TORCH_OK = False


def _logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))



def load_all(cfg):
    """Döndürür: dict[sid] = {'wrist':(deepX,featX,y), 'chest':(deepX,featX,y)}."""
    use_demo = cfg.demo_mode or not TORCH_OK or not os.path.isdir(cfg.wesad_root)
    cfg.demo_mode = use_demo
    subjects = {}
    if use_demo:
        for sid in range(2, 12):  # 10 sahte denek
            dX, fX, y = dw.synthetic_subject(cfg, sid, n_windows=60, n_channels=4)
            subjects[sid] = {"wrist": (dX, fX, y), "chest": (dX, fX, y)}
        return subjects, True
    for sid in dw.discover_subjects(cfg.wesad_root):
        data = dw.load_wesad_subject(cfg.wesad_root, sid)
        w = dw.build_windows(data, cfg, site="wrist")
        c = dw.build_windows(data, cfg, site="chest") if cfg.run_site_shift else None
        if w is None:
            continue
        subjects[sid] = {"wrist": w, "chest": c}
    return subjects, False



def evaluate_probs(y_true, prob):
    out = dict(
        auc=float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        f1=float(f1_score(y_true, (prob >= 0.5).astype(int), zero_division=0)),
        acc=float(accuracy_score(y_true, (prob >= 0.5).astype(int))),
    )
    out.update(cal.all_calibration_metrics(y_true, prob, CFG.ece_bins))
    out.update(st.spiegelhalter_z(y_true, prob))  # z, p_value (kalibrasyon)
    return out


def subsample_labels(y, frac, rng):

    if frac >= 1.0:
        return np.arange(len(y))
    idx = []
    for cls in np.unique(y):
        c = np.where(y == cls)[0]
        k = max(1, int(round(len(c) * frac)))
        idx.extend(rng.choice(c, size=k, replace=False))
    return np.array(sorted(idx))



def run_loso(subjects, cfg, pred_rows, result_rows, demo):
    sids = sorted(subjects.keys())
    rng = np.random.default_rng(cfg.seed)


    enc_ssl = None
    if TORCH_OK and not demo:
        pool = np.concatenate([subjects[s]["wrist"][0] for s in sids], axis=0)
        in_ch = pool.shape[1]
        ckpt = os.path.join(cfg.ckpt_dir, "ssl_loso.pt")
        enc_ssl = M.pretrain_ssl(pool, cfg, in_ch, ckpt)

    for test_sid in sids:
        tr = [s for s in sids if s != test_sid]
        Xtr_d = np.concatenate([subjects[s]["wrist"][0] for s in tr], axis=0)
        Xtr_f = np.concatenate([subjects[s]["wrist"][1] for s in tr], axis=0)
        ytr = np.concatenate([subjects[s]["wrist"][2] for s in tr], axis=0)
        Xte_d, Xte_f, yte = subjects[test_sid]["wrist"]

        for frac in cfg.label_fractions:
            sel = subsample_labels(ytr, frac, rng)
            methods = {}


            methods["gb_feat"] = bl.fit_predict_proba(bl.gb_baseline(cfg.seed), Xtr_f[sel], ytr[sel], Xte_f)
            methods["logreg_feat"] = bl.fit_predict_proba(bl.logreg_baseline(cfg.seed), Xtr_f[sel], ytr[sel], Xte_f)


            if TORCH_OK and not demo:
                in_ch = Xtr_d.shape[1]
                # scratch
                enc0 = M.build_encoder(in_ch, cfg)
                clf0 = M.train_classifier(enc0, Xtr_d[sel], ytr[sel], cfg, freeze=False)
                methods["deep_scratch"] = 1 / (1 + np.exp(-M.predict_logits(clf0, Xte_d, cfg)))
                # SSL-init
                enc1 = M.build_encoder(in_ch, cfg); enc1.load_state_dict(enc_ssl.state_dict())
                clf1 = M.train_classifier(enc1, Xtr_d[sel], ytr[sel], cfg, freeze=False)
                logit_ssl = M.predict_logits(clf1, Xte_d, cfg)
                methods["deep_ssl"] = 1 / (1 + np.exp(-logit_ssl))


            for name, prob in list(methods.items()):

                m = evaluate_probs(yte, prob)
                m.update(dict(method=name, protocol="loso", fold=test_sid,
                              label_frac=frac, calibrated="raw", demo=demo,
                              n_test=len(yte), prevalence=float(yte.mean())))
                result_rows.append(m)
                for p, yt in zip(prob, yte):
                    pred_rows.append(dict(method=name, protocol="loso", fold=test_sid,
                                          label_frac=frac, calibrated="raw",
                                          prob=float(p), y_true=int(yt), demo=demo))


            if "deep_ssl" in methods and cfg.ablate_calibration_on_off and TORCH_OK and not demo:
                # basitçe test logit'inden T tahmini YAPMA (sızıntı); train-val ile yapılmalı.
                # Burada train'in bir kısmını val ayırıp T fit ediyoruz:
                val_idx = sel[: max(2, len(sel) // 5)]
                val_logit = M.predict_logits(clf1, Xtr_d[val_idx], cfg)
                T = cal.fit_temperature(val_logit, ytr[val_idx])
                prob_cal = cal.apply_temperature(logit_ssl, T)
                m = evaluate_probs(yte, prob_cal)
                m.update(dict(method="deep_ssl", protocol="loso", fold=test_sid,
                              label_frac=frac, calibrated=f"temp_T={T:.3f}", demo=demo,
                              n_test=len(yte), prevalence=float(yte.mean())))
                result_rows.append(m)



def run_site_shift(subjects, cfg, result_rows, pred_rows, demo):

    sids = [s for s in subjects if subjects[s]["chest"] is not None]
    if len(sids) < 2:
        return
    for direction in [("chest", "wrist"), ("wrist", "chest")]:
        src, dst = direction

        for test_sid in sids:
            tr = [s for s in sids if s != test_sid]
            Xtr = np.concatenate([subjects[s][src][1] for s in tr], axis=0)
            ytr = np.concatenate([subjects[s][src][2] for s in tr], axis=0)
            Xte, yte = subjects[test_sid][dst][1], subjects[test_sid][dst][2]
            if Xtr.shape[1] != Xte.shape[1]:
 )
                k = min(Xtr.shape[1], Xte.shape[1]); Xtr, Xte = Xtr[:, :k], Xte[:, :k]
            prob = bl.fit_predict_proba(bl.gb_baseline(cfg.seed), Xtr, ytr, Xte)
            m = evaluate_probs(yte, prob)
            m.update(dict(method="gb_feat", protocol=f"site_{src}2{dst}", fold=test_sid,
                          label_frac=1.0, calibrated="raw", demo=demo,
                          n_test=len(yte), prevalence=float(yte.mean())))
            result_rows.append(m)
            for p, yt in zip(prob, yte):
                pred_rows.append(dict(method="gb_feat", protocol=f"site_{src}2{dst}",
                                      fold=test_sid, label_frac=1.0, calibrated="raw",
                                      prob=float(p), y_true=int(yt), demo=demo))


# ----------------------------- ablasyon: LOSO vs random -----------------------------
def run_loso_vs_random(subjects, cfg, ablation_rows, demo):
    """DÜRÜST ablasyon: rastgele bölme, özne-dışıya kıyasla performansı şişirir mi?"""
    sids = sorted(subjects.keys())
    rng = np.random.default_rng(cfg.seed)
    Xf = np.concatenate([subjects[s]["wrist"][1] for s in sids], axis=0)
    y = np.concatenate([subjects[s]["wrist"][2] for s in sids], axis=0)
    grp = np.concatenate([[s] * len(subjects[s]["wrist"][2]) for s in sids])

    # random split
    idx = rng.permutation(len(y)); cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    p_rand = bl.fit_predict_proba(bl.gb_baseline(cfg.seed), Xf[tr], y[tr], Xf[te])
    auc_rand = roc_auc_score(y[te], p_rand) if len(np.unique(y[te])) > 1 else np.nan

    # LOSO (ortalama)
    aucs = []
    for test_sid in sids:
        trm = grp != test_sid; tem = grp == test_sid
        if len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        p = bl.fit_predict_proba(bl.gb_baseline(cfg.seed), Xf[trm], y[trm], Xf[tem])
        aucs.append(roc_auc_score(y[tem], p))
    auc_loso = float(np.mean(aucs)) if aucs else np.nan

    ablation_rows.append(dict(ablation="loso_vs_random", variant="random_split",
                              auc=float(auc_rand), demo=demo))
    ablation_rows.append(dict(ablation="loso_vs_random", variant="loso",
                              auc=auc_loso, demo=demo))



def paired_stats(pred_df, stats_rows, demo):
 
    key = ["protocol", "fold", "label_frac", "calibrated"]
    for keyvals, g in pred_df.groupby(key):
        g = g.sort_values("y_true") 
        methods = g["method"].unique()
 
        pivot = {}
        ytrue = None
        for mth in methods:
            sub = g[g["method"] == mth]
            pivot[mth] = sub["prob"].values
            ytrue = sub["y_true"].values
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a, b = methods[i], methods[j]
                if len(pivot[a]) != len(pivot[b]):
                    continue
                mc = st.mcnemar_test(ytrue, (pivot[a] >= 0.5).astype(int), (pivot[b] >= 0.5).astype(int))
                dl = st.delong_roc_test(ytrue, pivot[a], pivot[b])
                stats_rows.append(dict(
                    protocol=keyvals[0], fold=keyvals[1], label_frac=keyvals[2], calibrated=keyvals[3],
                    method_a=a, method_b=b,
                    mcnemar_p=mc["p_value"], mcnemar_method=mc["method"],
                    delong_auc_a=dl["auc1"], delong_auc_b=dl["auc2"], delong_p=dl["p_value"],
                    demo=demo,
                ))



def decision_curves(pred_df, cfg, dc_rows, demo):

    sel = pred_df[(pred_df.protocol == "loso") & (pred_df.label_frac == 1.0) & (pred_df.calibrated == "raw")]
    for mth, g in sel.groupby("method"):
        rows = dca.net_benefit_curve(g["y_true"].values, g["prob"].values, cfg.dca_thresholds)
        summ = dca.net_benefit_summary(rows)
        for r in rows:
            r.update(dict(method=mth, **summ, demo=demo))
            dc_rows.append(r)



def reliability(pred_df, cfg, rel_rows, demo):
    sel = pred_df[(pred_df.protocol == "loso") & (pred_df.label_frac == 1.0)]
    for (mth, calib), g in sel.groupby(["method", "calibrated"]):
        for row in cal.reliability_table(g["y_true"].values, g["prob"].values, cfg.ece_bins):
            row.update(dict(method=mth, calibrated=calib, demo=demo))
            rel_rows.append(row)



def main():
    cfg = CFG
    cfg.ensure_dirs()
    np.random.seed(cfg.seed)

    subjects, demo = load_all(cfg)
    print(f"[info] {'DEMO (SENTETİK — raporlanamaz)' if demo else 'WESAD'} | denek sayısı={len(subjects)}")

    pred_rows, result_rows, stats_rows, dc_rows, rel_rows, ablation_rows = [], [], [], [], [], []

    if cfg.run_loso:
        run_loso(subjects, cfg, pred_rows, result_rows, demo)
    if cfg.run_site_shift:
        run_site_shift(subjects, cfg, result_rows, pred_rows, demo)
    if cfg.ablate_loso_vs_random:
        run_loso_vs_random(subjects, cfg, ablation_rows, demo)

    pred_df = pd.DataFrame(pred_rows)
    if len(pred_df):
        paired_stats(pred_df, stats_rows, demo)
        decision_curves(pred_df, cfg, dc_rows, demo)
        reliability(pred_df, cfg, rel_rows, demo)


    od = cfg.out_dir
    pred_df.to_csv(os.path.join(od, "predictions.csv"), index=False)
    pd.DataFrame(result_rows).to_csv(os.path.join(od, "results_main.csv"), index=False)
    pd.DataFrame(stats_rows).to_csv(os.path.join(od, "stats_tests.csv"), index=False)
    pd.DataFrame(dc_rows).to_csv(os.path.join(od, "decision_curve.csv"), index=False)
    pd.DataFrame(rel_rows).to_csv(os.path.join(od, "reliability.csv"), index=False)
    pd.DataFrame(ablation_rows).to_csv(os.path.join(od, "ablation.csv"), index=False)

    return od


if __name__ == "__main__":
    main()
