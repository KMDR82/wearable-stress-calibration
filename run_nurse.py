
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

import nurse_deep as nd
import models as M


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _ece(y, p, nb=10):
    b = np.linspace(0, 1, nb + 1); e = 0.0; n = len(p)
    for i in range(nb):
        m = (p > b[i]) & (p <= b[i + 1]) if i > 0 else (p >= b[i]) & (p <= b[i + 1])
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def main(csv_path, cfg, out_dir="/kaggle/working", label_fractions=(1.0,), seed=1337):
    os.makedirs(out_dir, exist_ok=True)
    deep, feat, ys = nd.build_nurse_windows(csv_path)
    sids = list(deep.keys())
    print("özne:", len(sids), "toplam pencere:", sum(len(ys[s]) for s in sids), flush=True)
    in_ch = deep[sids[0]].shape[1]


    pool = np.concatenate([deep[s] for s in sids], 0)
    ckpt = os.path.join(out_dir, "ssl_nurse.pt")
    enc_ssl = M.pretrain_ssl(pool, cfg, in_ch, ckpt)
    print("SSL ön-eğitim bitti", flush=True)

    valid = [s for s in sids if len(np.unique(ys[s])) > 1]  # iki sınıflı test fold'ları
    print("geçerli LOSO fold:", len(valid), flush=True)
    rng = np.random.default_rng(seed)
    preds = []

    for test in valid:
        tr = [s for s in sids if s != test]
        Xd = np.concatenate([deep[s] for s in tr]); yd = np.concatenate([ys[s] for s in tr])
        Xf = np.concatenate([feat[s] for s in tr])
        Xte_d, Xte_f, yte = deep[test], feat[test], ys[test]

        for frac in label_fractions:
            if frac < 1.0:  # sınıf-dengeli az-etiket
                idx = []
                for c in (0, 1):
                    ci = np.where(yd == c)[0]
                    idx += list(rng.choice(ci, max(1, int(len(ci) * frac)), replace=False))
                idx = np.array(sorted(idx))
            else:
                idx = np.arange(len(yd))


            gb = GradientBoostingClassifier(random_state=seed).fit(Xf[idx], yd[idx])
            pgb = gb.predict_proba(Xte_f)[:, 1]

            enc0 = M.build_encoder(in_ch, cfg)
            clf0 = M.train_classifier(enc0, Xd[idx], yd[idx], cfg)
            psc = _sig(M.predict_logits(clf0, Xte_d, cfg))            enc1 = M.build_encoder(in_ch, cfg); enc1.load_state_dict(enc_ssl.state_dict())
            clf1 = M.train_classifier(enc1, Xd[idx], yd[idx], cfg)
            pssl = _sig(M.predict_logits(clf1, Xte_d, cfg))

            for name, p in (("gb", pgb), ("deep_scratch", psc), ("deep_ssl", pssl)):
                for pi, yi in zip(p, yte):
                    preds.append(dict(method=name, fold=test, label_frac=frac,
                                      prob=float(pi), y_true=int(yi)))
            print(f"{test} f={frac}: GB={roc_auc_score(yte,pgb):.3f} "
                  f"scratch={roc_auc_score(yte,psc):.3f} ssl={roc_auc_score(yte,pssl):.3f}", flush=True)

    P = pd.DataFrame(preds)
    P.to_csv(os.path.join(out_dir, "nurse_deep_pred.csv"), index=False)
    print("\n=== HAVUZ ÖZET (frac=1.0) ===", flush=True)
    for name in ("gb", "deep_scratch", "deep_ssl"):
        g = P[(P.method == name) & (P.label_frac == 1.0)]
        if len(g):
            y, p = g.y_true.values, g.prob.values
            print(f"[{name}] AUC={roc_auc_score(y,p):.3f} ECE={_ece(y,p):.3f} "
                  f"Brier={brier_score_loss(y,p):.3f}", flush=True)
    return P
