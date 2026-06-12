
import os, glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# ---------- CSV bulucu ----------
ROOTS = ["/kaggle/input", "/tmp/csvs", "."]
def find(name):
    for r in ROOTS:
        hits = glob.glob(os.path.join(r, "**", name), recursive=True)
        if hits: return hits[0]
    return None
def load(name):
    p = find(name)
    return (pd.read_csv(p), p) if p else (None, None)

# ===================== stats_tests.py (birebir) =====================
def mcnemar_test(y_true, pred1, pred2):
    y_true=np.asarray(y_true).astype(int); pred1=np.asarray(pred1).astype(int); pred2=np.asarray(pred2).astype(int)
    c1=(pred1==y_true); c2=(pred2==y_true)
    b=int(np.sum(c1&~c2)); c=int(np.sum(~c1&c2)); n=b+c
    if n==0: return dict(b=b,c=c,n_discordant=0,statistic=0.0,p_value=1.0,method="degenerate")
    if n<25:
        p=float(stats.binomtest(min(b,c),n,0.5,alternative="two-sided").pvalue)
        return dict(b=b,c=c,n_discordant=n,statistic=float(min(b,c)),p_value=p,method="exact_binomial")
    stat=(abs(b-c)-1)**2/n
    return dict(b=b,c=c,n_discordant=n,statistic=float(stat),p_value=float(stats.chi2.sf(stat,df=1)),method="chi2_continuity")

def _compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N); T2[J]=T; return T2

def _fast_delong(preds_sorted,m):
    k=preds_sorted.shape[0]; n=preds_sorted.shape[1]-m
    pos=preds_sorted[:,:m]; neg=preds_sorted[:,m:]
    tx=np.empty([k,m]); ty=np.empty([k,n]); tz=np.empty([k,m+n])
    for r in range(k):
        tx[r,:]=_compute_midrank(pos[r,:]); ty[r,:]=_compute_midrank(neg[r,:]); tz[r,:]=_compute_midrank(preds_sorted[r,:])
    aucs=tz[:,:m].sum(axis=1)/m/n-(m+1.0)/2.0/n
    v01=(tz[:,:m]-tx[:,:])/n; v10=1.0-(tz[:,m:]-ty[:,:])/m
    sx=np.cov(v01); sy=np.cov(v10); cov=np.atleast_2d(sx/m+sy/n)
    return aucs,cov

def delong_auc_variance(y_true,prob):
    y_true=np.asarray(y_true).astype(int); prob=np.asarray(prob,float)
    order=np.argsort(-y_true,kind="mergesort"); m=int(y_true.sum())
    aucs,cov=_fast_delong(prob[order][None,:],m)
    auc=float(aucs[0]); se=np.sqrt(max(float(cov[0,0]),0.0))
    return dict(auc=auc,ci_low=float(max(0,auc-1.96*se)),ci_high=float(min(1,auc+1.96*se)))

def delong_roc_test(y_true,prob1,prob2):
    y_true=np.asarray(y_true).astype(int); prob1=np.asarray(prob1,float); prob2=np.asarray(prob2,float)
    order=np.argsort(-y_true,kind="mergesort"); m=int(y_true.sum())
    preds=np.vstack((prob1,prob2))[:,order]; aucs,cov=_fast_delong(preds,m)
    var=cov[0,0]+cov[1,1]-2*cov[0,1]
    if var<=0: z,p=0.0,1.0
    else: z=(aucs[0]-aucs[1])/np.sqrt(var); p=float(2*stats.norm.sf(abs(z)))
    return dict(auc1=float(aucs[0]),auc2=float(aucs[1]),z=float(z),p_value=p)

def spiegelhalter_z(y_true,prob):
    y=np.asarray(y_true,float); p=np.clip(np.asarray(prob,float),1e-12,1-1e-12)
    num=np.sum((y-p)*(1-2*p)); den=np.sqrt(np.sum((1-2*p)**2*p*(1-p)))
    if den==0: return dict(z=0.0,p_value=1.0)
    z=num/den; return dict(z=float(z),p_value=float(2*stats.norm.sf(abs(z))))

# ===================== calibration.py (birebir) =====================
def _logit(p,eps=1e-7): p=np.clip(p,eps,1-eps); return np.log(p/(1-p))
def brier(y,p): return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))
def ece(y,p,nb):
    y=np.asarray(y,float); p=np.asarray(p,float); b=np.linspace(0,1,nb+1); e=0.0; n=len(p)
    for i in range(nb):
        m=(p>b[i])&(p<=b[i+1]) if i>0 else (p>=b[i])&(p<=b[i+1])
        if m.sum(): e+=m.sum()/n*abs(y[m].mean()-p[m].mean())
    return float(e)
def slope_intercept(y,p):
    y=np.asarray(y,int); x=_logit(np.asarray(p,float)).reshape(-1,1)
    if len(np.unique(y))<2: return (np.nan,np.nan)
    lr=LogisticRegression(C=1e12,solver="lbfgs",max_iter=1000).fit(x,y)
    return float(lr.coef_[0,0]),float(lr.intercept_[0])

# ===================== yardimcilar =====================
def per_fold_auc(df):
    a={}
    for f,g in df.groupby("fold"):
        if g.y_true.nunique()>1: a[f]=roc_auc_score(g.y_true,g.prob)
    return a
def disc_block(name,df):
    pa=per_fold_auc(df); vals=np.array(list(pa.values()))
    civ=delong_auc_variance(df.y_true.values,df.prob.values)
    nS=len(vals); sd=float(vals.std(ddof=1)) if nS>1 else float("nan")
    if nS>1:
        half=float(stats.t.ppf(0.975,nS-1))*sd/np.sqrt(nS)
        lo,hi=max(0.0,vals.mean()-half),min(1.0,vals.mean()+half)
    else:
        lo=hi=float("nan")
    print(f"  {name:16s} pooled AUROC={civ['auc']:.3f} (DeLong 95% CI {civ['ci_low']:.3f}-{civ['ci_high']:.3f})")
    print(f"  {'':16s} per-subject mean={vals.mean():.3f} (SD {sd:.3f}; subject-level 95% CI {lo:.3f}-{hi:.3f}) "
          f"| folds={nS} range={vals.min():.3f}-{vals.max():.3f} | prev={df.y_true.mean():.3f}")
def calib_block(name,df,):
    y,p=df.y_true.values,df.prob.values
    sl,ic=slope_intercept(y,p); sp=spiegelhalter_z(y,p)
    # per-subject ECE/Brier (M=10)
    es,bs=[],[]
    for f,g in df.groupby("fold"):
        if g.y_true.nunique()>1: es.append(ece(g.y_true.values,g.prob.values,10)); bs.append(brier(g.y_true.values,g.prob.values))
    es,bs=np.array(es),np.array(bs)
    print(f"  {name:16s} pooled: ECE@10={ece(y,p,10):.3f} ECE@15={ece(y,p,15):.3f} Brier={brier(y,p):.3f} "
          f"slope={sl:.3f} intercept={ic:.3f} Spiegelhalter z={sp['z']:.2f} (p={sp['p_value']:.2e})")
    print(f"  {'':16s} per-subject: meanECE@10={es.mean():.3f} (SD {es.std():.3f}) meanBrier={bs.mean():.3f} | folds={len(es)}")

# ===================== ANA =====================
def main():
    print("="*78,"\nDISCRIMINATION + DeLong CI\n"+"="*78)
    w,_=load("feat_loso_predictions.csv");  n,_=load("nurse_deep_pred.csv");  e,_=load("exstress_loso_pred.csv")
    if w is not None: disc_block("WESAD (GB)",w)
    ngb=None
    if n is not None:
        ngb=n[(n.method=="gb")&(n.label_frac==1.0)].copy(); disc_block("Nurse (GB)",ngb)
    if e is not None: disc_block("Exercise (GB)",e)

    if n is not None:
        print("\n"+"="*78,"\nDEEP vs BASELINE (Nurse, frac=1.0): DeLong + McNemar\n"+"="*78)
        d=n[n.label_frac==1.0]
        def aligned(m): return d[d.method==m].sort_values("fold", kind="stable").reset_index(drop=True)
        gb,ssl,sc=aligned("gb"),aligned("deep_ssl"),aligned("deep_scratch")
        ok = len(gb)==len(ssl)==len(sc) and (gb.y_true.values==ssl.y_true.values).all() and (gb.y_true.values==sc.y_true.values).all()
        print("  hizalama (ayni pencere sirasi):", "OK" if ok else "UYARI - y_true eslesmedi")
        y=gb.y_true.values
        for lab,other in (("gb vs ssl",ssl),("gb vs scratch",sc)):
            dl=delong_roc_test(y,gb.prob.values,other.prob.values)
            mc=mcnemar_test(y,(gb.prob.values>=.5).astype(int),(other.prob.values>=.5).astype(int))
            print(f"  {lab:14s} DeLong: AUC {dl['auc1']:.3f} vs {dl['auc2']:.3f} dz={dl['z']:.2f} p={dl['p_value']:.2e} | "
                  f"McNemar b={mc['b']} c={mc['c']} p={mc['p_value']:.2e} ({mc['method']})")
        # per-fold gb>ssl sayisi
        pf_gb=per_fold_auc(gb); pf_ssl=per_fold_auc(ssl)
        wins=sum(1 for f in pf_gb if f in pf_ssl and pf_gb[f]>pf_ssl[f])
        print(f"  per-fold: GB>SSL {wins}/{len(set(pf_gb)&set(pf_ssl))} fold")

    print("\n"+"="*78,"\nCALIBRATION (global, GB)\n"+"="*78)
    if w is not None: calib_block("WESAD",w)
    if ngb is not None: calib_block("Nurse",ngb)
    if e is not None: calib_block("Exercise",e)

    print("\n"+"="*78,"\nFEW-SHOT ECE-vs-k\n"+"="*78)
    for nm,fn in (("Nurse","fewshot_calib_curve.csv"),("Exercise","exstress_fewshot_calib.csv")):
        c,_=load(fn)
        if c is not None:
            print(f"  {nm}:"); print(c.to_string(index=False))
            try:
                s0=float(c[c.k==0].ece_std.iloc[0]); s20=float(c[c.k==20].ece_std.iloc[0])
                print(f"    between-subject SD: k=0 -> {s0:.3f} | k=20 -> {s20:.3f}")
            except Exception: pass

    print("\n"+"="*78,"\nDECISION CURVE (Nurse)\n"+"="*78)
    dca,_=load("nurse_dca_raw_vs_recal.csv")
    if dca is not None:
        th=dca.threshold.values; ref=np.maximum(dca.nb_all.values,0.0)
        fr=lambda c: float((dca[c].values>ref).mean()); ex=lambda c: float(np.mean(dca[c].values-ref))
        print(f"  threshold grid: [{th.min():.3f}, {th.max():.3f}], n={len(th)}")
        print(f"  superior-fraction: raw={fr('nb_raw'):.3f}  recal={fr('nb_recal'):.3f}")
        print(f"  mean excess NB:    raw={ex('nb_raw'):+.4f}  recal={ex('nb_recal'):+.4f}")
    print("\n[done]")

if __name__=="__main__":
    main()
