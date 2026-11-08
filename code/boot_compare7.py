"""Diagnostic for M2: on the same bootstrap samples, compare decisions of the
unweighted Pearson test (old bootstrap) and the design-based Rao-Scott test (new)."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json
from scipy import stats
import csm_upgrade as C, fast_rs as F
cfg=C.CFG; B=int(sys.argv[1])
df=pd.read_csv("analysis_7g.csv",low_memory=False)
for a in cfg["attrs"]: df[a]=df[a].map(C.norm_code)
G=["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]; g=pd.Categorical(df.mm_group,categories=G).codes
cands=C.candidates(C.build_items(df,cfg["attrs"]),2); cache={}
M=np.column_stack([C.mask_of(df,c,cache) for c in cands]); kp=M.sum(0)>=30; M=M[:,kp]; lab=[C.pattern_label(c) for c,k in zip(cands,kp) if k]
w=df._LLCPWT.values.astype(float); st=df._STSTR.values
fl=set(pd.read_csv("results7/final_patterns_weighted.csv").label); fmask=np.array([l in fl for l in lab])
sampler=C.PSUBootstrap(df,cfg,True); rng=np.random.default_rng(2026)
res=[]
for b in range(B):
    idx=sampler.draw(rng); Mb=M[idx]; gb=g[idx]; wb=w[idx]
    sp,_=pd.factorize(st[idx]); nh=np.bincount(sp); ok=np.where(Mb.sum(0)>=30)[0]
    P,rs,p_rs,de=F.rs1_batch(Mb[:,ok],gb,7,wb,np.arange(len(idx)),sp,nh)
    # unweighted Pearson on counts
    nk=np.bincount(gb,minlength=7); X=np.zeros(len(ok))
    for j,c in enumerate(ok):
        x=np.bincount(gb[Mb[:,c]],minlength=7); tab=np.vstack([x,nk-x])
        X[j]=stats.chi2_contingency(tab,correction=False)[1] if (tab.sum(1)>0).all() else 1.0
    d=P.max(1)-P.min(1)
    s_rs=(C.holm(p_rs)<.05)&(d>=.1); s_pe=(C.holm(X)<.05)&(d>=.1)
    fin=fmask[ok]
    res.append(dict(rs=int(s_rs.sum()),pe=int(s_pe.sum()),pe_not_rs=int((s_pe&~s_rs).sum()),rs_not_pe=int((s_rs&~s_pe).sum()),
                    final_flips=int(((s_pe!=s_rs)&fin).sum()),delta_pass=int((d>=.1).sum()),
                    holm_rs=int((C.holm(p_rs)<.05).sum()),holm_pe=int((C.holm(X)<.05).sum())))
r=pd.DataFrame(res); print(r.describe().round(2)); json.dump(r.sum().to_dict()|{"B":B},open("results7/E5_bootstrap_test_comparison.json","w"),indent=1,default=int)
print(r.sum())
