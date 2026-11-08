"""200-sample bootstrap, design-based Rao-Scott test (same as main analysis), seven groups.
Respondents are resampled within strata (each respondent is its own PSU in BRFSS 2024)."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time, json
import csm_upgrade as C, fast_rs as F
G=["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg=C.CFG; K=len(G); B=int(sys.argv[1])
df=pd.read_csv("analysis_7g.csv",low_memory=False)
for a in cfg["attrs"]: df[a]=df[a].map(C.norm_code)
g=pd.Categorical(df.mm_group,categories=G).codes
cands=C.candidates(C.build_items(df,cfg["attrs"]),2); cache={}
M=np.column_stack([C.mask_of(df,c,cache) for c in cands]); keep=M.sum(0)>=30; M=M[:,keep]
lab=[C.pattern_label(c) for c,k in zip(cands,keep) if k]
w=df._LLCPWT.values.astype(float); st=df._STSTR.values
# check full-data decisions equal the main run
pi,sp,nh=F.design_arrays(st,df._PSU.values)
P,rs,p,de=F.rs1_batch(M,g,K,w,pi,sp,nh); sig=(C.holm(p)<.05)&(P.max(1)-P.min(1)>=.1)
e1=pd.read_csv("results7/E1_candidates_weighted.csv"); assert set(np.array(lab)[sig])==set(e1[e1.significant].label), "mismatch"
final=pd.read_csv("results7/final_patterns_weighted.csv"); fl=final.label.tolist(); fmax=dict(zip(final.label,final.max_group))
pos={l:i for i,l in enumerate(lab)}; fidx=[pos[l] for l in fl]
sampler=C.PSUBootstrap(df,cfg,True); rng=np.random.default_rng(2026)
found=np.zeros(len(fl)); same=np.zeros(len(fl)); t0=time.time()
for b in range(B):
    idx=sampler.draw(rng); Mb=M[idx]; sp,_=pd.factorize(st[idx]); nh=np.bincount(sp)
    ok=np.where(Mb.sum(0)>=30)[0]
    P,rs,p,de=F.rs1_batch(Mb[:,ok],g[idx],K,w[idx],np.arange(len(idx)),sp,nh)
    s=(C.holm(p)<.05)&(P.max(1)-P.min(1)>=.1)
    smap={ok[j]:G[int(P[j].argmax())] for j in np.where(s)[0]}
    for t,(l,i) in enumerate(zip(fl,fidx)):
        if i in smap: found[t]+=1; same[t]+=(smap[i]==fmax[l])
    if b%25==0: print(b,round(time.time()-t0),flush=True)
pd.DataFrame(dict(pattern=fl,rediscovery_rate=found/B,same_max_group_rate=same/B)).to_csv("results7/E5_bootstrap_stability_raoscott.csv",index=False)
print("done",time.time()-t0)
