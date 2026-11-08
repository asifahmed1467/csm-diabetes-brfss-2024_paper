"""Permutation (null) check. Shuffle the group labels among respondents inside each stratum, then rerun the
design-based test on all 814 candidates with the same filters (Holm p < 0.05 and gap >= 0.10).
Usage: python perm_test.py 100"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json, time
import csm_upgrade as C, fast_rs as F
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG; K = len(G); B = int(sys.argv[1])
df = pd.read_csv("analysis_7g.csv", low_memory=False)
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
g = pd.Categorical(df.mm_group, categories=G).codes
cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); cache = {}
M = np.column_stack([C.mask_of(df, c, cache) for c in cands]); M = M[:, M.sum(0) >= 30]
w = df._LLCPWT.values.astype(float); st = df._STSTR.values
pi, sp, nh = F.design_arrays(st, df._PSU.values)
P, rs, p, de = F.rs1_batch(M, g, K, w, pi, sp, nh)
real = int(((C.holm(p) < .05) & (P.max(1) - P.min(1) >= .1)).sum())
rng = np.random.default_rng(2026); strata_idx = [np.where(st == s)[0] for s in np.unique(st)]
out = []; t0 = time.time()
for b in range(B):
    gp = g.copy()
    for ix in strata_idx: gp[ix] = g[rng.permutation(ix)]
    P, rs, p, de = F.rs1_batch(M, gp, K, w, pi, sp, nh)
    pa = C.holm(p) < .05; ga = (P.max(1) - P.min(1)) >= .1
    out.append(dict(holm_pass=int(pa.sum()), gap_pass=int(ga.sum()), both=int((pa & ga).sum())))
    if b % 10 == 0: print(b, round(time.time() - t0), out[-1], flush=True)
r = pd.DataFrame(out)
R = dict(real_both=real, B=B, shuffles_with_any_pattern=int((r.both > 0).sum()), max_both=int(r.both.max()),
         mean_both=float(r.both.mean()), max_holm_pass=int(r.holm_pass.max()), mean_holm_pass=float(r.holm_pass.mean()),
         mean_gap_pass=float(r.gap_pass.mean()))
json.dump(R, open("results7/perm_test.json", "w"), indent=1); r.to_csv("results7/perm_test_runs.csv", index=False); print(R)
