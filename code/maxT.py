"""Permutation max-statistic test (Section V-E, 'Null check', and the Perm. column of Table VI).
1. Compute the Rao-Scott chi-square statistic of all 814 candidates (count >= 30) on the real data.
2. Shuffle the group labels inside each stratum, recompute all 814 statistics, keep the LARGEST one per shuffle.
3. A final pattern passes if its real statistic is above the 95th percentile of the 100 shuffle maxima.
Also records the share of null p-values below 0.05 and 0.001 over all shuffles.
Usage: python maxT.py 100        (about 10 s per shuffle on one core)"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json
import csm_upgrade as C, fast_rs as F
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG; K = len(G); B = int(sys.argv[1])
df = pd.read_csv("analysis_7g.csv", low_memory=False)
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
g = pd.Categorical(df.mm_group, categories=G).codes
cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); cache = {}
M = np.column_stack([C.mask_of(df, c, cache) for c in cands]); keep = M.sum(0) >= 30
M = M[:, keep]; cands = [c for c, k in zip(cands, keep) if k]
w = df._LLCPWT.values.astype(float); st = df._STSTR.values
pi, sp, nh = F.design_arrays(st, df._PSU.values)
P, rs, p, de = F.rs1_batch(M, g, K, w, pi, sp, nh)            # real data
labels = [C.pattern_label(c) for c in cands]; pos = {l: i for i, l in enumerate(labels)}
fin = pd.read_csv("results7/final_patterns_weighted.csv")
fs = np.array([rs[pos[l]] for l in fin.label])                  # real statistic of each final pattern
rng = np.random.default_rng(2026); sidx = [np.where(st == s)[0] for s in np.unique(st)]
mx, f05, f001 = [], [], []
for b in range(B):
    gp = g.copy()
    for ix in sidx: gp[ix] = g[rng.permutation(ix)]
    P2, rs2, p2, _ = F.rs1_batch(M, gp, K, w, pi, sp, nh)
    mx.append(rs2.max()); f05.append(float((p2 < 0.05).mean())); f001.append(float((p2 < 0.001).mean()))
    if b % 10 == 0: print(b, round(mx[-1], 1), flush=True)
mx = np.array(mx); crit = float(np.percentile(mx, 95))
fin["stat"] = fs; fin["p_adj_perm"] = [(1 + (mx >= x).sum()) / (1 + len(mx)) for x in fs]; fin["pass_perm"] = fs > crit
fin["n_conditions_in_pattern"] = fin.label.str.count(" AND ") + 1
fin[["label", "count", "delta", "stat", "p_adj_perm", "pass_perm"]].to_csv("results7/maxT_final30.csv", index=False)
np.save("results7/maxT_null_max.npy", mx)
R = dict(B=B, real_candidates=int(len(rs)), frac_p_lt_05=float(np.mean(f05)), frac_p_lt_001=float(np.mean(f001)),
         crit95=crit, final30_above_crit=int((fs > crit).sum()),
         single_pass=int(fin.pass_perm[fin.n_conditions_in_pattern == 1].sum()), n_single=int((fin.n_conditions_in_pattern == 1).sum()),
         pairs_pass=int(fin.pass_perm[fin.n_conditions_in_pattern == 2].sum()), n_pairs=int((fin.n_conditions_in_pattern == 2).sum()))
print(R); json.dump(R, open("results7/maxT.json", "w"), indent=1)
