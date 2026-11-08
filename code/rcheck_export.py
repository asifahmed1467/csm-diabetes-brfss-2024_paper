"""Export data for the R cross-check (R 'survey' package, svychisq). Selected: the 30 final patterns and 50 random other
candidates (seed 5), each with the REAL group labels and one within-strata shuffle (seed 5) of the labels.
Also writes the adults removed for missing data (needed for the full-file domain design in rcheck_full.R)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import csm_upgrade as C, fast_rs as F
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG; K = 7
df = pd.read_csv("analysis_7g.csv", low_memory=False)
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
g = pd.Categorical(df.mm_group, categories=G).codes
cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); cache = {}
M = np.column_stack([C.mask_of(df, c, cache) for c in cands]); keep = M.sum(0) >= 30; M = M[:, keep]; cands = [c for c, k in zip(cands, keep) if k]
w = df._LLCPWT.values.astype(float); st = df._STSTR.values; pi, sp, nh = F.design_arrays(st, df._PSU.values)
P, rs, p, de = F.rs1_batch(M, g, K, w, pi, sp, nh)
pos = {C.pattern_label(c): i for i, c in enumerate(cands)}
fi = [pos[l] for l in pd.read_csv("results7/final_patterns_weighted.csv").label]
rng = np.random.default_rng(5); others = [int(i) for i in rng.choice([i for i in range(M.shape[1]) if i not in set(fi)], 150, replace=False)][:50]   # 150 drawn, first 50 used
idx = fi + others
sidx = [np.where(st == s)[0] for s in np.unique(st)]; gp = g.copy()
for ix in sidx: gp[ix] = g[rng.permutation(ix)]
P2, rs2, p2, _ = F.rs1_batch(M, gp, K, w, pi, sp, nh)
out = pd.DataFrame({"w": w, "st": st, "g": g, "gp": gp})
for n, j in enumerate(idx): out[f"x{n}"] = M[:, j].astype(int)
out.to_csv("rcheck_data.csv", index=False)
pd.DataFrame({"n": range(len(idx)), "is_final": [1] * 30 + [0] * 50, "py_p_real": p[idx], "py_p_perm": p2[idx]}).to_csv("rcheck_py.csv", index=False)
al = pd.read_csv("diabetes_all_7g.csv", low_memory=False)
al["key"] = al._STSTR.astype(str) + "|" + al._PSU.astype(str); df["key"] = df._STSTR.astype(str) + "|" + df._PSU.astype(str)
al[~al.key.isin(df.key)][["_LLCPWT", "_STSTR"]].rename(columns={"_LLCPWT": "w", "_STSTR": "st"}).to_csv("rcheck_dropped.csv", index=False)
print("exported", len(idx), "patterns,", len(out), "adults,", int((~al.key.isin(df.key)).sum()), "removed adults")
