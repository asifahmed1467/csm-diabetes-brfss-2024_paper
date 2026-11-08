"""Times the candidate build and the 814 design-based tests on the full data."""
import sys, time, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import csm_upgrade as C, fast_rs as F
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG
df = pd.read_csv("analysis_7g.csv", low_memory=False)
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
g = pd.Categorical(df.mm_group, categories=G).codes
t0 = time.time(); cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); cache = {}
M = np.column_stack([C.mask_of(df, c, cache) for c in cands]); M = M[:, M.sum(0) >= 30]; t1 = time.time()
pi, sp, nh = F.design_arrays(df._STSTR.values, df._PSU.values)
F.rs1_batch(M, g, len(G), df._LLCPWT.values.astype(float), pi, sp, nh); t2 = time.time()
R = {"n_tests": int(M.shape[1]), "candidate_build_s": round(t1 - t0, 1), "design_tests_s": round(t2 - t1, 1)}
json.dump(R, open("results7/runtime_this_pc.json", "w"), indent=1); print(R)
