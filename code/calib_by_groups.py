"""How the over-liberal null rate depends on the number and size of groups (Section V-E, 'Null check').
Shuffle group labels inside strata (seed 11) and report the share of design-based p-values below 0.05.
Runs: all 7 groups (3 shuffles), 5 groups without Diab.+CKD and Diab.+COPD (4 shuffles), the 3 largest groups (4 shuffles)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json
import csm_upgrade as C, fast_rs as F
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG
df = pd.read_csv("analysis_7g.csv", low_memory=False)
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); cache = {}
M0 = np.column_stack([C.mask_of(df, c, cache) for c in cands])
def run(groups, B, name):
    keep = df.mm_group.isin(groups).values; d = df[keep].reset_index(drop=True); M = M0[keep]; M = M[:, M.sum(0) >= 30]
    g = pd.Categorical(d.mm_group, categories=groups).codes; K = len(groups)
    w = d._LLCPWT.values.astype(float); st = d._STSTR.values; pi, sp, nh = F.design_arrays(st, d._PSU.values)
    rng = np.random.default_rng(11); sidx = [np.where(st == s)[0] for s in np.unique(st)]; ps = []
    for b in range(B):
        gp = g.copy()
        for ix in sidx: gp[ix] = g[rng.permutation(ix)]
        ps.append(F.rs1_batch(M, gp, K, w, pi, sp, nh)[2])
    p = np.concatenate(ps); r = dict(groups=K, shuffles=B, patterns=int(M.shape[1]), rate_p_lt_005=float((p < .05).mean()))
    print(name, r, flush=True); return r
R = {"all_7_groups": run(G, 3, "all 7"),
     "5_groups_drop_CKD_COPD": run([x for x in G if x not in ("diabetes_ckd", "diabetes_copd")], 4, "5 groups"),
     "3_largest_groups": run(["diabetes_only", "diabetes_arthritis", "diabetes_multi_2plus"], 4, "3 largest")}
json.dump(R, open("results7/calibration_by_groups.json", "w"), indent=1)
