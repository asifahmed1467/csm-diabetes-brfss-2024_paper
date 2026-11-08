"""Missing-data check by reweighting (Section III-C) and weight summary (Section V-E).
Complete-case adults are reweighted by 1 / P(complete | age group x sex x group), estimated among ALL adults with
diabetes. Then the gap D (max minus min share over the seven groups) of each of the 30 final patterns is recomputed.
Also writes the Kish design effect and effective sample sizes."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json
import csm_upgrade as C
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG
al = pd.read_csv("diabetes_all_7g.csv", low_memory=False); df = pd.read_csv("analysis_7g.csv", low_memory=False)
al["key"] = al._STSTR.astype(str) + "|" + al._PSU.astype(str); df["key"] = df._STSTR.astype(str) + "|" + df._PSU.astype(str)
assert al.key.is_unique and df.key.is_unique
al["kept"] = al.key.isin(df.key).astype(int)
cell = al.groupby([al.age_group.fillna(0), al._SEX.fillna(0), "mm_group"]).kept.transform("mean")
al["adj"] = al._LLCPWT / cell
df = df.merge(al[["key", "adj"]], on="key", how="left"); assert df.adj.notna().all()
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
g = pd.Categorical(df.mm_group, categories=G).codes
cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); cache = {}
f = pd.read_csv("results7/final_patterns_weighted.csv"); lab = {C.pattern_label(c): c for c in cands}
M = np.column_stack([C.mask_of(df, lab[l], cache) for l in f.label])
def sup(w): return np.array([[(w[g == k] @ M[g == k, j]) / w[g == k].sum() for k in range(7)] for j in range(M.shape[1])])
w0 = df._LLCPWT.values.astype(float); w1 = df.adj.values.astype(float); S0, S1 = sup(w0), sup(w1)
d0 = S0.max(1) - S0.min(1); d1 = S1.max(1) - S1.min(1)
print("max |paper delta - recomputed delta (original weights)|:", float(np.abs(d0 - f.delta.values).max()))
R = dict(max_abs_change_in_delta=float(np.abs(d1 - d0).max()), n_still_ge_010=int((d1 >= 0.10).sum()),
         same_top_group=int((S0.argmax(1) == S1.argmax(1)).sum()), same_bottom_group=int((S0.argmin(1) == S1.argmin(1)).sum()),
         weight_ratio_min=float((w1 / w0).min()), weight_ratio_max=float((w1 / w0).max()), n_patterns=len(f))
json.dump(R, open("results7/reweight_check.json", "w"), indent=1); print(R)
ne = {gn: round(float(w0[(df.mm_group == gn).values].sum() ** 2 / (w0[(df.mm_group == gn).values] ** 2).sum())) for gn in G}
st = df._STSTR.values; _, cnt = np.unique(st, return_counts=True)
W = dict(kish_deff=float(len(w0) * (w0 ** 2).sum() / w0.sum() ** 2), n_eff_all=float(w0.sum() ** 2 / (w0 ** 2).sum()),
         n_eff_by_group=ne, singleton_strata=int((cnt == 1).sum()))
json.dump(W, open("results7/weights_check.json", "w"), indent=1); print(W)
