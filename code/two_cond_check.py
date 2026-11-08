"""Check of the 'Diab.+2 or more' group. For every final pattern whose characteristic group is 'Diab.+2 or more',
compare adults with exactly 2 extra conditions, and adults with 3 or more, with all adults in the other six groups.
D = share in the subgroup minus share in the other six groups; 95% CI by Taylor linearization (each respondent is its
own sampling unit within a stratum)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, json
import csm_upgrade as C
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G); cfg = C.CFG
df = pd.read_csv("analysis_7g.csv", low_memory=False)
for a in cfg["attrs"]: df[a] = df[a].map(C.norm_code)
f = pd.read_csv("results7/final_patterns_weighted.csv"); b = pd.read_csv("results7/E2_best_group_per_pattern.csv").set_index("pattern")
pats = [l for l in f.label if b.loc[l, "group"] == "diabetes_multi_2plus"]
cache = {}; w = df._LLCPWT.values.astype(float); st = df._STSTR.values
rest = (df.mm_group != "diabetes_multi_2plus").values; multi = (df.mm_group == "diabetes_multi_2plus").values
def d_ci(y, dom):
    Wg, Wr = w[dom].sum(), w[rest].sum(); sg = (w[dom] * y[dom]).sum() / Wg; sr = (w[rest] * y[rest]).sum() / Wr
    z = np.zeros(len(w)); z[dom] = w[dom] * (y[dom] - sg) / Wg; z[rest] = -w[rest] * (y[rest] - sr) / Wr
    var = 0.0
    for s in np.unique(st):
        zz = z[st == s]; n = len(zz)
        if n > 1: var += n / (n - 1) * ((zz - zz.mean()) ** 2).sum()
    d = sg - sr; se = var ** .5; return sg, sr, d, d - 1.96 * se, d + 1.96 * se
rows = []
cands = C.candidates(C.build_items(df, cfg["attrs"]), 2); lab = {C.pattern_label(c): c for c in cands}
for l in pats:
    y = C.mask_of(df, lab[l], cache).astype(float)
    for name, dom in [("all 2+", multi), ("exactly 2", multi & (df.n_conditions == 2).values), ("3 or more", multi & (df.n_conditions >= 3).values)]:
        sg, sr, d, lo, hi = d_ci(y, dom); rows.append(dict(pattern=l, subgroup=name, n=int(dom.sum()), share=sg, rest=sr, D=d, lo=lo, hi=hi))
r = pd.DataFrame(rows); r.to_csv("results7/two_cond_check.csv", index=False)
main = r[r.subgroup == "all 2+"].set_index("pattern"); two = r[r.subgroup == "exactly 2"].set_index("pattern"); three = r[r.subgroup == "3 or more"].set_index("pattern")
same_sign = (np.sign(two.D) == np.sign(main.D)); excl = ((two.lo > 0) | (two.hi < 0))
R = dict(n_patterns=len(pats), n_two=int((multi & (df.n_conditions == 2).values).sum()), n_three_plus=int((multi & (df.n_conditions >= 3).values).sum()),
         two_same_sign=int(same_sign.sum()), two_ci_excludes_0=int(excl.sum()), two_both=int((same_sign & excl).sum()),
         three_same_sign=int((np.sign(three.D) == np.sign(main.D)).sum()),
         min_ratio_two_over_all=float((two.D / main.D).min()), max_ratio_two_over_all=float((two.D / main.D).max()))
json.dump(R, open("results7/two_cond_check.json", "w"), indent=1); print(r.round(3).to_string()); print(R)
