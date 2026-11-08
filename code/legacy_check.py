"""Missing-code check (Data section): keep BRFSS "don't know/refused" codes for smoking and age as
their own categories (legacy file) and compare the gap of the six-group single traits with the
corrected file. Same logic as the original side analysis. Unweighted, six groups."""
import sys, json; sys.path.insert(0, ".")
import pandas as pd
import csm_upgrade as C
cfg = C.CFG                                   # default config = the six groups
L = pd.read_csv("analysis_legacy.csv", low_memory=False); K = pd.read_csv("analysis_corrected.csv", low_memory=False)
for df in (L, K):
    for x in cfg["attrs"]: df[x] = df[x].map(C.norm_code)
rl = C.mine(L, cfg); rk = C.mine(K, cfg)
fl = rl[(rl.length == 1) & rl.significant][["label", "delta", "max_group"]]
cmp = fl.join(rk.set_index("label")[["delta", "max_group", "significant"]], on="label", rsuffix="_corr")
cmp.to_csv("results7/legacy_vs_corrected.csv", index=False)
R = {"n_single_traits": len(cmp), "max_abs_gap_change": float((cmp.delta - cmp.delta_corr).abs().max())}
json.dump(R, open("results7/legacy_vs_corrected.json", "w"), indent=1); print(cmp.round(4).to_string()); print(R)
