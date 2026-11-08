"""Rebuilds every table of the paper and every number quoted in its text, from the result files."""
import json, numpy as np, pandas as pd
R = "results7/"
G = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
GN = {"diabetes_only":"Diabetes only","diabetes_cvd":"Diab.+CVD","diabetes_ckd":"Diab.+CKD","diabetes_arthritis":"Diab.+arthritis",
      "diabetes_depression":"Diab.+depression","diabetes_copd":"Diab.+COPD","diabetes_multi_2plus":"Diab.+2 or more"}
GS = {"diabetes_only":"Only","diabetes_cvd":"CVD","diabetes_ckd":"CKD","diabetes_arthritis":"Arth.","diabetes_depression":"Dep.","diabetes_copd":"COPD","diabetes_multi_2plus":"2+"}
s = json.load(open(R+"summary.json")); cnt = json.load(open(R+"counts_7groups.json"))
f = pd.read_csv(R+"final_patterns_weighted.csv"); fi = f.set_index("label")
b = pd.read_csv(R+"E2_best_group_per_pattern.csv").set_index("pattern")
o = pd.read_csv(R+"E2_one_vs_rest_weighted.csv").set_index(["pattern","group"])
bs = pd.read_csv(R+"E5_bootstrap_stability_raoscott.csv").set_index("pattern")
ALL = pd.read_csv("diabetes_all_7g.csv", usecols=["mm_group"]).mm_group.value_counts()

def tables():
    T = {}
    g = pd.read_csv(R+"E0_group_sizes.csv").set_index("mm_group")
    t1 = pd.DataFrame({"Group":[GN[k] for k in G], "All":[int(ALL[k]) for k in G], "Analysis":[int(g.loc[k,"n"]) for k in G],
                       "Weighted %":[round(float(g.loc[k,"pct_weighted"]),1) for k in G]})
    t1.loc[len(t1)] = ["Total", int(ALL.sum()), int(g.n.sum()), round(float(g.pct_weighted.sum()),1)]
    T["Table I: group sizes"] = t1
    T["Table II: settings"] = pd.DataFrame({"Setting":["Records in raw file","Analysis set","Strata","Sampling units","Min count","delta","Holm alpha","R2 gap","Bootstrap","Seed"],
        "Value":["see cell 2 output", s["n_analysis"], s["n_strata"], f'{s["n_psu"]} (one per respondent)', 30, 0.10, 0.05, 0.02, "200 within strata", 2026]})
    c = pd.read_csv(R+"E7_cramers_v.csv")
    T["Table III: Cramer's V"] = pd.DataFrame({"Trait":c.attribute,"Lev.":c.levels,"df*":c.df_star,"V":c.V_unweighted.round(3),
        "V_w":c.V_weighted.round(3),"Size":c.label_weighted,"Deff":c.mean_deff.round(1)})
    pu = pd.read_csv(R+"E3_pruning_unweighted.csv"); pw = pd.read_csv(R+"E3_pruning_weighted.csv")
    T["Table IV: funnel"] = pd.DataFrame({"Step":["Candidates with count >= 30","Holm p<0.05","Holm p<0.05 and Delta>=0.10","After R1 and R2","  of which pairs"],
        "Unweighted":[s["n_candidates_after_min_count"]["1-item"]+s["n_candidates_after_min_count"]["2-item"], cnt["unweighted_holm_pass"],
                      s["n_significant_unweighted"], int(pu.kept.sum()), int((pu.kept & (pu.length==2)).sum())],
        "Weighted":[s["n_candidates_after_min_count"]["1-item"]+s["n_candidates_after_min_count"]["2-item"], cnt["weighted_holm_pass"],
                    s["n_significant_weighted"], int(pw.kept.sum()), int((pw.kept & (pw.length==2)).sum())]})
    e8 = pd.read_csv(R+"E8_conditions_by_group.csv").set_index("condition")[G].round(0).astype(int)
    e8.columns = [GS[x] for x in G]; T["Table V: conditions by group (weighted %)"] = e8
    F = lambda x, d: x.map(lambda v: f"{v:.{d}f}")          # same rounding as the paper
    t6 = pd.DataFrame({"Delta":F(fi.delta,3), "Highest":fi.max_group.map(GS), "Lowest":fi.min_group.map(GS),
        "Char.":b.group.map(GS), "D":F(b["diff"],3), "D lo":F(b.diff_lo,3), "D hi":F(b.diff_hi,3),
        "RR":F(b.rr,2), "Found":F(bs.rediscovery_rate,2), "Same":F(bs.same_max_group_rate,2)})
    t6["below 70%"] = np.where(bs.rediscovery_rate.reindex(t6.index) < .7, "dagger", "")
    t6 = t6.reindex(b["diff"].abs().sort_values(ascending=False).index)
    T["Table VI: all 30 final patterns"] = t6
    se = pd.read_csv(R+"E4_sensitivity.csv")
    se["cell"] = se.apply(lambda r: f"{int(r.n_final)} ({int(r.n_final_pairs)}) / {r.jaccard_vs_base:.2f}", axis=1)
    T["Table VII: threshold grid"] = se.pivot(index="delta", columns="min_count", values="cell")
    return T

def text_numbers():
    N = []; add = lambda sec, what, val: N.append((sec, what, val))
    n_all = int(ALL.sum())
    add("Data", "adults with diabetes", n_all)
    import os
    if os.path.exists("results6/E8_conditions_by_group.csv"):
        add("Data", "depression share of the old six-group 'other single' group (%)", round(float(pd.read_csv("results6/E8_conditions_by_group.csv").set_index("condition").loc["Depressive disorder","diabetes_other_single"]),1))
    add("Data", "analysis set", s["n_analysis"]); add("Data", "removed", s["n_dropped"]); add("Data", "removed (%)", round(100*s["n_dropped"]/n_all,1))
    ms = pd.read_csv(R+"E0_missing_share.csv").set_index("index").share_missing
    add("Data", "income missing (%)", round(100*float(ms["income_group"]),1))
    dd = pd.read_csv(R+"E0_dropout_check.csv"); dd["level"] = dd.level.astype(str)
    fem = dd[(dd.variable=="_SEX")&(dd.level.isin(["2","2.0"]))].iloc[0]; a80 = dd[(dd.variable=="age_group")&(dd.level.isin(["13","13.0"]))].iloc[0]
    add("Data", "female: removed vs kept (%)", f"{fem.pct_dropped:.1f} vs {fem.pct_kept:.1f}")
    add("Data", "aged 80+: removed vs kept (%)", f"{a80.pct_dropped:.1f} vs {a80.pct_kept:.1f}")
    add("Data", "missing-code check: max change in gap", round(json.load(open(R+"legacy_vs_corrected.json"))["max_abs_gap_change"],3))
    r = o.loc[("Smoking: Never smoked","diabetes_copd")]; fr = fi.loc["Smoking: Never smoked"]
    add("Method", "worked example: COPD never smoked vs rest, D", f"{r.supp_group:.3f} vs {r.supp_rest:.3f}, D = {r['diff']:.3f}")
    add("Method", "worked example: Diabetes only share, Delta", f"{fr.supp_diabetes_only:.3f}, Delta = {fr.delta:.3f}")
    add("Method", "single values / pairs generated / candidates", f'{s["n_candidates_after_min_count"]["1-item"]} / {s["pairs_generated"]} / {s["n_candidates_after_min_count"]["1-item"]+s["n_candidates_after_min_count"]["2-item"]}')
    rt = json.load(open(R+"runtime_this_pc.json")); add("Method", "time for the 814 design-based tests, this PC (s)", rt["design_tests_s"])
    c = pd.read_csv(R+"E7_cramers_v.csv")
    add("Results A", "design effect range / SE ratio", f"{c.mean_deff.min():.1f} to {c.mean_deff.max():.1f} / {np.sqrt(c.mean_deff.min()):.1f} to {np.sqrt(c.mean_deff.max()):.1f}")
    add("Results A", "largest Cramer p-value", f"{c.p_rao_scott.max():.2e}")
    add("Results B", "gap passes that fail the design-based test", cnt["weighted_delta_pass_fail_holm"])
    pr = s["pruning_reasons_weighted"]; add("Results B", "removed by R1 / R2", f'{pr.get("R1 complement of a kept item",0)} / {pr.get("R2 not surprising given parents",0)}')
    add("Results B", "final: total / singles / pairs", f'{len(f)} / {int((f.length==1).sum())} / {int((f.length==2).sum())}')
    fu = set(pd.read_csv(R+"E3_pruning_unweighted.csv").query("kept").label)
    add("Results B", "unweighted final / in common", f"{len(fu)} / {len(fu & set(f.label))}")
    add("Results C", "CIs for D that exclude 0", int(((b.diff_lo>0)|(b.diff_hi<0)).sum()))
    vc = b.group.value_counts()
    add("Results C", "characteristic group count: 2+ / COPD / Dep.", f'{vc.get("diabetes_multi_2plus",0)} / {vc.get("diabetes_copd",0)} / {vc.get("diabetes_depression",0)}')
    for p, nm in [("Physical activity: Any leisure activity","2+ active D"),("Smoking: Never smoked","2+ never smoked D")]:
        add("Results C", nm, round(float(o.loc[(p,"diabetes_multi_2plus"),"diff"]),3))
    r = o.loc[("General health: Poor","diabetes_multi_2plus")]; add("Results C", "2+ poor health: RR (share vs rest)", f"{r.rr:.2f} ({100*r.supp_group:.1f}% vs {100*r.supp_rest:.1f}%)")
    r = o.loc[("Smoking: Current, daily","diabetes_copd")]
    add("Results C", "COPD daily smoking: share vs rest, RR [CI]", f"{100*r.supp_group:.1f}% vs {100*r.supp_rest:.1f}%, RR {r.rr:.2f} [{r.rr_lo:.2f}, {r.rr_hi:.2f}]")
    add("Results C", "COPD college graduate D", round(float(o.loc[("Education: College graduate","diabetes_copd"),"diff"]),3))
    r = o.loc[("Smoking: Former smoker","diabetes_copd")]; add("Results C", "COPD former smoker D CI", f"{r.diff_lo:.3f} to {r.diff_hi:.3f}")
    add("Results C", "COPD analysis n / CKD analysis n", f'{int((pd.read_csv(R+"E0_group_sizes.csv").set_index("mm_group").n)["diabetes_copd"])} / {int((pd.read_csv(R+"E0_group_sizes.csv").set_index("mm_group").n)["diabetes_ckd"])}')
    r = o.loc[("Poor mental health days: 0 days","diabetes_depression")]
    add("Results C", "Dep. no poor MH days: share vs rest, RR", f"{100*r.supp_group:.1f}% vs {100*r.supp_rest:.1f}%, RR {r.rr:.2f}")
    add("Results C", "patterns without the two outcome traits", s["n_final_without_outcome_attrs"])
    add("Results D", "characteristic = highest / lowest group", f'{s["ovr_group_equals_max_group"]} / {s["ovr_group_equals_min_group"]} of {len(f)}')
    add("Results D", "bootstrap found in all / >=90% / >=70%", f'{int((bs.rediscovery_rate==1).sum())} / {int((bs.rediscovery_rate>=.9).sum())} / {int((bs.rediscovery_rate>=.7).sum())}')
    add("Results D", "patterns below 70%", "; ".join(f"{p} ({v:.3f})" for p, v in bs.rediscovery_rate[bs.rediscovery_rate<.7].items()))
    copd = [l for l in f.label if "diabetes_copd" in (fi.loc[l,"max_group"], fi.loc[l,"min_group"], b.loc[l,"group"])]
    add("Results D", "patterns involving Diab.+COPD: number / lowest found rate", f"{len(copd)} / {bs.rediscovery_rate[copd].min():.3f}")
    add("Results D", "same highest group in >=90% of samples", int((bs.same_max_group_rate>=.9).sum()))
    add("Results D", "no poor MH days: same highest group rate", round(float(bs.loc["Poor mental health days: 0 days","same_max_group_rate"]),3))
    tc = json.load(open(R+"E5_bootstrap_test_comparison.json"))
    add("Results D", "20 samples: Pearson-only passes / final disagreements", f'{tc["pe_not_rs"]} / {tc["final_flips"]}')
    se = pd.read_csv(R+"E4_sensitivity.csv")
    add("Results E", "delta 0.05 finals / delta 0.15 finals / Jaccard range", f'{se[se.delta==.05].n_final.min()} to {se[se.delta==.05].n_final.max()} / {se[se.delta==.15].n_final.min()} to {se[se.delta==.15].n_final.max()} / {se[se.delta!=.1].jaccard_vs_base.min():.2f} to {se[se.delta!=.1].jaccard_vs_base.max():.2f}')
    add("Results E", "age strata finals: under 65 / 65+", f'{s["age_under65_final_after_pruning"]} / {s["age_65plus_final_after_pruning"]}')
    u = pd.read_csv(R+"E9_final_under65.csv").set_index("label"); v = pd.read_csv(R+"E9_final_65plus.csv").set_index("label")
    k = "Poor mental health days: 14-30 days"; add("Results E", "14-30 MH days Delta: under 65 / 65+", f"{u.loc[k,'delta']:.3f} / {v.loc[k,'delta']:.3f}")
    if os.path.exists(R+"compare_6_vs_7_groups.json"):
      c6 = json.load(open(R+"compare_6_vs_7_groups.json"))
      add("Results E", "six groups: found / kept / new / dropped / Jaccard", f'{c6["six_group_final"]} / {c6["kept"]} / {len(c6["new"])} / {len(c6["dropped"])} / {c6["jaccard"]:.2f}')
    add("Results E", "multinomial McFadden R2", round(s["multinomial_mcfadden_r2"],2))
    m = pd.read_csv(R+"E6_multinomial_logit.csv"); singles = f[f.length==1].label; inm = [x for x in singles if x in set(m.term)]
    sig = set(m[(m.lo>1)|(m.hi<1)].term)
    add("Results E", "final single values in model / with OR CI excluding 1", f"{len(inm)} / {sum(x in sig for x in inm)}")
    w = pd.read_csv(R+"E6_wracc_top10.csv")
    add("Results E", "WRAcc top-10 patterns in our set", f"{int(w.pattern.isin(f.label).sum())} of {len(w)}")
    top = w[w['rank']==1]; add("Results E", "WRAcc top pattern is a mental health pattern in", f'{int(top.pattern.str.contains("mental health").sum())} of {len(top)} groups')
    add("Discussion", "highest-share group not the one that stood out", f'{len(f)-s["ovr_group_equals_max_group"]} of {len(f)}')
    return pd.DataFrame(N, columns=["section", "number in the paper", "your run"])
