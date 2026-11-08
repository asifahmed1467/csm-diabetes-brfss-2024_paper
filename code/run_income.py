"""Missing-data check: keep records with a missing income answer and give them their own level, 'Not given'.
All other traits must still be complete. Same pipeline and thresholds as the main run."""
import sys; sys.path.insert(0, '.')
import pandas as pd
import csm_upgrade as C
G7 = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
d = pd.read_csv("diabetes_all_7g.csv", low_memory=False)
others = [a for a in C.CFG["attrs"] if a != "income_group"]
d = d[d[others].notna().all(axis=1)].copy()
d["income_group"] = d["income_group"].fillna(8)
d.to_csv("analysis_inc.csv", index=False)
print("records:", len(d), "| income 'Not given':", int((d.income_group == 8).sum()))
C.LABELS["income_group"][8] = "Not given"
C.CFG.update(groups=G7, single_condition_groups=G7[1:6])
C.GROUP_NAMES.update({"diabetes_depression": "Diab.+depression", "diabetes_copd": "Diab.+COPD"})
sys.argv = ["csm_upgrade.py", "--data", "analysis_inc.csv", "--full", "diabetes_all_7g.csv", "--out", "results_inc", "--boot", "2"]
C.main()

# Summary of the check (compared with the 30 patterns of the main run)
import json
a = pd.read_csv("results_inc/final_patterns_weighted.csv"); m = pd.read_csv("results7/final_patterns_weighted.csv")
A, M = set(a.label), set(m.label)
o = pd.read_csv("results_inc/E2_one_vs_rest_weighted.csv").set_index(["pattern", "group"])
bs = pd.read_csv("results7/E5_bootstrap_stability_raoscott.csv").set_index("pattern").rediscovery_rate
least5 = set(bs.sort_values().index[:5])
R = dict(n_records=len(d), n_income_not_given=int((d.income_group == 8).sum()), final_patterns=len(A), main_patterns_kept=len(A & M),
         main_patterns_dropped=len(M - A), new_patterns=len(A - M), least_stable5_dropped=len(least5 - A),
         poor_health_2plus_rr=round(float(o.loc[("General health: Poor", "diabetes_multi_2plus")].rr), 2),
         daily_smoking_copd_rr=round(float(o.loc[("Smoking: Current, daily", "diabetes_copd")].rr), 2))
json.dump(R, open("results7/income_check.json", "w"), indent=1); print(R)
