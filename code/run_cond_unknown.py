"""Check on the group definition. In build_data.py a condition counts only if the answer is 'yes' (code 1), so
'don't know', 'refused' and missing answers on the 7 condition questions count as 'no'. This affects 3,355 of the 65,809
adults with diabetes (5.1%). Here those adults are removed from the analysis (they stay in the full file for the design
variance), and the whole pipeline is rerun with the same thresholds. Usage: python run_cond_unknown.py RAW_CSV"""
import sys; sys.path.insert(0, '.')
import pandas as pd, json
import csm_upgrade as C
RAW = sys.argv[1]
CV = ["CVDINFR4", "CVDCRHD4", "CVDSTRK3", "CHCKDNY2", "HAVARTH4", "CHCCOPD3", "ADDEPEV3"]
raw = pd.read_csv(RAW, usecols=["_PSU", "_STSTR", "DIABETE4"] + CV, low_memory=False); raw = raw[raw.DIABETE4 == 1].copy()
raw["unknown"] = (~raw[CV].isin([1, 2]).all(axis=1)).astype(int)
raw["key"] = raw._STSTR.astype(str) + "|" + raw._PSU.astype(str)
a = pd.read_csv("analysis_7g.csv", low_memory=False); a["key"] = a._STSTR.astype(str) + "|" + a._PSU.astype(str)
assert raw.key.is_unique and a.key.isin(raw.key).all()
unk = set(raw.key[raw.unknown == 1])
d = a[~a.key.isin(unk)].drop(columns="key"); d.to_csv("analysis_cu.csv", index=False)
print("adults with diabetes:", len(raw), "| with an unknown condition answer:", int(raw.unknown.sum()), "| in the analysis set:", int(a.key.isin(unk).sum()), "| analysis set now:", len(d))
G7 = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G7, single_condition_groups=G7[1:6])
C.GROUP_NAMES.update({"diabetes_depression": "Diab.+depression", "diabetes_copd": "Diab.+COPD"})
sys.argv = ["csm_upgrade.py", "--data", "analysis_cu.csv", "--full", "diabetes_all_7g.csv", "--out", "results_cu", "--boot", "2"]
C.main()
n = pd.read_csv("results_cu/final_patterns_weighted.csv"); m = pd.read_csv("results7/final_patterns_weighted.csv")
A, M = set(n.label), set(m.label)
R = dict(n_diabetes=len(raw), n_unknown_condition=int(raw.unknown.sum()), pct_unknown=round(100 * raw.unknown.mean(), 2), n_records_analysis=len(d),
         final_patterns=len(A), main_patterns_kept=len(A & M), main_patterns_dropped=len(M - A), new_patterns=len(A - M),
         group_sizes=d.mm_group.value_counts().to_dict())
json.dump(R, open("results7/cond_unknown_check.json", "w"), indent=1); print(R)
