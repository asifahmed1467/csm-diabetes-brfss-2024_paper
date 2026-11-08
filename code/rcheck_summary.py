"""Summary of the R cross-check. Writes results7/R_crosscheck_summary.json and results7/R_crosscheck.csv."""
import pandas as pd, numpy as np, json
py = pd.read_csv("rcheck_py.csv"); R = pd.read_csv("rcheck_R.csv"); Rf = pd.read_csv("rcheck_R_full.csv")
m = py.merge(R, on="n"); fin = m[m.is_final == 1]; mf = m.merge(Rf, on="n")
S = dict(n_patterns=len(m),
 python_null_rate_p_lt_005=float((m.py_p_perm < .05).mean()), R_null_rate_RS1=float((m.R_p_perm_RS1 < .05).mean()), R_null_rate_F=float((m.R_p_perm_F < .05).mean()),
 R_full_file_null_rate_RS1=float((mf.R_full_perm_RS1 < .05).mean()), R_full_file_null_rate_F=float((mf.R_full_perm_F < .05).mean()), n_patterns_full_file=len(mf),
 final30_python_p_lt_1e4=int((fin.py_p_real < 1e-4).sum()), final30_R_RS1_p_lt_1e4=int((fin.R_p_real_RS1 < 1e-4).sum()),
 final30_R_F_p_lt_005_over_814=int((fin.R_p_real_F < 0.05 / 814).sum()))
m.to_csv("results7/R_crosscheck.csv", index=False); json.dump(S, open("results7/R_crosscheck_summary.json", "w"), indent=1); print(S)
