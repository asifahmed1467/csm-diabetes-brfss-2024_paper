"""Counts quoted in the text: Holm passes, Delta passes, and patterns that fail only the design-based test."""
import json, pandas as pd
K = {"adults_with_diabetes": int(len(pd.read_csv("diabetes_all_7g.csv", usecols=["mm_group"])))}
K["group_sizes_all_diabetes"] = pd.read_csv("diabetes_all_7g.csv", usecols=["mm_group"]).mm_group.value_counts().to_dict()
for n in ["weighted", "unweighted"]:
    e = pd.read_csv(f"results7/E1_candidates_{n}.csv"); h = e.p_holm < .05; d = e.delta >= .1
    K[f"{n}_holm_pass"] = int(h.sum()); K[f"{n}_delta_pass"] = int(d.sum())
    K[f"{n}_delta_pass_fail_holm"] = int((d & ~h).sum())
json.dump(K, open("results7/counts_7groups.json", "w"), indent=1); print(json.dumps(K, indent=1))
