"""Six-group vs seven-group final patterns."""
import json, pandas as pd
a = set(pd.read_csv("results6/final_patterns_weighted.csv").label)
b = set(pd.read_csv("results7/final_patterns_weighted.csv").label)
R = dict(six_group_final=len(a), seven_group_final=len(b), kept=len(a & b), new=sorted(b - a),
         dropped=sorted(a - b), jaccard=len(a & b) / len(a | b))
json.dump(R, open("results7/compare_6_vs_7_groups.json", "w"), indent=1); print(json.dumps(R, indent=1))
