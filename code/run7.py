import sys; sys.path.insert(0,'.')
import csm_upgrade as C
G7=["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G7,single_condition_groups=["diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd"])
C.GROUP_NAMES.update({"diabetes_depression":"Diab.+depression","diabetes_copd":"Diab.+COPD"})
sys.argv=["csm_upgrade.py","--data","analysis_7g.csv","--full","diabetes_all_7g.csv","--out","results7","--boot","2"]
C.main()
