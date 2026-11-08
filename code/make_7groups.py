"""Task A: split 'diabetes_other_single' into diabetes_depression and diabetes_copd."""
import pandas as pd
for src,dst in [("analysis_corrected.csv","analysis_7g.csv"),("diabetes_all_corrected.csv","diabetes_all_7g.csv")]:
    d=pd.read_csv(src,low_memory=False)
    os_=d.mm_group=="diabetes_other_single"
    assert ((d.cond_depression+d.cond_copd)[os_]==1).all()
    d.loc[os_&(d.cond_depression==1),"mm_group"]="diabetes_depression"
    d.loc[os_&(d.cond_copd==1),"mm_group"]="diabetes_copd"
    d.to_csv(dst,index=False); print(dst,d.mm_group.value_counts().to_dict())
