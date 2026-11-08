"""Rebuild the diabetes multimorbidity analysis set from raw BRFSS 2024.
legacy=True reproduces the original mining_ready_data.csv (keeps smoker=9 and age=14,
which are BRFSS 'don't know/refused' codes). legacy=False treats them as missing."""
import sys, numpy as np, pandas as pd
RAW = sys.argv[1]
COLS=["_PSU","_STSTR","_LLCPWT","GENHLTH","MENTHLTH","CVDINFR4","CVDCRHD4","CVDSTRK3","ASTHMA3","CHCSCNC1",
      "CHCOCNC1","CHCCOPD3","ADDEPEV3","CHCKDNY2","HAVARTH4","DIABETE4","EDUCA","_TOTINDA","_SEX","_AGEG5YR",
      "_BMI5CAT","_INCOMG1","_SMOKER3"]
A=["bmi_cat","smoker_status","physical_activity","general_health","mh_days_cat","income_group","education","age_group"]
def build(legacy):
    d=pd.read_csv(RAW,usecols=COLS,low_memory=False)
    d=d[d.DIABETE4==1].copy()                       # diagnosed diabetes only (excl. gestational, prediabetes)
    y=lambda c:(d[c]==1)
    d["cond_cvd"]=(y("CVDINFR4")|y("CVDCRHD4")|y("CVDSTRK3")).astype(int)
    d["cond_ckd"]=y("CHCKDNY2").astype(int); d["cond_arthritis"]=y("HAVARTH4").astype(int)
    d["cond_copd"]=y("CHCCOPD3").astype(int); d["cond_depression"]=y("ADDEPEV3").astype(int)
    d["cond_asthma"]=y("ASTHMA3").astype(int); d["cond_cancer"]=(y("CHCSCNC1")|y("CHCOCNC1")).astype(int)
    cnt=d[["cond_cvd","cond_ckd","cond_arthritis","cond_copd","cond_depression"]].sum(axis=1)
    d["n_conditions"]=cnt
    d["mm_group"]=np.select([cnt==0,cnt>=2,(cnt==1)&(d.cond_cvd==1),(cnt==1)&(d.cond_ckd==1),(cnt==1)&(d.cond_arthritis==1)],
        ["diabetes_only","diabetes_multi_2plus","diabetes_cvd","diabetes_ckd","diabetes_arthritis"],"diabetes_other_single")
    m=d.MENTHLTH
    d["mh_days_cat"]=pd.Series(np.select([m==88,(m>=1)&(m<=5),(m>=6)&(m<=13),(m>=14)&(m<=30)],
        ["none","1-5 days","6-13 days","14-30 days"],"NA"),index=d.index).replace("NA",np.nan)
    k=lambda s,ok:s.where(s.isin(ok))
    d["general_health"]=k(d.GENHLTH,[1,2,3,4,5]); d["physical_activity"]=k(d._TOTINDA,[1,2])
    d["age_group"]=k(d._AGEG5YR,range(1,15) if legacy else range(1,14))
    d["smoker_status"]=k(d._SMOKER3,[1,2,3,4,9] if legacy else [1,2,3,4])
    d["income_group"]=k(d._INCOMG1,range(1,8)); d["bmi_cat"]=k(d._BMI5CAT,[1,2,3,4]); d["education"]=k(d.EDUCA,range(1,7))
    return d, A
if __name__=="__main__":
    for legacy,name in [(True,"legacy"),(False,"corrected")]:
        d,A=build(legacy)
        d.to_csv(f"diabetes_all_{name}.csv",index=False)
        ok=d[A].notna().all(axis=1)
        d[ok].to_csv(f"analysis_{name}.csv",index=False)
        print(name, "diabetes:",len(d),"analysis:",ok.sum())
