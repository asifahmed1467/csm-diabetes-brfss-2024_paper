"""Tables and figures for the seven-group paper. All values read from results7/."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})
R="results7/"
G=["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
GS={"diabetes_only":"Only","diabetes_cvd":"CVD","diabetes_ckd":"CKD","diabetes_arthritis":"Arth.","diabetes_depression":"Dep.","diabetes_copd":"COPD","diabetes_multi_2plus":"2+"}
GH={"diabetes_only":"Only","diabetes_cvd":"CVD","diabetes_ckd":"CKD","diabetes_arthritis":"Arth.","diabetes_depression":"Dep.","diabetes_copd":"COPD","diabetes_multi_2plus":"2+"}
GN={"diabetes_only":"Diabetes only","diabetes_cvd":"Diab.+CVD","diabetes_ckd":"Diab.+CKD","diabetes_arthritis":"Diab.+arthritis","diabetes_depression":"Diab.+depression","diabetes_copd":"Diab.+COPD","diabetes_multi_2plus":"Diab.+2 or more"}
SH={"Poor mental health days: ":"MH days ","General health: ":"Health ","Physical activity: Any leisure activity":"Active",
    "Physical activity: No leisure activity":"Inactive","Smoking: Never smoked":"Never smoked","Smoking: Former smoker":"Former smoker",
    "Smoking: Current, daily":"Daily smoker","Education: College graduate":"College grad.","Income: ":"Income ","BMI: ":"","Age: ":"Age ",
    " AND ":" + ","0 days":"0","14-30 days":"14-30","1-5 days":"1-5","6-13 days":"6-13","Very good":"very good","Good":"good","Fair":"fair","Poor":"poor"}
def short(x,tex=False):
    for a,b in SH.items(): x=x.replace(a,b)
    return x.replace("$","\\$") if tex else x
T={}
f=pd.read_csv(R+"final_patterns_weighted.csv")
b=pd.read_csv(R+"E2_best_group_per_pattern.csv").set_index("pattern")
bs=pd.read_csv(R+"E5_bootstrap_stability_raoscott.csv").set_index("pattern")
# Table V: 12 largest |D|
b["absD"]=b["diff"].abs(); top=b.sort_values("absD",ascending=False).head(12); fi=f.set_index("label")
T["patterns"]="\n".join(f"{short(l,True)} & {fi.loc[l,'delta']:.3f} & {GS[fi.loc[l,'max_group']]} & {GS[fi.loc[l,'min_group']]} & {GS[x.group]} & ${x['diff']:+.3f}$ $[{x.diff_lo:+.3f},{x.diff_hi:+.3f}]$ & {x.rr:.2f} & {bs.loc[l,'rediscovery_rate']:.2f} & {bs.loc[l,'same_max_group_rate']:.2f}\\\\" for l,x in top.iterrows())
T["top12_min_absD"]=float(top.absD.min()); T["13th_absD"]=float(b.sort_values("absD",ascending=False).absD.iloc[12])
c=pd.read_csv(R+"E7_cramers_v.csv").replace({"Poor mental health days":"Poor MH days"})
T["cramer"]="\n".join(f"{r.attribute} & {r.levels} & {r.df_star} & {r.V_unweighted:.3f} & {r.V_weighted:.3f} & {r.label_weighted} & {r.mean_deff:.1f}\\\\" for r in c.itertuples())
e8=pd.read_csv(R+"E8_conditions_by_group.csv").set_index("condition")
order=[("CVD (heart attack, CHD or stroke)","CVD"),("Kidney disease","Kidney disease"),("Arthritis","Arthritis"),("Depressive disorder","Depressive disorder"),("COPD","COPD"),("Asthma (not used for groups)","Asthma$^\\ast$"),("Cancer (not used for groups)","Cancer$^\\ast$")]
T["comp"]="\n".join(n+" & "+" & ".join(f"{e8.loc[k,g]:.0f}" for g in G)+"\\\\" for k,n in order)
g=pd.read_csv(R+"E0_group_sizes.csv").set_index("mm_group"); allg=pd.read_csv("diabetes_all_7g.csv",usecols=["mm_group"]).mm_group.value_counts()
T["groups"]="\n".join(f"{GN[k]} & {allg[k]:,} & {int(g.loc[k,'n']):,} & {g.loc[k,'pct_weighted']:.1f}\\\\" for k in G)
json.dump(T,open("tables7.json","w"),indent=1)
# Fig 1 heatmap
fs=f.sort_values(["length","delta"],ascending=[True,False]); S=fs[["supp_"+x for x in G]].to_numpy()
fig,ax=plt.subplots(figsize=(3.45,5.3))
im=ax.imshow(S,cmap="Greys",vmin=0,vmax=1,aspect="auto")
for i in range(S.shape[0]):
    hi,lo=S[i].argmax(),S[i].argmin()
    for j in range(S.shape[1]):
        ax.text(j,i,f"{S[i,j]:.2f}"[1:] if S[i,j]<1 else "1.0",ha="center",va="center",fontsize=7,color="white" if S[i,j]>.55 else "black",fontweight="bold" if j in(hi,lo) else "normal")
    ax.add_patch(plt.Rectangle((hi-.5,i-.5),1,1,fill=False,lw=1.1,ec="black"))
    ax.add_patch(plt.Rectangle((lo-.5,i-.5),1,1,fill=False,lw=0.9,ec="black",ls=(0,(2,1))))
ax.axhline((fs.length==1).sum()-.5,color="black",lw=1.4)
ax.set_yticks(range(len(fs))); ax.set_yticklabels([short(l) for l in fs.label],fontsize=7)
ax.set_xticks(range(7)); ax.set_xticklabels([GH[x] for x in G],fontsize=7); ax.xaxis.tick_top(); ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig("fig_heatmap_weighted.png",dpi=400); plt.close(fig)
# Fig 2 forest
bb=b.sort_values("diff"); fig,ax=plt.subplots(figsize=(3.45,4.9)); y=np.arange(len(bb))
ax.errorbar(bb["diff"],y,xerr=[bb["diff"]-bb.diff_lo,bb.diff_hi-bb["diff"]],fmt="o",color="black",ms=2.5,capsize=1.5,lw=0.8)
ax.axvline(0,color="0.5",lw=0.7,ls="--")
ax.set_yticks(y); ax.set_yticklabels([f"{short(p)} [{GS[q]}]" for p,q in zip(bb.index,bb.group)],fontsize=7)
ax.set_xlabel("Group minus rest (95% CI)",fontsize=7); ax.tick_params(axis="x",labelsize=7)
fig.tight_layout(pad=0.3); fig.savefig("fig_forest_one_vs_rest.png",dpi=400); plt.close(fig)
print(json.dumps({k:v for k,v in T.items() if not isinstance(v,str)}))
