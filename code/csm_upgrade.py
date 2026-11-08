#!/usr/bin/env python3
"""
Upgrade experiments for:
  "Contrast Set Mining for Discovering Distinctive Multimorbidity Patterns in
   Diabetes-Related Chronic Conditions Using CDC BRFSS Health Indicators"

What this script does (one section per experiment, E0 to E10):
  E0  Group sizes for the analysis set + check of who was dropped (missing data)
  E1  Survey-weighted supports + design-based tests (Wald F and first-order
      Rao-Scott), side by side with the unweighted analysis
  E2  One-vs-rest contrasts: support difference, risk ratio, odds ratio,
      95% CIs, and WRAcc
  E3  Explicit redundancy pruning (complement rule + STUCCO-style
      "surprising" rule for two-item patterns); reports which pairs survive
  E4  Threshold sensitivity (delta x minimum count)
  E5  Bootstrap stability (PSU bootstrap within strata, B = 200)
  E6  Baselines: multinomial logistic regression and WRAcc subgroup discovery
  E7  Cramer's V with correct Cohen labels (uses df* = min(r, c) - 1)
  E8  Circularity check (which conditions define each group) and a rerun
      without outcome-type attributes
  E9  Age-stratified mining with the SAME pipeline (age_group removed)
  E10 Condition-specific contrast among the four single-condition groups
  FIG All figures with readable labels

Usage:
  python csm_upgrade.py --data analysis_ready.csv --full diabetes_all.csv --out results

  --data : the 49,757-row analysis set (one row per respondent, columns below)
  --full : (optional) the 65,809-row diabetes file BEFORE dropping missing
           values; needed only for E0's dropout check.

Required columns (rename in CFG if yours differ):
  mm_group, the 8 mining attributes, _LLCPWT, _STSTR, _PSU
Optional columns: _SEX, and condition flags listed in CFG["conditions"].

Every number this script writes comes from your data. Nothing is hard-coded.
Random seed is fixed (SEED) so every run gives the same output.
"""
import argparse
import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 2026

# --------------------------------------------------------------------------
# CONFIGURATION. Check every entry against your own preprocessing code.
# --------------------------------------------------------------------------
CFG = dict(
    group="mm_group",
    groups=["diabetes_only", "diabetes_cvd", "diabetes_ckd",
            "diabetes_arthritis", "diabetes_other_single",
            "diabetes_multi_2plus"],
    single_condition_groups=["diabetes_cvd", "diabetes_ckd",
                             "diabetes_arthritis", "diabetes_other_single"],
    attrs=["mh_days_cat", "general_health", "physical_activity", "age_group",
           "smoker_status", "income_group", "bmi_cat", "education"],
    # Attributes that are closer to health OUTCOMES than to risk factors.
    # E8 reruns the mining without them.
    outcome_attrs=["mh_days_cat", "general_health"],
    weight="_LLCPWT", strata="_STSTR", psu="_PSU", sex="_SEX",
    age_attr="age_group",
    under65_codes=[1, 2, 3, 4, 5, 6, 7, 8, 9],   # _AGEG5YR 1..9 = 18 to 64
    # 0/1 flags for each chronic condition used to build the groups (E8).
    # Put the column names of YOUR processed file here.
    conditions={
        "CVD (heart attack, CHD or stroke)": "cond_cvd",
        "Kidney disease": "cond_ckd",
        "Arthritis": "cond_arthritis",
        "Depressive disorder": "cond_depression",
        "COPD": "cond_copd",
        "Asthma (not used for groups)": "cond_asthma",
        "Cancer (not used for groups)": "cond_cancer",
    },
    min_count=30, delta=0.10, alpha=0.05,
    surprise_eps=0.02,       # E3: min. |observed - expected| support gap
    boot_B=200,
    sens_delta=[0.05, 0.10, 0.15], sens_min_count=[30, 100, 500],
)

# Readable labels. Codes follow the 2024 BRFSS codebook for GENHLTH,
# _TOTINDA, _SMOKER3, _BMI5CAT, _AGEG5YR, EDUCA and _INCOMG1.
# VERIFY each map against the codebook and your own recoding.
LABELS = {
    "general_health": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    "physical_activity": {1: "Any leisure activity", 2: "No leisure activity"},
    "smoker_status": {1: "Current, daily", 2: "Current, some days",
                      3: "Former smoker", 4: "Never smoked"},
    "bmi_cat": {1: "Underweight", 2: "Normal weight", 3: "Overweight", 4: "Obese"},
    "age_group": {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
                  6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
                  11: "70-74", 12: "75-79", 13: "80+"},
    "education": {1: "No school", 2: "Grades 1-8", 3: "Grades 9-11",
                  4: "High school", 5: "Some college", 6: "College graduate"},
    "income_group": {1: "<$15k", 2: "$15k-25k", 3: "$25k-35k", 4: "$35k-50k",
                     5: "$50k-100k", 6: "$100k-200k", 7: "$200k+"},
    "mh_days_cat": {"none": "0 days", "1-5 days": "1-5 days",
                    "6-13 days": "6-13 days", "14-30 days": "14-30 days"},
}
ATTR_NAMES = {"mh_days_cat": "Poor mental health days",
              "general_health": "General health",
              "physical_activity": "Physical activity",
              "age_group": "Age", "smoker_status": "Smoking",
              "income_group": "Income", "bmi_cat": "BMI",
              "education": "Education"}
GROUP_NAMES = {"diabetes_only": "Diabetes only", "diabetes_cvd": "Diab.+CVD",
               "diabetes_ckd": "Diab.+CKD", "diabetes_arthritis": "Diab.+arthritis",
               "diabetes_other_single": "Diab.+other single",
               "diabetes_multi_2plus": "Diab.+2 or more"}


def norm_code(v):
    """Turn 4.0 / np.float64(4.0) / '4' into 4; keep text codes as text."""
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def item_label(attr, val):
    v = norm_code(val)
    lab = LABELS.get(attr, {}).get(v, str(v))
    return f"{ATTR_NAMES.get(attr, attr)}: {lab}"


def pattern_label(items):
    return " AND ".join(item_label(a, v) for a, v in items)


# --------------------------------------------------------------------------
# Survey helpers (Taylor linearization, stratified with-replacement PSUs)
# --------------------------------------------------------------------------
class Design:
    """Holds weights, strata and PSU ids; computes linearized covariances."""

    def __init__(self, df, cfg, weighted=True):
        n = len(df)
        self.w = df[cfg["weight"]].to_numpy(float) if weighted else np.ones(n)
        if weighted:
            st = df[cfg["strata"]].astype(str).to_numpy()
            ps = df[cfg["psu"]].astype(str).to_numpy()
        else:  # simple random sampling: each row is its own PSU, one stratum
            st = np.zeros(n, dtype=int).astype(str)
            ps = np.arange(n).astype(str)
        key = pd.Series(st) + "|" + pd.Series(ps)
        self.psu_idx, psu_keys = pd.factorize(key)
        psu_stratum = pd.Series(psu_keys).str.split("|").str[0]
        self.str_of_psu, _ = pd.factorize(psu_stratum)
        self.n_psu = len(psu_keys)
        self.n_str = self.str_of_psu.max() + 1
        nh = np.bincount(self.str_of_psu, minlength=self.n_str)
        self.nh = nh
        self.df_design = self.n_psu - self.n_str   # design degrees of freedom
        self.weighted = weighted

    def cov_of_totals(self, U):
        """U: n x m matrix of influence values. Returns m x m covariance."""
        m = U.shape[1]
        T = np.zeros((self.n_psu, m))
        np.add.at(T, self.psu_idx, U)                       # PSU totals
        S = np.zeros((self.n_str, m))
        np.add.at(S, self.str_of_psu, T)
        mean_h = S / self.nh[:, None]
        D = T - mean_h[self.str_of_psu]                     # centred in stratum
        f = np.where(self.nh > 1, self.nh / np.maximum(self.nh - 1, 1), 0.0)
        D = D * np.sqrt(f[self.str_of_psu])[:, None]        # lonely PSU -> 0
        return D.T @ D


def domain_props(Y, g, design, levels):
    """Weighted proportions of the columns of Y within each domain level, with
    their joint linearized covariance. Y: n x J (0/1). Returns p (K*J,), V."""
    w = design.w
    K, J = len(levels), Y.shape[1]
    P = np.zeros((K, J))
    U = np.zeros((len(g), K * J))
    for k, lev in enumerate(levels):
        d = (g == lev)
        Wk = w[d].sum()
        P[k] = (w[d, None] * Y[d]).sum(0) / Wk
        U[:, k * J:(k + 1) * J] = (d * w)[:, None] * (Y - P[k]) / Wk
    return P.ravel(), design.cov_of_totals(U), P


def wald_homogeneity(Y, g, design, levels):
    """Design-based test that the distribution of Y is the same in all
    groups. Returns Wald F, its p-value, first-order Rao-Scott chi2 and p."""
    p, V, P = domain_props(Y, g, design, levels)
    K, J = P.shape
    # contrasts: p_{k,j} - p_{K,j}, k < K
    C = np.zeros(((K - 1) * J, K * J))
    for k in range(K - 1):
        for j in range(J):
            C[k * J + j, k * J + j] = 1
            C[k * J + j, (K - 1) * J + j] = -1
    q = C.shape[0]
    cp = C @ p
    CVC = C @ V @ C.T
    W = float(cp @ np.linalg.pinv(CVC) @ cp)
    d = max(design.df_design, q + 1)
    F = W * (d - q + 1) / (d * q)
    pF = stats.f.sf(F, q, d - q + 1)
    # SRS covariance (multinomial within each domain, unweighted n_k)
    Vs = np.zeros((K * J, K * J))
    for k, lev in enumerate(levels):
        nk = (g == lev).sum()
        pk = P[k]
        Vs[k * J:(k + 1) * J, k * J:(k + 1) * J] = (np.diag(pk) - np.outer(pk, pk)) / nk
    CVsC = C @ Vs @ C.T
    Ws = float(cp @ np.linalg.pinv(CVsC) @ cp)           # ~ Pearson X2
    deff_mean = np.trace(np.linalg.pinv(CVsC) @ CVC) / q
    rs1 = Ws / max(deff_mean, 1e-12)
    prs = stats.chi2.sf(rs1, q)
    return dict(wald_F=F, p_wald=pF, rs1_chi2=rs1, p_rs1=prs,
                pearson_w=Ws, mean_deff=deff_mean, df=q)


# --------------------------------------------------------------------------
# Candidate generation and counting
# --------------------------------------------------------------------------
def build_items(df, attrs):
    items = []
    for a in attrs:
        for v in sorted(df[a].dropna().unique(), key=lambda x: str(x)):
            items.append((a, v))
    return items


def mask_of(df, items, cache):
    m = None
    for it in items:
        if it not in cache:
            cache[it] = (df[it[0]] == it[1]).to_numpy()
        m = cache[it] if m is None else (m & cache[it])
    return m


def candidates(items, max_len=2):
    cands = [(it,) for it in items]
    if max_len >= 2:
        for a, b in itertools.combinations(items, 2):
            if a[0] != b[0]:
                cands.append((a, b))
    return cands


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def mine(df, cfg, attrs=None, groups=None, weighted=False, delta=None,
         min_count=None, max_len=2, design=None, test="auto"):
    """STUCCO-style mining. Returns a table with one row per candidate that
    passed the minimum count, with supports, Delta, max/min group, p, Holm p."""
    attrs = attrs or cfg["attrs"]
    groups = groups or cfg["groups"]
    delta = cfg["delta"] if delta is None else delta
    min_count = cfg["min_count"] if min_count is None else min_count
    df = df[df[cfg["group"]].isin(groups)].reset_index(drop=True)
    g = df[cfg["group"]].to_numpy()
    gidx = pd.Categorical(g, categories=groups).codes
    K = len(groups)
    w = df[cfg["weight"]].to_numpy(float) if weighted else np.ones(len(df))
    n_g = np.bincount(gidx, minlength=K)
    W_g = np.bincount(gidx, weights=w, minlength=K)
    if weighted and design is None:
        design = Design(df, cfg, weighted=True)
    cache = {}
    rows = []
    items = build_items(df, attrs)
    n_cand_total = 0
    for c in candidates(items, max_len):
        n_cand_total += 1
        m = mask_of(df, c, cache)
        cnt = m.sum()
        if cnt < min_count:
            continue
        x_g = np.bincount(gidx[m], minlength=K)
        s = np.bincount(gidx[m], weights=w[m], minlength=K) / W_g
        if weighted and test != "chi2":
            t = wald_homogeneity(m[:, None].astype(float), g, design, groups)
            p = t["p_rs1"]
        else:
            tab = np.vstack([x_g, n_g - x_g])
            p = stats.chi2_contingency(tab, correction=False)[1] if (tab.min(1) > 0).all() else 1.0
        row = dict(pattern=c, length=len(c), label=pattern_label(c), count=int(cnt),
                   delta=s.max() - s.min(), max_group=groups[int(s.argmax())],
                   min_group=groups[int(s.argmin())], p=p)
        for k, gr in enumerate(groups):
            row["supp_" + gr] = s[k]
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs["n_candidates_generated"] = n_cand_total
    if len(res):
        res["p_holm"] = holm(res["p"].to_numpy())
        res["significant"] = (res["p_holm"] < cfg["alpha"]) & (res["delta"] >= delta)
    return res


# --------------------------------------------------------------------------
# E3. Redundancy pruning (explicit rules)
# --------------------------------------------------------------------------
def prune(res, df, cfg, groups=None, weighted=False):
    """Rule R1 (complement): for a binary attribute, A=a and A=b carry the same
    contrast; keep only the level with the larger Delta (ties: first code).
    Rule R2 (STUCCO-style surprise, Bay & Pazzani 2001): a two-item pattern
    A AND B is kept only if (i) its group supports are significantly different
    from what A and B predict if they were independent within each group
    (Breslow-Day test of equal A-B odds ratios across groups, Holm-corrected),
    and (ii) max_g |s_obs(g) - s(A|g)s(B|g)| >= surprise_eps.
    Single-item patterns are only subject to R1."""
    from statsmodels.stats.contingency_tables import StratifiedTable
    groups = groups or cfg["groups"]
    sig = res[res["significant"]].copy()
    sig["drop_reason"] = ""
    d = df[df[cfg["group"]].isin(groups)].reset_index(drop=True)
    gcol = d[cfg["group"]].to_numpy()
    w = d[cfg["weight"]].to_numpy(float) if weighted else np.ones(len(d))
    # R1 complements
    singles = sig[sig["length"] == 1]
    for a in singles["pattern"].map(lambda c: c[0][0]).unique():
        if d[a].nunique() == 2:
            rows = singles[singles["pattern"].map(lambda c: c[0][0] == a)]
            if len(rows) == 2:
                drop = rows.sort_values("delta", ascending=False).index[1]
                sig.loc[drop, "drop_reason"] = "R1 complement of a kept item"
    # R2 surprise test for pairs
    pairs = sig[sig["length"] == 2]
    pvals, surpr, idx = [], [], []
    for i, r in pairs.iterrows():
        (a, b) = r["pattern"]
        # Pairs are tested even when a parent is not a contrast on its own:
        # a pure interaction is exactly what STUCCO wants to keep.
        A = (d[a[0]] == a[1]).to_numpy()
        B = (d[b[0]] == b[1]).to_numpy()
        tabs, gap = [], 0.0
        for gr in groups:
            k = gcol == gr
            wk = w[k]
            Ak, Bk = A[k], B[k]
            t = np.array([[np.sum(Ak & Bk), np.sum(Ak & ~Bk)],
                          [np.sum(~Ak & Bk), np.sum(~Ak & ~Bk)]], float) + 0.5
            tabs.append(t)
            sA = wk[Ak].sum() / wk.sum()
            sB = wk[Bk].sum() / wk.sum()
            sAB = wk[Ak & Bk].sum() / wk.sum()
            gap = max(gap, abs(sAB - sA * sB))
        st = StratifiedTable(tabs)
        pvals.append(st.test_equal_odds().pvalue)
        surpr.append(gap)
        idx.append(i)
    if idx:
        ph = holm(pvals)
        for i, p, s in zip(idx, ph, surpr):
            sig.loc[i, "p_surprise_holm"] = p
            sig.loc[i, "surprise_gap"] = s
            if not (p < cfg["alpha"] and s >= cfg["surprise_eps"]):
                sig.loc[i, "drop_reason"] = "R2 not surprising given parents"
    sig["kept"] = sig["drop_reason"] == ""
    return sig


# --------------------------------------------------------------------------
# E2. One-vs-rest contrasts with CIs and WRAcc
# --------------------------------------------------------------------------
def one_vs_rest(df, cfg, pattern, groups=None, weighted=True, design=None):
    groups = groups or cfg["groups"]
    d = df[df[cfg["group"]].isin(groups)].reset_index(drop=True)
    if design is None:
        design = Design(d, cfg, weighted=weighted)
    m = np.ones(len(d), bool)
    for a, v in pattern:
        m &= (d[a] == v).to_numpy()
    y = m.astype(float)
    g = d[cfg["group"]].to_numpy()
    w = design.w
    N = w.sum()
    out = []
    z = stats.norm.ppf(0.975)
    for gr in groups:
        ind = np.where(g == gr, "in", "rest")
        p, V, P = domain_props(y[:, None], ind, design, ["in", "rest"])
        s1, s0 = p[0], p[1]
        diff = s1 - s0
        se_d = np.sqrt(max(V[0, 0] + V[1, 1] - 2 * V[0, 1], 0))
        rr = s1 / s0 if s0 > 0 else np.nan
        grad = np.array([1 / s1, -1 / s0]) if s1 > 0 and s0 > 0 else np.array([np.nan, np.nan])
        se_lrr = np.sqrt(max(grad @ V @ grad, 0))
        orr = (s1 / (1 - s1)) / (s0 / (1 - s0)) if 0 < s1 < 1 and 0 < s0 < 1 else np.nan
        go = np.array([1 / (s1 * (1 - s1)), -1 / (s0 * (1 - s0))]) if np.isfinite(orr) else grad
        se_lor = np.sqrt(max(go @ V @ go, 0))
        cov = w[m].sum() / N
        prec = w[m & (g == gr)].sum() / max(w[m].sum(), 1e-12)
        prior = w[g == gr].sum() / N
        out.append(dict(pattern=pattern_label(pattern), group=gr,
                        supp_group=s1, supp_rest=s0,
                        diff=diff, diff_lo=diff - z * se_d, diff_hi=diff + z * se_d,
                        rr=rr, rr_lo=np.exp(np.log(rr) - z * se_lrr), rr_hi=np.exp(np.log(rr) + z * se_lrr),
                        or_=orr, or_lo=np.exp(np.log(orr) - z * se_lor), or_hi=np.exp(np.log(orr) + z * se_lor),
                        wracc=cov * (prec - prior)))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# E7. Cramer's V with Cohen labels
# --------------------------------------------------------------------------
def cohen_label(v, dfstar):
    s, m, l = 0.10 / np.sqrt(dfstar), 0.30 / np.sqrt(dfstar), 0.50 / np.sqrt(dfstar)
    return "negligible" if v < s else "small" if v < m else "medium" if v < l else "large"


def cramers_table(df, cfg, design_w, design_u):
    rows = []
    g = df[cfg["group"]].to_numpy()
    for a in cfg["attrs"]:
        levels = sorted(df[a].dropna().unique(), key=str)
        Y = np.column_stack([(df[a] == v).to_numpy(float) for v in levels[:-1]])
        tab = pd.crosstab(df[a], df[cfg["group"]]).to_numpy()
        chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
        n = tab.sum()
        r, c = tab.shape
        dfstar = min(r, c) - 1
        V = np.sqrt(chi2 / (n * dfstar))
        tw = wald_homogeneity(Y, g, design_w, cfg["groups"])
        Vw = np.sqrt(tw["pearson_w"] / (n * dfstar))  # weighted proportions
        rows.append(dict(attribute=ATTR_NAMES[a], levels=r, df_star=dfstar,
                         V_unweighted=V, chi2=chi2, p=p, label_unweighted=cohen_label(V, dfstar),
                         V_weighted=Vw, rao_scott_chi2=tw["rs1_chi2"], p_rao_scott=tw["p_rs1"],
                         wald_F=tw["wald_F"], p_wald=tw["p_wald"], mean_deff=tw["mean_deff"],
                         label_weighted=cohen_label(Vw, dfstar),
                         small_cut=0.10 / np.sqrt(dfstar), medium_cut=0.30 / np.sqrt(dfstar)))
    return pd.DataFrame(rows).sort_values("V_unweighted", ascending=False)


# --------------------------------------------------------------------------
# E5. Bootstrap
# --------------------------------------------------------------------------
class PSUBootstrap:
    """Resample PSUs with replacement within each stratum (n_h draws per
    stratum), keeping all rows of a drawn PSU. Unweighted: row bootstrap."""

    def __init__(self, df, cfg, weighted):
        self.n = len(df)
        self.weighted = weighted
        if not weighted:
            return
        d = Design(df, cfg, weighted=True)
        order = np.argsort(d.psu_idx, kind="stable")             # rows by PSU
        counts = np.bincount(d.psu_idx, minlength=d.n_psu)
        self.row_order = order
        self.row_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.row_count = counts
        psu_by_str = np.argsort(d.str_of_psu, kind="stable")     # PSUs by stratum
        self.psu_by_str = psu_by_str
        self.str_start = np.concatenate([[0], np.cumsum(d.nh)[:-1]])
        self.nh = d.nh
        self.str_of_slot = d.str_of_psu[psu_by_str]

    def draw(self, rng):
        if not self.weighted:
            return rng.integers(0, self.n, self.n)
        h = self.str_of_slot
        pick = self.psu_by_str[self.str_start[h] + (rng.random(len(h)) * self.nh[h]).astype(int)]
        starts, cnts = self.row_start[pick], self.row_count[pick]
        rep = np.repeat(starts - np.concatenate([[0], np.cumsum(cnts)[:-1]]), cnts)
        return self.row_order[np.arange(cnts.sum()) + rep]


def bootstrap_stability(df, cfg, final, weighted, B, rng):
    keys = [tuple(p) for p in final["pattern"]]
    found = {k: 0 for k in keys}
    same_max = {k: 0 for k in keys}
    max_ref = dict(zip(keys, final["max_group"]))
    sampler = PSUBootstrap(df, cfg, weighted)
    t0 = time.time()
    for b in range(B):
        idx = sampler.draw(rng)
        db = df.iloc[idx].reset_index(drop=True)
        r = mine(db, cfg, weighted=weighted, test="chi2")   # chi2 for speed
        s = r[r["significant"]]
        smap = dict(zip(s["pattern"].map(tuple), s["max_group"]))
        for k in keys:
            if k in smap:
                found[k] += 1
                same_max[k] += (smap[k] == max_ref[k])
        if b == 0:
            print(f"    one bootstrap replicate took {time.time()-t0:.1f}s")
    return pd.DataFrame(dict(pattern=[pattern_label(k) for k in keys],
                             rediscovery_rate=[found[k] / B for k in keys],
                             same_max_group_rate=[same_max[k] / B for k in keys]))


# --------------------------------------------------------------------------
# E6. Baselines
# --------------------------------------------------------------------------
def baseline_multinomial(df, cfg):
    """Multinomial logit of group on all attributes (one-hot, first level as
    reference). Unweighted fit for SEs; weighted point estimates from sklearn.
    Reports odds ratios vs diabetes_only."""
    import statsmodels.api as sm
    X = pd.get_dummies(df[cfg["attrs"]].astype(str), drop_first=True, dtype=float)
    y = pd.Categorical(df[cfg["group"]], categories=cfg["groups"]).codes
    model = sm.MNLogit(y, sm.add_constant(X)).fit(disp=False, maxiter=200, method="newton")
    rows = []
    for k in range(1, len(cfg["groups"])):
        par = model.params[k - 1]
        se = model.bse[k - 1]
        for name in X.columns:
            a, v = name.rsplit("_", 1)
            rows.append(dict(group=cfg["groups"][k], term=item_label(a, v),
                             OR=np.exp(par[name]),
                             lo=np.exp(par[name] - 1.96 * se[name]),
                             hi=np.exp(par[name] + 1.96 * se[name]),
                             p=model.pvalues[k - 1][name]))
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=2000, C=1e6)
        lr.fit(X.to_numpy(), y, sample_weight=df[cfg["weight"]].to_numpy(float))
        ref = lr.coef_[0]
        Wt = {}
        for k in range(1, len(cfg["groups"])):
            for j, name in enumerate(X.columns):
                a, v = name.rsplit("_", 1)
                Wt[(cfg["groups"][k], item_label(a, v))] = np.exp(lr.coef_[k][j] - ref[j])
        for r in rows:
            r["OR_weighted_point"] = Wt.get((r["group"], r["term"]), np.nan)
    except Exception as e:  # sklearn missing: skip weighted column
        print("    sklearn weighted fit skipped:", e)
    return pd.DataFrame(rows), model.prsquared


def baseline_wracc(df, cfg, top_k=10, weighted=True):
    """Exhaustive subgroup discovery (depth <= 2) with WRAcc per target group.
    Same search space as the contrast miner, so the comparison is fair."""
    d = df.reset_index(drop=True)
    w = d[cfg["weight"]].to_numpy(float) if weighted else np.ones(len(d))
    g = d[cfg["group"]].to_numpy()
    N = w.sum()
    cache = {}
    items = build_items(d, cfg["attrs"])
    cands = candidates(items, 2)
    out = []
    for gr in cfg["groups"]:
        tgt = g == gr
        prior = w[tgt].sum() / N
        rows = []
        for c in cands:
            m = mask_of(d, c, cache)
            if m.sum() < cfg["min_count"]:
                continue
            cov = w[m].sum() / N
            prec = w[m & tgt].sum() / w[m].sum()
            rows.append((c, cov * (prec - prior), prec, cov))
        rows.sort(key=lambda r: -r[1])
        for rank, (c, wr, prec, cov) in enumerate(rows[:top_k], 1):
            out.append(dict(group=gr, rank=rank, pattern=pattern_label(c), length=len(c),
                            wracc=wr, precision=prec, coverage=cov, prior=prior))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# E0. Group table and dropout check
# --------------------------------------------------------------------------
def group_table(df, cfg):
    w = df[cfg["weight"]]
    t = df.groupby(cfg["group"]).agg(n=(cfg["group"], "size"))
    t["pct_unweighted"] = 100 * t["n"] / t["n"].sum()
    t["pct_weighted"] = 100 * df.groupby(cfg["group"])[cfg["weight"]].sum() / w.sum()
    return t.reindex(cfg["groups"]).reset_index()


def dropout_check(full, cfg):
    """Compare respondents kept vs dropped (missing in any mining attribute)."""
    kept = full[cfg["attrs"]].notna().all(axis=1)
    full = full.assign(status=np.where(kept, "kept", "dropped"))
    rows = []
    for col in [cfg["group"], cfg["age_attr"], cfg["sex"]]:
        if col not in full:
            continue
        tab = pd.crosstab(full[col], full["status"])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        V = np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1)))
        pct = 100 * tab / tab.sum()
        for lev in tab.index:
            rows.append(dict(variable=col, level=str(lev),
                             pct_kept=pct.loc[lev, "kept"], pct_dropped=pct.loc[lev, "dropped"],
                             cramers_v=V, p=p))
    miss = full[cfg["attrs"]].isna().mean().rename("share_missing").reset_index()
    return pd.DataFrame(rows), miss, int((~kept).sum()), int(kept.sum())


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_heatmap(final, cfg, path, title_suffix=""):
    import matplotlib.pyplot as plt
    supp = final[["supp_" + g for g in cfg["groups"]]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(final) + 1.4))
    im = ax.imshow(supp, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for i in range(supp.shape[0]):
        jmax, jmin = supp[i].argmax(), supp[i].argmin()
        for j in range(supp.shape[1]):
            txt = f"{supp[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if supp[i, j] > 0.55 else "black",
                    fontweight="bold" if j in (jmax, jmin) else "normal")
        ax.add_patch(plt.Rectangle((jmax - .5, i - .5), 1, 1, fill=False, lw=1.6, ec="black"))
        ax.add_patch(plt.Rectangle((jmin - .5, i - .5), 1, 1, fill=False, lw=1.2, ec="black", ls="--"))
    ax.set_yticks(range(len(final)))
    ax.set_yticklabels(final["label"], fontsize=8)
    ax.set_xticks(range(len(cfg["groups"])))
    ax.set_xticklabels([GROUP_NAMES[g] for g in cfg["groups"]], fontsize=8, rotation=30, ha="right")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Support (share of group)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_forest(ovr, path):
    """One row per final pattern: one-vs-rest support difference with 95% CI
    for the group with the largest |difference|."""
    import matplotlib.pyplot as plt
    best = ovr.loc[ovr.groupby("pattern")["diff"].apply(lambda s: s.abs().idxmax())]
    best = best.sort_values("diff")
    fig, ax = plt.subplots(figsize=(3.5, 0.25 * len(best) + 0.9))
    y = np.arange(len(best))
    ax.errorbar(best["diff"], y, xerr=[best["diff"] - best["diff_lo"], best["diff_hi"] - best["diff"]],
                fmt="o", color="black", ms=3, capsize=2, lw=1)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} [{GROUP_NAMES[g]}]" for p, g in zip(best["pattern"], best["group"])], fontsize=6)
    ax.set_xlabel("Group minus rest (95% CI)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_age(r_u65, r_65, path, top=10):
    import matplotlib.pyplot as plt
    a = r_u65.set_index("label")["delta"]
    b = r_65.set_index("label")["delta"]
    labs = (pd.concat([a, b], axis=1).max(axis=1).sort_values(ascending=False).index[:top])
    fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(labs) + 0.9))
    y = np.arange(len(labs))
    ax.barh(y - 0.18, [a.get(l, 0) for l in labs], 0.36, color="0.25", label="Under 65")
    ax.barh(y + 0.18, [b.get(l, 0) for l in labs], 0.36, color="0.7", hatch="///", label="65 and over")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ (max minus min group support)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--full", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=CFG["boot_B"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = CFG
    rng = np.random.default_rng(SEED)
    out = lambda f: os.path.join(args.out, f)
    summary = {}

    df = pd.read_csv(args.data, low_memory=False)
    for a in cfg["attrs"]:
        df[a] = df[a].map(norm_code)
    df = df[df[cfg["group"]].isin(cfg["groups"])].reset_index(drop=True)
    print(f"Loaded {len(df):,} rows")
    dw = Design(df, cfg, weighted=True)
    du = Design(df, cfg, weighted=False)
    summary["n_analysis"] = len(df)
    summary["n_psu"], summary["n_strata"] = int(dw.n_psu), int(dw.n_str)

    print("E0 group table / dropout")
    group_table(df, cfg).to_csv(out("E0_group_sizes.csv"), index=False)
    if args.full:
        full = pd.read_csv(args.full, low_memory=False)
        dd, miss, n_drop, n_keep = dropout_check(full, cfg)
        dd.to_csv(out("E0_dropout_check.csv"), index=False)
        miss.to_csv(out("E0_missing_share.csv"), index=False)
        summary.update(n_dropped=n_drop, n_kept=n_keep)

    print("E7 Cramer's V")
    cramers_table(df, cfg, dw, du).to_csv(out("E7_cramers_v.csv"), index=False)

    print("E1 mining, unweighted and weighted")
    ru = mine(df, cfg, weighted=False)
    rw = mine(df, cfg, weighted=True, design=dw)
    summary["n_candidates_generated"] = ru.attrs["n_candidates_generated"]
    summary["n_candidates_after_min_count"] = {"1-item": int((ru["length"] == 1).sum()),
                                               "2-item": int((ru["length"] == 2).sum())}
    summary["n_significant_unweighted"] = int(ru["significant"].sum())
    summary["n_significant_weighted"] = int(rw["significant"].sum())
    summary["n_holm_only_unweighted"] = int((ru["p_holm"] < cfg["alpha"]).sum())
    ru.drop(columns="pattern").to_csv(out("E1_candidates_unweighted.csv"), index=False)
    rw.drop(columns="pattern").to_csv(out("E1_candidates_weighted.csv"), index=False)

    print("E3 pruning")
    pu = prune(ru, df, cfg, weighted=False)
    pw = prune(rw, df, cfg, weighted=True)
    pu.drop(columns="pattern").to_csv(out("E3_pruning_unweighted.csv"), index=False)
    pw.drop(columns="pattern").to_csv(out("E3_pruning_weighted.csv"), index=False)
    final_u = pu[pu["kept"]].sort_values("delta", ascending=False)
    final_w = pw[pw["kept"]].sort_values("delta", ascending=False)
    summary["n_final_unweighted"] = int(len(final_u))
    summary["n_final_weighted"] = int(len(final_w))
    summary["n_final_pairs_weighted"] = int((final_w["length"] == 2).sum())
    summary["pruning_reasons_weighted"] = pw["drop_reason"].value_counts().to_dict()
    # agreement between weighted and unweighted results
    ku = set(final_u["label"]); kw = set(final_w["label"])
    summary["final_overlap_jaccard"] = len(ku & kw) / max(len(ku | kw), 1)
    mg = final_u.set_index("label")["max_group"]
    mgw = rw.set_index("label")["max_group"]
    summary["same_max_group_weighted_vs_unweighted"] = int(sum(mgw.get(l) == g for l, g in mg.items()))

    print("E2 one-vs-rest")
    ovr = pd.concat([one_vs_rest(df, cfg, p, design=dw) for p in final_w["pattern"]], ignore_index=True)
    ovr.to_csv(out("E2_one_vs_rest_weighted.csv"), index=False)
    best = ovr.loc[ovr.groupby("pattern")["diff"].apply(lambda s: s.abs().idxmax())]
    best.to_csv(out("E2_best_group_per_pattern.csv"), index=False)
    # how often the max-support group differs from the one-vs-rest group
    lab2max = final_w.set_index("label")["max_group"]
    lab2min = final_w.set_index("label")["min_group"]
    summary["ovr_group_equals_max_group"] = int(sum(lab2max[r.pattern] == r.group for r in best.itertuples()))
    summary["ovr_group_equals_min_group"] = int(sum(lab2min[r.pattern] == r.group for r in best.itertuples()))

    print("E4 sensitivity")
    rows = []
    base = set(final_w["label"])
    for dlt in cfg["sens_delta"]:
        for mc in cfg["sens_min_count"]:
            r = mine(df, cfg, weighted=True, design=dw, delta=dlt, min_count=mc)
            p = prune(r, df, cfg, weighted=True)
            k = p[p["kept"]]
            labs = set(k["label"])
            rows.append(dict(delta=dlt, min_count=mc, n_significant=int(r["significant"].sum()),
                             n_final=len(k), n_final_pairs=int((k["length"] == 2).sum()),
                             jaccard_vs_base=len(labs & base) / max(len(labs | base), 1)))
    pd.DataFrame(rows).to_csv(out("E4_sensitivity.csv"), index=False)

    print(f"E5 bootstrap (B={args.boot})")
    bs = bootstrap_stability(df, cfg, final_w, weighted=True, B=args.boot, rng=rng)
    bs.to_csv(out("E5_bootstrap_stability.csv"), index=False)

    print("E6 baselines")
    mn, r2 = baseline_multinomial(df, cfg)
    mn.to_csv(out("E6_multinomial_logit.csv"), index=False)
    summary["multinomial_mcfadden_r2"] = r2
    wr = baseline_wracc(df, cfg, weighted=True)
    wr.to_csv(out("E6_wracc_top10.csv"), index=False)
    summary["wracc_top10_pairs"] = int((wr["length"] == 2).sum())
    summary["wracc_top_overlap_with_final"] = int(wr["pattern"].isin(final_w["label"]).sum())

    print("E8 circularity")
    present = {k: v for k, v in cfg["conditions"].items() if v in df}
    if present:
        rows = []
        for name, col in present.items():
            for gr in cfg["groups"]:
                k = df[cfg["group"]] == gr
                rows.append(dict(condition=name, group=gr,
                                 pct=100 * df.loc[k, col].mean(),
                                 pct_weighted=100 * np.average(df.loc[k, col], weights=df.loc[k, cfg["weight"]])))
        pd.DataFrame(rows).pivot(index="condition", columns="group", values="pct_weighted") \
          .reindex(columns=cfg["groups"]).to_csv(out("E8_conditions_by_group.csv"))
    no_out = [a for a in cfg["attrs"] if a not in cfg["outcome_attrs"]]
    r8 = mine(df, cfg, attrs=no_out, weighted=True, design=dw)
    p8 = prune(r8, df, cfg, weighted=True)
    p8[p8["kept"]].drop(columns="pattern").to_csv(out("E8_final_without_outcome_attrs.csv"), index=False)
    summary["n_final_without_outcome_attrs"] = int(p8["kept"].sum())

    print("E9 age-stratified (same pipeline, age removed)")
    no_age = [a for a in cfg["attrs"] if a != cfg["age_attr"]]
    u65 = df[df[cfg["age_attr"]].isin(cfg["under65_codes"])].reset_index(drop=True)
    o65 = df[~df[cfg["age_attr"]].isin(cfg["under65_codes"])].reset_index(drop=True)
    res_age = {}
    for name, sub in [("under65", u65), ("65plus", o65)]:
        dsub = Design(sub, cfg, weighted=True)
        r = mine(sub, cfg, attrs=no_age, weighted=True, design=dsub)
        p = prune(r, sub, cfg, weighted=True)
        k = p[p["kept"]]
        k.drop(columns="pattern").to_csv(out(f"E9_final_{name}.csv"), index=False)
        summary[f"age_{name}_n"] = len(sub)
        summary[f"age_{name}_significant_before_pruning"] = int(r["significant"].sum())
        summary[f"age_{name}_final_after_pruning"] = int(len(k))
        res_age[name] = r[r["significant"]]
    fig_age(res_age["under65"], res_age["65plus"], out("fig_age_comparison.png"))

    print("E10 single-condition groups only")
    r10 = mine(df, cfg, groups=cfg["single_condition_groups"], weighted=True)
    p10 = prune(r10, df, cfg, groups=cfg["single_condition_groups"], weighted=True)
    p10[p10["kept"]].drop(columns="pattern").to_csv(out("E10_single_condition_groups.csv"), index=False)
    summary["n_final_single_condition_groups"] = int(p10["kept"].sum())

    print("Synergy recount (explains 793 vs 828)")
    pairs = ru[ru["length"] == 2]
    summary["pairs_generated"] = int(sum(1 for c in candidates(build_items(df, cfg["attrs"]), 2) if len(c) == 2))
    summary["pairs_after_min_count"] = int(len(pairs))

    print("Figures")
    fig_heatmap(final_w, cfg, out("fig_heatmap_weighted.png"))
    fig_forest(ovr, out("fig_forest_one_vs_rest.png"))
    final_w.drop(columns="pattern").to_csv(out("final_patterns_weighted.csv"), index=False)

    with open(out("summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
