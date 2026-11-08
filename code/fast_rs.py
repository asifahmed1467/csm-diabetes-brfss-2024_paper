"""Vectorized design-based test for many binary patterns at once.

For each candidate pattern (a 0/1 column), tests H0: equal weighted support in all
K groups. Uses the same statistic as csm_upgrade.wald_homogeneity with J = 1:
Taylor-linearized covariance of the K domain proportions under a stratified
with-replacement PSU design, and the first-order Rao-Scott correction
RS1 = X2_srs / mean_deff, referred to chi2 with K-1 df.
Used for the main analysis check and for every bootstrap replicate (M2)."""
import numpy as np
from scipy import stats


def design_arrays(strata, psu):
    """Return PSU index per row, stratum index per PSU, and n_h."""
    import pandas as pd
    key = pd.Series(strata).astype(str) + "|" + pd.Series(psu).astype(str)
    psu_idx, keys = pd.factorize(key)
    str_of_psu, _ = pd.factorize(pd.Series(keys).str.split("|").str[0])
    nh = np.bincount(str_of_psu)
    return psu_idx, str_of_psu, nh


def rs1_batch(Y, gidx, K, w, psu_idx, str_of_psu, nh, chunk=64):
    """Y: n x C (0/1). gidx: group code 0..K-1 per row. Returns supports (C x K),
    RS1 statistic, p-value and mean design effect per candidate."""
    n, C = Y.shape
    n_psu, H = len(str_of_psu), len(nh)
    f = np.where(nh > 1, nh / np.maximum(nh - 1, 1), 0.0)
    sqf = np.sqrt(f)[str_of_psu]                      # per PSU
    Wk = np.bincount(gidx, weights=w, minlength=K)
    nk = np.bincount(gidx, minlength=K)
    q = K - 1
    Cm = np.zeros((q, K)); Cm[np.arange(q), np.arange(q)] = 1; Cm[:, K - 1] = -1
    P_all = np.zeros((C, K)); rs1 = np.zeros(C); deff = np.zeros(C)
    for s in range(0, C, chunk):
        Yc = Y[:, s:s + chunk].astype(float); c = Yc.shape[1]
        P = np.zeros((K, c)); D = []
        for k in range(K):
            m = gidx == k
            P[k] = (w[m] @ Yc[m]) / Wk[k]
            U = np.zeros((n, c))
            U[m] = (w[m, None] * (Yc[m] - P[k])) / Wk[k]
            T = np.zeros((n_psu, c)); np.add.at(T, psu_idx, U)       # PSU totals
            S = np.zeros((H, c)); np.add.at(S, str_of_psu, T)
            T -= (S / nh[:, None])[str_of_psu]                         # centre in stratum
            D.append(T * sqf[:, None])
        V = np.zeros((c, K, K))
        for k in range(K):
            for l in range(k, K):
                v = np.einsum("ij,ij->j", D[k], D[l]); V[:, k, l] = v; V[:, l, k] = v
        Vs = np.zeros((c, K, K))
        for k in range(K):
            Vs[:, k, k] = P[k] * (1 - P[k]) / nk[k]
        cp = (Cm @ P).T                                                # c x q
        CVC = Cm @ V @ Cm.T
        CVsC = Cm @ Vs @ Cm.T
        iS = np.linalg.pinv(CVsC)
        Ws = np.einsum("ci,cij,cj->c", cp, iS, cp)
        dm = np.trace(iS @ CVC, axis1=1, axis2=2) / q
        P_all[s:s + c] = P.T; rs1[s:s + c] = Ws / np.maximum(dm, 1e-12); deff[s:s + c] = dm
    return P_all, rs1, stats.chi2.sf(rs1, q), deff
