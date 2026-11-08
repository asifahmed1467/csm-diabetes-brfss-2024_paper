"""Sensitivity of rule R2 (Breslow-Day test) to survey design.
In the paper's pipeline the Breslow-Day statistic uses UNWEIGHTED counts. Here the statistic is divided by an assumed
design effect D before its p-value is computed (chi-square, 6 df), and the whole pipeline is rerun.
D = 1 must reproduce the paper (30 patterns, 12 pairs). Usage: python r2_sens.py D outdir"""
import sys; sys.path.insert(0, '.')
import csm_upgrade as C
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable
D = float(sys.argv[1]); out = sys.argv[2]
orig = StratifiedTable.test_equal_odds
class R: pass
def patched(self, adjust=False):
    r = orig(self, adjust=adjust); s = r.statistic / D; o = R(); o.statistic = s; o.pvalue = stats.chi2.sf(s, 6); o.df = 6; return o
StratifiedTable.test_equal_odds = patched
G7 = ["diabetes_only","diabetes_cvd","diabetes_ckd","diabetes_arthritis","diabetes_depression","diabetes_copd","diabetes_multi_2plus"]
C.CFG.update(groups=G7, single_condition_groups=G7[1:6])
C.GROUP_NAMES.update({"diabetes_depression": "Diab.+depression", "diabetes_copd": "Diab.+COPD"})
sys.argv = ["csm_upgrade.py", "--data", "analysis_7g.csv", "--full", "diabetes_all_7g.csv", "--out", out, "--boot", "2"]
C.main()
