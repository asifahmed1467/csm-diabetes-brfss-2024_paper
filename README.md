# Anonymous code repository for double-blind review

Paper: *Survey-Aware Contrast Set Mining of Diabetes Multimorbidity Groups in BRFSS 2024* (under review). This repository is anonymous. The authors are not named here.

This repository holds **all the code** behind every number, table and figure in the final paper. The BRFSS data is **not** included.
Nothing here is tuned to match the paper: the notebook only runs the scripts and then compares **your** numbers with the numbers of the reference run (the numbers in the paper) (PASS or FAIL per row).

## How to verify (recommended)
1. Download the 2024 BRFSS file as CSV (457,670 rows) with the columns listed on top of the notebook.
2. Open `CSM_diabetes_BRFSS2024_full_reproduction_v2.ipynb`. Set `RAW_CSV` (use an `r"..."` string on Windows) in the first code cell.
3. Run all cells from top to bottom. Use `BOOT_B = 10` and `PERM_B = 10` for a quick test (the bootstrap and permutation rows then differ on purpose).
4. Read the PASS and FAIL tables in Sections 9, 10, 11 and 12. The notebook is the same as the `code/` folder: every `%%writefile` cell writes one file of `code/`.

Run time on one core: about 5 minutes for the main analysis, 34 minutes for the 200-sample bootstrap, about 17 minutes for the 100 permutations, about 25 minutes for the other Section 11 and 12 checks. Random seeds: 2026 (bootstrap, permutations), 11 (null-rate runs), 5 (R cross-check export).

## Files in `code/`
| File | Purpose |
|---|---|
| `build_data.py` | Builds the diabetes file and the analysis file from the raw BRFSS CSV |
| `make_7groups.py` | Splits the old "other single condition" group into Diab.+depression and Diab.+COPD |
| `csm_upgrade.py` | Main pipeline: candidates, design-based tests (Rao-Scott), Holm, gap filter, pruning R1 and R2, one-vs-rest effect sizes, sensitivity, baselines, figures |
| `fast_rs.py` | Fast vectorised version of the same Rao-Scott test (bootstrap, permutation, calibration) |
| `run7.py` | Runs the main pipeline with the seven groups |
| `boot7.py`, `boot_compare7.py` | 200-sample design-based bootstrap; Pearson vs design-based test on 20 samples |
| `gen7.py` | Tables and the two figures |
| `counts7.py`, `compare6v7.py`, `runtime7.py`, `legacy_check.py`, `paper_all.py` | Group counts, six-vs-seven comparison, run time, missing-code check, every number quoted in the paper |
| `run_income.py` | Missing income kept as its own level, pipeline rerun |
| `perm_test.py` | Permutation null check with the gap rule (0.07 patterns per shuffle against 78 real) |
| `two_cond_check.py` | The 9 "2 or more" patterns inside the group with exactly 2 extra conditions |
| `maxT.py` | **Permutation max-statistic test: the Perm. column of Table VI** (15 of 30 pass) |
| `ipw.py` | Reweighting check for missing data; Kish design effect; effective group sizes |
| `calib_by_groups.py` | Null rate with 3, 5 and 7 groups (7.7%, 12.5%, 24.9%) |
| `r2_sens.py` | Breslow-Day (R2) statistic divided by a design effect D, pipeline rerun |
| `run_cond_unknown.py` | Adults with a "don't know", refused or missing condition answer removed (28 of 30 patterns remain) |
| `rcheck_export.py`, `rcheck.R`, `rcheck_full.R`, `rcheck_summary.py` | Cross-check of the Rao-Scott tests in R (`survey` package, `svychisq`); needs R |

## Design choices and limits you should know about (all are stated or measured in the paper)
1. **Thresholds are practical choices**, not derived: gap 0.10, minimum count 30, R2 gap 0.02, Holm 0.05. Table VII shows other values. They were kept fixed through all review rounds.
2. **Complete-case analysis**: 27.3% of adults with diabetes are removed for a missing trait (income 18.7%). Reweighting changes no gap by more than 0.009; the missing-income rerun keeps 20 of the 30 patterns (the headline contrasts hold).
3. **Group definition**: a condition counts only if the answer is "yes". "Don't know", refused and missing answers on the 7 condition questions (5.1% of adults) therefore count as "no condition". `run_cond_unknown.py` removes these adults: 28 of the 30 patterns remain.
4. **The design-based test is too liberal** in the permutation null (23.7% of null p-values below 0.05). The cause is mainly small effective group sizes (COPD 79, CKD 103). This is why the Perm. column exists; the 15 patterns marked No are descriptive. The R `survey` package shows the same behaviour.
5. **Rule R2 (Breslow-Day) uses unweighted counts.** Dividing its statistic by a design effect of 2, 3 or 5.8 leaves 27, 21 or 20 patterns; all 18 single-value patterns stay in every case. The 12 pairs are the less certain part.
6. **Each respondent is its own sampling unit** in the public file (stratum plus `_PSU` is unique), so the variance is stratified with-replacement at the respondent level.
7. **One survey year, US data**, groups defined by the authors; the Diab.+depression result for poor mental health is built in by the group definition and is used only as a check of the method.
8. The code is deterministic given the seeds. If a number does not match, report it; do not adjust it.

## Folders
- `CSM_diabetes_BRFSS2024_full_reproduction_v2.ipynb`: the notebook that runs everything.
- `code/`: the same scripts as plain files.
- `reference_results/`: result files of the reference run that the paper reports (`results7`), so numbers can be compared without running the code.
- `reference_results_missing_income/`: result files of the missing-income rerun.
- The BRFSS data is not included. It is public at https://www.cdc.gov/brfss/annual_data/annual_2024.html (a CSV export of the file with the columns listed in the notebook is needed).

## Software
See `requirements.txt`. R (optional, only for the cross-check with the `survey` package) 4.2.1, `survey` 4.2.1.
