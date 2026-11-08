# R cross-check 1: Rao-Scott tests from the 'survey' package on the same data (subset design, each adult its own PSU).
suppressMessages(library(survey)); options(survey.lonely.psu = "adjust")
d <- read.csv("rcheck_data.csv"); d$gf <- factor(d$g); d$gpf <- factor(d$gp)
n <- 80
for (i in 0:(n - 1)) d[[paste0("f", i)]] <- factor(d[[paste0("x", i)]])
des <- svydesign(ids = ~1, strata = ~st, weights = ~w, data = d)
res <- data.frame(n = 0:(n - 1), R_p_real_RS1 = NA, R_p_real_F = NA, R_p_perm_RS1 = NA, R_p_perm_F = NA)
for (i in 0:(n - 1)) {
  fx <- as.formula(paste0("~f", i, "+gf")); fp <- as.formula(paste0("~f", i, "+gpf"))
  res$R_p_real_RS1[i + 1] <- svychisq(fx, des, statistic = "Chisq")$p.value
  res$R_p_real_F[i + 1]   <- svychisq(fx, des, statistic = "F")$p.value
  res$R_p_perm_RS1[i + 1] <- svychisq(fp, des, statistic = "Chisq")$p.value
  res$R_p_perm_F[i + 1]   <- svychisq(fp, des, statistic = "F")$p.value
  if (i %% 10 == 9) { write.csv(res[1:(i + 1), ], "rcheck_R.csv", row.names = FALSE); cat(i + 1, "of", n, "\n") }
}
