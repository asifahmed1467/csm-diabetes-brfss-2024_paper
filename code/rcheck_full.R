# R cross-check 2: same shuffled data, but the variance uses the FULL file (domain estimation: removed adults stay in the design).
suppressMessages(library(survey)); options(survey.lonely.psu = "adjust")
d <- read.csv("rcheck_data.csv"); dr <- read.csv("rcheck_dropped.csv")
n <- 40; cols <- paste0("x", 0:(n - 1))
dr$gp <- NA; for (c in cols) dr[[c]] <- 0
a <- rbind(d[, c("w", "st", "gp", cols)], dr[, c("w", "st", "gp", cols)])
a$gpf <- factor(a$gp); for (c in cols) a[[paste0("f", c)]] <- factor(a[[c]])
des <- svydesign(ids = ~1, strata = ~st, weights = ~w, data = a); sub <- subset(des, !is.na(gp))
res <- data.frame(n = 0:(n - 1), R_full_perm_RS1 = NA, R_full_perm_F = NA)
for (i in 0:(n - 1)) {
  fp <- as.formula(paste0("~fx", i, "+gpf"))
  res$R_full_perm_RS1[i + 1] <- svychisq(fp, sub, statistic = "Chisq")$p.value
  res$R_full_perm_F[i + 1]   <- svychisq(fp, sub, statistic = "F")$p.value
  if (i %% 10 == 9) { write.csv(res[1:(i + 1), ], "rcheck_R_full.csv", row.names = FALSE); cat(i + 1, "of", n, "\n") }
}
