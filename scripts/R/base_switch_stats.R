source("src/R/setup.R")
library(knitr)
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)

classifier = 'svm_radial'

df = read.csv(file.path(processed_datadir,'analyses',
    paste0('baseline-switch-by-hit_classifier=',classifier,'.csv')))
df$correct_mean = (df$correct_mean - 0.5)*0.99 + 0.5

model = stan_glmer(correct_mean ~ hit * switchclass * condition + (1 | sid),
    family = mgcv::betar, data = df)

coefnames = p_direction(model)[[1]]
pval = pd_to_p(p_direction(model)[[2]])
# Yes: p-values are terrible, so bad in fact that stan_glmer refuses to compute them for you
# Yes: we need to report them; that's the reality of the publication process right now
knitr::kable(cbind(
    round(summary(model)[coefnames,c(1,3)], digits = 2),
    "p-value" = ifelse(pval < 1e-3, "<1e-3", round(pval, digits=3)),
    "sig" = ifelse(pval <= 0.001,"***",
            ifelse(pval <= 0.01, "**",
            ifelse(pval <= 0.05, "*",
            ifelse(pval <= 0.1,  "~",""))))))

# TODO: create an ANOVA table here
