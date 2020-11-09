source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)
library(gamm4)

options(mc.cores = parallel::detectCores())

df = read.csv(file.path(processed_datadir,'analyses','behavioral_condition.csv'))
df$prop = 0.99*(df$prop - 0.5) + 0.5

model = stan_glmer(prop ~ condition * type + (1 | sid),
    family = mgcv::betar,
    data = df,
    iter = 2000)

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

# TODO: do a table of planned comparisons across the conditions for fr and hr
# TODO: we'll also need to compute effects sizes, and ideally we report mangitudse and
# uncertainties
