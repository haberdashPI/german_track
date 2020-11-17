source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(gamm4)

options(mc.cores = parallel::detectCores())

df = read.csv(file.path(processed_datadir,'analyses','eeg_condition.csv'))

model = stan_glmer(shrinkmean ~ comparison + (1 + logitnullmean | sid),
    family = mgcv::betar,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    adapt_delta = 0.99,
    iter = 4000)

coefs = as.data.frame(model) %>%
    mutate(global_v_object = `(Intercept)`,
           global_v_spatial = `(Intercept)` + `comparisonglobal-v-spatial`,
           object_v_spatial = `(Intercept)` + `comparisonobject-v-spatial`) %>%
    gather(global_v_object:object_v_spatial, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(value)

write.csv(coefs, file.path(processed_datadir, 'analyses', 'eeg_condition_coefs.csv'))
