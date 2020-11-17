source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)
library(gamm4)
library(purrr)

options(mc.cores = parallel::detectCores())

df = read.csv(file.path(processed_datadir,'analyses','behavioral_condition.csv'))
df$prop = 0.99*(df$prop - 0.5) + 0.5

model = stan_glmer(prop ~ condition * type + (1 | sid),
    family = mgcv::betar,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

p = pp_check(model)
ggsave(file.path(plot_dir, 'condition', 'behavior_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

coefs = as.data.frame(model) %>%
    mutate(gvo_hr = typehr + conditionobject + `conditionobject:typehr`,
           gvs_hr = typehr + conditionspatial + `conditionspatial:typehr`,
           ovs_hr = conditionobject - conditionspatial +
                `conditionobject:typehr` - `conditionspatial:typehr`,
           gvo_fr = conditionobject,
           gvs_fr = conditionspatial,
           ovs_fr = conditionobject - conditionspatial) %>%
    gather(gvo_hr:ovs_fr, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    summarize(
        mean = mean(value),
        meanint05 = posterior_interval(matrix(value))[,1],
        meanint95 = posterior_interval(matrix(value))[,2],
        pval = pd_to_p(p_direction(value))[[1]],
        d = mean(value / `(phi)`),
        dint05 = posterior_interval(matrix(value / `(phi)`))[,1],
        dint95 = posterior_interval(matrix(value / `(phi)`))[,2],
    )
knitr::kable(coefs, digits = 3)
print(coefs)

df = read.csv(file.path(processed_datadir,'analyses','eeg_condition.csv'))

model = stan_glmer(shrinkmean ~ comparison + (1 + logitnullmean | sid),
    family = mgcv::betar,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

p = pp_check(model)
ggsave(file.path(plot_dir, 'condition', 'eeg_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

coefs = as.data.frame(model) %>%
    mutate(global_v_object = `(Intercept)`,
           global_v_spatial = `(Intercept)` + `comparisonglobal-v-spatial`,
           object_v_spatial = `(Intercept)` + `comparisonobject-v-spatial`) %>%
    gather(global_v_object:object_v_spatial, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    summarize(
        mean = mean(value),
        meanint05 = posterior_interval(matrix(value))[,1],
        meanint95 = posterior_interval(matrix(value))[,2],
        pval = pd_to_p(p_direction(value))[[1]],
        d = mean(value / `(phi)`),
        dint05 = posterior_interval(matrix(value / `(phi)`))[,1],
        dint95 = posterior_interval(matrix(value / `(phi)`))[,2],
    )
knitr::kable(coefs, digits = 3)
print(coefs)

