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
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'behavior_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

coefdf = as.data.frame(model) %>%
    mutate(
        global_hr = typehr + `(Intercept)`,
        object_hr = typehr + `(Intercept)` + conditionobject + `conditionobject:typehr`,
        spatial_hr = typehr + `(Intercept)` + conditionspatial + `conditionspatial:typehr`,
        global_fr = `(Intercept)`,
        object_fr = `(Intercept)` + conditionobject,
        spatial_fr = `(Intercept)` + conditionspatial,
    ) %>%
    select(global_hr:spatial_fr, `(phi)`) %>%
    mutate(sample = 1:n()) %>%
    gather(global_hr:spatial_fr, key = 'condition', value = 'value')

coefdf %>%
    filter(str_detect(as.character(condition), 'hr')) %>%
    group_by(condition) %>%
    effect_summary(value) %>%
    mutate(across(-condition,exp)) %>%
    print

coefdf %>%
    filter(str_detect(as.character(condition), 'fr')) %>%
    group_by(condition) %>%
    effect_summary(invlogit(value)) %>%
    print

pairdf = coefdf %>%
    select(-`(phi)`) %>%
    filter(str_detect(as.character(condition), 'hr')) %>%
    spread(condition, value) %>%
    select(-sample) %>%
    pairwise() %>%
    gather(everything(), key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(value) %>%
    mutate(across(-condition, function(x){exp(-x)})) %>%
    print

pairdf = coefdf %>%
    select(-`(phi)`) %>%
    filter(str_detect(as.character(condition), 'hr')) %>%
    spread(condition, value) %>%
    select(-sample) %>%
    pairwise() %>%
    gather(everything(), key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(value) %>%
    mutate(across(-condition, function(x){exp(x)})) %>%
    print

pairdf = coefdf %>%
    select(-`(phi)`) %>%
    filter(str_detect(as.character(condition), 'fr')) %>%
    spread(condition, value) %>%
    select(-sample) %>%
    pairwise() %>%
    gather(everything(), key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(value) %>%
    mutate(across(-condition, function(x){exp(-x)})) %>%
    print

df = read.csv(file.path(processed_datadir,'analyses','eeg_condition.csv'))

model = stan_glmer(shrinkmean ~ comparison + logitnullmean + (1 | sid),
    family = mgcv::betar,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

p = pp_check(model)
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'eeg_pp_check.svg'), p)

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

