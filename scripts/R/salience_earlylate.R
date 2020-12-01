source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)

options(mc.cores = parallel::detectCores())

# df = read.csv(file.path(processed_datadir,'analyses','behavioral_condition.csv'))
# df$prop = 0.99*(df$prop - 0.5) + 0.5

# model = stan_glmer(prop ~ condition * type + (1 | sid),
#     family = mgcv::betar,
#     prior = normal(0,2.5),
#     prior_intercept = normal(0,2.5),
#     prior_aux = exponential(autoscale = TRUE),
#     prior_covariance = decov(),
#     data = df,
#     iter = 2000)

# p = pp_check(model)
# ggsave(file.path(plot_dir, 'condition', 'behavior_pp_check.svg'), p)

# posterior_interval(matrix(c(predictive_error(model))))

# coefs = as.data.frame(model) %>%
#     mutate(gvo_hr = typehr + conditionobject + `conditionobject:typehr`,
#            gvs_hr = typehr + conditionspatial + `conditionspatial:typehr`,
#            ovs_hr = conditionobject - conditionspatial +
#                 `conditionobject:typehr` - `conditionspatial:typehr`,
#            gvo_fr = conditionobject,
#            gvs_fr = conditionspatial,
#            ovs_fr = conditionobject - conditionspatial) %>%
#     gather(gvo_hr:ovs_fr, key = 'comparison', value = 'value') %>%
#     group_by(comparison) %>%
#     summarize(
#         mean = mean(value),
#         meanint05 = posterior_interval(matrix(value))[,1],
#         meanint95 = posterior_interval(matrix(value))[,2],
#         pval = pd_to_p(p_direction(value))[[1]],
#         d = mean(value / `(phi)`),
#         dint05 = posterior_interval(matrix(value / `(phi)`))[,1],
#         dint95 = posterior_interval(matrix(value / `(phi)`))[,2],
#     )
# knitr::kable(coefs, digits = 3)
# print(coefs)

df = read.csv(file.path(processed_datadir,'analyses','eeg_salience_earlylate.csv'))
bad_sids = df %>% group_by(sid) %>% summarize(bad = any(abs(logitnullmean) > 2.9))

fit_lin1 = stan_glm(mean ~ condition*target_time_label,
    family = binomial(link = "logit"),
    weight = count,
    data = df,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_lin1))))
p = pp_check(fit_lin1)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck_fit_lin1.svg'), p)

# prior_mean = rep(0, 6)
# prior_mean[4] = 1
# prior_scale = rep(2.5, 6)
# prior_scale[4] = 0.1
fit_lin2 = stan_glm(mean ~ condition*target_time_label + logitnullmean,
    family = binomial(link = "logit"),
    weight = count,
    # prior = normal(location = prior_mean, scale = prior_scale),
    data = df %>% filter(abs(logitnullmean) < 2.5),
    iter = 2000)

nullslope = as.data.frame(fit_lin2)$logitnullmean %>% mean
intercept = as.data.frame(fit_lin2) %>%
    mutate(intercept = (3*`(Intercept)`+conditionobject+conditionspatial)/3) %>%
    {mean(.$intercept)}

shrink = function(x) { 0.99*(x-0.5) + 0.5 }
pl = df %>% filter(abs(logitnullmean) < 2.5) %>%
    ggplot(aes(x = logitnullmean, y = logit(shrink(mean)), color = target_time_label)) +
    facet_wrap(~condition) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1) +
    geom_abline(intercept = intercept, slope = nullslope, linetype = 2)
ggsave(file.path(plot_dir, 'figure5_parts', 'supplement', 'inddata.svg'), width = 8, height = 3)

posterior_interval(matrix(c(predictive_error(fit_lin2))))
p = pp_check(fit_lin2)
ggsave(file.path(plot_dir, 'figure5_parts', 'supplement', 'eeg_salience_modelcheck_fit_lin2.svg'), p)

fit_mm = stan_glmer(mean ~ condition*target_time_label + logitnullmean + (1 | sid),
    family = binomial(link = "logit"),
    weight = count,
    data = df %>% filter(abs(logitnullmean) < 2.5),
    iter = 4000)
posterior_interval(matrix(c(predictive_error(fit_mm))))
p = pp_check(fit_mm)
ggsave(file.path(plot_dir, 'figure5_parts', 'supplement', 'salience_early_late_check_fit_mm.svg'), p)

coefs = as.data.frame(fit_mm) %>%
    mutate(
        timeglobal = target_time_labellate,
        timeobject = target_time_labellate + `conditionobject:target_time_labellate`,
        timespatial = target_time_labellate + `conditionspatial:target_time_labellate`,
    ) %>%
    gather(timeglobal:timespatial, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(value)

coefs = as.data.frame(fit_mm) %>%
    mutate(
        timeglobal = target_time_labellate,
        timeobject = target_time_labellate + `conditionobject:target_time_labellate`,
        timespatial = target_time_labellate + `conditionspatial:target_time_labellate`,
    ) %>%
    gather(timeglobal:timespatial, key = 'condition', value = 'value') %>%
    mutate(percent = 100*(exp(value) - 1)) %>%
    group_by(condition) %>%
    effect_summary(percent)

print(coefs)

nullmean = mean(df$logitnullmean)
coefs = as.data.frame(fit_mm) %>%
    mutate(
        timeglobal = target_time_labellate,
        timeobject = target_time_labellate + `conditionobject:target_time_labellate`,
        timespatial = target_time_labellate + `conditionspatial:target_time_labellate`,
    ) %>%
    mutate(
        global_early = `(Intercept)`,
        object_early = `(Intercept)` + conditionobject,
        spatial_early = `(Intercept)` + conditionspatial,
        global_late = `(Intercept)` + timeglobal,
        object_late = `(Intercept)` + conditionobject + timeobject,
        spatial_late = `(Intercept)` + conditionspatial + timespatial,
    ) %>%
    gather(global_early:spatial_late, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(value) %>%
    mutate(across(matches('(med|[0-9]+)'), ~invlogit(.x + nullmean))) %>%
    write.csv(file.path(processed_datadir, 'analyses', 'eeg_salience_earlylate_coefs.csv'))

