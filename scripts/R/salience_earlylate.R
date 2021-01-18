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

df = read.csv(file.path(processed_datadir,'analyses','eeg_salience_earlylate.csv')) %>%
    mutate(meancor = invlogit(logitmean - logitnullmean))

fitmm = stan_glmer(meancor ~ condition * target_time_label + (1 | sid),
    data = df,
    family = mgcv::betar,
)

effects = as.data.frame(fitmm) %>%
    mutate(
        global_early = `(Intercept)`,
        object_early = global_early + conditionobject,
        spatial_early = global_early + conditionspatial,
    ) %>%
    mutate(
        global_late = global_early + target_time_labellate,
        object_late = global_early + `conditionobject:target_time_labellate`,
        spatial_late = global_early + `conditionspatial:target_time_labellate`,
    ) %>%
    mutate(
        global_diff = global_early - global_late,
        object_diff = object_early - object_late,
        spatial_diff = spatial_early - spatial_late,
    )

logitnullmean = df$logitnullmean %>% mean
table = effects %>%
    select(global_early:spatial_late) %>%
    gather(global_early:spatial_late, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = invlogit(value + logitnullmean))

table %>% write.csv(file.path(processed_datadir, 'analyses', 'eeg_salience_earlylate_coefs.csv'))
table %>% effect_table()

