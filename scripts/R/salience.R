source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)
library(gamm4)

options(mc.cores = parallel::detectCores())

df = read.csv(file.path(processed_datadir,'analyses','behavior_salience.csv'))
df$prop = 0.99*(df$prop - 0.5) + 0.5

model = stan_glmer(prop ~ condition * salience_label + (1 | sid),
    family = mgcv::betar,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

p = pp_check(model)
ggsave(file.path(plot_dir, 'category_salience', 'behavior_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

coefs = as.data.frame(model) %>%
    mutate(gvo_saldiff = `conditionobject:salience_labellow`,
           gvs_saldiff = `conditionspatial:salience_labellow`,
           ovs_saldiff = `conditionobject:salience_labellow` -
            `conditionspatial:salience_labellow`) %>%
    gather(gvo_saldiff:ovs_saldiff, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(value)
knitr::kable(coefs, digits = 3)
print(coefs)

df = read.csv(file.path(processed_datadir, 'analyses', 'eeg_salience_timeline.csv')) %>%
    filter(hittype == 'hit')

pl = ggplot(df, aes(x = winstart, y = mean, color = condition)) +
    stat_summary(fun = mean, geom='line') +
    stat_smooth(fun = mean, geom='line', method='lm') +
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'lineplot.svg'))

# use a subset of the[data, because the full data set is too big to
# test out different modeling approaches
usestart = df$winstart %>% unique %>% sort %>% .[seq(1,length(.),by=3)]
dfsubset = filter(df, winstart %in% usestart, fold == 1)

model = stan_glm(mean ~ winstart:condition, family = binomial(link = "logit"),
    weights = weight, data = dfsubset, iter = 2000)
posterior_interval(matrix(c(predictive_error(model))))
p = pp_check(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck.svg'), p)

model = stan_glmer(mean ~ winstart:condition + (1 + logitnullmean | sid),
    family = binomial(link = "logit"), weights = weight,
    data = dfsubset, iter = 2000)
posterior_interval(matrix(c(predictive_error(model))))
p = pp_check(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck2.svg'), p)

coefs = as.data.frame(model) %>%
    mutate(
        gvo_slope = `winstart:conditionglobal` - `winstart:conditionobject`,
        gvs_slope = `winstart:conditionglobal` - `winstart:conditionspatial`,
        ovs_slope = `winstart:conditionobject` - `winstart:conditionspatial`,
    ) %>%
    gather(gvo_slope:ovs_slope, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(value)
knitr::kable(coefs, digits = 3)
print(coefs)

# TODO: save the correction due to null model accuracy so we can make good plots for
# fig 3

# Failed models
# -----------------------------------------------------------------

# These didn't fit the data in a way that made sense, so I went with the simpler
# model above

model = stan_gamm4(mean ~ s(winstart, by = condition),
    random = ~ (1 + logitnullmean | sid),
    family = binomial(link = "logit"), weights = dfsubset$weight,
    # TODO: gamm4 has additional priors
    # prior = normal(0,2.5),
    # prior_intercept = normal(0,2.5),
    # prior_aux = exponential(autoscale = TRUE),
    # prior_covariance = decov(),
    data = dfsubset,
    iter = 2000)

posterior_interval(matrix(c(predictive_error(model))))
p = pp_check(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck3.svg'), p)

p = plot_nonlinear(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'splinenonlienar.svg'))

# FULL DATASET VERSION (does it help?)

model = stan_gamm4(mean ~ s(winstart, by = condition),
    random = ~ (1 + logitnullmean | sid),
    family = binomial(link = "logit"), weights = df$weight,
    # TODO: gamm4 has additional priors
    # prior = normal(0,2.5),
    # prior_intercept = normal(0,2.5),
    # prior_aux = exponential(autoscale = TRUE),
    # prior_covariance = decov(),
    data = df,
    # adapt_delta = 0.99,
    iter = 2000)

posterior_interval(matrix(c(predictive_error(model))))
p = pp_check(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck3.5.svg'), p)

p = plot_nonlinear(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'splinenonlienar1.5.svg'))

# within-subject slopes...

model = stan_gamm4(mean ~ s(winstart, by = condition),
    random = ~ (1 + logitnullmean + winstart:condition | sid),
    family = binomial(link = "logit"), weights = dfsubset$weight,
    # TODO: gamm4 has additional priors
    # prior = normal(0,2.5),
    # prior_intercept = normal(0,2.5),
    # prior_aux = exponential(autoscale = TRUE),
    # prior_covariance = decov(),
    data = dfsubset,
    iter = 2000)

posterior_interval(matrix(c(predictive_error(model))))
p = pp_check(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck4.svg'), p)

p = plot_nonlinear(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'splinenonlienar2.svg'))

plotdf = ggplot_build(p)
