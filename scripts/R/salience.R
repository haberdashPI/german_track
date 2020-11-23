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
# use a subset of the[data, because the full data set is too big to
# test out different modeling approaches
usestart = df$winstart %>% unique %>% sort %>% .[seq(1,length(.),by=3)]
dfsubset = filter(df, winstart %in% usestart, fold == 1)

pl = ggplot(df, aes(x = winstart, y = mean, color = condition)) +
    stat_summary(fun = mean, geom='line') +
    stat_smooth(fun = mean, geom='line', method='lm')
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'lineplot.svg'))


fit_lin1 = stan_glm(mean ~ winstart:condition, family = binomial(link = "logit"),
    weights = count, data = dfsubset, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_lin1))))
p = pp_check(fit_lin1)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck.svg'), p)

fit_lin1.5 = stan_glm(mean ~ winstart:condition + logitnullmean, family = binomial(link = "logit"),
    weights = count, data = dfsubset, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_lin1.5))))
p = pp_check(fit_lin1.5)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck1.5.svg'), p)

fit_lin2.0 = stan_glmer(mean ~ winstart:condition + (1 | sid),
    family = binomial(link = "logit"),
    weights = count, data = dfsubset, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_lin2.0))))
p = pp_check(fit_lin2.0)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck2.0.svg'), p)

fit_lin2 = stan_glmer(mean ~ winstart:condition + logitnullmean + (1 | sid),
    family = binomial(link = "logit"), weights = count,
    data = dfsubset, iter = 6000)
posterior_interval(matrix(c(predictive_error(fit_lin2))))
p = pp_check(fit_lin2)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck2.svg'), p)

coefs = as.data.frame(fit_lin2) %>%
    select(contains('winstart:condition')) %>%
    gather(contains('winstart:condition'), key = 'condition', value = 'value') %>%
    mutate(condition = str_replace(condition, 'winstart:condition([a-z]+)', '\\1')) %>%
    group_by(condition) %>%
    effect_summary(value)
knitr::kable(coefs, digits = 3)
print(coefs)

coefs = as.data.frame(fit_lin2) %>%
    select(contains('winstart:condition')) %>%
    pairwise() %>%
    gather(contains('winstart:condition'), key = 'compare', value = 'value') %>%
    mutate(compare = str_replace_all(compare, 'winstart:condition([a-z]+)', '\\1')) %>%
    group_by(compare) %>%
    effect_summary(value)
knitr::kable(coefs, digits = 3)
print(coefs)

# the gamm approach doesn't seme to work: I think because the non-linear trend
# varies by listener, and it assumes the non-linear trend is followed by everyone
# so, we are just going to bin the data and treate time as a categorical variable
dfbin = dfsubset %>% mutate(winbin = cut(winstart, 5))

fit_bin = stan_glmer(mean ~ winbin*condition + (1 | sid),
    family = binomial(link = "logit"), weights = count,
    data = dfbin, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_bin))))
p = pp_check(fit_bin)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck_bin.svg'), p)

fit_bin1.25 = stan_glmer(mean ~ winbin*condition + logitnullmean + (1 | sid),
    family = binomial(link = "logit"), weights = count,
    data = dfbin, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_bin1.25))))
p = pp_check(fit_bin1.25)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck_bin1.25.svg'), p)

fit_bin1.5 = stan_glmer(mean ~ winbin*condition + (winbin | sid),
    family = binomial(link = "logit"), weights = count,
    data = dfbin, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_bin1.5))))
p = pp_check(fit_bin1.5)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck_bin1.5.svg'), p)

fit_bin2 = stan_glmer(mean ~ winbin*condition + logitnullmean + (winbin | sid),
    family = binomial(link = "logit"), weights = count,
    data = dfbin, iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_bin2))))
p = pp_check(fit_bin2)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck_bin2.svg'), p)


dfbinfull = df %>% mutate(winbin = cut(winstart, 5))

fit_bin2_full = stan_glmer(mean ~ winbin*condition + logitnullmean + (winbin | sid),
    family = binomial(link = "logit"), weights = count,
    data = dfbinfull, iter = 3000, adapt_delta = 0.99)
posterior_interval(matrix(c(predictive_error(fit_bin2_full))))
p = pp_check(fit_bin2_full)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_modelcheck_bin2_full.svg'), p)

nullslope = as.data.frame(fit_bin2_full)$logitnullmean %>% mean

grandnull = df %>%
    summarize(logitnullmean = mean(logitnullmean)) %>%
    {mean(.$logitnullmean)}

df %<>% mutate(corrected_mean = invlogit(logit(mean) - nullslope*logitnullmean + grandnull))

write.csv(df,
    file.path(processed_datadir, 'analyses', 'eeg_salience_timeline_correct.csv'))

# Other attempted model variants (they don't go anywhere fruitful, y_rep starts to look really wonky)
# and the model that don't look wonkey seem too unrealistic to me (no within-subject errors)
# -----------------------------------------------------------------

fit_spline = stan_gamm4(mean ~ s(winstart, by = condition),# + logitnullmean,
    # random = ~ (1 | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    data = dfsubset,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_spline))))
p = pp_check(fit_spline)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_splinecheck1.svg'), p)

fit_spline1.5 = stan_gamm4(mean ~ s(winstart, by = condition, bs = 'gp'),# + logitnullmean,
    # random = ~ (1 | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    data = dfsubset,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_spline1.5))))
p = pp_check(fit_spline1.5)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_spline1.5check1.svg'), p)

fit_spline2 = stan_gamm4(mean ~ s(winstart, by = condition) + logitnullmean,
    # random = ~ (1 | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    data = dfsubset,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_spline2))))
p = pp_check(fit_spline2)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_spline2.svg'), p)

fit_spline3 = stan_gamm4(mean ~ s(winstart, by = condition),
    random = ~ (1 | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    data = dfsubset,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_spline3))))
p = pp_check(fit_spline3)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_spline3.svg'), p)

fit_spline3.25 = stan_gamm4(mean ~ s(winstart, by = condition, bs = 'gp'),
    random = ~ (winstart | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    data = dfsubset,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_spline3.25))))
p = pp_check(fit_spline3.25)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_spline3.25.svg'), p)

fit_spline3.5 = stan_gamm4(mean ~ s(winstart, by = condition),
    random = ~ (1 + winstart | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    data = dfsubset,
    iter = 2000)
posterior_interval(matrix(c(predictive_error(fit_spline3.5))))
p = pp_check(fit_spline3.5)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_spline3.5.svg'), p)
p = plot_nonlinear(fit_spline3.5)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_nonlinear_spline3.5.svg'), p)

fit_spline4 = stan_gamm4(mean ~ s(winstart, by = condition) + logitnullmean,
    random = ~ (1 | sid),
    family = binomial(link = "logit"), weights = dfsubset$count,
    # family = mgcv::betar,
    # TODO: gamm4 has additional priors
    # prior = normal(0,2.5),
    # prior_intercept = normal(0,2.5),
    # prior_aux = exponential(autoscale = TRUE),
    # prior_covariance = decov(),
    data = dfsubset,
    iter = 2000)

# mu <- posterior_linpred(model, transform = TRUE)
# phi <- as.data.frame(model)$`(phi)`
# PPD <- matrix(rbeta(prod(dim(mu)), shape1 = mu * phi, shape2 = (1 - mu) * phi),
#                          nrow = nrow(mu), ncol = ncol(mu))
# dimnames(PPD) <- dimnames(mu)
# posterior_interval(matrix(c(PPD)))
posterior_interval(matrix(c(predictive_error(fit_spline))))
p = pp_check(fit_spline)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'eeg_salience_fit_splinecheck3.svg'), p)

p = plot_nonlinear(fit_spline)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'splinenonlienar.svg'))

draws = posterior_predict(fit_spline)

drawdf = dfsubset %>% mutate(rowid = 1:n()) %>%
    group_by(winstart, condition) %>%
    summarize(
        predict = mean(draws[,rowid]),
        pi = posterior_interval(rray_mean(draws[,rowid], axes = 2)))

pl = ggplot(drawdf, aes(x = winstart, color = condition, y = predict)) +
    geom_ribbon(aes(ymin = pi[,1], ymax = pi[,2])) +
    geom_line()
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'splinepredict.svg'))


# FULL DATASET VERSION (does it help?)

model = stan_gamm4(mean ~ s(winstart, by = condition, bs = 'gp'),
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
