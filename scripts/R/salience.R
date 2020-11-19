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

# model = stan_gamm4(shrinkmean ~ s(winstart, by = condition),
#     random = ~ (1 + logitnullmean | sid),
#     family = mgcv::betar,
#     # TODO: gamm4 has additional priors
#     # prior = normal(0,2.5),
#     # prior_intercept = normal(0,2.5),
#     # prior_aux = exponential(autoscale = TRUE),
#     # prior_covariance = decov(),
#     data = df,
#     adapt_delta = 0.99,
#     iter = 2000)

model = stan_glm(shrinkmean ~ winstart:condition, family = mgcv::betar, data = df, iter = 2000)

model = stan_glmer(shrinkmean ~ winstart:condition + (logitnullmean + winstart:condition | sid),
    data = df, iter = 2000)

# TODO: stopped here

pl = ggplot(df, aes(x = winstart, y = logit(shrinkmean), color = logitnullmean)) +
    stat_summary(fun = mean, geom='line') +
    stat_smooth(fun = mean, geom='line', method='lm') +
    geom_point() +
    facet_wrap(~condition)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'lineplot.svg'))

# other models

model = stan_glmer(shrinkmean ~ winstart:condition +
    (logitnullmean + winstart:condition | sid),
    family = mgcv::betar, data = df, iter = 2000)

p = plot_nonlinear(model)
ggsave(file.path(plot_dir, 'figure3_parts', 'supplement', 'splinenonlienar.svg'))

mu <- posterior_epred(model)
phi <- as.data.frame(model)$`(phi)`
PPD <- matrix(rbeta(prod(dim(mu)), shape1 = mu * phi, shape2 = (1 - mu) * phi),
                         nrow = nrow(mu), ncol = ncol(mu))
posterior_interval(matrix(c(PPD)))

posterior_interval(matrix(c(predictive_error(model))))


## compute regions where the three splines are different from one another
## TODO: create a hypothetical participant at the mean(logitnullmean)
## and compute points where each condition differs from the other three

nullmean = df %>%
    group_by(sid,condition) %>% summarize(logitnullmean = first(logitnullmean)) %>%
    ungroup() %>%
    summarize(logitnullmean = mean(logitnullmean)) %>% first

times = sort(unique(df$winstart))
pdf = data.frame(
    winstart = rep(times,3),
    condition = rep(levels(df$condition), each = length(times)),
    logitnullmean = nullmean,
    sid = 0
)

draws = posterior_predict(model, newdata = pdf)

# WIP: this doesn't seem to be resulting in sensible values
# I need to plot the results and see if sid 'new' is giving something
# reasonable
result = data.frame()
for(t in times){
    object = draws[,pdf$condition == 'object' & pdf$winstart == t]
    global = draws[,pdf$condition == 'global' & pdf$winstart == t]
    spatial = draws[,pdf$condition == 'spatial' & pdf$winstart == t]
    vals = cbind(pmin(abs(object - global), abs(object - spatial)),
                pmin(abs(global - object), abs(global - spatial)),
                pmin(abs(spatial - object), abs(spatial - global)))
    dfvals = as.data.frame(vals)
    colnames(dfvals) = c('objectdiff','globaldiff','spatialdiff')
    result_ = dfvals %>%
        gather(objectdiff:spatialdiff, key = 'conditiondiff', value = 'value') %>%
        group_by(conditiondiff) %>%
        summarize(
            mean = mean(value),
            meanint05 = posterior_interval(matrix(value))[,1],
            meanint95 = posterior_interval(matrix(value))[,2],
            pval = pd_to_p(p_direction(value))[[1]],
            # d = mean(value / `(phi)`),
            # dint05 = posterior_interval(matrix(value / `(phi)`))[,1],
            # dint95 = posterior_interval(matrix(value / `(phi)`))[,2],
        ) %>%
        mutate(winstart = t)
    result = rbind(result, result_)
}
# TODO: how are we going to run the analysis for salience?
# pick two time points, run a regression? using exponential curves??? splines?
