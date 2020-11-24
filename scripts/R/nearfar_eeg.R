source("src/R/setup.R")


shrink = function(x){0.99*(x-0.5) + 0.5}
df = read.csv(file.path(processed_datadir,'analyses','eeg_nearfar.csv')) %>%
    mutate(meancor = invlogit(logit(shrink(mean)) - logitnullmean))

model = stan_glmer(meancor ~ condition*target_time_label + (1 | sid),
    family = mgcv::betar,
    # weight = df$count,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

posterior_interval(matrix(c(predictive_error(model))))

p = pp_check(model)
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'eeg_pp_check.svg'), p)

p = ggplot(df, aes(x = logitnullmean, y = logit(shrink(mean)), color = target_time_label)) +
    geom_point() + facet_wrap(~condition)
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'eeg_data.svg'), p, width = 8, height = 3)

# Something is wrong with this data

# TODO: effects
