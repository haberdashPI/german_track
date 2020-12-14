source("src/R/setup.R")

df = read.csv(file.path(processed_datadir,'analyses','eeg_condition.csv')) %>%
    mutate(meancor = invlogit(logit(shrinkmean) - logitnullmean))

model = stan_glmer(meancor ~ comparison + (1 | sid),
    family = mgcv::betar,
    weight = df$count,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

pl = ggplot(df, aes(x = logitnullmean, y = logit(shrinkmean))) + geom_point() +
    facet_wrap(~comparison)
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'eeg_data.svg'), pl)

p = pp_check(model)
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'eeg_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

effects = as.data.frame(model) %>%
    mutate(global_v_object = `(Intercept)`,
           global_v_spatial = `(Intercept)` + `comparisonglobal-v-spatial`,
           object_v_spatial = `(Intercept)` + `comparisonobject-v-spatial`) %>%
    select(global_v_object:object_v_spatial, `(phi)`) %>%
    gather(global_v_object:object_v_spatial, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = value, d = value / `(phi)`)

coeftable = effects %>%
    mutate(across(matches('r_[med0-9]+'), list(odds = exp), .names = '{.fn}{.col}'))
coeftable %>% effect_table()

logitnullmean = df$logitnullmean %>% mean
effects %>% mutate(across(matches('r_[med0-9]+'),
    list(prop = ~ invlogit(.x + logitnullmean)), .names = '{.fn}{.col}')) %>%
    select(comparison, contains('prop')) %>%
    write.csv(file.path(processed_datadir,'analyses','eeg_condition_coefs.csv'))

coeftable %>%
    select(comparison,
        value = oddsr_med, pi05 = oddsr_05, pi95 = oddsr_95, pd = d_pd, D = d_med) %>%
    effect_json('condition_eeg', comparison)
