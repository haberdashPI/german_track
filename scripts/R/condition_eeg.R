source("src/R/setup.R")

df = read.csv(file.path(processed_datadir,'analyses','eeg_condition.csv')) %>%
    mutate(meancor = invlogit(logit(shrinkmean) - logitnullmean))

model = stan_glmer(meancor ~ comparison * compare_hit +(1 | sid),
    family = mgcv::betar,
    weight = df$count,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

pl = ggplot(df, aes(x = logitnullmean, y = logit(shrinkmean))) + geom_point() +
    facet_wrap(compare_hit~comparison) +
    geom_abline(intercept = 0, slope = 1)
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'eeg_data.svg'), pl)

p = pp_check(model)
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'eeg_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

effects = as.data.frame(model) %>%
    mutate(
        global_v_object_all = `(Intercept)`,
        global_v_spatial_all = `(Intercept)` + `comparisonglobal-v-spatial`,
        object_v_spatial_all = `(Intercept)` + `comparisonobject-v-spatial`
    ) %>%
    mutate(
        global_v_object_hit = global_v_object_all + compare_hittrue,
        global_v_spatial_hit = global_v_spatial_all + `comparisonglobal-v-spatial:compare_hittrue`,
        object_v_spatial_hit = object_v_spatial_all + `comparisonobject-v-spatial:compare_hittrue`
    ) %>%
    select(global_v_object_all:object_v_spatial_hit, `(phi)`) %>%
    gather(global_v_object_all:object_v_spatial_hit, key = 'comparison', value = 'value') %>%
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

# as an extra double check
df %>% filter(comparison == 'object-v-spatial', compare_hit == 'true') %>%
    {t.test(.$shrinkmean, .$meancor, paired = T)}

df %>% filter(comparison == 'object-v-spatial', compare_hit == 'true') %>%
    {wilcox.test(.$shrinkmean, .$meancor, paired = T)}
