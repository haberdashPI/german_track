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

effects = as.data.frame(model) %>%
    mutate(
        global_early = `(Intercept)`,
        object_early = `(Intercept)` + conditionobject,
        spatial_early = `(Intercept)` + conditionspatial
    ) %>%
    mutate(
        global_late = global_early + target_time_labellate,
        object_late = object_early + target_time_labellate + `conditionobject:target_time_labellate`,
        spatial_late = spatial_early + target_time_labellate + `conditionspatial:target_time_labellate`
    ) %>%
    mutate(
        globaldiff = global_early - global_late,
        objectdiff = object_early - object_late,
        spatialdiff = spatial_early - spatial_late
    ) %>%
    select(global_early:spatialdiff, `(phi)`) %>%
    gather(-`(phi)`, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(r = value, d = value / `(phi)`)

effects %>%
    mutate(across(matches('r_[med0-9]+'), list(odds = exp), .names = '{.fn}{.col}')) %>%
    effect_table()

logitnullmean = df$logitnullmean %>% mean

effects %>%
    filter(!str_detect(condition, 'diff')) %>%
    mutate(across(matches('r_[med0-9]+'), ~ invlogit(.x + logitnullmean))) %>%
    write.csv(file.path(processed_datadir, 'analyses', 'eeg_nearfar_coefs.csv'))

# TODO: effects
