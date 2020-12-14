source("src/R/setup.R")

df = read.csv(file.path(processed_datadir,'analyses','behavioral_condition.csv'))
df$prop = 0.99*(df$prop - 0.5) + 0.5

rfit = stan_glmer(prop ~ condition * type + (1 | sid),
    family = mgcv::betar,
    prior = normal(0,2.5),
    prior_intercept = normal(0,2.5),
    prior_aux = exponential(autoscale = TRUE),
    prior_covariance = decov(),
    data = df,
    iter = 2000)

posterior_interval(matrix(c(predictive_error(rfit))))

p = pp_check(rfit)
ggsave(file.path(plot_dir, 'figure2_parts', 'supplement', 'behavior_pp_check.svg'), p)

coefdf = as.data.frame(rfit) %>%
    mutate(
        global_hr = typehr + `(Intercept)`,
        object_hr = typehr + `(Intercept)` + conditionobject + `conditionobject:typehr`,
        spatial_hr = typehr + `(Intercept)` + conditionspatial + `conditionspatial:typehr`,
        global_fr = `(Intercept)`,
        object_fr = `(Intercept)` + conditionobject,
        spatial_fr = `(Intercept)` + conditionspatial,
    ) %>%
    select(global_hr:spatial_fr, `(phi)`) %>%
    pairwise(global_hr:spatial_hr, bothdir = T) %>%
    pairwise(global_fr:spatial_fr, bothdir = T) %>%
    gather(-`(phi)`, key = 'condition', value = 'value') %>%
    mutate(type = ifelse(str_detect(condition, 'fr'), 'fr', 'hr')) %>%
    mutate(condition = str_replace_all(condition, '([a-z]+)_(fr|hr)', '\\1'))

effects = coefdf %>%
    group_by(type, condition) %>%
    filter(!str_detect(condition, '-')) %>%
    effect_summary(r = value, d = value / `(phi)`) %>%
    arrange(desc(type), condition)

effects %>%
    mutate(across(matches('r_[med0-9]+'), list(odds = exp), .names = '{.fn}{.col}')) %>%
    effect_table()

effects %>% mutate(across(matches('r_[med0-9]+'),
    list(prop = invlogit), .names = '{.fn}{.col}')) %>%
    select(type, condition, contains('prop')) %>%
    write.csv(file.path(processed_datadir,'analyses','behavioral_condition_coefs.csv'))

coefdf %>%
    group_by(type, condition) %>%
    filter(str_detect(condition, '-')) %>%
    effect_summary(r = value, d = value / `(phi)`) %>%
    select(-r_pd) %>%
    mutate(across(matches('r_[med0-9]+'), list(odds = exp), .names = '{.fn}{.col}')) %>%
    arrange(desc(type), condition) %>%
    effect_table()

coeftable = coefdf %>%
    group_by(type, condition) %>%
    filter(str_detect(condition, '-')) %>%
    effect_summary(r = -value, d = -value / `(phi)`) %>%
    # select(-r_p) %>%
    mutate(across(matches('r_[med0-9]+'), list(odds = exp), .names = '{.fn}{.col}')) %>%
    arrange(desc(type), condition) %>%
    select(-matches('^r_')) %>% ungroup()

coeftable %>% effect_table()

coeftable %>%
    mutate(condition = str_replace(condition, ' - ', '_vs_')) %>%
    select(type, condition,
        value = oddsr_med, pi05 = oddsr_05, pi95 = oddsr_95, pd = d_pd, D = d_med) %>%
    effect_json('condition_behavior', type, condition)

