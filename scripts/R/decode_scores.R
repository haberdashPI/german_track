source("src/R/setup.R")

df = read.csv(file.path(processed_datadir, 'analyses', 'decode', 'decode_scores.csv'))

fit1 = stan_glmer(score ~ target_window + (1 | sid),
    data = filter(df, target_window %in% c('athit-hit', 'athit-miss')))

fit2 = stan_glmer(score ~ target_window * condition +
    (target_window * condition | sid),
    data = filter(df, target_window %in% c('athit-hit', 'athit-miss')))

fit3 = stan_glmer(score ~ target_window * condition +
    (target_window * condition | sid) +
    (target_window * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = filter(df, target_window %in% c('athit-hit', 'athit-miss')))

fit4 = stan_glmer(score ~ target_window * condition * target_time_label *
    target_switch_label * target_salience +
    (target_window * condition | sid) +
    (target_window * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = filter(df, target_window %in% c('athit-hit', 'athit-miss')))

means = df %>%
    filter(target_window %in% c('athit-hit', 'athit-miss')) %>%
    group_by(condition, sid, target_window) %>%
    summarize(score = mean(score))

stan_glm(score ~ target_window * condition, data = means)
