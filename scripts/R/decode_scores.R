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

# we can use sum to zero coding to simplify interpretability of main effects
ct = function(x) { C(x, contr.sum) }
dfc = filter(df, target_window %in% c('athit-hit', 'athit-miss')) %>%
        mutate(
            condition = ct(condition),
            target_time_label = ct(target_time_label),
            target_switch_label = ct(target_switch_label),
            target_salience = ct(target_salience)
        )

fit4 = stan_glmer(score ~ target_window * condition + target_time_label +
    (target_window * condition | sid) +
    (target_window * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = dfc)

# does target timing matter?
fit5 = stan_glmer(score ~ target_window * condition * target_time_label +
    (target_window * condition | sid) +
    (target_window * condition | stim_id),
    adapt_delta = 0.998, # prevents divergent transitions after warm-up
    data = dfc)


newdf = dfc %>% group_by(target_window,condition,target_time_label) %>%
    summarize(mean_score = mean(score))
pr = posterior_predict(fit5, newdf, re.form = ~0)
mean(pr)
newdf$pred_score = apply(pr , 2 , median)
int = posterior_interval(pr)
newdf$pred_lower = int[,1]
newdf$pred_upper = int[,2]

effects = as.data.frame(fit5) %>%
    mutate(

    )
    select(global_early:spatialdiff, `(phi)`) %>%
    gather(-`(phi)`, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(r = value, d = value / `(phi)`)


# how many data points per cell for each subject is this final model?
# somewhere around 6 points per subject, we wouldn't want to go much lower - so no more interactions
dfc %>% group_by(target_window,condition,target_time_label,stim_id) %>%
    summarize(c = length(score)) %>%
    ungroup() %>%
    summarize(min = min(c), max = max(c), mean = mean(c))

# does target_salience matter?
fit6 = stan_glmer(score ~ target_window * condition * target_salience +
    (target_window * condition | sid) +
    (target_window * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = dfc)

# does switch proximity matter?
fit7 = stan_glmer(score ~ target_window * condition * target_switch_label +
    (target_window * condition | sid) +
    (target_window * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = dfc)

means = df %>%
    filter(target_window %in% c('athit-hit', 'athit-miss')) %>%
    group_by(condition, sid, target_window) %>%
    summarize(score = mean(score))

fitmean = lm(score ~ target_window * condition,
    data = mutate(means, condition = ct(condition)))

t.test(score ~ target_window, data = filter(means, condition == 'global'), paired = T)
t.test(score ~ target_window, data = filter(means, condition == 'object'), paired = T)
t.test(score ~ target_window, data = filter(means, condition == 'spatial' & sid != 10), paired = T)

condmeans = means %>%
    group_by(sid, target_window) %>%
    summarize(score = mean(score))

t.test(score ~ target_window, condmeans, paired = T)
