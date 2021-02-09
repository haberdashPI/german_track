source("src/R/setup.R")

df = read.csv(file.path(processed_datadir, 'analyses', 'decode', 'decode_scores.csv'))

fit1 = stan_glmer(score ~ train_kind + (1 | sid),
    data = df)

effects = as.data.frame(fit1) %>%
    transform(
        other = `(Intercept)`,
        target = `(Intercept)` + train_kindathittarget
    ) %>%
    gather(other:target, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(cor = value)

fit2 = stan_glmer(score ~ train_kind * condition +
    (train_kind * condition | sid),
    data = df)

effects = as.data.frame(fit2) %>%
    gather(`(Intercept)`:`train_kindatmisstarget:conditionspatial`,
        key = 'coef', value = 'value') %>%
    group_by(coef) %>%
    effect_summary(cor = value)

# we can use sum to zero coding to simplify interpretability of main effects
ct = function(x) { C(x, contr.sum) }
dfc = filter(df, train_kind %in% c('athittarget', 'athitother')) %>%
        mutate(
            condition = ct(condition),
            target_time_label = ct(target_time_label),
            target_switch_label = ct(target_switch_label),
            target_salience = ct(target_salience)
        )

fit3 = stan_glmer(score ~ train_kind * condition +
    (train_kind * condition | sid) +
    (train_kind * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = filter(dfc, train_kind %in% c('athittarget', 'athitother')))

effects = as.data.frame(fit3) %>%
    gather(`(Intercept)`:`train_kindathittarget:condition2`,
        key = 'coef', value = 'value') %>%
    group_by(coef) %>%
    effect_summary(cor = value)

fit4 = stan_glmer(score ~ train_kind * condition + target_time_label +
    (train_kind * condition | sid) +
    (train_kind * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = dfc)

# does target timing matter?
fit5 = stan_glmer(score ~ train_kind * condition * target_time_label +
    (train_kind * condition | sid) +
    (train_kind * condition | stim_id),
    adapt_delta = 0.998, # prevents divergent transitions after warm-up
    data = dfc)

newdf = dfc %>% group_by(train_kind,condition,target_time_label) %>%
    summarize(mean_score = mean(score))
pr = posterior_predict(fit5, newdf, re.form = ~0)
mean(pr)
newdf$pred_score = apply(pr , 2 , median)
int = posterior_interval(pr)
newdf$pred_lower = int[,1]
newdf$pred_upper = int[,2]

effects = as.data.frame(fit5) %>%
    select(`train_kindathit-miss`:`train_kindathit-miss:condition2:target_time_label1`) %>%
    gather(`train_kindathit-miss`:`train_kindathit-miss:condition2:target_time_label1`, key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(r = value)

# how many data points per cell for each subject is this final model?
# somewhere around 6 points per subject, we wouldn't want to go much lower - so no more interactions
dfc %>% group_by(train_kind,condition,target_time_label,stim_id) %>%
    summarize(c = length(score)) %>%
    ungroup() %>%
    summarize(min = min(c), max = max(c), mean = mean(c))

# does target_salience matter?
fit6 = stan_glmer(score ~ train_kind * condition * target_salience +
    (train_kind * condition | sid) +
    (train_kind * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = dfc)

# does switch proximity matter?
fit7 = stan_glmer(score ~ train_kind * condition * target_switch_label +
    (train_kind * condition | sid) +
    (train_kind * condition | stim_id),
    adapt_delta = 0.99, # prevents divergent transitions after warm-up
    data = dfc)

means = df %>%
    filter(test_type == 'hit-target') %>%
    group_by(condition, sid, train_type, train_kind, test_type) %>%
    summarize(score = mean(score)) %>%
    group_by(condition, sid, train_kind, test_type) %>%
    summarize(score = mean(score))

fitmeans = stan_glmer(score ~ train_kind + (1 | sid), means)

effects = as.data.frame(fitmeans) %>%
    mutate(
        hitother = `(Intercept)`,
        hittarget = `(Intercept)` + train_kindathittarget,
        premiss = `(Intercept)` + train_kindpremiss
    ) %>%
    select(hitother:premiss) %>%
    pairwise(hitother:premiss) %>%
    gather(, key = 'train_kind', value = 'value') %>%
    group_by(train_kind) %>%
    effect_summary(r = value)

fitmean = lm(score ~ train_kind * condition,
    data = mutate(means, condition = ct(condition)))

t.test(score ~ train_kind, data = filter(means, condition == 'global'), paired = T)
t.test(score ~ train_kind, data = filter(means, condition == 'object'), paired = T)
t.test(score ~ train_kind, data = filter(means, condition == 'spatial' & sid != 10), paired = T)

condmeans = means %>%
    group_by(sid, train_kind) %>%
    summarize(score = mean(score))

t.test(score ~ train_kind, condmeans, paired = T)


pretargetdf = read.csv(file.path(cache_dir, 'eeg', 'decoding', 'pretarget_attend.csv')) %>%
    filter(condition != 'spatial')

prefit = stan_glmer(score ~ condition + (1 | sid) + (1 | sound_index), pretargetdf)

effectspre = as.data.frame(prefit) %>%
    mutate(
        global = `(Intercept)`,
        object = `(Intercept)` + conditionobject,
        # spatial = `(Intercept)` + conditionspatial,
    ) %>%
    select(global:object) %>%
    pairwise(global:object) %>%
    gather(everything(), key = 'condition', value = 'value') %>%
    group_by(condition) %>%
    effect_summary(prop = value)

t.test(score ~ condition, pretargetdf, paired = TRUE)
