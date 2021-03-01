source("src/R/setup.R")
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = T)

df = read.csv(file.path(processed_datadir,'analyses','hit_by_switch.csv')) %>%
    filter(perf %in% c('hit', 'miss')) %>%
    mutate(id = paste0(sid, exp_id))

# STEP one, plot an stimulus-condition pair, and sho0w all subjects
# hits and missed on that trial, look at the markers for switch distance

# then compare to the actual stimulus
select_groups <- function(dd, gr, ...) dd[sort(unlist(attr(dd, "groups")$.rows[ gr ])), ]

singledf = df %>% filter(dev_time > 0) %>%
    group_by(condition, exp_id, sid) %>%
    select_groups(1)

p = ggplot(singledf, aes(x = dev_time, y = direction_timing, color = perf, shape = perf)) +
    geom_point()
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'data_slice.svg'), p)

# OKAY: this convinces me that we really do need to avoid using a simple logistic
# regression. We need something to make the analysis robust: i.e. an error parameter of some
# sort that allows for misses even when we're at a very high logit

# Let's compare plain logistic regression to the robust logistic model

df = df %>% mutate(dtz = (direction_timing - mean(direction_timing, na.rm = T) /
    sd(direction_timing, na.rm =T)))

mf = model.frame(perf ~ direction_timing, singledf)
MM = model.matrix(perf ~ direction_timing, singledf)
fit1 = stan(file = file.path(standir, 'robust_logit.stan'), data = list(
    n = nrow(MM),
    k = ncol(MM),
    y = model.response(mf) == 'hit',
    A = MM,
    theta_prior = 5,
    r = 0.1
))

fit2 = stan(file = file.path(standir, 'robust_logit.stan'), data = list(
    n = nrow(MM),
    k = ncol(MM),
    y = model.response(mf) == 'hit',
    A = MM,
    theta_prior = 5,
    r = 0.001
))

df$dtz = (df$direction_timing)
mf = model.frame(perf ~ direction_timing * condition * target_time_label, df)
MM = model.matrix(perf ~ direction_timing * condition * target_time_label, df)
fit2 = stan(file = file.path(standir, 'robust_logit.stan'), data = list(
    n = nrow(MM),
    k = ncol(MM),
    y = model.response(mf) == 'hit',
    A = MM,
    theta_prior = 5,
    r = 0.001
))

dft = filter(df, !is.na(switch_distance))
df$time_condition = interaction(df$target_time_label, df$condition)
fit_add = stan_gamm4(sbj_answer ~ s(switch_distance),
    family = binomial(),
    adapt_delta = 0.99,
    data = dft, iter = 2000) #, random = ~(1 | id))

fit_add = stan_gamm4(sbj_answer ~ s(switch_distance, by = target_time_label),
    family = binomial(),
    adapt_delta = 0.99,
    data = dft, iter = 2000) #, random = ~(1 | id))

fit_add = stan_gamm4(sbj_answer ~ s(switch_distance, by = time_condition),
    family = binomial(),
    adapt_delta = 0.99,
    data = dft, iter = 2000) #, random = ~(1 | id))

p = plot_nonlinear(fit_add)
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'behavior_nonlinear.svg'), p)

# conclusion: non of the conditions are particularly non-linear: I think it's safe to
# use a linear model

fitmm = stan_glmer(sbj_answer ~ switch_distance * condition * target_time_label
    + (switch_distance * condition * target_time_label | id),
    # switches last for 0.6 seconds, skip overlapping targets
    filter(df, switch_distance > 0.6, switch_distance < 1.6),
    adapt_delta = 0.99,
    family = binomial(link = "logit"))

p = pp_check(fitmm)
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'behavior_mm_pp_check.svg'), p)

effects = as.data.frame(fitmm) %>%
    mutate(
        global_early = switch_distance,
        object_early = switch_distance + `switch_distance:conditionobject`,
        spatial_early = switch_distance + `switch_distance:conditionspatial`
    ) %>%
    mutate(
        global_late = global_early + `switch_distance:target_time_labellate`,
        object_late = global_early + `switch_distance:target_time_labellate` + `switch_distance:conditionobject:target_time_labellate`,
        spatial_late = global_early + `switch_distance:target_time_labellate` + `switch_distance:conditionspatial:target_time_labellate`
    ) %>%
    mutate(
        global_diff = global_early - global_late,
        object_diff = object_early - object_late,
        spatial_diff = spatial_early - spatial_late,
        early = (global_early + object_early + spatial_early)/3,
        late = (global_late + object_late + spatial_late)/3,
        all_diff = early - late,
    )

table = effects %>%
    select(global_early:spatial_late) %>%
    gather(global_early:spatial_late, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = atan(value))

table %>% write.csv(file.path(processed_datadir, 'analyses', 'nearfar_behavior_coefs.csv'))
table %>% effect_table()

table2 = effects %>%
    select(early:late) %>%
    gather(early:late, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = value)

diff_table = effects %>%
    select(matches('_diff')) %>%
    gather(matches('_diff'), key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = value)
diff_table %>% effect_table()
