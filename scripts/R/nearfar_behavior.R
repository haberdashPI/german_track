source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)
library(gamm4)

options(mc.cores = parallel::detectCores())

df = read.csv(file.path(processed_datadir,'analyses','hit_by_switch.csv')) %>%
    filter(perf %in% c('hit', 'miss')) %>%
    mutate(id = paste0(sid, exp_id))


df = df %>% mutate(dtz = (switch_distance - mean(switch_distance, na.rm = T) /
    sd(switch_distance, na.rm =T)))

dft = filter(df, !is.na(switch_distance))
df$time_condition = interaction(df$target_time_label, df$condition)
fit_add = stan_gamm4(sbj_answer ~ s(dtz),
    family = binomial(),
    adapt_delta = 0.99,
    data = dft, iter = 2000) #, random = ~(1 | id))

fit_add = stan_gamm4(sbj_answer ~ s(dtz, by = target_time_label),
    family = binomial(),
    adapt_delta = 0.99,
    data = dft, iter = 2000) #, random = ~(1 | id))

fit_add = stan_gamm4(sbj_answer ~ s(dtz, by = time_condition),
    family = binomial(),
    adapt_delta = 0.99,
    data = dft, iter = 2000) #, random = ~(1 | id))

p = plot_nonlinear(fit_add)
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'behavior_nonlinear.svg'), p)

# conclusion: non of the conditions are particularly non-linear: I think it's safe to
# use a linear model

fitmm = stan_glmer(sbj_answer ~ dtz * condition * target_time_label
    + (dtz * condition * target_time_label | id), df, family = binomial(link = "logit"))

p = pp_check(fitmm)
ggsave(file.path(plot_dir, 'figure4_parts', 'supplement', 'behavior_mm_pp_check.svg'), p)

effects = as.data.frame(fitmm) %>%
    mutate(
        global_early = dtz,
        object_early = dtz + `dtz:conditionobject`,
        spatial_early = dtz + `dtz:conditionspatial`
    ) %>%
    mutate(
        global_late = global_early + `dtz:target_time_labellate`,
        object_late = global_early + `dtz:target_time_labellate` + `dtz:conditionobject:target_time_labellate`,
        spatial_late = global_early + `dtz:target_time_labellate` + `dtz:conditionspatial:target_time_labellate`
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
    effect_summary(r = value)

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
