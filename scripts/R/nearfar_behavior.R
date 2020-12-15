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


df = df %>% mutate(dtz = (direction_timing - mean(direction_timing) / sd(direction_timing)))

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
    )

table = effects %>%
    select(global_early:spatial_late) %>%
    gather(global_early:spatial_late, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = atan(value))

table %>% write.csv(file.path(processed_datadir, 'analyses', 'nearfar_behavior_coefs.csv'))
table %>% effect_table()

diff_table = effects %>%
    select(matches('_diff')) %>%
    gather(matches('_diff'), key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    effect_summary(r = value)
diff_table %>% effect_table()
