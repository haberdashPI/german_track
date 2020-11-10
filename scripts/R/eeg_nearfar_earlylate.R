source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)
library(gamm4)

options(mc.cores = parallel::detectCores())

df = read.csv(file.path(processed_datadir,'analyses','nearfar_earlylate.csv'))

ggplot(df, aes(x = logitnullmean, y = shrinkmean, color = target_time_label)) + facet_wrap(~condition)

model = stan_glmer(shrinkmean ~ condition * target_time_label + (1 + logitnullmean | sid),
    family = mgcv::betar,
    data = df,
    iter = 4000)

p = pp_check(model)
ggsave(file.path(plot_dir, 'category_nearfar_target', 'eeg_pp_check.svg'), p)

posterior_interval(matrix(c(predictive_error(model))))

coefs = as.data.frame(model) %>%
    mutate(
        global_latediff = target_time_labellate,
        object_latediff = target_time_labellate + `conditionobject:target_time_labellate`,
        spatial_latediff = target_time_labellate + `conditionspatial:target_time_labellate`,
    ) %>%
    gather(global_latediff:spatial_latediff, key = 'comparison', value = 'value') %>%
    group_by(comparison) %>%
    summarize(
        mean = mean(value),
        meanint05 = posterior_interval(matrix(value))[,1],
        meanint95 = posterior_interval(matrix(value))[,2],
        pval = pd_to_p(p_direction(value))[[1]],
        d = mean(value / `(phi)`),
        dint05 = posterior_interval(matrix(value / `(phi)`))[,1],
        dint95 = posterior_interval(matrix(value / `(phi)`))[,2],
    )
knitr::kable(coefs, digits = 3)
print(coefs)
