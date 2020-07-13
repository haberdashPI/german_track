source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)

df = read.csv(file.path(processed_datadir,'analyses','target-time.csv'))
df$correct_mean = (df$correct_mean - 0.5)*0.99 + 0.5

model = stan_glmer(correct_mean ~ target_time_label + (1 | sid),
    family = mgcv::betar, data = df)

summary(model)
p_map(model)
