source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)

df = read.csv(file.path(processed_datadir,'analyses','target-time.csv'))
df$correct_mean = (df$correct_mean - 0.5)*0.99 + 0.5

summary(aov(correct_mean ~ target_time_label * condition +
    Error(sid / (target_time_label/condition)), data = df))

model = stan_glmer(correct_mean ~ target_time_label * condition + (1 | sid),
    family = mgcv::betar, data = df)

summary(model)
pdir = p_direction(model)
pd_to_p(p_direction(model)[[2]])
pdir

df = read.csv(file.path(processed_datadir,'analyses','salience-target-time.csv'))
df$correct_mean = (df$correct_mean - 0.5)*0.99 + 0.5

summary(aov(correct_mean ~ target_time_label * condition * salience_label +
    Error(sid / (target_time_label/condition/salience_label)), data = df))

model = stan_glmer(correct_mean ~ salience_label * target_time_label * condition + (1 | sid),
    family = mgcv::betar, data = df)

summary(model)
pdir = p_direction(model)
pd_to_p(p_direction(model)[[2]])
pdir
