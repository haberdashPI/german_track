source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)

df = read.csv(file.path(processed_datadir,'analyses','spatial-timing.csv'))
df$correct_mean = (df$correct_mean - 0.5)*0.99 + 0.5

objmodel = stan_glmer(correct_mean ~ salience_label * winstart_label + (1 | sid),
    family = mgcv::betar,
    data = filter(df, condition == 'object'))

print(objmodel)
p_map(objmodel)

ci95 <- posterior_interval(objmodel, prob = 0.95, pars = "salience_labellow:winstart_labellate")
print(ci95)

ci95 <- posterior_interval(objmodel, prob = 0.95, pars = "salience_labellow")
print(ci95)

spmodel = stan_glmer(correct_mean ~ salience_label * winstart_label + (1 | sid),
    family = mgcv::betar,
    data = filter(df, condition == 'spatial'))

print(spmodel)

ci95 <- posterior_interval(spmodel, prob = 0.95, pars = "salience_labellow")
print(ci95)

ci95 <- posterior_interval(spmodel, prob = 0.95, pars = "salience_labellow:winstart_labellate")
print(ci95)
