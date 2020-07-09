source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)

df = read.csv(file.path(processed_datadir,'analyses','spatial-timing.csv'))

objmodel = stan_lmer(correct_mean ~ salience_label * winstart_label + (1 | sid),
    data = filter(df, condition == 'object'))

print(objmodel)

ci95 <- posterior_interval(objmodel, prob = 0.95, pars = "salience_labellow:winstart_labellate")
print(ci95)

ci95 <- posterior_interval(objmodel, prob = 0.95, pars = "salience_labellow")
print(ci95)

spmodel = stan_lmer(correct_mean ~ salience_label * winstart_label + (1 | sid),
    data = filter(df, condition == 'spatial'))

ci95 <- posterior_interval(spmodel, prob = 0.95, pars = "salience_labellow")
print(ci95)

ci95 <- posterior_interval(spmodel, prob = 0.95, pars = "salience_labellow:winstart_labellate")
print(ci95)
