source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)
library(rstanarm)
library(bayestestR)
library(gamm4)

options(mc.cores = parallel::detectCores())

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

classfile = read.csv(file.path(processed_datadir, 'svm_params',
    'timeline-classify_absolute=true.csv'))

hits = classfile %>% filter(winstart > 0, hit == 'hit', condition == 'object')

ggplot(hits, aes(x = winstart, y = correct_mean, group = interaction(salience_label, target_time_label))) +
    stat_summary(geom='line', aes(color = interaction(salience_label, target_time_label))) +
    stat_summary(geom='ribbon', alpha = 0.4, aes(fill = interaction(salience_label, target_time_label))) +
    coord_cartesian(xlim=c(0,2))

hits$correct_mean = (hits$correct_mean - 0.5)*0.99 + 0.5
hitmodel = stan_glm(correct_mean ~ winstart * salience_label * target_time_label,
    data = hits)
hitmeans = hits %>% group_by(winstart, salience_label, target_time_label) %>%
    summarize(correct_mean = mean(correct_mean))
pred = predictive_interval(hitmodel, prob=0.5, newdata = hitmeans)
hitmeans$lower = pred[,1]
hitmeans$upper = pred[,2]

ggplot(hits, aes(x = winstart, y = correct_mean, group = interaction(salience_label, target_time_label))) +
    stat_summary(geom='line', aes(color = interaction(salience_label, target_time_label))) +
    # stat_summary(geom='ribbon', alpha = 0.4, aes(fill = interaction(salience_label, target_time_label))) +
    geom_ribbon(data = hitmeans, aes(ymin=lower,ymax=upper, fill = interaction(salience_label, target_time_label)), alpha=0.3) +
    coord_cartesian(xlim=c(0,2))

hitmodel2 = stan_gamm4(correct_mean ~ s(winstart),data = hits, family = mgcv::betar)

hitmodel3 = stan_gamm4(correct_mean ~ s(winstart), data = hits)
hitmodel3 = stan_gamm4(correct_mean ~ s(winstart) + salience_label, data = hits)
hits$salience_label = factor(hits$salience_label)
hitmodel3 = stan_gamm4(correct_mean ~ s(winstart, by = salience_label), data = hits)

hitmodel4 = stan_gamm4(correct_mean ~ s(winstart, by = salience_label),
    random = ~(1 | sid), data = hits)
