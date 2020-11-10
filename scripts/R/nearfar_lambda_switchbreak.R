source("src/R/setup.R")
library(ggplot2)
library(cowplot)
library(dplyr)

df = read.csv(file.path(processed_datadir,'analyses','nearfar_lambda_switchbreak.csv'))

pl = ggplot(df, aes(x = logλ, y = switch_break, fill = logitmeandiff)) +
    geom_raster() + facet_wrap(target_time_label~condition) +
    scale_fill_distiller(type = 'div')
ggsave(file.path(plot_dir, 'category_nearfar_target', 'switchbreak_lambdas.svg'), pl,
    width = 11, height = 8)
