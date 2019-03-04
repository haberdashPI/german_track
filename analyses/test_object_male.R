
source('util/setup.R')

df = gather(read.csv(file.path(cache_dir,'testobj.csv')),
            label,cor,male_C:fem2_C)

ggplot(df,aes(x=label,y=cor,color=label)) +
  geom_point(position=position_jitter(width=0.2),color='black',alpha=0.2) +
  stat_summary(geom='pointrange',fun.data='mean_cl_boot') +
  facet_wrap(~sid)
