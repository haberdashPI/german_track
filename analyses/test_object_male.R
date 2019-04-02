
source("util/setup.R")

use_fake_data = FALSE #TRUE

dir = file.path(plot_dir,paste("run",Sys.Date(),sep="_"))
dir.exists(dir) || dir.create(dir)

if(use_fake_data){
  df = gather(read.csv(file.path(cache_dir,"fake_testobj.csv")),
              label,cor,male_C:other_male_C)
}else{
  df = gather(read.csv(file.path(cache_dir,"testobj.csv")),
              label,cor,male_C:other_male_C)
}

ggplot(df,aes(x=label,y=cor,color=label)) +
  geom_point(position=position_jitter(width=0.2),color="black",alpha=0.2) +
  stat_summary(geom="pointrange",fun.data="mean_cl_boot") +
  geom_abline(slope=0,intercept=0,linetype=2) +
  facet_wrap(~sid)

if(use_fake_data){
  ggsave(file.path(dir,"fake_object_condition.pdf"))
}else{
  ggsave(file.path(dir,"object_condition.pdf"))
}
