using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack
using VegaLite

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir()))

df = mapreduce(vcat,eeg_files) do file
    df_, sid = events_for_eeg(file,stim_info)
    df_[!,:sid] .= sid
    df_
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

meanb(x,n=1) = (sum(x)+(n/2))/(length(x)+n)
function dprime(hits,falarm,n=1)
    quantile(Normal(),meanb(hits,n)) - quantile(Normal(),meanb(falarm,n))
end

dfsum = df |>
    @groupby({_.sid,_.condition}) |>
    @map({key(_)...,
          dp = dprime(_.target_present .& _.correct,
                      .!_.target_present .& .!_.correct),
          mean = mean(_.correct),
          truepos = mean(_.target_present .& _.correct),
          trueneg = mean(.!_.target_present .& _.correct),
          falsepos = mean(.!_.target_present .& .!_.correct),
          falseneg = mean(_.target_present .& .!_.correct)}) |>
    DataFrame

condition = dfsum |>
    @vlplot(x="condition:o") +
    @vlplot(mark={:point, filled=true, size=100},
        y={"mean(dp)", scale={zero=false}, title="d'"},
        color={value=:black}) +
    @vlplot(mark={:errorbar}, y={"dp:q", title="d'"}) +
    @vlplot(mark={:point}, y={:dp, scale={zero=false}, title="d'"})

save(joinpath(dir,"behavior_summary.pdf"),condition)

condition_bytrues = dfsum |>
    @vlplot(
        mark={:point,filled=true}, column=:condition,
        x=:falsepos, y=:truepos)
save(joinpath(dir,"behavior_summary_splitcor.pdf"),condition_bytrues)

condition_bytrues_sid = dfsum |>
    @vlplot(
        mark={:text}, column=:condition,
        text=:sid,
        x=:falsepos, y=:truepos)

condition_byfalse = dfsum |>
    @vlplot(
        mark={:point,filled=true}, column=:condition,
        x=:falsepos, y=:dp)

dftiming = df |>
    @groupby({_.sid,_.condition,time_bin = 1.2*floor.(Int,_.target_time/1.2)}) |>
    @map({key(_)...,
          dp = dprime(_.target_present .& _.correct,
                 .!_.target_present .& .!_.correct,1)}) |>
    DataFrame

timing = dftiming |>
    @vlplot(:line,x=:time_bin,y=:dp,color="sid:o",column=:condition)

save(joinpath(dir,"behavior_bytype.pdf"),timing)
