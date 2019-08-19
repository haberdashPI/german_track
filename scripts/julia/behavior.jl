using DrWatson; quickactivate(@__DIR__,"german_track")
include(joinpath(srcdir(),"julia","setup.jl"))
using VegaLite

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))

df = mapreduce(vcat,eeg_files) do file
    df_, sid = events_for_eeg(file,stim_info)
    df_[!,:sid] .= sid
    df_
end

dir = joinpath(plotsdir(),string("results_",Date(now())))
isdir(dir) || mkdir(dir)

meanb(x) = (sum(x)+1)/length(x)
dprime(hits,falarm) =
    quantile(Normal(),meanb(hits)) - quantile(Normal(),meanb(falarm))

dfsum = df |>
    @groupby({_.sid,_.condition}) |>
    @map({key(_)...,
          dp = dprime(_.target_present .& _.correct,
                      .!_.target_present .& .!_.correct),
          mean = mean(_.correct)}) |>
    DataFrame

condition = dfsum |>
    @vlplot(x="condition:o") +
    @vlplot(mark={:point, filled=true, size=100},
        y={"mean(dp)", scale={zero=false}, title="d'"},
        color={value=:black}) +
    @vlplot(mark={:errorbar}, y={"dp:q", title="d'"}) +
    @vlplot(mark={:point}, y={:dp, scale={zero=false}, title="d'"})

save(joinpath(dir,"behavior_summary.pdf"),condition)
