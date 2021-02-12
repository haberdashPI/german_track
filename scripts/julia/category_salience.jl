# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson
@quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, Colors, CSV, DataFramesMeta, Lasso, ColorSchemes, StatsBase

using GermanTrack: colors, gray, patterns, lightdark, darkgray, seqpatterns, neutral

dir = mkpath(plotsdir("figure3_parts"))

# Hit rate x salience level (Figure 3A-B)
# =================================================================

summaries = CSV.read(joinpath(processed_datadir("behavioral", "merve_summaries"),
    "export_salience.csv"), DataFrame)

ascondition = Dict(
    "test" => "global",
    "feature" => "spatial",
    "object" => "object"
)

indmeans = @_ summaries |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition) |>
    rename(__,:sbj_id => :sid, :hr_highsal => :high, :hr_lowsal => :low) |>
    select(__, :condition, :sid, :high, :low) |>
    stack(__, [:high, :low], [:condition, :sid],
        variable_name = :salience_label, value_name = :prop)

CSV.write(joinpath(processed_datadir("analyses"), "behavior_salience.csv"), indmeans)

# Absolute values (Figure 3A)
# -----------------------------------------------------------------

means = @_ indmeans |>
    groupby(__, [:condition, :salience_label]) |>
    combine(__,
        :prop => mean => :mean,
        :prop => lowerboot => :lower,
        :prop => upperboot => :upper
    ) |>
    transform(__, [:condition, :salience_label] => ByRow(string) => :condition_salience)

barwidth = 18
ytitle = "Hit Rate"
yrange = [0, 1]
pl = means |>
    @vlplot(
        height = 160, width = 242, autosize = "fit",
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {titlePadding = 13, labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        },
    ) +
    @vlplot({:bar, xOffset = -(barwidth/2), clip = true},
        transform = [{filter = "datum.salience_label == 'high'"}],
        color = {:condition_salience, title = nothing, scale = {range = "#".*hex.(lightdark)}},
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}},
        y = {:mean, title = ytitle, scale = {domain = yrange}}
    ) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.salience_label == 'high'"}],
        color = {value = "black"},
        x = {:condition, title = nothing},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:bar, xOffset = (barwidth/2), clip = true},
        transform = [{filter = "datum.salience_label == 'low'"}],
        color = {:condition_salience, title = nothing},
        x = {:condition, title = nothing},
        y = {:mean, title = ytitle}
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.salience_label == 'low'"}],
        x = {:condition, title = nothing},
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.salience_label == 'high' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "High Salience"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.salience_label == 'low' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "Low Salience"},
    ); # +
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-top",
    #         dx = -2barwidth - 17, dy = 22},
    #         color = {value = "#"*hex(darkgray)},
    #         x = {datum = "global"},
    #         y = {datum = yrange[1]},
    #         text = {datum = ["Less distinct", "response during switch"]}
    #     ) +
    #     @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-bottom",
    #         dx = -2barwidth - 17, dy = -24},
    #         color = {value = "#"*hex(darkgray)},
    #         x = {datum = "global"},
    #         y = {datum = yrange[2]},
    #         text = {datum = ["More distinct", "response during switch"]}
    #     )
    # ) +
pl |> save(joinpath(dir, "fig3a.svg"))

# Differences (Figure 3B)
# -----------------------------------------------------------------

diffmeans = @_ indmeans |>
    unstack(__, [:condition, :sid], :salience_label, :prop) |>
    transform!(__, [:high, :low] => (-) => :meandiff) |>
    groupby(__, :condition) |>
    combine(__,
        :meandiff => mean => :meandiff,
        :meandiff => lowerboot => :lower,
        :meandiff => upperboot => :upper
    )

barwidth = 18
pl = @_ diffmeans |>
    @vlplot(
        width = 60, height = 140,
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }
    ) +
    @vlplot(:bar,
        x = {:condition,
            title = "",
            axis = {labelAngle = -32, labelAlign = "right",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
        },
        color = {:condition, scale = {
            range = urlcol.(keys(seqpatterns))}},
        y = {:meandiff,
            title = ["High - Low Salience", "(Hit Rate)"]
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    );
plotfile =  joinpath(dir, "fig3b.svg")
pl |> save(plotfile)
addpatterns(plotfile, seqpatterns, size = 10)

# Find hyperparameters (λ and winlen)
# =================================================================

file = joinpath(processed_datadir("analyses"), "salience-hyperparams.json")
GermanTrack.@cache_results file fold_map hyperparams begin

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    lambdas = 10.0 .^ range(-2, 0, length=100)
    windows = [
        windowtarget(len = len, start = start)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in [0; 2.0 .^ range(-2, 2, length = 10)]
    ]

    classdf = @_ events |>
        filter(findresponse(_) == "hit", __) |>
        groupby(__, [:sid, :condition, :salience_label]) |>
        repeatby(__, :window => windows) |>
        tcombine(__, desc = "Computing features...",
            df -> compute_powerbin_features(df, subjects, df.window[1]))

    resultdf = @_ classdf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :salience_hyper_folds)) |>
        groupby(__, [:winstart, :winlen, :condition]) |>
        repeatby(__, :cross_fold => 1:10) |>
        tcombine(__, desc = "Evaluating hyperparameters...", function(sdf)
            rng = stableRNG(2019_11_18, :validate_salience_lambda, NamedTuple(sdf[1, Not(r"channel")]))
            test, model = traintest(sdf, sdf.cross_fold[1], y = :salience_label, weight = :weight,
                selector = m -> AllSeg(), λ = lambdas,
                validate_rng = rng)
            test[:, Not(r"channel")]
        end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    param_means = @_ resultdf |>
        groupby(__, [:λ, :condition, :fold, :winlen, :winstart]) |>
        @combine(__, mean = mean(:val_accuracy), mean_sem = mean(:val_se)) |>
        groupby(__, [:λ, :condition, :fold]) |>
        @combine(__, mean = maximum(:mean), mean_sem = maximum(:mean_sem))

    pl = @_ param_means |>
        @transform(__, lower = :mean .- :mean_sem, upper = :mean .+ :mean_sem) |>
        groupby(__, [:λ, :condition]) |>
        @combine(__, mean = maximum(:mean), mean_sem = mean(:mean_sem)) |>
    @vlplot(
        x = {:λ, scale = {type = :log}},
        color = :condition
    ) + @vlplot(:line, y = :mean) +
    @vlplot(:errorband, y = {:upper, type = :quantitative}, y2 = :lower);
    pl |> save(joinpath(dir, "supplement", "lambdas.svg"))

    folder = @_ resultdf |>
        repeatby(__, :cross_fold => 1:10) |>
        @where(__, :fold .!= :cross_fold)
    hyperparamsdf = combine(folder, function(sdf)
        fold = sdf.cross_fold[1]
        meandf = @_ sdf |> groupby(__, [:λ, :winlen, :condition, :winstart]) |>
            @combine(__, mean = mean(:val_accuracy)) |>
            groupby(__, [:λ, :winlen, :condition]) |>
            @combine(__, mean = maximum(:mean)) |>
            groupby(__, [:λ, :winlen]) |>
            @combine(__, mean = mean(:mean))
        thresh = maximum(meandf.mean)
        λ = maximum(meandf.λ[meandf.mean .>= thresh])
        meandf_ = meandf[meandf.λ .== λ, :]
        best = maximum(meandf_.mean[(meandf_.λ .== λ)])
        (fold = fold, mean = best, λ = λ, winlen = only(meandf_.winlen[meandf_.mean .== best]))
    end)

    hyperparams = @_ hyperparamsdf |>
        Dict(row.fold => (;row[Not(:fold)]...) for row in eachrow(__))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Plot timeline (Fig 3d)
# =================================================================

# Compute classificaiton accuracy
# -----------------------------------------------------------------

file = joinpath(processed_datadir("analyses"), "salience-freqmeans-timeline.json")
GermanTrack.@cache_results file resultdf_timeline begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    lens = @_ getindex.(values(hyperparams), :winlen) |> unique |>
        GermanTrack.spread.(__, 0.5, n_winlens) |> reduce(vcat, __) |> unique
    windows = [
        windowtarget(start = start, len = len)
        for len in lens
        for start in range(0, 4, length = 48)
    ]

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        groupby(__, [:sid, :fold, :condition, :salience_label, :hittype]) |>
        repeatby(__, :window => windows) |>
        tcombine(__, desc = "Computing frequency bins...",
            df -> compute_powerbin_features(df, subjects, df.window[1]))

    resultdf_timeline = @_ classdf |>
        groupby(__, [:hittype, :condition, :winstart]) |>
        repeatby(__, :cross_fold => 1:10, :modeltype => ["full", "null"]) |>
        tcombine(__, desc = "Classifying salience...", function(sdf)
            fold = sdf.cross_fold[1]
            selector = sdf.modeltype[1] == "null" ? m -> NullSelect() :
                hyperparams[fold][:λ]
            lens = hyperparams[fold][:winlen] |> GermanTrack.spread(0.5, n_winlens)

            sdf = filter(x -> x.winlen ∈ lens, sdf)
            combine(groupby(sdf, :winlen)) do sdf_len
                test, model = traintest(sdf_len, fold, y = :salience_label,
                    selector = selector, weight = :weight)
                test[:, Not(r"channel")]
            end
        end)
end

# Display classification timeline
# -----------------------------------------------------------------

# nz = @_ filter(occursin(r"nz", _), names(resultdf_timeline)) |> Symbol.(__)
classmeans = @_ resultdf_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :modeltype, :fold, :condition, :hittype]) |>
        combine(__,
            [:correct, :weight] => GermanTrack.wmean   => :mean,
            :weight             => sum                 => :weight,
            :correct            => length              => :count) |>
        transform(__, :weight => (x -> x ./ mean(x)) => :weight)

classmeans_sum = @_ classmeans |>
    groupby(__, [:winstart, :sid, :modeltype, :fold, :condition, :hittype]) |>
    combine(__,
        [:mean, :weight] => GermanTrack.wmean => :mean,
        :weight          => sum               => :weight,
        :count           => length            => :count)

nullmeans = @_ classmeans_sum |>
    @where(__, :modeltype .== "null")  |>
    rename(__, :mean => :nullmean) |>
    select!(__, Not([:modeltype, :weight, :count]))
statdata = @_ classmeans_sum |>
    @where(__, :modeltype .!= "null") |>
    innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype]) |>
    @transform(__, logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)))

CSV.write(processed_datadir("analyses", "eeg_salience_timeline.csv"), statdata)

logitnullmean = mean(statdata.logitnullmean)
nullmean = logistic.(logitnullmean)
corrected_data = @_ statdata |>
    @transform(__,
        mean = logistic.(logit.(shrinktowards.(:mean, 0.5, by = 0.01)) .-
                :logitnullmean .+ logitnullmean),
        condition_label = uppercasefirst.(:condition))

maxdiff(xs) = maximum(abs(xs[i] - xs[j]) for i in 1:(length(xs)-1) for j in (i+1):length(xs))
timeslice = @_ resultdf_timeline |>
    @where(__, (:hittype .== "hit") .& (:modeltype .== "full")) |>
    groupby(__, [:winstart, :fold]) |>
    @combine(__, mean = mean(:correct)) |> # TODO: use validation accuracy here
    groupby(__, :fold) |>
    @combine(__, best = :winstart[argmax(:mean)])
labelfn(fold) = "fold $fold"

# plcoef = @_ statdata |>
#     filter(_.hittype == "hit", __) |>
#     stack(__, nz, [:winstart, :sid, :modeltype, :condition]) |>
#     filter(_.variable != "nzero", __) |>
#     @vlplot(
#         facet = {column = {field = :condition}}
#     ) +
#     @vlplot(:line,
#         x = :winstart,
#         y = {:value, aggregate = :mean, type = :quantitative},
#         color = {:variable, type = :ordinal,
#             scale = {range = "#".*hex.(ColorSchemes.lajolla[range(0.2,0.9, length = 5)])}}
#     );
# plcoef |> save(joinpath(dir, "supplement", "fig3d_coefes.svg"))

ytitle = ["High/Low Salience", "Classification"]
target_len_y = 0.8
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    groupby(__, [:condition, :condition_label, :winstart]) |>
    combine(__, :mean => boot(alpha = sqrt(0.05)) => AsTable) |>
    @vlplot(
        width = 130, height = 140,
        config = {
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        },
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal, scale = {range = "#".*hex.(colors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round, clip = true},
        strokeDash = {:condition, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
        x = {:winstart, type = :quantitative, title = "Time relative to target onset (s)"},
        y = {:value, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0.5,1.0]}}) +
    # data errorbands
    @vlplot({:errorband, clip = true},
        x = {:winstart, type = :quantitative},
        y = {:lower, type = :quantitative, title = ytitle},
        y2 = :upper
    ) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "(datum.winstart > 3 && datum.winstart <= 3.1)"}],
        x = {datum = 4.0},
        y = {:value, aggregate = :mean, type = :quantitative},
        text = :condition_label
    ) +
    # "Time Slice" annotation
    (
        @transform(timeslice, fold_label = labelfn.(:fold)) |>
        @vlplot() +
        @vlplot({:rule, strokeDash = [2 2]},
            x = {:best, aggregate = :mean},
            color = {value = "black"}
        )
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot({:text, align = "right", dx = -1},
            x = {datum = mean(timeslice.best)},
            y = {datum = 0.98},
            text = {value = "Panel B slice →"},
            color = {value = "black"}
        )
    ) +
    # "Null Model" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        # white rectangle to give text a background
        @vlplot(mark = {:text, size = 11, baseline = "middle", dy = -5, dx = 5,
            align = "left"},
            x = {datum = 4}, y = {datum = nullmean},
            text = {value = ["Null Model", "Accuracy"]},
            color = {value = "black"}
        )
    ) +
    # Dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"})
    ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.05, y = target_len_y, dir = 270},
            {x = 0.95, y = target_len_y, dir = 90}]}) +
        @vlplot(mark = {:line, size = 1.5},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            color = {value = "black"},
        ) +
        @vlplot(mark = {:point, shape = "triangle", opacity = 1.0, size = 10},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            angle = {:dir, type = :quantitative, scale = {domain = [0, 360], range = [0, 360]}},
            color = {value = "black"}
        )
    ) +
    # "Target Length" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left, yOffset = -3},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "fig3d.svg"))

# Compute classificaiton accuracy across early/late
# =================================================================

file = joinpath(processed_datadir("analyses"), "salience-freqmeans-earlylate-timeline.json")
GermanTrack.@cache_results file resultdf_earlylate_timeline begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    lens = @_ getindex.(values(hyperparams), :winlen) |> unique |>
        GermanTrack.spread.(__, 0.5, n_winlens) |> reduce(vcat, __) |> unique
    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        groupby(__, [:sid, :fold, :condition, :salience_label, :hittype, :target_time_label]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [
                windowtarget(start = start, len = len)
                for len in lens
                for start in range(0, 4, length = 24)
            ],
            compute_powerbin_features(_1, subjects, _2)) |>
        delete!(__, :window)

    resultdf_earlylate_timeline = @_ classdf |>
        groupby(__, [:hittype, :condition, :winstart]) |>
        filteringmap(__, desc = "Classifying salience...",
            :cross_fold => 1:10, folder = foldxt,
            :modeltype => ["full", "null"],
            function (sdf, fold, modeltype)
                selector = modeltype == "null" ? m -> NullSelect() : hyperparams[fold][:λ]
                lens = hyperparams[fold][:winlen] |> GermanTrack.spread(0.5, n_winlens)

                sdf = filter(x -> x.winlen ∈ lens, sdf)
                combine(groupby(sdf, :winlen)) do sdf_len
                    test, model = traintest(sdf_len, fold, y = :salience_label,
                        selector = selector, weight = :weight)
                    test[:, Not(r"channel")]
                end
            end)
end

# plotting (trained on early late separate)
# -----------------------------------------------------------------

classmeans_earlylate = @_ resultdf_earlylate_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :modeltype, :fold, :condition, :hittype, :target_time_label]) |>
        combine(__, [:correct, :weight] => GermanTrack.wmean => :mean,
                    :weight => sum => :weight,
                    :correct => length => :count) |>
        transform(__, :weight => (x -> x ./ mean(x)) => :weight)

classmeans_earlylate_sum = @_ classmeans_earlylate |>
    groupby(__, [:winstart, :sid, :modeltype, :fold, :condition, :hittype, :target_time_label]) |>
    combine(__,
        [:mean, :weight] => GermanTrack.wmean => :mean,
        :weight => sum => :weight,
        :count => length => :count)

nullmeans = @_ classmeans_earlylate_sum |>
    @where(__, :modeltype .== "null")  |>
    rename(__, :mean => :nullmean) |>
    delete!(__, [:modeltype, :weight, :count])
statdata = @_ classmeans_earlylate_sum |>
    @where(__, :modeltype .!= "null") |>
    innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype, :target_time_label]) |>
    @transform(__, logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)))

logitnullmean = mean(statdata.logitnullmean)
nullmean = logistic.(logitnullmean)
corrected_data = @_ statdata |>
    @transform(__,
        corrected_mean =
            logistic.(logit.(shrinktowards.(:mean, 0.5, by = 0.01)) .-
                :logitnullmean .+ logitnullmean),
        condition_label = uppercasefirst.(:condition))

ytitle = ["High/Low Salience", "Classification"]
target_len_y = 0.8
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    groupby(__, [:condition, :condition_label, :winstart, :target_time_label]) |>
    @combine(__,
        corrected_mean = mean(:corrected_mean),
        lower = lowerboot(:corrected_mean, alpha = 0.318),
        upper = upperboot(:corrected_mean, alpha = 0.318),
    ) |>
    @vlplot(
        facet = {field = :target_time_label, type = :nominal},
        width = 130, height = 140,
        config = {
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        },
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal, scale = {range = "#".*hex.(colors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round, clip = true},
        strokeDash = {:condition, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
        x = {:winstart, type = :quantitative, title = "Time relative to target onset (s)"},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0.4,1.0]}}) +
    # data errorbands
    @vlplot({:errorband, clip = true},
        x = {:winstart, type = :quantitative},
        y = {:lower, type = :quantitative, title = ytitle},
        y2 = :upper
    ) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "(datum.winstart > 3.95 && datum.winstart <= 4.0)"}],
        x = {datum = 4.0},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative},
        text = :condition_label
    ) +
    # # "Time Slice" annotation
    # (
    #     @transform(timeslice, fold_label = labelfn.(:fold)) |>
    #     @vlplot() +
    #     @vlplot({:rule, strokeDash = [2 2]},
    #         x = {:best, aggregate = :mean},
    #         color = {value = "black"}
    #     )
    # ) +
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot({:text, align = "right", dx = -1},
    #         x = {datum = mean(timeslice.best)},
    #         y = {datum = 0.98},
    #         text = {value = "Panel B slice →"},
    #         color = {value = "black"}
    #     )
    # ) +
    # "Null Model" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        # white rectangle to give text a background
        @vlplot(mark = {:text, size = 11, baseline = "middle", dy = -5, dx = 5,
            align = "left"},
            x = {datum = 4}, y = {datum = nullmean},
            text = {value = ["Null Model", "Accuracy"]},
            color = {value = "black"}
        )
    ) +
    # Dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"})
    ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.05, y = target_len_y, dir = 270},
            {x = 0.95, y = target_len_y, dir = 90}]}) +
        @vlplot(mark = {:line, size = 1.5},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            color = {value = "black"},
        ) +
        @vlplot(mark = {:point, shape = "triangle", opacity = 1.0, size = 10},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            angle = {:dir, type = :quantitative, scale = {domain = [0, 360], range = [0, 360]}},
            color = {value = "black"}
        )
    ) +
    # "Target Length" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left, yOffset = -3},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "supplement", "fig3d_earlylate.svg"))


# plotting (trained on early late together)
# -----------------------------------------------------------------

nz = @_ filter(occursin(r"nz", _), names(resultdf_timeline)) |> Symbol.(__)
classmeans_earlylate = @_ resultdf_timeline |>
    groupby(__, [:winstart, :winlen, :sid, :modeltype, :fold, :condition, :hittype, :target_time_label]) |>
        combine(__, [:correct, :weight] => GermanTrack.wmean => :mean,
                    :weight => sum => :weight,
                    :correct => length => :count) |>
        transform(__, :weight => (x -> x ./ mean(x)) => :weight)

classmeans_earlylate_sum = @_ classmeans_earlylate |>
    groupby(__, [:winstart, :sid, :modeltype, :fold, :condition, :hittype, :target_time_label]) |>
    combine(__,
        [:mean, :weight] => GermanTrack.wmean => :mean,
        :weight => sum => :weight,
        :count => length => :count)

nullmeans = @_ classmeans_earlylate_sum |>
    @where(__, :modeltype .== "null")  |>
    rename(__, :mean => :nullmean) |>
    delete!(__, [:modeltype, :weight, :count])
statdata = @_ classmeans_earlylate_sum |>
    @where(__, :modeltype .!= "null") |>
    innerjoin(__, nullmeans, on = [:winstart, :condition, :sid, :fold, :hittype, :target_time_label]) |>
    @transform(__, logitnullmean = logit.(shrinktowards.(:nullmean, 0.5, by = 0.01)))

logitnullmean = mean(statdata.logitnullmean)
nullmean = logistic.(logitnullmean)
corrected_data = @_ statdata |>
    @transform(__,
        corrected_mean =
            logistic.(logit.(shrinktowards.(:mean, 0.5, by = 0.01)) .-
                :logitnullmean .+ logitnullmean),
        condition_label = uppercasefirst.(:condition))

ytitle = ["High/Low Salience", "Classification"]
target_len_y = 0.8
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    groupby(__, [:condition, :condition_label, :winstart, :target_time_label]) |>
    @combine(__,
        corrected_mean = mean(:corrected_mean),
        lower = lowerboot(:corrected_mean, alpha = 0.318),
        upper = upperboot(:corrected_mean, alpha = 0.318),
    ) |>
    @vlplot(
        facet = {field = :target_time_label, type = :nominal},
        width = 130, height = 140,
        config = {
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        },
    ) +
    (@vlplot(
        color = {field = :condition, type = :nominal, scale = {range = "#".*hex.(colors)}},
    ) +
    # data lines
    @vlplot({:line, strokeCap = :round, clip = true},
        strokeDash = {:condition, type = :nominal, scale = {range = [[1, 0], [6, 4], [2, 4]]}},
        x = {:winstart, type = :quantitative, title = "Time relative to target onset (s)"},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative, title = ytitle,
            scale = {domain = [0.4,1.0]}}) +
    # data errorbands
    @vlplot({:errorband, clip = true},
        x = {:winstart, type = :quantitative},
        y = {:lower, type = :quantitative, title = ytitle},
        y2 = :upper
    ) +
    # condition labels
    @vlplot({:text, align = :left, dx = 5},
        transform = [{filter = "(datum.winstart > 3.95 && datum.winstart <= 4.0)"}],
        x = {datum = 4.0},
        y = {:corrected_mean, aggregate = :mean, type = :quantitative},
        text = :condition_label
    ) +
    # # "Time Slice" annotation
    # (
    #     @transform(timeslice, fold_label = labelfn.(:fold)) |>
    #     @vlplot() +
    #     @vlplot({:rule, strokeDash = [2 2]},
    #         x = {:best, aggregate = :mean},
    #         color = {value = "black"}
    #     )
    # ) +
    # (
    #     @vlplot(data = {values = [{}]}) +
    #     @vlplot({:text, align = "right", dx = -1},
    #         x = {datum = mean(timeslice.best)},
    #         y = {datum = 0.98},
    #         text = {value = "Panel B slice →"},
    #         color = {value = "black"}
    #     )
    # ) +
    # "Null Model" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        # white rectangle to give text a background
        @vlplot(mark = {:text, size = 11, baseline = "middle", dy = -5, dx = 5,
            align = "left"},
            x = {datum = 4}, y = {datum = nullmean},
            text = {value = ["Null Model", "Accuracy"]},
            color = {value = "black"}
        )
    ) +
    # Dotted line
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"})
    ) +
    # "Target Length" arrow annotation
    (
        @vlplot(data = {values = [
            {x = 0.05, y = target_len_y, dir = 270},
            {x = 0.95, y = target_len_y, dir = 90}]}) +
        @vlplot(mark = {:line, size = 1.5},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            color = {value = "black"},
        ) +
        @vlplot(mark = {:point, shape = "triangle", opacity = 1.0, size = 10},
            x = {:x, type = :quantitative}, y = {:y, type = :quantitative},
            angle = {:dir, type = :quantitative, scale = {domain = [0, 360], range = [0, 360]}},
            color = {value = "black"}
        )
    ) +
    # "Target Length" text annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left, yOffset = -3},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = "Target Length"},
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "supplement", "fig3d_earlylate.svg"))

# Salience class accuracy at fixed time point (fig 3c)
# =================================================================

timeslice_map = Dict(row.fold => row.best for row in eachrow(timeslice))

barwidth = 16

ytitle = ["High/Low Salience", "Classification"]
yrange = [0.5, 1]
pl = @_ corrected_data |>
    filter(_.hittype == "hit", __) |>
    filter(_.winstart == timeslice_map[_.fold], __) |>
    groupby(__, [:condition]) |>
    @combine(__,
        corrected_mean = mean(:corrected_mean),
        lower = lowerboot(:corrected_mean, alpha = 0.05),
        upper = upperboot(:corrected_mean, alpha = 0.05),
    ) |>
    @vlplot(
        width = 60, height = 140,
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }
    ) +
    @vlplot(:bar,
        x = {:condition,
            title = "",
            axis = {labelAngle = -32, labelAlign = "right",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
        },
        color = {:condition, scale = {range = urlcol.(keys(seqpatterns))}},
        y = {:corrected_mean,
            title = ytitle,
            scale = {domain = yrange}
        }
    ) +
    @vlplot(:errorbar,
        x = {:condition, title = ""},
        y = {:lower, title = ""}, y2 = :upper,
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"})
    );
plotfile = joinpath(dir, "fig3c.svg")
pl |> save(plotfile)
addpatterns(plotfile, seqpatterns, size = 10)

# Final, combined plots for data fig 2
# =================================================================

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

fig = svg.Figure("178mm", "75mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3b.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(90,15)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3c.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(90,15)
    ).move(130, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig3d.svg")).move(0,15),
        svg.Text("C", 2, 10, size = 12, weight = "bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(190,0)
    ).move(260, 0)
).scale(1.333).save(joinpath(plotsdir("figures"), "fig3.svg"))
