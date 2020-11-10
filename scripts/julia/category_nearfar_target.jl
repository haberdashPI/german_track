# Setup
# =================================================================

n_winlens = 12
n_folds = 10

using DrWatson; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures, Dates, Underscores,
    Printf, ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers,
    Infiltrator, Peaks, Distributions, DSP, Random, CategoricalArrays, StatsModels,
    StatsFuns, CSV, Colors

dir = mkpath(plotsdir("category_nearfar_target"))

using GermanTrack: neutral, colors, lightdark, darkgray

# Find λ
# =================================================================

# Mean Frequency Bin Analysis
# -----------------------------------------------------------------

classdf_file = joinpath(cache_dir("features"), "nearfartarget-freqmeans.csv")

if isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :target_switch_label])

    windows = [windowtarget(len = len, start = start)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in [0; 2.0 .^ range(-2, 2, length = 10)]]
    classdf = compute_freqbins(subjects, classdf_groups, windows)

    CSV.write(classdf_file, classdf)
end

# Compute classification accuracy
# -----------------------------------------------------------------

resultdf_file = joinpath(cache_dir("models"), "nearfartarget-target-time.csv")

λ_folds = folds(2, unique(classdf.sid), rng = stableRNG(2019_11_18, :lambda_folds,
    :nearfartarget))
classdf[!,:fold] = in.(classdf.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_file) && mtime(resultdf_file) > mtime(classdf_file)
    resultdf = CSV.read(resultdf_file)
else
    lambdas = 10.0 .^ range(-2, 0, length=100)
    factors = [:fold, :winlen, :winstart, :condition]
    groups = groupby(classdf, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        result = testclassifier(LassoPathClassifiers(lambdas),
            data = sdf, y = :target_switch_label, X = r"channel", crossval = :sid,
            n_folds = n_folds, seed = stablehash(:nearfartarget_classification, 2019_11_18),
            maxncoef = size(sdf[:,r"channel"], 2),
            irls_maxiter = 1200, weight = :weight, on_model_exception = :throw)
        result[!, keys(key)] .= permutedims(collect(values(key)))
        next!(progress)

        result
    end

    resultdf = @_ groups |> pairs |> collect |>
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_file, resultdf)

    alert("Completed salience/target-time classification!")
end

# λ selection
# -----------------------------------------------------------------

fold_map, λ_map = pickλ(resultdf, 2, [:condition, :winlen, :winstart], :condition,
    smoothing = 0.8, slope_thresh = 0.15, flat_thresh = 0.05, dir = dir)

# Plot near/far performance for early/late targets
# =================================================================

# -----------------------------------------------------------------

classdf_earlylate_file = joinpath(cache_dir("features"), "switch-target-freqmeans-earlylate.csv")

if isfile(classdf_earlylate_file)
    classdf_earlylate = CSV.read(classdf_earlylate_file)
else
    nbins = 30

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    breaks = range(0,1,length = nbins+1)[2:(end-1)]
    switchclass = @_ GermanTrack.load_stimulus_metadata().switch_distance |>
        skipmissing |> quantile(__, breaks)
    events[!, :switch_class] = @_ map(sum(_ .> switchclass), events.switch_distance)

    classdf_earlylate = mapreduce(append!!, 1:(nbins-2)) do switch_break
        label(x) = ismissing(x) ? missing : x > switch_break ? "early" : "late"
        events[!, :target_switch_label] = label.(events.switch_class)
        classdf_earlylate_groups = @_ events |>
            filter(ishit(_, region = "target") ∈ ["hit"], __) |>
            groupby(__, [:sid, :condition, :target_time_label, :target_switch_label])

        windows = [windowtarget(len = len, start = start)
            for len in 2.0 .^ range(-1, 1, length = 10),
                start in range(0, 2.5, length = 12)]

        result = compute_freqbins(subjects, classdf_earlylate_groups, windows)
        result[!, :switch_break] .= switch_break

        result
    end
    # TASK: set condition on infiltrate to look at missing data case
    CSV.write(classdf_earlylate_file, classdf_earlylate)
end

# Compute classification accuracy
# -----------------------------------------------------------------

resultdf_earlylate_file = joinpath(cache_dir("models"), "switch-target-earlylate.csv")
classdf_earlylate[!,:fold] = getindex.(Ref(fold_map), classdf_earlylate.sid)

if isfile(resultdf_earlylate_file) && mtime(resultdf_earlylate_file) > mtime(classdf_earlylate_file)
    resultdf_earlylate = CSV.read(resultdf_earlylate_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label, :switch_break]
    groups = groupby(classdf_earlylate, factors)
    lambdas = 10.0 .^ range(-2, 0, length=100)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = λ_map[first(sdf.fold)]
        result = testclassifier(LassoPathClassifiers(lambdas),
            data               = sdf,
            y                  = :target_switch_label,
            X                  = r"channel",
            crossval           = :sid,
            n_folds            = n_folds,
            seed               = stablehash(:salience_classification,
                                            :target_time, 2019_11_18),
            maxncoef           = size(sdf[:,r"channel"], 2),
            irls_maxiter       = 600,
            weight             = :weight,
            on_model_exception = :throw,
            on_missing_case    = :missing,
        )
        if !isempty(result)
            result[!, keys(key)] .= permutedims(collect(values(key)))
        end
        next!(progress)

        result
    end

    resultdf_earlylate = @_ groups |> pairs |> collect |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_earlylate_file, resultdf_earlylate)
end

# Plot by λ and switch_break
# -----------------------------------------------------------------

classmeans = @_ resultdf_earlylate |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label, :switch_break]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label, :switch_break]) |>
    combine(__, :mean => mean => :mean) |>
    transform!(__, :λ => ByRow(log) => :logλ)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, [:λ, :logλ])

logitshrink = logit ∘ shrinktowards(0.5, by = 0.01)

let l = logitshrink, C = mean(l.(nullmeans.nullmean)), tocor = x -> logistic(x + C)
    rawdata = @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label, :switch_break]) |>
        transform!(__, :nullmean => ByRow(l) => :logitnullmean) |>
        transform!(__, :mean => ByRow(l) => :logitmean) |>
        transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff)

    CSV.write(joinpath(processed_datadir("analyses"), "nearfar_lambda_switchbreak.csv"), rawdata)
end

# Plot salience by early/late targets
# -----------------------------------------------------------------

classmeans = @_ resultdf_earlylate |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label, :switch_break]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label, :switch_break]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

logitshrink = logit ∘ shrinktowards(0.5, by = 0.01)

nullmean, classdiffs, rawdata =
    let l = logitshrink, C = mean(l.(nullmeans.nullmean)), tocor = x -> logistic(x + C)

    rawdata = @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label, :switch_break]) |>
        transform!(__, :nullmean => ByRow(l) => :logitnullmean) |>
        transform!(__, :mean => ByRow(l) => :logitmean) |>
        transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff)

    meandata = @_ rawdata |>
        groupby(__, [:condition, :target_time_label, :switch_break]) |>
        combine(__,
            :logitmeandiff => tocor ∘ mean => :mean,
            :logitmeandiff => (x -> tocor(lowerboot(x, alpha = 0.318))) => :lower,
            :logitmeandiff => (x -> tocor(upperboot(x, alpha = 0.318))) => :upper,
        ) |>
        transform!(__, [:condition, :target_time_label] => ByRow(string) => :condition_time)

    logistic(C), meandata, rawdata
end

classdiffs[!, :early_break] = switchclass[classdiffs.switch_break.+1]

# select the best break value by fold
best_break_df = @_ rawdata |>
    groupby(__, [:fold, :switch_break]) |>
    combine(__, :logitmeandiff => mean => :mean,
                [:condition, :target_time_label] =>
                ((x,y) -> string.(x,y) |> unique |> length) => :condcount) |>
    filter(_.condcount == 6, __) |> # only consider values with the complete conditions covered
    groupby(__, :fold) |>
    combine(__, [:switch_break, :mean] => ((b, mean) -> b[argmax(mean)]) => :best) |>
    transform!(__, :best => reverse => :best) |>
    transform!(__, :best => ByRow(x -> switchclass[x + 1]) => :early_break)

best_breaks = @_ best_break_df |> Dict(r.fold => r.best for r in eachrow(__))

classdiffs |>
    @vlplot(
        facet = {row = {field = :condition, title = ""}},
    ) + (
        @vlplot() +
        @vlplot(:line,
            color = :target_time_label,
            x = {:early_break, title = "Early/Late Divide (s)", scale = {domain = [0, 2.5]}},
            y = {:mean, title = ["Classsification Accuracy", "(Null Mean Corrected)"]}
        ) +
        @vlplot(:point,
            color = :target_time_label,
            x = :early_break,
            y = {:mean, title = ""}
        ) +
        @vlplot(:errorband,
            color = :target_time_label,
            x = :early_break,
            y = {:lower, title = ""}, y2 = :upper
        ) + (
            best_break_df |> @vlplot() +
            @vlplot({:rule, strokeDash = [2,2]},
                x = :early_break,
            ) +
            @vlplot({:text, align = :left, fontSiz = 9, xOffset = 2},
                transform = [{calculate = "'Fold '+datum.fold", as = :fold_label}],
                x = :early_break,
                y = {datum = 0.2},
                text = :fold_label
            )
        )
    ) |> save(joinpath(dir, "switch_target_earlylate_multibreak.svg"))

classdiff_best =
    let l = logitshrink, C = mean(l.(nullmeans.nullmean)), tocor = x -> logistic(x + C)

    @_ rawdata |>
        filter(_.switch_break == best_breaks[_.fold], __) |>
        CSV.write(joinpath(processed_datadir("analyses"), "nearfar_earlylate.csv"), __)

    meandata = @_ rawdata |>
        filter(_.switch_break == best_breaks[_.fold], __) |>
        groupby(__, [:condition, :target_time_label]) |>
        combine(__,
            :logitmeandiff => tocor ∘ mean => :mean,
            :logitmeandiff => (x -> tocor(lowerboot(x, alpha = 0.318))) => :lower,
            :logitmeandiff => (x -> tocor(upperboot(x, alpha = 0.318))) => :upper,
        ) |>
        transform!(__, [:condition, :target_time_label] => ByRow(string) => :condition_time)
end

rawbest = @_ rawdata |> filter(_.switch_break == best_breaks[_.fold], __)
pl = rawbest |>
    @vlplot(
        facet = {column = {field = :condition, type = :nominal}}
    ) +
    (
        @vlplot() +
        @vlplot({:point, filled = true},
            x = {"logitnullmean:q", title = "Null Model Accuracy (Logit Scale)"},
            y = {"logitmean:q", title = "Full Model Accuracy (Logit Scale)"},
            color = :target_time_label, shape = :target_time_label
        ) +
        (
            @vlplot(data = {values = [{x = -6, y = -6}, {x = 6, y = 6}]}) +
            @vlplot({:line, clip = true, strokeDash = [2 2]},
                color = {value = "black"},
                x = {"x:q", scale = {domain = collect(extrema(rawbest.logitnullmean))}},
                y = {"y:q", type = :quantitative, scale = {domain = collect(extrema(rawbest.logitmean))}}
            )
        )
    );
pl |>  save(joinpath(dir, "nearfar_earlylate_ind.svg"))

ytitle = ["Neural Switch-Classification", "Accuracy (Null Model Corrected)"]
barwidth = 14
yrange = [0, 1.0]
pl = classdiff_best |>
    @vlplot(
        height = 175, width = 242, autosize = "fit",
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth},
            axis = {titlePadding = 13}
        },
    ) +
    @vlplot({:bar, xOffset = -(barwidth/2), clip = true},
        transform = [{filter = "datum.target_time_label == 'early'"}],
        color = {:condition_time, title = nothing, scale = {range = "#".*hex.(lightdark)}},
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}},
        y = {:mean, title = ytitle, scale = {domain = yrange}}
    ) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.target_time_label == 'early'"}],
        color = {value = "black"},
        x = {:condition, title = nothing},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:bar, xOffset = (barwidth/2), clip = true},
        transform = [{filter = "datum.target_time_label == 'late'"}],
        color = {:condition_time, title = nothing},
        x = {:condition, title = nothing},
        y = {:mean, title = ytitle}
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.target_time_label == 'late'"}],
        x = {:condition, title = nothing},
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.target_time_label == 'early' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "Early"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.target_time_label == 'late' && datum.condition == 'global'"}],
        # x = {datum = "spatial"}, y = {datum = },
        x = {:condition, axis = {title = ""}},
        y = {datum = yrange[1]},
        text = {value = "Late"},
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-top",
            dx = -2barwidth - 17, dy = 22},
            color = {value = "#"*hex(darkgray)},
            x = {datum = "global"},
            y = {datum = yrange[1]},
            text = {datum = ["Less distinct", "response near switch"]}
        ) +
        @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-bottom",
            dx = -2barwidth - 17, dy = -24},
            color = {value = "#"*hex(darkgray)},
            x = {datum = "global"},
            y = {datum = yrange[2]},
            text = {datum = ["More distinct", "response near switch"]}
        )
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"}) +
        @vlplot({:text, align = "left", dx = 5, dy = 0, baseline = "line-bottom", fontSize = 9},
            y = {datum = nullmean},
            x = {datum = "spatial"},
            text = {value = ["Null Model", "Accuracy"]}
        )
    );
pl |> save(joinpath(dir, "switch_target_earlylate.svg"))

# Behavior data
# -----------------------------------------------------------------

nearsplit = mean(switchclass[collect(values(best_breaks))])

target_labels = OrderedDict(
    "early" => ["Early Target", "(before 3rd Switch)"],
    "late"  => ["Late Target", "(after 3rd Switch)"]
)

target_timeline = @_ CSV.read(joinpath(processed_datadir("plots"),
    "hitrate_timeline_bytarget.csv")) |>
    groupby(__, :condition) |>
    transform!(__, :err => (x -> replace(x, NaN => 0.0)) => :err,
                   [:pmean, :err] => (+) => :upper,
                   [:pmean, :err] => (-) => :lower,
                   :target_time => ByRow(x -> target_labels[x]) => :target_time_label)



pl = @_ target_timeline |>
    filter(_.time < 1.5, __) |>
    @vlplot(
        config = {legend = {disable = true}},
        transform = [{calculate = "upper(slice(datum.condition,0,1)) + slice(datum.condition,1)",
                        as = :condition}],
        spacing = 1,
        facet = {
            column = {
                field = :target_time_label, type = :ordinal, title = nothing,
                sort = collect(values(target_labels)),
                header = {labelFontWeight = "bold"}
            }
        }
    ) +
    (
        @vlplot(width = 80, autosize = "fit", height = 130, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot({:trail, clip = true},
            transform = [{filter = "datum.time < 1.25 || datum.target_time == 'early'"}],
            x = {:time, type = :quantitative, scale = {domain = [0, 1.5]},
                title = ["Time after", "Switch (s)"]},
            y = {:pmean, type = :quantitative, scale = {domain = [0.5, 1]}, title = "Hit Rate"},
            size = {:weight, type = :quantitative, scale = {range = [0, 2]}},
        ) +
        @vlplot({:errorband, clip = true},
            transform = [{filter = "datum.time < 1.25 || datum.target_time == 'early'"}],
            x = {:time, type = :quantitative, scale = {domain = [0, 1.5]}},
            y = {:upper, type = :quantitative, title = "", scale = {domain = [0.5, 1]}}, y2 = :lower,
            # opacity = :weight,
            color = :condition,
        ) +
        @vlplot({:text, align = :left, dx = 5},
            transform = [
                {filter = "datum.time > 1 && datum.time < 1.1 && datum.target_time == 'late'"},
            ],
            x = {datum = 1.2},
            y = {:pmean, aggregate = :mean, type = :quantitative},
            color = :condition,
            text = {:condition, }
        ) +
        (
            @vlplot(data = {values = [{}]}) +
            # @vlplot({:rule, strokeDash = [4 4], size = 1},
            #     x = {datum = nearsplit},
            #     color = {value = "black"}
            # ) +
            @vlplot({:text, fontSize = 9, align = :right, dx = 2, baseline = "bottom"},
                x = {datum = 1.5},
                y = {datum = 0.5},
                text = {value = "Far"},
                color = {value = "#"*hex(darkgray)}
            ) +
            @vlplot({:text, fontSize = 9, align = :left, dx = 2, baseline = "bottom"},
                x = {datum = 0},
                y = {datum = 0.5},
                text = {value = "Near"},
                color = {value = "#"*hex(darkgray)}
            )
        )
    );
pl |> save(joinpath(dir, "behavior_timeline.svg"))

# Combine early/late plots
# -----------------------------------------------------------------

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

for (suffix, file) in [
    ("behavior_timeline", "behavior_timeline.svg"),
    ("neural", "switch_target_earlylate.svg")]
    filereplace(joinpath(dir, file), r"\bclip([0-9]+)\b" =>
        SubstitutionString("clip\\1_$suffix"))
end

fig = svg.Figure("89mm", "160mm", # "240mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "behavior_timeline.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(115,50),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,50)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "switch_target_earlylate.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,30)
    ).move(0, 250),
).scale(1.333).save(joinpath(dir, "fig3.svg"))

# Plot near/far performance for early/late targets by salience
# =================================================================

# -----------------------------------------------------------------

classdf_sal_earlylate_file = joinpath(cache_dir("features"), "switch-salience-target-freqmeans-earlylate.csv")

if isfile(classdf_sal_earlylate_file)
    classdfsal_earlylate_ = CSV.read(classdf_sal_earlylate_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_sal_earlylate_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :target_time_label, :target_switch_label, :salience_label])

    windows = [windowtarget(len = len, start = start)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in range(0, 2.5, length = 12)]
    classdfsal_earlylate_ = compute_freqbins(subjects, classdf_sal_earlylate_groups, windows)

    CSV.write(classdf_sal_earlylate_file, classdfsal_earlylate_)
end

# Compute classification accuracy
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

resultdf_sal_earlylate_file = joinpath(cache_dir("models"), "switch-salience-target-earlylate.csv")
classdfsal_earlylate_[!,:fold] = getindex.(Ref(fold_map), classdfsal_earlylate_.sid)

if isfile(resultdf_sal_earlylate_file) && mtime(resultdf_sal_earlylate_file) > mtime(classdf_sal_earlylate_file)
    resultdf_sal_earlylate = CSV.read(resultdf_sal_earlylate_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label, :salience_label]
    groups = groupby(classdfsal_earlylate_, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = λ_map[first(sdf.fold)]
        result = testclassifier(LassoPathClassifiers([1.0, λ]),
            data               = sdf,
            y                  = :target_switch_label,
            X                  = r"channel",
            crossval           = :sid,
            n_folds            = n_folds,
            seed               = stablehash(:salience_classification,
                                            :target_time, 2019_11_18),
            maxncoef           = size(sdf[:,r"channel"], 2),
            irls_maxiter       = 600,
            weight             = :weight,
            on_model_exception = :debug,
            on_missing_case    = :missing,
        )
        if !isempty(result)
            result[!, keys(key)] .= permutedims(collect(values(key)))
        end
        next!(progress)

        result
    end

    resultdf_sal_earlylate = @_ groups |> pairs |> collect |>
        # foldl(append!!, Map(findclass), __)
        foldxt(append!!, Map(findclass), __)

    ProgressMeter.finish!(progress)
    CSV.write(resultdf_sal_earlylate_file, resultdf_sal_earlylate)
end

# Plot salience by early/late targets
# -----------------------------------------------------------------

classmeans = @_ resultdf_sal_earlylate |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label, :salience_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean, :salience_label)

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label, :salience_label]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

classdiffs = let l = logit ∘ shrinktowards(0.5, by = 0.01)
    @_ classmeans_sum |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label, :salience_label]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff)
end

ytitle = "Model - Null Model Accuracy (logit scale)"
pl = classdiffs |>
    @vlplot(
        facet = {
            column = {field = :target_time_label, title = nothing},
            row    = {field = :salience_label, title = nothing},
        },
        title = ["Near/Far from Switch Target Classification","Accuracy by Target Time"],
        config = {legend = {disable = true}}
    ) + (
        @vlplot(color = {:condition, title = nothing}, x = {:condition, title = nothing}) +
        @vlplot(:bar,
            y = {:logitmeandiff, aggregate = :mean, type = :quantitative, title = ytitle}
        ) +
        @vlplot(:errorbar,
            color = {value = "black"},
            y = {:logitmeandiff, aggregate = :ci, type = :quantitative, title = ytitle}
        )
    );
pl |> save(joinpath(dir, "switch_target_sal_earlylate_.svg"))
