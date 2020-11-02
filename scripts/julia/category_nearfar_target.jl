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

using GermanTrack: gray, colors, lightdark, darkgray

# Behavioral Data
# =================================================================

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
        @vlplot({:trail, #= clip = true =#},
            x = {:time, type = :quantitative, scale = {domain = [0, 1.5]},
                title = ["Time after", "Switch (s)"]},
            y = {:pmean, type = :quantitative, scale = {domain = [0.5, 1]}, title = "Hit Rate"},
            size = {:weight, type = :quantitative, scale = {range = [0, 2]}},
        ) +
        @vlplot({:errorband, #= clip = true =#},
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
        )
    );
pl |> save(joinpath(dir, "behavior_timeline.svg"))

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
            irls_maxiter = 600, weight = :weight, on_model_exception = :throw)
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

means = @_ resultdf |>
    groupby(__, [:condition, :λ, :sid, :fold, :winstart, :winlen]) |>
    combine(__,
        :nzcoef => mean => :nzcoef,
        [:correct, :weight] => GermanTrack.wmean => :mean)

bestmeans = @_ means |>
    groupby(__, [:condition, :λ, :sid, :fold]) |>
    combine(__, :nzcoef => mean => :nzcoef,
                :mean => maximum => :mean,
                :mean => logit ∘ shrinktowards(0.5, by = 0.01) ∘ maximum => :logitmean)


logitmeandiff = @_ filter(_.λ == 1.0, bestmeans) |>
    deletecols!(__, [:λ, :nzcoef, :mean]) |>
    rename!(__, :logitmean => :logitnullmean) |>
    innerjoin(__, bestmeans, on = [:condition, :sid, :fold]) |>
    transform!(__, [:logitmean,:logitnullmean] => (-) => :logitmeandiff)

grandlogitmeandiff = @_ logitmeandiff |>
    groupby(__, [:λ, :fold]) |>
    combine(__, :logitmeandiff => mean => :logitmeandiff) |>
    sort!(__, [:λ]) |>
    groupby(__, [:fold]) |>
    transform!(__, :logitmeandiff =>
        (x -> filtfilt(digitalfilter(Lowpass(0.2), Butterworth(5)), x)) => :logitmeandiff)

pl = grandlogitmeandiff |> @vlplot() +
    @vlplot(:line,
        config = {},
        color = {:fold, type = :nominal,
            legend = {orient = :none, legendX = 175, legendY = 0.5}},
        x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
        title = "Regularization Parameter (λ)"},
        y     = {:logitmeandiff, aggregate = :mean, type = :quantitative,
                title = "Model - Null Model Accuracy (logit scale)"}) |>
    save(joinpath(dir, "grandmean.svg"))

# pick the largest valued λ, with a non-negative peak for meandiff
function pickλ(df)
    peaks = @_ maxima(df.logitmeandiff) |>
        filter(df.logitmeandiff[_] > 0.1, __)
    maxλ = argmax(df[peaks,:λ])
    df[peaks[maxλ],[:λ]]
end
λs = @_ grandlogitmeandiff |> groupby(__,:fold) |> combine(pickλ,__)

λs[!,:fold_text] .= string.("Fold: ",λs.fold)
λs[!,:yoff] = [0.1,0.15]

final_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)

pl = @vlplot() +
    vcat(
        logitmeandiff |> @vlplot(
        :line, width = 750, height = 100,
            color = {field = :condition, type = :nominal},
            x     = {:λ, scale = {type = :log, domain = [0.01, 0.35]},
                     title = "Regularization Parameter (λ)"},
            y     = {:nzcoef, aggregate = :max, type = :quantitative,
                     title = "# of non-zero coefficients (max)"}
        ),
        (
            @_ bestmeans |> DataFrames.transform(__, :mean => ByRow(x -> 100x) => :mean) |>
            @vlplot(
                width = 750, height = 200,
                x = {:λ, scale = {type = :log}},
                color = {field = :condition, type = :nominal},
            ) +
            @vlplot(
                :line,
                y = {:mean, aggregate = :mean, type = :quantitative,
                    title = "% Correct", scale = {domain = [50, 100]}},
            ) +
            @vlplot(
                :errorband,
                y = {:mean, aggregate = :ci, type = :quantitative}
            )
        ),
        (
            @vlplot() +
            (
                logitmeandiff |> @vlplot(
                    width = 750, height = 200,
                    x     = {:λ, scale = {type = :log},
                             title = "Regularization Parameter (λ)"},
                    color = {field = :condition, type = :nominal}) +
                @vlplot(:errorband,
                    y = {:logitmeandiff, aggregate = :ci,   type = :quantitative,
                         title = "Model - Null Model Accuracy (logit scale)"}) +
                @vlplot(:line,
                    y = {:logitmeandiff, aggregate = :mean, type = :quantitative})
            ) +
            (
                @vlplot(data = {values = [{}]}, encoding = {y = {datum = 0}}) +
                @vlplot(mark = {type = :rule, strokeDash = [2, 2], size = 2})
            ) +
            (
                @vlplot(data = λs) +
                @vlplot({:rule, strokeDash = [4, 4], size = 3}, x = :λ,
                    color = {value = "green"}) +
                @vlplot({:text, align = :left, dy = -8, size =  12, angle = 90},
                    text = :fold_text, x = :λ, y = :yoff)
            )
        )
    ) |> save(joinpath(dir, "switchtarget_lambdas.svg"))

#= # Compute the best window length
# -----------------------------------------------------------------

best_λs = vcat((DataFrame(sid = sid, λ = λ, fold = fi)
    for (fi, (λ, λ_fold)) in enumerate(zip(λs.λ, λ_folds))
    for sid in λ_fold[2])...)
λsid = groupby(best_λs, :sid)

windowmeans = @_ resultdf |>
    filter(_.λ ∈ (1.0, first(λsid[(sid = _.sid,)].λ)), __) |>
    groupby(__,[:condition, :sid, :fold, :λ, :winlen, :winstart]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean) |>
    transform!(__, :mean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitmean)

nullmeans = @_ windowmeans |>
    filter(_.λ == 1.0, __) |>
    deletecols!(__, [:λ, :mean]) |>
    rename!(__, :logitmean => :logitnullmean)

windowdiff = @_ windowmeans |>
    filter(_.λ != 1.0, __) |>
    innerjoin(nullmeans, __, on = [:condition, :sid, :fold, :winlen, :winstart]) |>
    transform!(__, [:logitmean, :logitnullmean] => (-) => :logitmeandiff)

windavg = @_ windowdiff |> groupby(__, [:condition, :fold, :winlen, :winstart]) |>
    combine(__, :logitmeandiff => mean => :logitmeandiff) |>
    groupby(__, [:fold, :winlen]) |>
    combine(__, :logitmeandiff => maximum => :logitmeandiff)

bestlens = @_ windavg |> groupby(__, [:fold]) |>
    combine(__, [:logitmeandiff, :winlen] =>
        ((m,l) -> l[argmax(m)]) => :winlen,
        :logitmeandiff => maximum => :logitmeandiff)

bestlen_bysid = @_ bestlens |>
    groupby(__, [:fold, :winlen, :logitmeandiff]) |>
    combine(__, :fold => (f -> λ_folds[f |> first][2]) => :sid) |>
    groupby(__, :sid)
    winlen_bysid(sid) = bestlen_bysid[(sid = sid,)].winlen |> first

pl = windowdiff |>
    @vlplot(:rect,
        config =  {view = {stroke = :transparent}},
        column = :condition,
        # row = :fold,
        y = {:winlen, type = :ordinal, axis = {format = ".2f"}, sort = :descending,
            title = "Length (s)"},
        x = {:winstart, type = :ordinal, axis = {format = ".2f"}, title = "Start (s)"},
        color = {:logitmeandiff, aggregate = :mean, type = :quantitative,
            scale = {scheme = "redblue", domainMid = 0}}) |>
    save(joinpath(dir, "switchtarget_windows.svg"))
 =#

# Plot near/far performance for early/late targets
# =================================================================

# -----------------------------------------------------------------

classdf_earlylate_file = joinpath(cache_dir("features"), "switch-target-freqmeans-earlylate.csv")

if isfile(classdf_earlylate_file)
    classdf_earlylate = CSV.read(classdf_earlylate_file)
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf_earlylate_groups = @_ events |>
        filter(ishit(_, region = "target") ∈ ["hit"], __) |>
        groupby(__, [:sid, :condition, :target_time_label, :target_switch_label])

    windows = [windowtarget(len = len, start = start)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in range(0, 2.5, length = 12)]
    classdf_earlylate = compute_freqbins(subjects, classdf_earlylate_groups, windows)

    # TASK: set condition on infiltrate to look at missing data case
    CSV.write(classdf_earlylate_file, classdf_earlylate)
end

# Compute classification accuracy
# -----------------------------------------------------------------

λsid = groupby(final_λs, :sid)

resultdf_earlylate_file = joinpath(cache_dir("models"), "switch-target-earlylate.csv")
classdf_earlylate[!,:fold] = in.(classdf_earlylate.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_earlylate_file) && mtime(resultdf_earlylate_file) > mtime(classdf_earlylate_file)
    resultdf_earlylate = CSV.read(resultdf_earlylate_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label]
    groups = groupby(classdf_earlylate, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
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
            on_model_exception = :throw,
            # on_missing_case    = :debug,
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

# Plot salience by early/late targets
# -----------------------------------------------------------------

classmeans = @_ resultdf_earlylate |>
    groupby(__, [:winstart, :winlen, :sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :mean)

classmeans_sum = @_ classmeans |>
    groupby(__, [:sid, :λ, :fold, :condition, :target_time_label]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ classmeans_sum |>
    filter(_.λ == 1.0, __) |>
    rename!(__, :mean => :nullmean) |>
    deletecols!(__, :λ)

nullmean, classdiffs =
    let l = logit ∘ shrinktowards(0.5, by = 0.01),
        C = mean(l.(nullmeans.nullmean)),
        tocor = x -> logistic(x + C)
    logistic(C),
    @_ classmeans_sum |>
        filter(_.λ != 1.0, __) |>
        innerjoin(__, nullmeans, on = [:condition, :sid, :fold, :target_time_label]) |>
        transform!(__, [:mean, :nullmean] => ByRow((x,y) -> (l(x)-l(y))) => :logitmeandiff) |>
        groupby(__, [:condition, :target_time_label]) |>
        combine(__,
            :logitmeandiff => tocor ∘ mean => :mean,
            :logitmeandiff => (x -> tocor(lowerboot(x, alpha = 0.318))) => :lower,
            :logitmeandiff => (x -> tocor(upperboot(x, alpha = 0.318))) => :upper,
        ) |>
        transform!(__, [:condition, :target_time_label] => ByRow(string) => :condition_time)
end

ytitle = ["Neural Switch-Classification", "Accuracy (Null Model Corrected)"]
barwidth = 14
yrange = [0.2, 0.8]
pl = classdiffs |>
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
            text = {datum = ["Less distinct", "response during switch"]}
        ) +
        @vlplot({:text, angle = 0, fontSize = 9, align = "left", baseline = "line-bottom",
            dx = -2barwidth - 17, dy = -24},
            color = {value = "#"*hex(darkgray)},
            x = {datum = "global"},
            y = {datum = yrange[2]},
            text = {datum = ["More distinct", "response during switch"]}
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

# Absolute values
# -----------------------------------------------------------------

summary = @_ classdiffs |>
    stack(__, [:mean, :nullmean], [:target_time_label, :condition, :sid],
        variable_name = :modeltype, value_name = :prop) |>
    groupby(__, [:target_time_label, :condition, :modeltype]) |>
    combine(__,
        :prop => mean => :prop,
        :prop => (x -> lowerboot(x; alpha = 0.318)) => :lower,
        :prop => (x -> upperboot(x; alpha = 0.318)) => :upper
    ) |>
    transform!(__, :modeltype => (x -> replace(string.(x), "mean" => "pmean")) => :modeltype)

ytitle = "Accuracy"
barwidth = 10
pl = summary |>
    @vlplot(
        # width = 121, #autosize = "fit",
        facet = {column = {field = :target_time_label, type = "nominal", title = nothing}},
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth}
        }
    ) + (
        @vlplot() +
        @vlplot({:bar, xOffset = -(barwidth/2)},
            transform = [{filter = "datum.modeltype == 'pmean'"}],
            # color = {value = "#"*hex(gray)},
            color = {:condition, title = nothing, scale = {range ="#".*hex.(colors)}},
            x = {:condition, title = nothing},
            y = {:prop, title = ytitle, scale = {domain = [0.3,0.8]}}
        ) +
        @vlplot({:rule, xOffset = -(barwidth/2)},
            transform = [{filter = "datum.modeltype == 'pmean'"}],
            color = {value = "black"},
            x = {:condition, title = nothing},
            y = {:lower, title = ytitle}, y2 = :upper
        ) +
        @vlplot({:bar, xOffset = (barwidth/2)},
            transform = [{filter = "datum.modeltype == 'nullmean'"}],
            color = {value = "#"*hex(gray)},
            x = {:condition, title = nothing},
            y = {:prop, title = ytitle}
        ) +
        @vlplot({:rule, xOffset = (barwidth/2)},
            transform = [{filter = "datum.modeltype == 'nullmean'"}],
            color = {value = "black"},
            x = {:condition, title = nothing},
            y = {:lower, title = ytitle}, y2 = :upper
        )
    );
pl |> save(joinpath(dir, "switch_target_absolute_earlylate.svg"))

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

fig = svg.Figure("89mm", "160mm", # "240mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "behavior_timeline.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold")
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "switch_target_earlylate.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold")
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
classdfsal_earlylate_[!,:fold] = in.(classdfsal_earlylate_.sid, Ref(Set(λ_folds[1][1]))) .+ 1

if isfile(resultdf_sal_earlylate_file) && mtime(resultdf_sal_earlylate_file) > mtime(classdf_sal_earlylate_file)
    resultdf_sal_earlylate = CSV.read(resultdf_sal_earlylate_file)
else
    factors = [:fold, :winlen, :winstart, :condition, :target_time_label, :salience_label]
    groups = groupby(classdfsal_earlylate_, factors)

    progress = Progress(length(groups))
    function findclass((key, sdf))
        λ = first(λsid[(sid = first(sdf.sid),)].λ)
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
