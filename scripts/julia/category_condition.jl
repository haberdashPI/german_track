# Setup
# =================================================================

using DrWatson; @quickactivate("german_track")

using GermanTrack, DataFrames, Statistics, Dates, Underscores, Random, Printf,
    ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers, Infiltrator, Peaks,
    StatsFuns, Distributions, DSP, DataStructures, Colors, Bootstrap, CSV, EEGCoding,
    JSON3, DataFramesMeta, Lasso
wmean = GermanTrack.wmean
n_winlens = 6

dir = mkpath(joinpath(plotsdir(), "figure2_parts"))

using GermanTrack: colors, neutral, patterns

# Fig 2A: behavioarl hit rate
# =================================================================

ascondition = Dict(
    "test" => "global",
    "feature" => "spatial",
    "object" => "object"
)

file = joinpath(raw_datadir("behavioral", "export_ind_data.csv"))
rawdata = @_ CSV.read(file, DataFrame) |>
    transform!(__, :block_type => ByRow(x -> ascondition[x]) => :condition)

meansraw = @_ rawdata |>
    groupby(__, [:sid, :condition, :exp_id]) |>
    @combine(__,
        hr = sum(:perf .== "hit") / sum(:perf .∈ Ref(Set(["hit", "miss"]))),
        fr = sum(:perf .== "false") / sum(:perf .∈ Ref(Set(["false", "reject"]))),
    )

bad_sids = @_ meansraw |>
    @where(__, :condition .== "global") |>
    @where(__, (:hr .<= :fr) .| (:fr .>= 1)) |> __.sid |> Ref |> Set

meansclean = @_ meansraw |>
    @where(__, :sid .∉ bad_sids) |>
    stack(__, [:hr, :fr], [:sid, :condition, :exp_id], variable_name = :type, value_name = :prop)


CSV.write(joinpath(processed_datadir("analyses"), "behavioral_condition.csv"), meansclean)

run(`Rscript $(joinpath(scriptsdir("R"), "condition_behavior.R"))`)

file = joinpath(processed_datadir("analyses"), "behavioral_condition_coefs.csv")
means = @_ CSV.read(file, DataFrame) |>
    rename(__, :propr_med => :prop, :propr_05 => :lower, :propr_95 => :upper) |>
    @transform(__, condition = categorical(:condition,
        levels = ["global", "spatial", "object"], ordered = true)) |>
    sort!(__, :condition)

barwidth = 20
pl = means |> @vlplot(
        width = 242, height = 150, autosize = "fit",
        # width = {step = 50},
        x = {:condition,
            type = :nominal,
            sort = ["global", "spatial", "object"],
            axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }) +
    @vlplot({:bar, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {:condition, scale = {range = "#".*hex.(colors)}}) +
    @vlplot({:line, xOffset = -(barwidth/2), size = 1},
        transform = [{filter = "datum.type == 'hr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {value = "black"}) +
    @vlplot({:bar, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'fr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean},
        color = {value = "#"*hex(neutral)}) +
    @vlplot({:line, xOffset = (barwidth/2), size = 1},
        transform = [{filter = "datum.type == 'fr'"}],
        y = {:prop, type = :quantitative, aggregate = :mean,
                scale = {domain = [0, 1]}, title = "Response Rate"},
        color = {value = "black"}) +
    @vlplot({:rule, xOffset = -(barwidth/2)},
        transform = [{filter = "datum.type == 'hr'"}],
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:rule, xOffset = (barwidth/2)},
        transform = [{filter = "datum.type == 'fr'"}],
        color = {value = "black"},
        y = {:lower, title = ""}, y2 = :upper
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-2},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'hr'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        y = {:prop, aggregate = :mean, type = :quantitative},
        text = {value = "Hits"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'fr'"}],
        # x = {datum = "spatial"}, y = {datum = },
        y = {datum = 0},
        text = {value = "False Positives"},
    );
pl |> save(joinpath(dir, "fig2a.svg"))

# Find best λs
# =================================================================

file = joinpath(processed_datadir("analyses"), "condition_lambdas.json")
GermanTrack.@cache_results file fold_map hyperparams begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    classdf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(len = len, start = start)
                for len in 2.0 .^ range(-1, 1, length = 10),
                    start in [0; 2.0 .^ range(-2, 2, length = 10)]],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    resultdf = @_ classdf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :condition_lambda_folds)) |>
        groupby(__, [:winstart, :winlen]) |>
        filteringmap(__, desc = "Evaluating lambdas...", folder = foldxt,
            :cross_fold => 1:10,
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            function(sdf, fold, comparison)
                train = filter(x -> x.fold != fold, sdf)
                test  = filter(x -> x.fold == fold, sdf)
                conds = split(comparison, "-v-")

                y = train.condition .== first(conds)
                model = fit(ZScoring(LassoPath, [(0:29) .+ i for i in 1:30:150]),
                    Array(train[:,r"channel"]), y, Bernoulli(), standardize = false)

                y = test.condition .== first(conds)
                ŷ = predict(model, Array(test[:,r"channel"]), select = MinAICc())
                test.predict = conds[ifelse.(ŷ .> 0.5, 1, 2)]
                test.correct = test.predict .== test.condition
                test.nzero = sum(!iszero, coef(model, MinAICc()))

                test
            end)

    fold_map = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold) |>
        Dict(row.sid => row.fold for row in eachrow(__))

    win_means = @_ resultdf |>
        groupby(__, [:comparison, :winlen, :winstart, :sid, :fold]) |>
        @combine(__, mean = GermanTrack.wmean(:correct, :weight)) |>
        groupby(__, [:winlen, :winstart]) |>
        filteringmap(__, desc = nothing,
            :fold => cross_folds(1:10),
            (sdf, fold) -> DataFrame(mean = mean(sdf.mean)))

    @_ win_means |>
        @vlplot(:line,
            # column = :comparison,
            x = :winstart, y = :mean,
            color = :winlen
        ) |> save(joinpath(dir, "supplement", "window_means.svg"))

    hyperparams = @_ win_means |>
        groupby(__, :fold) |>
        @combine(__,
            best = maximum(:mean),
            winlen = :winlen[argmax(:mean)],
            winstart = :winstart[argmax(:mean)]
        ) |>
        Dict(row.fold => (;row[Not(:fold)]...) for row in eachrow(__))

    @info "Saving plots to $(joinpath(dir, "supplement"))"
end

# Compute condition categorization, and several baselines (sanity checks)
# =================================================================

file = joinpath(processed_datadir("analyses"), "condition-and-baseline.json")
GermanTrack.@cache_results file predictbasedf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    windowtypes = [
        "target"    => (;kwds...) -> windowtarget(name = "target"; kwds...)
        "rndbefore" => (;kwds...) -> windowbase_bytarget(>; name = "rndbefore",
            mindist = 0.5, minlength = 0.5, onempty = missing, kwds...)
    ]

    start_len = unique((x.winstart, x.winlen) for x in values(hyperparams))

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(ishit) => :hittype) |>
        transform!(__, :sid => ByRow(x -> fold_map[x]) => :fold) |>
        groupby(__, [:sid, :condition, :hittype]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [winfn(start = start, len = len)
                for (name, winfn) in windowtypes
                for (start, len) in start_len],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)

    modeltypes = (
        "full" => x -> x.windowtype == "target",
        "null" => x -> x.windowtype == "target",
        "random-labels" => x -> x.windowtype == "target",
        "random-window-before" => x -> x.windowtype == "rndbefore",
        "random-trialtype" => x -> x.windowtype == "target"
    )

    modelshufflers = Dict(
        "random-labels" => (df -> @_ df |>
            groupby(__, [:sid, :winlen, :windowtype, :hittype]) |>
            transform(__, :condition => shuffle => :condition)),
        "random-trialtype" => (df -> @_ df |>
            groupby(__, [:sid, :condition, :winlen, :windowtype]) |>
            transform(__, :hittype => shuffle => :hittype))
    )

    predictbasedf = @_ classdf |>
        transform!(__, :sid => ByRow(sid -> fold_map[sid]) => :fold) |>
        groupby(__, :hittype) |>
        filteringmap(__, folder = foldl, desc = "Classifying conditions...",
            :cross_fold => 1:10,
            :comparison => (
                "global-v-object"  => x -> x.condition ∈ ["global", "object"],
                "global-v-spatial" => x -> x.condition ∈ ["global", "spatial"],
                "object-v-spatial" => x -> x.condition ∈ ["object", "spatial"],
            ),
            :modeltype => modeltypes,
            function (sdf, fold, comparison, modeltype)
                selector = modeltype == "null" ? NullSelect() : MinAICc()
                sdf = get(modelshufflers, modeltype, identity)(sdf)

                win = hyperparams[fold]
                sdf = filter(x -> x.winstart == win.winstart && x.winlen == win.winlen, sdf)
                train = filter(x -> x.fold != fold, sdf)
                test  = filter(x -> x.fold == fold, sdf)
                conds = split(comparison, "-v-")

                y = train.condition .== first(conds)

                model = fit(ZScoring(LassoPath, [(0:29) .+ i for i in 1:30:150]),
                    Array(train[:,r"channel"]), y, Bernoulli(), standardize = false,
                    maxncoef = size(view(train,:,r"channel"), 2)
                )

                y = test.condition .== first(conds)
                ŷ = predict(model, Array(test[:,r"channel"]), select = selector)
                test.predict = conds[ifelse.(ŷ .> 0.5, 1, 2)]
                test.correct = test.predict .== test.condition
                test.nzero = sum(!iszero, coef(model, selector))

                test
            end)
end

# Main EEG results (Figure 2B)
# =================================================================

predictmeans = @_ predictbasedf |>
    filter(_.modeltype ∈ ["null", "full"] && _.hittype == "hit", __) |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype, :hittype]) |>
    combine(__,
        :correct => mean => :mean,
        :correct => logit ∘ shrinktowards(0.5, by=0.01) ∘ mean => :logitcorrect)

nullmeans = @_ predictmeans |>
    filter(_.modeltype == "null", __) |>
    deletecols!(__, [:logitcorrect, :modeltype]) |>
    rename!(__, :mean => :nullmean)

statdata = @_ predictmeans |>
    filter(_.modeltype == "full", __) |>
    innerjoin(__, nullmeans, on = [:sid, :comparison, :hittype]) |>
    transform!(__, :mean => ByRow(shrinktowards(0.5, by = 0.01)) => :shrinkmean) |>
    transform!(__, :mean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitmean) |>
    transform!(__, :nullmean => ByRow(logit ∘ shrinktowards(0.5, by = 0.01)) => :logitnullmean)

file = joinpath(processed_datadir("analyses"), "eeg_condition.csv")
CSV.write(file, statdata)
run(`Rscript $(joinpath(scriptsdir("R"), "condition_eeg.R"))`)

compnames = OrderedDict(
    "global_v_object"  => "Global vs.\n Object",
    "global_v_spatial" => "Global vs.\n Spatial",
    "object_v_spatial" => "Object vs.\n Spatial")

statfile = joinpath(processed_datadir("analyses"), "eeg_condition_coefs.csv")
plotdata = @_ CSV.read(statfile, DataFrame) |>
    rename(__, :propr_med => :mean, :propr_05 => :lower, :propr_95 => :upper) |>
    @transform(__, compname = map(x -> compnames[x], :comparison))
nullmean = logistic(mean(statdata.logitnullmean))

ytitle = "Condition Classification"
barwidth = 25
plhit = @_ plotdata |>
    @vlplot(
        # facet = { column = { field = :hittype, type = :nominal} },
        width = 237, height = 150,
        autosize = "fit",
        config = {
            bar = {discreteBandSize = barwidth},
            axis = {labelFont = "Helvetica", titleFont = "Helvetica"},
            legend = {disable = true, labelFont = "Helvetica", titleFont = "Helvetica"},
            header = {labelFont = "Helvetica", titleFont = "Helvetica"},
            mark = {font = "Helvetica"},
            text = {font = "Helvetica"},
            title = {font = "Helvetica", subtitleFont = "Helvetica"}
        }
    ) + (
    @vlplot(x = {:compname, axis = {
            labelAngle = 0,
            title = "",
            labelExpr = "split(datum.label, '\\n')"}},
        color = {
            :compname, title = nothing,
            scale = {range = ["url(#mix1_2)", "url(#mix1_3)", "url(#mix2_3)"]}}) +
    @vlplot({:bar},
        y = {:mean,
            scale = {domain = [0.5, 1]},
            title = ytitle}) +
    @vlplot({:rule},
        color = {value = "black"},
        y2 = :upper,
        y = {:lower,
            scale = {domain = [0.5, 1]},
            title = ytitle})
    ) +
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:rule, strokeDash = [4 4], size = 2},
            y = {datum = nullmean},
            color = {value = "black"}) +
        @vlplot({:text, align = "left", dx = 15, dy = 0, baseline = "line-bottom", fontSize = 9},
            y = {datum = nullmean},
            x = {datum = "Object vs.\n Spatial"},
            text = {value = ["Null Model", "Accuracy"]}
        )
    );
plotfile = joinpath(dir, "fig2b.svg")
plhit |> save(plotfile)
addpatterns(plotfile, patterns, size = 10)

# EEG Cross-validated Features
# =================================================================

GermanTrack.@cache_results file coefdf begin
    classdf = @_ events |>
        filter(ishit(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        filteringmap(__, desc = "Computing features...",
            :window => [windowtarget(len = len, start = start)
                for len in 2.0 .^ range(-1, 1, length = 10),
                    start in [0; 2.0 .^ range(-2, 2, length = 10)]],
            compute_powerbin_features(_1, subjects, _2)) |>
        deletecols!(__, :window)
end

# Combine Figures (Full Figure 2)
# =================================================================

GermanTrack.@usepython

svg = pyimport("svgutils").compose

background_file = tempname()

background = pyimport("svgutils").transform.fromstring("""
    <svg>
        <rect width="100%" height="100%" fill="white"/>
    </svg>
""").save(background_file)

fig = svg.Figure("89mm", "125mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig2a.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(220,15)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig2b.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,15)
    ).move(0, 180)
).scale(1.333).save(joinpath(plotsdir("figures"), "fig2.svg"))




