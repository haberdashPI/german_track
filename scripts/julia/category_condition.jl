# Setup
# =================================================================

using DrWatson
@quickactivate("german_track")

using GermanTrack, DataFrames, Statistics, Dates, Underscores, Random, Printf,
    ProgressMeter, VegaLite, FileIO, StatsBase, BangBang, Transducers, Infiltrator, Peaks,
    StatsFuns, Distributions, DSP, DataStructures, Colors, Bootstrap, CSV, EEGCoding,
    JSON3, DataFramesMeta, Lasso, Indexing
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
        width = 190, height = 130,
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

# Presentation
# -----------------------------------------------------------------

barwidth = 13
pl = means |> @vlplot(
        width = 145, height = 75,
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
    @vlplot({:text, angle = -90, fontSize = 9, align = "right", baseline = "bottom", dx = 0, dy = -barwidth-1},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'hr'"}],
        # x = {datum = "spatial"}, y = {datum = 0.},
        y = {:prop, aggregate = :mean, type = :quantitative},
        text = {value = "Hits"},
    ) +
    @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+1},
        transform = [{filter = "datum.condition == 'global' && datum.type == 'fr'"}],
        # x = {datum = "spatial"}, y = {datum = },
        y = {datum = 0},
        text = {value = "False Positives"},
    );
pl |> save(joinpath(dir, "present", "fig2a.svg"))


# Find best λs
# =================================================================

prefix = joinpath(processed_datadir("analyses"), "condition-lambdas")
GermanTrack.@use_cache prefix foldmap hyperparamsdf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    windows = [
        windowtarget(len = len, start = start)
        for len in 2.0 .^ range(-1, 1, length = 10),
            start in [0; 2.0 .^ range(-2, 2, length = 10)]
    ]

    classdf = @_ events |>
        filter(findresponse(_) == "hit", __) |>
        groupby(__, [:sid, :condition]) |>
        repeatby(__, :window => windows) |>
        tcombine(__, desc = "Computing features...",
            df -> compute_powerbin_features(df, subjects, df.window[1]))

    lambdas = 10.0 .^ range(-2, 0, length=100)
    resultsetup = @_ classdf |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :condition_lambda_folds)) |>
        groupby(__, [:winstart, :winlen]) |>
        repeatby(__,
            :cross_fold => 1:10,
            :comparison => [
                ("global", "object"),
                ("global", "spatial"),
                ("object", "spatial")
            ]
        ) |>
        @where(__, :condition .∈ :comparison)

    resultdf = tcombine(resultsetup, desc = "Evaluating lambdas...") do sdf
        rng = stableRNG(2019_11_18, :condition_lambda_folds,
            NamedTuple(sdf[1, Not(r"channel")]))
        test, model = traintest(sdf, sdf.cross_fold[1], y = :condition,
            selector = m -> AllSeg(), λ = lambdas,
            validate_rng = rng)
        test[:, Not(r"channel")]
    end

    foldmap = @_ resultdf |>
        groupby(__, :sid) |> combine(__, :fold => first => :fold)

    folder = @_ resultdf |>
        repeatby(__, :cross_fold => 1:10) |>
        @where(__, :fold .!= :cross_fold)
    hyperparamsdf = combine(folder) do sdf
        meandf = @_ sdf |> groupby(__, [:comparison, :winlen, :winstart, :sid, :λ]) |>
            @combine(__, mean = mean(:val_accuracy)) |>
            groupby(__, [:winlen, :winstart, :λ]) |>
            @combine(__, mean = mean(:mean))

        thresh = maximum(meandf.mean)
        λ = maximum(meandf.λ[meandf.mean .>= thresh])

        meandf_ = meandf[meandf.λ .== λ, :]
        best = maximum(meandf_.mean)
        selected = meandf_[findfirst(meandf_.mean .== best), :]

        (fold = sdf.cross_fold[1], λ = λ, mean = best, selected[[:winlen, :winstart]]...)
    end

    # GermanTrack.@save_cache prefix foldmap hyperparamsdf
end

# Compute condition categorization, and several baselines (sanity checks)
# =================================================================

prefix = joinpath(processed_datadir("analyses"), "condition-and-baseline")
GermanTrack.@use_cache prefix predictbasedf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    windowtypes = [
        "target"    => (;kwds...) -> windowtarget(name = "target"; kwds...)
        "rndbefore" => (;kwds...) -> windowbase_bytarget(>; name = "rndbefore",
            mindist = 0.5, minlength = 0.5, onempty = missing, kwds...)
    ]

    start_len = unique(hyperparamsdf[:, [:winstart, :winlen]]) |> eachrow

    windows = [winfn(start = start, len = len)
        for (name, winfn) in windowtypes
        for (start, len) in start_len]

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
        groupby(__, [:sid, :condition, :hittype]) |>
        repeatby(__, :window => windows) |>
        tcombine(__, desc = "Computing features...",
            df -> compute_powerbin_features(df, subjects, df.window[1])) |>
        innerjoin(__, foldmap, on = :sid)

    modeltypes = OrderedDict(
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

    predictsetup = @_ classdf |>
        groupby(__, :hittype) |>
        repeatby(__,
            :cross_fold => 1:10,
            :modeltype => keys(modeltypes),
            :comparison => [
                ("global", "object"),
                ("global", "spatial"),
                ("object", "spatial")
            ]
        ) |>
        @where(__, :condition .∈ :comparison)

    predictbasedf = tcombine(predictsetup, desc = "Classifying conditions...") do sdf
        modeltype = sdf.modeltype[1]
        hyper = only(eachrow(filter(x -> x.fold == sdf.cross_fold[1], hyperparamsdf)))

        sdf = @_ filter(modeltypes[modeltype], sdf) |>
            get(modelshufflers, modeltype, identity) |>
            filter(x -> x.winstart == hyper.winstart && x.winlen == hyper.winlen, __)

        selector = modeltype == "null" ? m -> NullSelect() : hyper.λ
        test, model = traintest(sdf, sdf.cross_fold[1], y = :condition, selector = selector)

        test[:, Not(r"channel")]
    end

    # GermanTrack.@save_cache prefix predictbasedf
end

# Main EEG results (Figure 2B)
# =================================================================

predictmeans = @_ predictbasedf |>
    filter(_.modeltype ∈ ["null", "full"] && _.hittype == "hit", __) |>
    @transform(__, comparison = join.(:comparison, "-v-")) |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen, :hittype]) |>
    combine(__, [:correct, :weight] => GermanTrack.wmean => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype, :hittype]) |>
    combine(__,
        :correct => mean => :mean,
        :correct => logit ∘ shrinktowards(0.5, by=0.01) ∘ mean => :logitcorrect)

nullmeans = @_ predictmeans |>
    filter(_.modeltype == "null", __) |>
    select!(__, Not([:logitcorrect, :modeltype])) |>
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
        width = 180, height = 130,
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

# Presentation version
# -----------------------------------------------------------------

ytitle = "Accuracy"
barwidth = 25
plhit = @_ plotdata |>
    @vlplot(
        # facet = { column = { field = :hittype, type = :nominal} },
        width = 170, height = 75,
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
    );
plotfile = joinpath(dir, "present", "fig2b.svg")
plhit |> save(plotfile)
addpatterns(plotfile, patterns, size = 10)

# EEG Cross-validated Features
# =================================================================

prefix = joinpath(processed_datadir("analyses"), "condition_coefs")
GermanTrack.@use_cache prefix coefdf begin
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

    start_len = unique(hyperparamsdf[:, [:winstart, :winlen]]) |> eachrow

    classdf = @_ events |>
        transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
        @where(__, :hittype .== "hit") |>
        groupby(__, [:sid, :condition]) |>
        repeatby(__, :window => [windowtarget(start = start, len = len)
            for (start, len) in start_len]) |>
        tcombine(__, desc = "Computing features...",
            df -> compute_powerbin_features(df, subjects, df.window[1])) |>
        innerjoin(__, foldmap, on = :sid)

    coefsetup = @_ classdf |>
        repeatby(__,
            :cross_fold => 1:10,
            :comparison => [
                ("global", "object"),
                ("global", "spatial"),
                ("object", "spatial")
            ]
        ) |>
        @where(__, :condition .∈ :comparison)

    coefdf = tcombine(coefsetup, desc = "Evaluating lambdas...") do sdf
        hyper = only(eachrow(filter(x -> x.fold == sdf.cross_fold[1], hyperparamsdf)))
        sdf = filter(x -> x.winstart == hyper.winstart && x.winlen == hyper.winlen, sdf)

        test, model = traintest(sdf, sdf.cross_fold[1], y = :condition, selector = hyper.λ)
        coefs = coef(model)'
        result = DataFrame(coefs, vcat("I", names(view(sdf, :, r"channel"))))
        result[!, :cross_fold] .= sdf.cross_fold[1]
        result[!, :comparison] .= Ref(sdf.comparison[1])

        result
    end

    GermanTrack.@save_cache prefix coefdf
end

# Plotting
# -----------------------------------------------------------------

longdf = @_ coefdf |>
stack(__, All(r"channel"), [:cross_fold, :comparison]) |>
@transform(__,
    channel = parse.(Int, getindex.(match.(r"channel_([0-9]+)", :variable), 1)),
    freqbin = getindex.(match.(r"channel_[0-9]+_(.*)$", :variable), 1)
)

pl = @_ longdf |>
    groupby(__, [:freqbin, :comparison, :cross_fold]) |>
    @combine(__, value = mean(abs, :value)) |>
    groupby(__, [:freqbin, :comparison]) |>
    @combine(__,
        value = median(:value),
        lower2 = quantile(:value, 0.05),
        upper2 = quantile(:value, 0.95),
        lower1 = quantile(:value, 0.25),
        upper1 = quantile(:value, 0.75)
    ) |>
    @vlplot(
        facet = {column = {field = :comparison, type = :nominal}}
    ) + (
        @vlplot() +
        (
            @vlplot(x = {:freqbin,
                sort = string.(keys(GermanTrack.default_freqbins)),
                type = :ordinal,
            }) +
            @vlplot(:rule, y = :lower2, y2 = :upper2) +
            @vlplot({:rule, size = 4}, y = :lower1, y2 = :upper1) +
            @vlplot({:point, size = 50, filled = true}, y = :value, color = {value = "black"})
        )
    );
pl |> save(joinpath(dir, "fig2c.svg"))

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

fig = svg.Figure("174mm", "67mm",
    svg.SVG(background_file),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig2a.svg")).move(0,15),
        svg.Text("A", 2, 10, size = 12, weight="bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "behavior.svg")).
            scale(0.1).move(210,15)
    ).move(0, 0),
    svg.Panel(
        svg.SVG(joinpath(dir, "fig2b.svg")).move(0,15),
        svg.Text("B", 2, 10, size = 12, weight = "bold", font = "Helvetica"),
        svg.SVG(joinpath(plotsdir("icons"), "eeg.svg")).
            scale(0.1).move(220,15)
    ).move(240, 0)
).scale(1.333).save(joinpath(plotsdir("figures"), "fig2.svg"))
