# Setup
# =================================================================

if Threads.nthreads() == 1
    @warn "This script is designed to working using multiple threads; start julia using "*
        "`julia -t auto`"
end

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_absolute_features = true
n_winlens = 6

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux,
    JSON3, JSONTables, Tables, Infiltrator, FileIO, BlackBoxOptim, RCall, Peaks,
    Distributions

R"library(ggplot2)"
R"library(dplyr)"

DrWatson._wsave(file, data::Dict) = open(io -> JSON3.write(io, data), file, "w")

# local only pac kages
using Formatting

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# then, whatever choice we make, run an analysis  to evaluate
# the tradeoff of λ and % correct

wmeanish(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))

dir = joinpath(plotsdir(), string("results_", Date(now())))
isdir(dir) || mkdir(dir)

best_λs = CSV.read(joinpath(processed_datadir("classifier_params"),"best-lambdas.json"))

# Mean Frequency Bin Analysis (Timeline)
# =================================================================

classdf_file = joinpath(cache_dir("features"), savename("cond-freaqmeans-timeline",
    (absolute = use_absolute_features, n_winlens = n_winlens), "csv"))

spread(scale,npoints)   = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))

if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    windows = [(len = len, start = start, before = -len)
        for len in spread(1, 0.5, n_winlens),
            start in [0.0]]

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    classdf_groups = @_ events |>
        filter(_.target_present, __) |>
        filter(ishit(_, region = "target") == "hit", __) |>
        groupby(__, [:sid, :condition])

    classdf = compute_freqbins(subjects, classdf_groups, windowtarget, windows)
    CSV.write(classdf_file, classdf)
end
classdf = innerjoin(classdf, best_λs, on = [:sid])

# Timeline Plots
# =================================================================

classcomps = [
    "global-v-object"  => @_(classdf |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf |> filter(_.condition in ["object", "spatial"], __))
]

resultdf = mapreduce(append!!, classcomps) do (comp, data)
    function findclass((key, sdf))
        result = testclassifier(LassoPathClassifiers([1.0, sdf.λ |> first]),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 10,
            seed = stablehash(:cond_coef_timeline,2019_11_18),
            irls_maxiter = 100,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, [:winstart, :winlen, :fold]))
    foldxt(append!!, Map(findclass), collect(groups))
end

means = @_ resultdf |>
    groupby(__, [:winlen, :winstart, :comparison, :sid, :fold, :λ]) |>
    combine(__, [:correct, :weight] => wmeanish => :mean)

winstart_means = @_ means |>
    groupby(__, [:winstart, :comparison, :sid, :fold, :λ]) |>
    combine(__, :mean => mean => :mean)

nullmeans = @_ winstart_means |>
    filter(_.λ == 1.0, __) |>
    deletecols!(__, :λ) |>
    rename!(__, :mean => :nullmean)

meandiff = @_ winstart_means |>
    filter(_.λ != 1.0, __) |>
    innerjoin(nullmeans, __, on = [:comparison, :sid, :fold, :winstart]) |>
    transform!(__, [:mean, :nullmean] => (-) => :meandiff)

pl = meandiff |>
    @vlplot(width = 600, height = 300,
        color = {field = :comparison, type = :nominal}, x = :winstart) +
    @vlplot(:line,  y = {:meandiff, aggregate = :mean, type = :quantitative}) +
    @vlplot(:errorband,  y = {:meandiff, aggregate = :ci, type = :quantitative})

pl |> save(joinpath(dir, "condition_timeline.svg"))

# select data at the central time point (and length)
# -----------------------------------------------------------------

centerlen = @_ classdf.winlen |> unique |> sort! |> __[4]
centerstart = @_ classdf.winstart |> unique |> __[argmin(abs.(__ .- 0.0))]

classdf_atlen = @_ classdf |> filter(_.winlen == centerlen && _.winstart == centerstart, __)

classcomps_atlen = [
    "global-v-object"  => @_(classdf_atlen |> filter(_.condition in ["global", "object"],  __)),
    "global-v-spatial" => @_(classdf_atlen |> filter(_.condition in ["global", "spatial"], __)),
    "object-v-spatial" => @_(classdf_atlen |> filter(_.condition in ["object", "spatial"], __))
]

# test accuracy
# -----------------------------------------------------------------

nullmodels = mapreduce(append!!, classcomps_atlen) do (comp, data)
    function findclass((key, sdf))
        result = testclassifier(LassoClassifier(0.5),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 10,
            seed = stablehash(:cond_coef_timeline,2019_11_18),
            irls_maxiter = 100,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, :fold))
    foldl(append!!, Map(findclass), collect(groups))
end

nullmeans = @_ nullmodels |>
    groupby(__, [:sid, :comparison]) |>
    combine(__, [:correct, :weight] => wmeanish => :correct)

nullmeans |>
    @vlplot(:bar,
        x = :comparison,
        y = {:correct, aggregate = :mean, type = :quantitative,
            scale = {domain = [0.4, 1]}})


# Different baseline models
# =================================================================

# Features
# -----------------------------------------------------------------

classbasedf_file = joinpath(cache_dir("features"), savename("baseline-freqmeans",
    (n_winlens = n_winlens, ), "csv"))

spread(scale,npoints)   = x -> spread(x,scale,npoints)
spread(x,scale,npoints) = quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints))

if use_cache && isfile(classbasedf_file)
    classbasedf = CSV.read(classbasedf_file)
else
    windows = [(len = len, start = 0.0)
        for len in spread(1, 0.5, n_winlens)]

    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
    classbasedf_groups = @_ events |>
        transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
        groupby(__, [:sid, :condition, :hittype])

    baseparams = (mindist = 0.5, minlength = 0.5, onempty = missing)
    windowtypes = [
        "target"    => windowtarget,
        "rndbefore" => windowbase_bytarget(>; baseparams...),
        "rndafter"  => windowbase_bytarget(<; baseparams...),
        "baseline"  => windowbaseline(; baseparams...)
    ]
    classbasedf = mapreduce(append!!, windowtypes) do (windowtype, windowfn)
        result = compute_freqbins(subjects, classbasedf_groups, windowfn, windows)
        result[!, :windowtype] .= windowtype
        result
    end
    CSV.write(classbasedf_file, classbasedf)
    alert("Feature computation complete!")
end

classbasedf = innerjoin(classbasedf, best_λs, on = [:sid])

# Classification for different baselines
# -----------------------------------------------------------------

modeltype = [
    "full"            => (
        filterfn = @_(filter(_.windowtype == "target" && _.hittype == "hit", __)),
        λfn = df -> df.λ |> first
    ),
    "null"            => (
        filterfn = @_(filter(_.windowtype == "target" && _.hittype == "hit", __)),
        λfn = df -> 1.0,
    ),
    "random-labels" => (
        filterfn = df ->
            @_(df |> filter(_.windowtype == "target" && _.hittype == "hit", __) |>
                     groupby(__, [:sid, :winlen, :windowtype]) |>
                     transform!(__, :condition => shuffle => :condition)),
        λfn = df -> df.λ |> first
    ),
    "random-window"   => (
        filterfn = @_(filter(_.windowtype == "baseline" && _.hittype == "hit", __)),
        λfn = df -> df.λ |> first
    ),
    "random-window-before"   => (
        filterfn = @_(filter(_.windowtype == "rndbefore" && _.hittype == "hit", __)),
        λfn = df -> df.λ |> first
    ),
    "random-window-after"   => (
        filterfn = @_(filter(_.windowtype == "rndafter" && _.hittype == "hit", __)),
        λfn = df -> df.λ |> first
    ),
    "random-trialtype" => (
        filterfn = df ->
            @_(df |> filter(_.windowtype == "target", __) |>
                     groupby(__, [:sid, :condition, :winlen, :windowtype]) |>
                     transform!(__, :hittype => shuffle => :hittype) |>
                     filter(_.hittype == "hit", __)),
        λfn = df -> df.λ |> first
    )
]

comparisons = [
    "global-v-object"  => @_(filter(_.condition ∈ ["global", "object"],  __)),
    "global-v-spatial" => @_(filter(_.condition ∈ ["global", "spatial"], __)),
    "object-v-spatial" => @_(filter(_.condition ∈ ["object", "spatial"], __)),
]

function bymodeltype(((key, df), type, comp))
    df = df |> comp[2] |> type[2].filterfn

    result = testclassifier(LassoClassifier(type[2].λfn(df)),
        data = df, y = :condition, X = r"channel",
        crossval = :sid, n_folds = 10,
        seed = stablehash(:cond_baseline,2019_11_18),
        irls_maxiter = 100,
        weight = :weight, on_model_exception = :throw
    )
    result[!, :modeltype]  .= type[1]
    result[!, :comparison] .= comp[1]
    result[!, keys(key)] .= permutedims(collect(values(key)))

    result
end

predictbasedf = @_ classbasedf |>
    groupby(__, [:fold, :winlen]) |> pairs |>
    Iterators.product(__, modeltype, comparisons) |>
    collect |> foldxt(append!!, Map(bymodeltype), __)

# Plot results
# -----------------------------------------------------------------

predictmeans = @_ predictbasedf |>
    groupby(__, [:sid, :comparison, :modeltype, :winlen]) |>
    combine(__, [:correct, :weight] => wmeanish => :correct) |>
    groupby(__, [:sid, :comparison, :modeltype]) |>
    combine(__, :correct => mean => :correct)


predictmeans |>
    @vlplot(:bar,
        column = :modeltype,
        x = :comparison,
        y = {:correct, aggregate = :mean, type = :quantitative,
            scale = {domain = [0.2, 1]}})

baselines = @_ predictmeans |>
    filter(_.modeltype != "full", __) |>
    rename!(__, :correct => :baseline)

Nmtypes = length(baselines.modeltype |> unique)
refline = DataFrame(x = repeat([0,1],Nmtypes), y = repeat([0,1],Nmtypes),
    modeltype = repeat(baselines.modeltype |> unique, inner=2))
nrefs = size(refline,1)
refline = @_ refline |>
    repeat(__, 3) |>
    insertcols!(__, :comparison => repeat(baselines.comparison |> unique, inner=nrefs))

compnames = Dict(
    "global-v-object"  => "Global vs. Object",
    "global-v-spatial" => "Global vs. Spatial",
    "object-v-spatial" => "Object vs. Spatial")

modelnames = OrderedDict(
    # "random-window" => "Random\nWindow",
    "random-window-before" => "Random\nPre-target\nWindow",
    # "random-window-after" => "Random\nPost-target Window",
    "null" => "Null Model",
    "random-labels" => "Random\nLabels",
    "random-trialtype" => "Random\nTrial Type",
)

plotmeans = @_ predictmeans |>
    filter(_.modeltype == "full", __) |>
    deletecols(__, :modeltype) |>
    innerjoin(__, baselines, on = [:sid, :comparison]) |>
    filter(_.modeltype ∈ keys(modelnames), __) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :compname) |>
    transform!(__, :modeltype => ByRow(x -> modelnames[x]) => :mtypename) |>
    transform!(__, [:correct, :baseline] => ((x,y) -> 100(x-y)) => :correctdiff)

xtitle = "Baseline Accuracy"
ytitle = "Full-model Accuracy"
pl = @vlplot(data = plotmeans,
        facet = {field = :mtypename, title = "Basline Method",
                sort = values(modelnames)},
        config = {facet = {columns = 3}}
    ) + (
        @vlplot() +
        @vlplot(:point,
            color = :comparison,
            y = {:correct, title = ytitle}, x = {:baseline, title = xtitle}) +
        @vlplot(data = refline, mark = {:line, strokeDash = [2, 2], size = 2},
            y = {:y, title = ytitle},
            x = {:x, title = xtitle},
            color = {value = "black"})
    );
pl |> save(joinpath(dir, "baseline_individual.svg"))

pl = plotmeans |>
    @vlplot(
        facet = {
            column = {
                field = :mtypename, sort = collect(values(modelnames)),
                header = {
                    title = "Baseline Method",
                    labelExpr = "split(datum.value, '\\n')"
                }
            }
        }) + (
        @vlplot(x = {:compname, axis = nothing},
            color = {
                :compname, title = nothing,
                scale = {range = ["url(#blue_orange)", "url(#blue_red)", "url(#orange_red)"]},
                legend = {legendX = 5, legendY = 5, orient = "none"}}) +
        @vlplot(:bar,
            y = {:correctdiff, aggregate = :mean, type = :quantitative,
                 title = "% Correct (Full Model - Baseline)"}) +
        @vlplot({:errorbar, size = 1, ticks = {size = 5}, tickSize = 2.5},
            color = {value = "black"},
            y = {:correctdiff, aggregate = :ci, type = :quantitative,
                title = "% Correct (Full Model - Baseline)"}) +
        @vlplot({:point, filled = true, size = 15, opacity = 0.25, xOffset = -2},
            color = {value = "black"},
            y = :correctdiff)
    );

plotfile = joinpath(dir, "baseline_models.svg")
pl |> save(plotfile)

# customize the fill with some low-level svg coding
let blue = "#4c78a8", orange = "#f58518", red = "#e45756"
    stripes = @_ """
    <defs>
        <pattern id="blue_orange" x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect x="0" y="0" width="4" height="8" style="stroke:none; fill:$blue;" />
            <rect x="4" y="0" width="4" height="8" style="stroke:none; fill:$orange;" />
        </pattern>
        <pattern id="blue_red" x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect x="0" y="0" width="4" height="8" style="stroke:none; fill:$blue;" />
            <rect x="4" y="0" width="4" height="8" style="stroke:none; fill:$red;" />
        </pattern>
        <pattern id="orange_red" x="0" y="0" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect x="0" y="0" width="4" height="8" style="stroke:none; fill:$orange;" />
            <rect x="4" y="0" width="4" height="8" style="stroke:none; fill:$red;" />
        </pattern>
    </defs>
    """ |> parsexml |> __.node |> elements |> first |> unlink!
    vgplot = readxml(plotfile)
    @_ vgplot.root |> elements |> first |> linkprev!(__, stripes)
    open(io -> prettyprint(io, vgplot), joinpath(dir, "baseline_models.svg"), write = true)
end

# Display of model coefficients
# =================================================================

# Compare coefficients across folds
# -----------------------------------------------------------------

coefdf = mapreduce(append!!, classcomps_atlen) do (comp, data)
    function findclass((key, sdf))
        result = testclassifier(LassoClassifier(sdf.λ |> first),
            data = sdf, y = :condition, X = r"channel",
            crossval = :sid, n_folds = 10,
            seed = stablehash(:cond_coef_timeline,2019_11_18),
            irls_maxiter = 100, include_model_coefs = true,
            weight = :weight, on_model_exception = :throw)

        result[!, keys(key)] .= permutedims(collect(values(key)))
        result[!, :comparison] .= comp

        result
    end

    groups = pairs(groupby(data, :fold))
    foldl(append!!, Map(findclass), collect(groups))
end

coefnames_ = pushfirst!(propertynames(coefdf[:,r"channel"]), :C)

coefvals = @_ coefdf |>
    groupby(__, [:label_fold, :fold, :comparison]) |>
    combine(__, coefnames_ .=> (only ∘ unique) .=> coefnames_)

function parsecoef(coef)
    parsed = match(r"channel_([0-9]+)_([a-z]+)",string(coef))
    if !isnothing(parsed)
        chanstr, freqbin = parsed[1], parsed[2]
        chan = parse(Int,chanstr)
        chan, freqbin
    elseif string(coef) == "C"
        missing, missing
    else
        error("Unexpected coefficient $coef")
    end
end

coef_spread = @_ coefvals |>
    stack(__, coefnames_, [:label_fold, :fold, :comparison],
        variable_name = :coef) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[1]) => :channel) |>
    transform!(__, :coef => ByRow(x -> parsecoef(x)[2]) => :freqbin)

minabs(x) = x[argmin(abs.(x))]

coef_spread_means = @_ coef_spread |>
    filter(!ismissing(_.channel), __) |>
    groupby(__, [:freqbin, :channel, :comparison]) |>
    combine(__, :value => mean => :value,
                :value => length => :N,
                :value => (x -> quantile(x, 0.75)) => :innerhigh,
                :value => (x -> quantile(x, 0.25)) => :innerlow,
                :value => (x -> quantile(x, 0.95)) => :outerhigh,
                :value => (x -> quantile(x, 0.05)) => :outerlow)

compnames = Dict(
    "global-v-object"  => "Global vs. Object",
    "global-v-spatial" => "Global vs. Spatial",
    "object-v-spatial" => "Object vs. Spatial")

coefmeans_rank = @_ coef_spread_means |>
    groupby(__, [:comparison, :channel]) |>
    combine(__, :value => minimum => :minvalue,
                :outerlow => minimum => :minouter) |>
    sort!(__, [:comparison, :minvalue, :minouter]) |>
    groupby(__, [:comparison]) |>
    transform!(__, :minvalue => (x -> 1:length(x)) => :rank) |>
    innerjoin(coef_spread_means, __, on = [:comparison, :channel]) |>
    transform!(__, :channel => ByRow(x -> string("MCCA ",x)) => :channelstr) |>
    filter(!(_.value == 0 && _.outerlow == 0 && _.outerhigh == 0), __) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :comparisonstr)

ytitle = "Median cross-validated coefficient value"
plcoefs = coefmeans_rank |>
    @vlplot(facet =
        {column = {field = :comparisonstr, title = "Comparison", type = :ordinal}},
        spacing = {column = 45}) +
     (@vlplot(x = {:rank, title = "Coefficient Rank (low-to-high)"},
        color = {:freqbin, type = :ordinal, sort = ["delta","theta","alpha","beta","gamma"],
                 scale = {scheme = "magma"}}) +
      @vlplot({:rule, size = 3}, y = :innerlow, y2 = :innerhigh) +
      @vlplot({:errorbar, size = 1, ticks = {size = 5}, tickSize = 2.5},
        y = {:outerlow, title = ytitle, type = :quantitative}, y2 = "outerhigh:q") +
      @vlplot({:point, filled = true, size = 75},
        y = :value,
        color = {
            field = :freqbin,
            legend = {title = nothing},
            type = :ordinal, sort = ["delta","theta","alpha","beta","gamma"]}) +
      @vlplot(
          transform = [
            {aggregate = [{op = :min, field = :value, as = :min_mvalue}],
             groupby = [:channelstr, :rank]},
            {filter = "(datum.rank <= 3) && (datum.min_mvalue != 0)"}],
          x = :rank,
          mark = {type = :text, align = :left, dx = 5, dy = 5}, text = :channelstr,
          y = {field = :min_mvalue, title = ytitle},
          color = {value = "black"}))

# MCCA visualization
# =================================================================

# plot the features of top two components
# -----------------------------------------------------------------


chpat = r"channel_([0-9]+)_([a-z]+)"
bincenter(x) = @_ GermanTrack.default_freqbins[Symbol(x)] |> log.(__) |> mean |> exp
classdf_long = @_ classdf_atlen |>
    stack(__, r"channel", [:sid, :condition, :weight], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> match(chpat, x)[2])            => :bin) |>
    transform!(__, :bin        => ByRow(bincenter)                          => :frequency) |>
    groupby(__, :bin) |>
    transform!(__, :value => zscore => :zvalue)

# plot individual for two best dimensions

function bestchan_features(classdf)
    function (row)
        result = @_ filter(occursin(_.condition,row.comparison) &&
                  _.channel == row.channel &&
                  _.bin == row.freqbin, classdf)
        for col in propertynames(row)
            result[:,col] = row[col]
        end
        result
    end
end

best_channel_df = @_ coef_spread_means |>
    groupby(__, :comparison) |>
    combine(__,
        [:channel, :value] =>
            ((chan,x) -> chan[sortperm(x)[1:2]]) => :channel,
        [:freqbin, :value] =>
            ((bin,x) -> bin[sortperm(x)][1:2]) => :freqbin,
        :channel => (x -> 1:2) => :rank)

classdf_best = @_ foldl(append!!, Map(bestchan_features(classdf_long)),
    eachrow(best_channel_df))

classdf_best_long = @_ classdf_best |>
    unstack(__, [:sid, :condition, :comparison], :rank, :zvalue,
        renamecols = x -> Symbol(:value,x)) |>
    innerjoin(__, unstack(classdf_best, [:sid, :condition, :comparison], :rank, :channelbin,
        renamecols = x -> Symbol(:feat,x)), on = [:sid, :condition, :comparison]) |>
    transform!(__, :comparison => ByRow(x -> compnames[x]) => :comparisonstr) |>
    transform!(__, :feat1 => ByRow(string) => :feat1) |>
    transform!(__, :feat2 => ByRow(string) => :feat2)

function maketitle(x)
    m = match(r"channel_([0-9]+)_([a-z]+)", x)
    chn, bin = m[1], m[2]
    "MCCA Component $chn $(uppercasefirst(bin))"
end
titles = @_ classdf_best_long |>
    groupby(__, :comparison) |>
    combine(__, :feat1 => (maketitle ∘ first) => :title1,
                :feat2 => (maketitle ∘ first) => :title2) |>
    groupby(__, :comparison)

plfeats = @vlplot() + hcat(
    (classdf_best_long |> @vlplot(
        transform = [{filter = "datum.comparison == '$comparison'"}],
        {:point, filled = true},
        x = {:value1, title = titles[(comparison = comparison,)].title1[1],
            scale = {domain = [-2.5, 2.5]}},
        y = {:value2, title = titles[(comparison = comparison,)].title2[1],
            scale = {domain = [-2.5, 2.5]}},
        shape = :condition,
        color = {:condition, scale = {scheme = "dark2"}})
        for comparison in unique(classdf_best_long.comparison))...)

pl = @vlplot(align = "all",
        resolve = {scale = {color = "independent", shape = "independent"}}) +
    vcat(plcoefs, plfeats)

pl |> save(joinpath(dir, "condition_features.svg"))
pl |> save(joinpath(dir, "condition_features.png"))

# Plot spectrum of all components
# -----------------------------------------------------------------

best_channels = skipmissing(best_channel_df.channel) |> unique |> sort!
spectdf_file = joinpath(cache_dir("features"), savename("cond-freaqmeans-spect",
    (channels = best_channels,), "csv", allowedtypes = [Array]))

binsize = 100 / 128
finebins = OrderedDict(string("bin",i) => ((i-1)*binsize, i*binsize) for i in 1:128)
windows = [(len = 2.0, start = 0.0)]

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")
spectdf_groups = @_ events |>
    filter(_.target_present, __) |>
    filter(ishit(_, region = "target") == "hit", __) |>
    groupby(__, [:sid, :condition]);

spectdf = compute_freqbins(subjects, spectdf_groups, windowtarget, windows, foldxt,
    freqbins = finebins, channels = @_ best_channel_df.channel |> unique |> sort! |>
        push!(__, 1, 2))

chpat = r"channel_([0-9]+)_bin([0-9]+)"
spectdf_long = @_ spectdf |>
    stack(__, r"channel", [:sid, :condition, :weight], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[2])) => :bin) |>
    transform!(__, :bin => ByRow(bin -> (bin-1)*binsize + binsize/2) => :frequency)

@_ spectdf_long |>
    filter(_.frequency > 3, __) |>
    @vlplot(:line, column = :channel, color = :condition,
        x = {:frequency},
        y = {:value, aggregate = :median, type = :quantitative})

# maybe divide by median value

spectdf_norm = @_ spectdf_long |>
    groupby(__, [:frequency]) |>
    transform!(__, :value => (x -> x ./ median(x)) => :normvalue)

@_ spectdf_norm |>
    filter(_.frequency > 1, __) |>
    @vlplot(facet = {column = {field = :channel}}) +
    (@vlplot() +
        @vlplot(:line, color = :condition,
            x = {:frequency, scale = {type = :log}},
            y = {:normvalue, aggregate = :mean, type = :quantitative}) +
        @vlplot(:errorband, color = :condition,
            x = {:frequency, scale = {type = :log}},
            y = {:normvalue, aggregate = :ci, type = :quantitative}))

# Examine the power across bins/channels near a target
# -----------------------------------------------------------------

subjects, events = load_all_subjects(processed_datadir("eeg"), "h5")

classhitdf_groups = @_ events |>
    transform!(__, AsTable(:) => ByRow(x -> ishit(x, region = "target")) => :hittype) |>
    groupby(__, [:sid, :condition, :hittype])

windows = [(len = 2.0, start = 0.0)]
classhitdf = compute_freqbins(subjects, classhitdf_groups, windowtarget, windows)

chpat = r"channel_([0-9]+)_([a-z]+)"
bincenter(x) = @_ GermanTrack.default_freqbins[Symbol(x)] |> log.(__) |> mean |> exp
classhitdf_long = @_ classhitdf |>
    stack(__, r"channel", [:sid, :condition, :weight, :hittype], variable_name = "channelbin") |>
    transform!(__, :channelbin => ByRow(x -> parse(Int,match(chpat, x)[1])) => :channel) |>
    transform!(__, :channelbin => ByRow(x -> match(chpat, x)[2])            => :bin) |>
    transform!(__, :bin        => ByRow(bincenter)                          => :frequency) |>
    groupby(__, :bin) |>
    transform!(__, :value => zscore => :zvalue)

classhitdf_summary = @_ classhitdf_long |>
    groupby(__, [:hittype, :channel, :frequency, :condition]) |>
    combine(__, :zvalue => median => :medvalue,
                :zvalue => minimum => :minvalue,
                :zvalue => maximum => :maxvalue,
                :zvalue => (x -> quantile(x, 0.75)) => :q50u,
                :zvalue => (x -> quantile(x, 0.25)) => :q50l,
                :zvalue => (x -> quantile(x, 0.975)) => :q95u,
                :zvalue => (x -> quantile(x, 0.025)) => :q95l)

ytitle = "Median Power"
@_ classhitdf_summary |>
    filter(_.channel in 1:5, __) |>
    @vlplot(facet = {
            column = {field = :hittype, type = :ordinal},
            row = {field = :channel, type = :ordinal}
        }) + (
        @vlplot() +
        @vlplot(:line, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:medvalue, title = ytitle}) +
        @vlplot(:point, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:medvalue, title = ytitle}) +
        @vlplot(:errorband, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:q50u, title = ytitle}, y2 = :q50l) +
        @vlplot(:errorbar, color = :condition,
            x = {:frequency, scale = {type = :log}}, y = {:q95u, title = ytitle}, y2 = :q95l)
    )

ytitle = "Median Power"
@_ classhitdf_summary |>
    @vlplot(facet = {
            field = :channel,
            type = :ordinal
        },
        config = {facet = {columns = 10}}
    )+(@vlplot() +
        @vlplot(:point, color = :condition,
            x = :hittype,
            y = {:medvalue, title = ytitle, aggregate = :mean, type = :quantitative}) +
        @vlplot(:errorbar, color = :condition,
            x = :hittype,
            y = {:q95u, title = ytitle, aggregate = :mean, type = :quantitative},
            y2 = :q95l)
    )


classhitdf_summary = @_ classhitdf_long |>
    groupby(__, [:hittype, :condition]) |>
    combine(__, :zvalue => median => :medvalue,
                :zvalue => minimum => :minvalue,
                :zvalue => maximum => :maxvalue,
                :zvalue => (x -> quantile(x, 0.75)) => :q50u,
                :zvalue => (x -> quantile(x, 0.25)) => :q50l,
                :zvalue => (x -> quantile(x, 0.975)) => :q95u,
                :zvalue => (x -> quantile(x, 0.025)) => :q95l)

ytitle = "Median Power"
@_ classhitdf_summary |>
    @vlplot(facet = {
            field = :hittype,
            type = :ordinal
        },
        config = {facet = {columns = 10}}
    )+(@vlplot() +
        @vlplot(:point, color = :condition,
            x = :condition, y = {:medvalue, title = ytitle}) +
        @vlplot(:errorbar, color = :condition,
            x = :condition, y = {:q95u, title = ytitle}, y2 = :q95l)
    )

classhitdf_stats = @_ classhitdf_long |>
    groupby(__, [:hittype, :condition, :sid]) |>
    combine(__, :zvalue => median => :medvalue)

pl = @_ classhitdf_stats |>
    @vlplot(facet = {
        field = :hittype,
        type = :ordinal
    },
    config = {legend = {disable = true}, facet = {columns = 10}}
    )+(@vlplot(width = 50, height = 300) +
    @vlplot({:point, filled = true, size = 75}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :mean}) +
    @vlplot({:errorbar, ticks = {size = 5}}, color = :condition,
        x = :condition,
        y = {:medvalue, title = ytitle, type = :quantitative, aggregate = :ci}) +
    @vlplot({:point, filled = true, size = 15, opacity = 0.25, xOffset = -5},
        color = {value = "black"},
        x = :condition, y = {:medvalue, title = ytitle})
    )

pl |> save(joinpath(dir, "medpower_hittype.svg"))
pl |> save(joinpath(dir, "medpower_hittype.png"))


