# Setup
# =================================================================

# STEPS:
# veriyf measure that I'm using
# reproduce with flux model
# try adding more data to flux model

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn, ShiftedArrays, ColorSchemes, Flux, CUDA, GLM, SparseArrays,
    JLD, Arrow, FFTW

dir = mkpath(joinpath(plotsdir(), "figure6_parts"))

using GermanTrack: colors

nfolds = 5

# STEPS: maybe we should consider cross validating across stimulus type
# rather than subject id?

# Setup EEG Data
# -----------------------------------------------------------------

# eeg_encoding = FFTFilteredPower("freqbins", Float32[1, 3, 7, 15, 30, 100])
eeg_encoding = JointEncoding(
    RawEncoding(),
    FilteredPower("delta", 1,  3),
    FilteredPower("theta", 3,  7),
    FilteredPower("alpha", 7,  15),
    FilteredPower("beta",  15, 30),
    FilteredPower("gamma", 30, 100),
)
# eeg_encoding = RawEncoding()

sr = 32
file = joinpath(cache_dir("eeg", "features"), "subject-data-decoding.jld")
if isfile(file)
    data = load(file)
    subjects = data["subjects"]
    events = data["events"]
else
    subjects, events = load_all_subjects(processed_datadir("eeg"), "h5",
        encoding = eeg_encoding, framerate = sr)
    save(file, "subjects", subjects, "events", events)
end


meta = GermanTrack.load_stimulus_metadata()

target_length = 1.0
max_lag = 3

seed = 2019_11_18
target_samples = round(Int, sr*target_length)
function event2window(event, windowing)
    triallen     = size(subjects[event.sid].eeg[event.trial], 2)
    start_time = if windowing == "target"
        meta.target_times[event.sound_index]
    else
        max_time = meta.target_times[event.sound_index]-1.5
        if max_time <= 0
            0.0
        else
            # generates a random number that has the same value for
            # the same event (so windows are shared across subjects)
            max_time*rand(GermanTrack.trialrng((:decode_windowing, seed), event))
        end
    end
    start = clamp(round(Int, sr*start_time), 1, triallen)
    len   = clamp(target_samples, 1, triallen-start)
    (
        windowing = windowing,
        start     = start,
        len       = len,
        trialnum  = event.trial,
        event[[:condition, :sid, :target_source, :sound_index, :hittype]]...
    )
end

windows = @_ events |>
    transform!(__, AsTable(:) => ByRow(ishit) => :hittype) |>
    filter(_.hittype ∈ ["hit", "miss"], __) |> eachrow |>
    Iterators.product(__, ["target", "pre-target"]) |>
    map(event2window(_...), __) |> vec |>
    DataFrame |>
    transform!(__, :len => (x -> lag(cumsum(x), default = 1)) => :offset)

nobs = sum(windows.len)
starts = vcat(1,1 .+ cumsum(windows.len))
nfeatures = size(first(subjects)[2].eeg[1],1)
nlags = round(Int,sr*max_lag)
lags = -(nlags-1):1:0
x = Array{Float32}(undef, nfeatures*nlags, nobs)

progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
Threads.@threads for (i, trial) in collect(enumerate(eachrow(windows)))
    tstart = trial.start
    tstop = trial.start + trial.len - 1
    xstart = trial.offset
    xstop = trial.offset + trial.len - 1

    trialdata = withlags(subjects[trial.sid].eeg[trial.trialnum]', lags)
    x[:, xstart:xstop] = @view(trialdata[tstart:tstop, :])'
    next!(progress)
end
x_μ = mean(x, dims = 2)
x .-= x_μ
x_σ = std(x, dims = 2)
x ./= x_σ

# Setup stimulus data
# -----------------------------------------------------------------

stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
encoding_map = Dict("pitch" => PitchSurpriseEncoding(), "envelope" => ASEnvelope())
encodings = ["pitch", "envelope"]
sources = [
    male_source,
    fem1_source,
    fem2_source,
    # male_fem1_sources,
    # male_fem2_sources,
    # fem1_fem2_sources
]

stimuli = Empty(Vector)

progress = Progress(size(windows, 1), desc = "Organizing stimulus data...")
for (i, trial) in enumerate(eachrow(windows))
    for (j, encoding) in enumerate(encodings)
        for source in sources
            stim, stim_id = load_stimulus(source, trial, stim_encoding, sr, meta)
            start = trial.start
            stop = min(size(stim,1), trial.start + trial.len - 1)
            fullrange = starts[i] : (starts[i+1] - 1)

            stimulus = if stop >= start
                stimulus = Float32.(@view(stim[start:stop, j]))
            else
                Float32[]
            end

            stimuli = push!!(stimuli, (
                trial...,
                source           = string(source),
                encoding         = encoding,
                start            = start,
                stop             = stop,
                len              = stop - start + 1,
                data             = stimulus,
                is_target_source = trial.target_source == string(source),
                stim_id          = stim_id,
            ))
        end
    end
    next!(progress)
end

stimulidf = @_ DataFrame(stimuli) |>
    # @where(__, :condition .== "global") |>
    # @where(__, :is_target_source) |>
    # @where(__, :windowing .== "target") |>
    # train on quarter of subjects
    # @where(__, :sid .<= sort!(unique(:sid))[div(end,4)]) |>
    addfold!(__, nfolds, :sid, rng = stableRNG(2019_11_18, :decoding)) |>
    # insertcols!(__, :predict => Ref(Float32[])) |>
    groupby(__, [:encoding]) |>
    transform!(__, :data => (x -> mean(reduce(vcat, x))) => :datamean, ungroup = false) |>
    transform!(__, [:data, :datamean] => ByRow((x, μ) -> x .-= μ) => :data, ungroup = false) |>
    transform!(__, :data => (x -> std(reduce(vcat, x))) => :datastd, ungroup = false) |>
    transform!(__, [:data, :datastd] => ByRow((x, σ) -> x .-= σ) => :data)

fold_map = @_ stimulidf |> groupby(__, :sid) |>
    @combine(__, fold = first(:fold)) |>
    Dict(row.sid => row.fold for row in eachrow(__))

# Train Model
# =================================================================

eegindices(row::DataFrameRow) = (row.offset):(row.offset + row.len - 1)
function eegindices(df::AbstractDataFrame)
    mapreduce(eegindices, vcat, eachrow(df))
end

function decode_scores(predictions)
    score(x,y) = cor(x,y)
    meta = GermanTrack.load_stimulus_metadata()
    scores = @_ predictions |>
        @transform(__, score = score.(:predict, :data)) |>
        # @where(__, :encoding .== "envelope") |>
        # groupby(__, [:encoding, :λ]) |>
        # @transform(__, score = zscoresafe(:score)) |>
        groupby(__, [:sid, :condition, :source, :train_type, :is_target_source,
            :trialnum, :stim_id, :windowing, :λ, :hittype, :fold]) |>
        @combine(__, score = mean(:score)) |>
        transform!(__,
            :stim_id => (x -> meta.target_time_label[x]) => :target_time_label,
            :stim_id => (x -> meta.target_switch_label[x]) => :target_switch_label,
            :stim_id => (x -> meta.target_times[x]) => :target_time,
            :stim_id => (x -> cut(meta.target_salience[x], 2)) => :target_salience,
            :stim_id => (x -> meta.target_salience[x]) => :target_salience_level,
            [:hittype, :windowing] => ByRow((x,y) -> string(x, "-", y)) => :test_type
        )
end

datafile = processed_datadir("analyses", "decode-predict-freqbin")
if !isfile(string(datafile,"-model.bson"))
    @info "Generating cross-validated predictions, this could take a bit..."

    groupings = [:source, :encoding]
    groups = groupby(stimulidf, groupings)

    max_steps = 50
    nλ = 24
    batchsize = 2048
    train_types = ["athit-other", "athit-target", "atmiss-target"]
    progress = Progress(max_steps * length(groups) * nfolds * nλ * length(train_types))
    validate_fraction = 0.2

    predictions, coefs, models = filteringmap(groups, folder = foldl, streams = 3, desc = nothing,
        :fold => 1:nfolds,
        :λ => exp.(range(log(1e-4),log(1e-1),length=nλ)),
        :train_type => train_types,
        function(sdf, fold, λ, train_type)
            hittype, is_target =
                train_type == "athit-target" ? ("hit", true) :
                train_type == "athit-other" ? ("hit", false) :
                train_type == "atmiss-target" ? ("miss", false) :
                error("Unexpected `train_type` value of $train_type.")

            sdf = view(sdf, sdf.is_target_source .== is_target, :)
            isempty(sdf) && return (Empty(DataFrame), Empty(DataFrame), Empty(DataFrame))

            nontest = @_ filter((_1.fold != fold) &&
                            (_1.hittype == hittype) &&
                            (_1.windowing == "target"), sdf)
            test  = @_ filter((_1.fold == fold) &&
                              (_1.hittype == "hit") &&
                              (_1.windowing == "target"), sdf)

            sids = levels(nontest.sid)
            nval = max(1, round(Int, validate_fraction * length(sids)))
            rng = stableRNG(2019_11_18, :validate_flux, fold, λ,
                Tuple(sdf[1, groupings]))
            validate_ids = sample(rng, sids, nval, replace = false)

            train    = @_ filter(_.sid ∉ validate_ids, nontest)
            validate = @_ filter(_.sid ∈ validate_ids, nontest)

            encodings = groupby(train, :encoding)
            firstencoding = first(encodings).encoding |> first
            xᵢ = x[:, eegindices(first(encodings))]
            yᵢ = @_ [
                row.data
                for rows in encodings
                for row in eachrow(rows)
            ] |> reduce(vcat, __) |> reshape(__, length(encodings), :)

            xⱼ = x[:, eegindices(first(groupby(validate, :encoding)))]
            yⱼ = @_ [
                row.data
                for rows in groupby(validate, :encoding)
                for row in eachrow(rows)
            ] |> reduce(vcat, __) |> reshape(__, length(encodings), :)

            model = GermanTrack.decoder(xᵢ, yᵢ, λ, Flux.Optimise.RADAM(),
                progress = progress, batch = batchsize, max_steps = max_steps,
                min_steps = 20,
                patience = 6,
                inner = 64,
                validate = (xⱼ, yⱼ))

            test.predict = map(eachrow(test)) do testrow
                xⱼ = view(x, :, eegindices(testrow))
                yⱼ = model(xⱼ)
                view(yⱼ,testrow.encoding == firstencoding ? 1 : 2,:)
            end
            test.steps = GermanTrack.nsteps(model)
            C = GermanTrack.decode_weights(model) |> vec

            bins = ["raw", "delta", "theta", "alpha", "beta", "gamma"]
            mccai(i) = CartesianIndices((nlags, 30, 6))[i][2]
            lagi(i) = lags[CartesianIndices((nlags, 30, 6))[i][1]]
            bini(i) = bins[CartesianIndices((nlags, 30, 6))[i][3]]

            coefs = DataFrame(
                coef = C,
                lag = lagi.(eachindex(C)),
                bin = bini.(eachindex(C)),
                mcca = mccai.(eachindex(C)))

            test, coefs, DataFrame(model = model)
        end)

    ProgressMeter.finish!(progress)
    alert("Completed model training!")


    # score(x,y) = -sqrt(mean(abs2, xi - yi for (xi,yi) in zip(x,y)))
    scores = decode_scores(predictions)
    tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]

    function nanmean(xs)
        xs_ = (x for x in xs if !isnan(x))
        isempty(xs_) ? 0.0 : mean(xs_)
    end
    pldata = @_ scores |>
        @transform(__, condition = string.(:condition)) |>
        groupby(__, [:sid, :condition, :train_type, :test_type, :source, :λ]) |>
        @combine(__, score = nanmean(:score)) |>
        groupby(__, [:condition, :train_type, :test_type, :λ]) |>
        @combine(__, score = median(:score))

    best_λs = @_ scores |>
        @transform(__, condition = string.(:condition)) |>
        groupby(__, [:sid, :condition, :train_type, :test_type, :source, :λ, :fold]) |>
        @combine(__, score = nanmean(:score)) |>
        groupby(__, [:condition, :train_type, :test_type, :λ, :fold]) |>
        @combine(__, score = median(:score)) |>
        @where(__, (startswith.(:train_type, "athit-target")) .& (:test_type .== "hit-target")) |>
        groupby(__, [:fold, :condition, :λ]) |>
        @combine(__, score = mean(:score)) |>
        groupby(__, [:λ, :fold]) |>
        @combine(__, score = minimum(:score)) |>
        filteringmap(__, desc = nothing, :fold => cross_folds(1:nfolds),
            (sdf, fold) -> DataFrame(score = maximum(sdf.score), λ = sdf.λ[argmax(sdf.score)])
        )

    best_λ = Dict(row.fold => row.λ for row in eachrow(best_λs))
    # best_λ = lambdas[argmin(abs.(lambdas .- 0.002))]

    # TODO: plot all fold's λs
    tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 8)]
    pl = @_ pldata |>
        @where(__, :test_type .== "hit-target") |>
        @vlplot(
            facet = {column = {field = :condition, type = :nominal}}
        ) +
        (
            @vlplot() +
            @vlplot({:line, strokeCap = :round}, x = {:λ, scale = {type = :log}}, y = :score,
                color = {:train_type, scale = {range = "#".*hex.(tcolors)}}) +
            @vlplot({:point, filled = true}, x = {:λ, scale = {type = :log}}, y = :score,
                color = {:train_type, scale = {range = "#".*hex.(tcolors)}}) +
            (
                best_λs |> @vlplot() +
                @vlplot({:rule, strokeDash = [2 2], size = 1},
                    x = :λ
                )
            )
        );
    pl |> save(joinpath(dir, "decode_lambda.svg"))

    pl = @_ predictions |> select(__, :λ, :steps) |>
        @vlplot(:point, x = {:λ, scale = {type = :log}}, y = "mean(steps)");
    pl |> save(joinpath(dir, "steps_lambda.svg"))

    models_ = @_ filter(_.λ == best_λ[_.fold], models)
    coefs_ = @_ filter(_.λ == best_λ[_.fold], coefs)
    predictions_ = @_ filter(_.λ == best_λ[_.fold], predictions)

    save(string(datafile, "-model.bson"), Dict("models" => NamedTuple.(Tables.rows(models_))))
    Arrow.write(string(datafile, "-coef.feather"), coefs_, compress = :lz4)
    Arrow.write(string(datafile, "-predict.feather"), predictions_, compress = :lz4)
else
    @info "Loading models predictions from data file"
    coefs_ = DataFrame(Arrow.Table(string(datafile, "-coef.feather")))
    predictions_ = DataFrame(Arrow.Table(string(datafile, "-predict.feather")))
    models_ = DataFrame(load(string(datafile, "-model.bson"))["models"])
    scores = decode_scores(predictions_)
end


# Plotting
# -----------------------------------------------------------------

function zscoresafe(x)
    x = zscore(x)
    any(isnan, x) ? zero(x) : x
end

example = @_ predictions |>
    @where(__, (:λ .== first(best_λs.λ)) .& (:sid .== 33) .&
              (:windowing .== "target") .&
              (:hittype.== "hit") .&
              (:train_type .== "athit-target-male") .&
            #   (:encoding .== "envelope") .&
              (:condition .== "global")) |>
    mapreduce(row -> DataFrame(
        time = axes(row.predict,1) / sr,
        predict = row.predict,
        data = row.data;
        row[Not([:predict, :data])]...
    ), append!!, eachrow(__))

pl = @_ example |>
    @where(__, :trialnum .< 10) |>
    stack(__, [:data, :predict], [:time, :windowing, :trialnum, :condition, :sid, :source, :is_target_source, :encoding]) |>
    @vlplot(
        facet = {
            column = {field = :trialnum, type = :ordinal},
            row = {field = :encoding, type = :nominal}
        }
    ) +
    @vlplot() + (
        @vlplot(:line, x = :time, y = :value, color = :variable,
            strokeDash = :is_target_source)
    );
pl |> save(joinpath(dir, "example_predict.svg"))


@_ scores |>
    # filter(_.λ == best_λ[_.fold], __) |>
    transform!(__, :train_type => ByRow(x -> string(split(x, "-")[1:2]...)) => :train_kind) |>
    CSV.write(joinpath(processed_datadir("analyses", "decode"), "decode_scores.csv"))

mean_offset = 6
pl = @_ scores |>
filter(_.λ == best_λ[_.fold], __) |>
    @where(__, :test_type .== "hit-target") |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :train_type, :test_type, :source]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :test_type]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
            # row = {field = :train_type, type = :nominal}
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}},
            x = {:train_type, axis = {title = "Train Type", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, },
            y = {:score, title = ["Decoder score", "(For envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(score)",
        ) +
        @vlplot({:line, size = 1}, color = {value = "gray"},
            opacity = {value = 0.3},
            x = :train_type,
            y = :score,
            detail = :sid,
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(score)",
            y2 = "ci1(score)",  # {"score:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode.svg"))

# coefficient display

trfs = @_ coefs |>
    filter(_.λ == best_λ[_.fold], __) |>
    transform!(__, :train_type => ByRow(x -> string(split(x, "-")[1:2]...)) => :train_kind) |>
    transform!(__, :lag => ByRow(x -> -x / sr) => :time) |>
    groupby(__, [:time, :train_type, :encoding, :fold]) |>
    @combine(__, value = mean(abs, :coef)) |>
    groupby(__, [:time, :train_type, :encoding]) |>
    @combine(__,
        value = median(:value),
        lower = quantile(:value, 0.25),
        upper = quantile(:value, 0.75)
    )

density = @_ coefs |>
    filter(_.λ == best_λ[_.fold], __) |>
    groupby(__, [:lag, :train_type, :encoding, :fold]) |>
    @combine(__,
        density = mean(x -> abs(x) > 1e-4, :coef),
        sd = std(map(x -> abs(x) > 1e-4, :coef)),
        count = sum(x -> abs(x) > 1e-4, :coef)
    )

pl = trfs |>
    @vlplot(
        facet = {column = {field = :train_type}, row = {field = :encoding}}
    ) +
    (
        @vlplot() +
        @vlplot(:line, x = :time, y = :value) +
        @vlplot(:errorband, x = :time, y = :lower, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_combined_trf.svg"))

trfs = @_ coefs |>
    filter(_.λ == best_λ[_.fold], __) |>
    transform!(__, :train_type => ByRow(x -> string(split(x, "-")[1:2]...)) => :train_kind) |>
    groupby(__, [:mcca, :train_type, :fold]) |>
    @combine(__, value = mean(abs, :coef)) |>
    groupby(__, [:mcca, :train_type]) |>
    @combine(__,
        value = median(:value),
        lower = quantile(:value, 0.25),
        upper = quantile(:value, 0.75)
    )

pl = trfs |>
    @vlplot(
        facet = {column = {field = :train_type}}
    ) +
    (
        @vlplot() +
        @vlplot(:line, x = :mcca, y = :value) +
        @vlplot(:errorband, x = :mcca, y = :lower, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_combined_mcca.svg"))

trfs = @_ coefs |>
    filter(_.λ == best_λ[_.fold], __) |>
    transform!(__, :train_type => ByRow(x -> string(split(x, "-")[1:2]...)) => :train_kind) |>
    transform!(__, :lag => ByRow(x -> -x / sr) => :time) |>
    groupby(__, [:mcca, :train_kind, :time, :fold]) |>
    @combine(__, value = mean(abs, :coef)) |>
    groupby(__, [:mcca, :train_kind, :time]) |>
    @combine(__,
        value = median(:value),
        lower = quantile(:value, 0.25),
        upper = quantile(:value, 0.75)
    )

pl = trfs |>
    @vlplot(
        facet = {column = {field = :train_kind}}
    ) +
    (
        @vlplot(x = :time, color = :mcca) +
        @vlplot(:line, y = :value) +
        @vlplot(:errorband, y = :lower, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_combined_trf_mcca.svg"))


trfs = @_ coefs |>
    filter(_.λ == best_λ[_.fold], __) |>
    # transform!(__, :train_type => ByRow(x -> string(split(x, "-")[1:2]...)) => :train_kind) |>
    groupby(__, [:bin, :train_type, :fold]) |>
    @combine(__, value = mean(abs, :coef)) |>
    groupby(__, [:bin, :train_type]) |>
    @combine(__,
        value = median(:value),
        lower = quantile(:value, 0.25),
        upper = quantile(:value, 0.75)
    )

pl = trfs |>
    @vlplot(
        facet = {column = {field = :train_type}}
    ) +
    (
        @vlplot(x = {:bin, type = :ordinal, sort = ["raw", "delta", "theta", "alpha", "beta", "gamma"]}) +
        @vlplot(:point, y = :value) +
        @vlplot(:errorbar, y = :lower, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_combined_bin.svg"))


trfs = @_ coefs |>
    filter(_.λ == best_λ[_.fold], __) |>
    transform!(__, :train_type => ByRow(x -> string(split(x, "-")[1:2]...)) => :train_kind) |>
    transform!(__, :lag => ByRow(x -> -x / sr) => :time) |>
    groupby(__, [:bin, :train_kind, :time, :fold]) |>
    @combine(__, value = mean(:coef)) |>
    groupby(__, [:bin, :train_kind, :time]) |>
    @combine(__,
        value = median(:value),
        lower = quantile(:value, 0.25),
        upper = quantile(:value, 0.75)
    )

pl = trfs |>
    @vlplot(
        facet = {column = {field = :train_kind}, row = {field = :bin}}
    ) +
    (
        @vlplot(color = :bin, x = :time) +
        @vlplot(:line, y = :value) +
        @vlplot(:errorband, y = :lower, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_combined_bin_time.svg"))
# global only

mean_offset = 6
pl = @_ scores |>
    filter(_.λ == best_λ[_.fold], __) |>
    @where(__, :condition .== "global") |>
    groupby(__, [:sid, :train_type, :test_type, :source]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :train_type, :test_type]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :test_type, type = :nominal},
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}},
            x = {:train_type, axis = {title = "Training", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, },
            y = {:score, title = ["Decoder score", "(For envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(score)",
        ) +
        @vlplot({:line, size = 1}, color = {value = "gray"},
            opacity = {value = 0.3},
            x = :train_type,
            y = :score,
            detail = :sid,
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(score)",
            y2 = "ci1(score)",  # {"score:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_global.svg"))

mean_offset = 6
pl = @_ scores |>
    filter(_.λ == best_λ[_.fold], __) |>
    # @where(__, (:train_type .∈ Ref(["athit", "atmiss"])) .& (:test_type .== "hit-target")) |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :train_type, :source, :target_time_label]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :target_time_label]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
            row = {field = :target_time_label, type = :nominal}
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}},
            x = {:train_type, axis = {title = "Training", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, },
            y = {:score, title = ["Decoder score", "(For envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(score)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(score)",
            y2 = "ci1(score)",  # {"score:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_earlylate.svg"))

mean_offset = 6
pl = @_ scores |>
    filter(_.λ == best_λ[_.fold], __) |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :train_type, :source, :target_switch_label]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :target_switch_label]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
            row = {field = :target_switch_label, type = :nominal}
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}},
            x = {:train_type, axis = {title = "Source", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, },
            y = {:score, title = ["Decoder score", "(For envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(score)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(score)",
            y2 = "ci1(score)",  # {"score:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_switch.svg"))

mean_offset = 6
pl = @_ scores |>
    filter(_.λ == best_λ[_.fold], __) |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :train_type, :source, :target_salience]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :target_salience]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
            row = {field = :target_salience, type = :nominal}
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}},
            x = {:train_type, axis = {title = "Source", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, },
            y = {:score, title = ["Decoder score", "(For envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(score)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(score)",
            y2 = "ci1(score)",  # {"score:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_salience.svg"))

scolors = ColorSchemes.bamako[[0.2,0.8]]
mean_offset = 6
pldata = @_ scores |>
    filter(_.λ == best_λ[_.fold], __) |>
    @where(__, :train_type .∈ Ref(["athit-target", "athit-other"])) |>
    @transform(__,
        condition = string.(:condition),
        train_type = recode(:train_type,
            "athit-target" => "target", "athit-other" => "nontarget"),
        target_salience = string.(recode(:target_salience, (levels(:target_salience) .=> ["Low", "High"])...)),
    ) |>
    groupby(__, [:sid, :condition, :trialnum, :target_salience, :target_time_label, :target_switch_label, :train_type]) |>
    @combine(__, score = maximum(:score)) |>
    unstack(__, [:sid, :condition, :trialnum, :target_salience, :target_time_label, :target_switch_label], :train_type, :score) |>
    @transform(__, cordiff = :target .- :nontarget)

pl = @_ pldata |>
    groupby(__, [:sid, :condition]) |>
    @combine(__, cordiff = mean(:cordiff)) |>
    @vlplot(
        config = {legend = {disable = true}},
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:condition,
                type = :nominal,
                scale = {range = "#".*hex.(colors)}},
            x = {:condition, type = :nominal,
                type = :nominal,
                axis = {title = "", labelAngle = -45,
                    labelExpr = "slice(datum.label,'\\n')"}, },
            y = {:cordiff, title = ["Target - Non-target Score"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(cordiff)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(cordiff)",
            y2 = "ci1(cordiff)",  # {"cordiff:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_diff.svg"))

pl = @_ pldata |>
    groupby(__, [:sid, :condition, :target_salience]) |>
    @combine(__, cordiff = mean(:cordiff)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:target_salience,
                sort = ["Low", "High"],
                type = :ordinal,
                scale = {range = "#".*hex.(colors)}},
            x = {:target_salience, type = :nominal,
                sort = ["Low", "High"],
                type = :ordinal,
                axis = {title = "Salience", labelAngle = -45,
                    labelExpr = "slice(datum.label,'\\n')"}, },
            y = {:cordiff, title = ["Target - Non-target Score"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(cordiff)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(cordiff)",
            y2 = "ci1(cordiff)",  # {"cordiff:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_diff_salience.svg"))

scolors = ColorSchemes.imola[[0.2,0.7]]
pl = @_ pldata |>
    groupby(__, [:sid, :condition, :target_switch_label]) |>
    @combine(__, cordiff = mean(:cordiff)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:target_switch_label,
                sort = ["Low", "High"],
                type = :ordinal,
                scale = {range = "#".*hex.(scolors)}},
            x = {:target_switch_label, type = :nominal,
                sort = ["Low", "High"],
                type = :ordinal,
                axis = {title = "Switch Proximity", labelAngle = -45,
                    labelExpr = "slice(datum.label,'\\n')"}, },
            y = {:cordiff, title = ["Target - Non-target Score"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(cordiff)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(cordiff)",
            y2 = "ci1(cordiff)",  # {"cordiff:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_diff_switch.svg"))

scolors = ColorSchemes.imola[[0.2,0.7]]
pl = @_ pldata |>
    groupby(__, [:sid, :condition, :target_time_label]) |>
    @combine(__, cordiff = mean(:cordiff)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            column = {field = :condition, type = :nominal},
        },
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:target_time_label,
                sort = ["Low", "High"],
                type = :ordinal,
                scale = {range = "#".*hex.(colors)}},
            x = {:target_time_label, type = :nominal,
                sort = ["Low", "High"],
                type = :ordinal,
                axis = {title = "Target Time", labelAngle = -45,
                    labelExpr = "slice(datum.label,'\\n')"}, },
            y = {:cordiff, title = ["Target - Non-target Score"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(cordiff)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(cordiff)",
            y2 = "ci1(cordiff)",  # {"cordiff:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode_diff_earlylate.svg"))
# TODO: run stats on these various cases

tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 3)]

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
    ) |>
    groupby(__, [:sid, :condition, :train_type, :source, :target_salience_level]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :target_salience_level]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        facet = {
            column = {field = :condition, type = :ordinal},
            # row = {field = :train_type, type = :ordinal}
        }
    ) + (
        @vlplot({:point, filled = true, opacity = 0.6},
            x     = :target_salience_level,
            y     = {:score, type = :quantitative, aggregate = :mean},
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}}
        )
    );
pl |> save(joinpath(dir, "decode_salience_continuous.svg"))

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
    ) |>
    groupby(__, [:sid, :condition, :train_type, :target_salience_level, :trialnum]) |>
    @combine(__, score = mean(:score)) |>
    unstack(__, [:sid, :condition, :target_salience_level, :trialnum], :train_type, :score) |>
    @transform(__,
        hit = :var"athit-target" .- :var"athit-other", #"
        miss = :var"athit-target" .- :var"atmiss-target" #"
    ) |>
    stack(__, [:hit, :miss],
        [:sid, :condition, :target_salience_level, :trialnum],
        value_name = :score, variable_name = :trial_type) |>
    groupby(__, [:sid, :condition, :target_salience_level, :trial_type]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        facet = {column = {field = :trial_type}}
    ) + (
        @vlplot({:point, filled = true, opacity = 0.6, clip = true},
            x     = :target_salience_level,
            y     = {:score, type = :quantitative, aggregate = :mean},
            color = {:condition, scale = {range = "#".*hex.(colors)}}
        )
    );
pl |> save(joinpath(dir, "decode_salience_continuous_diff.svg"))

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
    ) |>
    groupby(__, [:sid, :condition, :train_type, :source, :target_time]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :train_type, :target_time]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        facet = {
            column = {field = :condition, type = :ordinal},
            # row = {field = :train_type, type = :ordinal}
        }
    ) + (
        @vlplot({:point, filled = true, opacity = 0.6},
            x     = :target_time,
            y     = {:score, type = :quantitative, aggregate = :mean},
            color = {:train_type, scale = {range = "#".*hex.(tcolors)}}
        )
    );
pl |> save(joinpath(dir, "decode_time_continuous.svg"))

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
    ) |>
    groupby(__, [:sid, :condition, :train_type, :target_time, :trialnum]) |>
    @combine(__, score = mean(:score)) |>
    unstack(__, [:sid, :condition, :target_time, :trialnum], :train_type, :score) |>
    @transform(__, scorediff = :var"athit-target" .- :var"athit-other") |> #"
    groupby(__, [:sid, :condition, :target_time]) |>
    @combine(__, scorediff = mean(:scorediff)) |>
    @vlplot() + (
        @vlplot({:point, filled = true, opacity = 0.6},
            x     = :target_time,
            y     = {:scorediff, type = :quantitative, aggregate = :mean},
            color = {:condition, scale = {range = "#".*hex.(colors)}}
        )
    );
pl |> save(joinpath(dir, "decode_time_continuous_diff.svg"))


# TODO: plot decoding scores vs. hit-rate

# Model timeline
# =================================================================

file = joinpath(cache_dir("eeg", "decoding"), "timeline.feather")
decode_sr = 1/ (round(Int, 0.1sr) / sr)
winlen_s = 1.0
if isfile(file)
    timelines = DataFrame(Arrow.Table(file))
else
    sid_trial = mapreduce(x -> x[1] .=> eachindex(x[2].eeg.data), vcat, pairs(subjects))
    progress = Progress(length(sid_trial), desc = "Evaluating decoder over timeline...")

    stimuli_stats = @_ stimulidf |> groupby(__, [:datamean, :datastd]) |>
        @combine(__, encoding = first(:encoding))

    function decode_timeline(sid, trial)
        event = subjects[sid].events[trial, :]
        if ishit(event) != "hit"
            next!(progress)
            return Empty(DataFrame)
        end
        stimid = event.sound_index

        trialdata = withlags(subjects[sid].eeg[trial]', lags)
        winstep = round(Int, sr/decode_sr)
        winlen = round(Int, winlen_s*sr)
        target_time = round(Int, event.target_time * sr)

        m = filter(x -> x.fold == fold_map[sid], models_)

        function modelpredict(modelgroup)
            train_type = modelgroup.source[1] == event.target_source ?
                "athit-target" : "athit-other"
            modelrow = only(eachrow(filter(x -> x.train_type == train_type, modelgroup)))
            sourcei = @_ findfirst(string(_) == modelrow.source, sources)
            source = sources[sourcei]
            encoding = encoding_map[modelrow.encoding]
            stim, = load_stimulus(source, stimid, encoding, sr, meta)

            y_μ, y_σ = only(eachrow(filter(x -> x.encoding == modelgroup.encoding[1],
                stimuli_stats)))
            maxlen = min(size(trialdata, 1), size(stim, 1))

            x = (view(trialdata, 1:maxlen, :) .- x_μ') ./ x_σ'
            y = vec((view(stim, 1:maxlen, :) .- y_μ') ./ y_σ')
            ŷ = vec(modelrow.model(x'))

            function scoreat(offset)
                start = clamp(1+offset+target_time, 1, maxlen)
                stop = clamp(winlen+offset+target_time, 1, maxlen)

                if stop > start
                    y_ = view(y, start:stop)
                    ŷ_ = view(ŷ, start:stop)

                    DataFrame(
                        score = cor(y_, ŷ_),
                        time = offset/sr,
                        sid = sid,
                        trial = trial,
                        ;merge(
                            modelrow[[:fold, :train_type]],
                            event[[:condition, :sound_index]]
                        )...
                    )
                else
                    Empty(DataFrame)
                end
            end

            foldxt(append!!, Map(scoreat), round(Int, -3*sr):winstep:round(Int, 3*sr),
                init = Empty(DataFrame))
        end
        result = mapgroups(m, [:encoding, :source], modelpredict, desc = nothing)
        next!(progress)

        result
    end

    timelines = foldl(append!!, MapSplat(decode_timeline), sid_trial)
    ProgressMeter.finish!(progress)

    Arrow.write(file, timelines, compress = :lz4)
end

# Plotting
# -----------------------------------------------------------------

plotdf = @_ timelines |>
    groupby(__, [:condition, :time, :sid, :train_type, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

labels = OrderedDict(
    "athit-target" => "Target",
    "athit-other" => "Other",
)
tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    @combine(__,
        score = mean(:score),
        lower = lowerboot(:score),
        upper = upperboot(:score)
    ) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) +
    (
        @vlplot(
            width = 60, height = 80,
            x = {:time, type = :quantitative, title = ""},
            color = {:train_type, sort = ["Target", "Other"], title = "Source", scale = {range = "#".*hex.(tcolors)}}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:score, title = "Mean Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_timeline.svg"))


tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__, time = :time .+ winlen_s, condition = uppercasefirst.(:condition)) |>
    @where(__, -1 .< :time .< 2.5) |>
    unstack(__, [:condition, :time, :sid], :train_type, :score) |>
    @transform(__, scorediff = :var"athit-target" .- :var"athit-other") |> #"
    groupby(__, [:condition, :time]) |>
    @combine(__,
        score = mean(:scorediff),
        lower = lowerboot(:scorediff, alpha = 0.318),
        upper = upperboot(:scorediff, alpha = 0.318)
    ) |>
    @vlplot(
        config = {legend = {disable = true}},
        height = 90, width = 100,
    ) +
    (
        @vlplot(x = {:time, type = :quantitative, title = "Time"}, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot({:line, strokeJoin = :round}, y = {:score, title = ["Target - Other", "Correlation"]}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3},
            transform = [
                {filter = "datum.time > 2.25 && datum.time < 2.5"},
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:score, aggregate = :mean},
            text = {field = :condition}
            # color = {value = "black"}
        )
    );
pl |> save(joinpath(dir, "decode_timeline_diff.svg"))

attend_thresh = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid, :trial, :fold]) |>
    @combine(__, score = mean(:score)) |>
    unstack(__, [:condition, :time, :sid, :trial, :fold], :train_type, :score) |>
    @transform(__,
        scorediff = :var"athit-target" .- :var"athit-other", #"
    ) |>
    @where(__, -0 .< :time .< 0.5) |>
    groupby(__, [:condition, :time, :fold, :sid, :trial]) |>
    @combine(__, scorediff = mean(:scorediff)) |>
    filteringmap(__, folder = foldl, desc = nothing,
        :cross_fold => cross_folds(1:nfolds),
        function(sdf, fold)
            DataFrame(threshold = quantile(sdf.scorediff,0.75))
        end
    ) |>
    Dict(row.cross_fold => row.threshold for row in eachrow(__))
above_thresh(score, fold) = score >= attend_thresh[fold]

tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
pretarget_df = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid, :fold, :sound_index, :trial]) |>
    @combine(__, score = mean(:score)) |>
    unstack(__, [:condition, :time, :sid, :fold, :sound_index, :trial], :train_type, :score) |>
    groupby(__, [:time]) |>
    @transform(__,
        scorediff = :var"athit-target" .- :var"athit-other", #"
        n = length(unique(:sid))
    ) |>
    @where(__, (:time .< 0.0) .& (:n .>= 24)) |>
    sort!(__, :time) |>
    groupby(__, [:condition, :sid, :fold, :sound_index, :trial]) |>
    @combine(__,
        var = std(:scorediff),
        # NOTE: ensure time steps use non-overlapping windows
        timevar = std(abs.(diff(:scorediff[1:ceil(Int,winlen_s*decode_sr):end]))),
        timecor = cor(:scorediff, lag(:scorediff, default = 0)),
        scoreother = mean(max.(0, :var"athit-other")), #"
        score = mean(above_thresh.(:scorediff,:fold))
    )
pretarget_df |> CSV.write(joinpath(cache_dir("eeg", "decoding"), "pretarget_attend.csv"))
# TODO: run stats using multi-levle on a per-trial basis to get best measurem

pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:condition]) |>
    @combine(__,
        score = mean(:score),
        lower = lowerboot(:score, alpha = 0.318),
        upper = upperboot(:score, alpha = 0.318),
    ) |>
    (
        @vlplot(
            config = {legend = {disable = true}},
            x = :condition, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:bar, y = {:score, title = ["Prop. of pre-target time ≥95th", " quantile of on-target time."]}) +
        @vlplot(:errorbar, y = {:lower, title = ""}, y2 = :upper, color = {value = "black"})
    );
pl |> save(joinpath(dir, "decode_pretarget_attend.svg"))

pcolors = ColorSchemes.bamako[range(0.3,0.7, length = 2)]
barwidth = 10
pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid]) |>
    @combine(__, var = mean(:var), timevar = mean(:timevar)) |>
    stack(__, [:var, :timevar], [:condition, :sid], variable_name = :measure) |>
    groupby(__, [:condition, :measure]) |>
    @combine(__,
        score = mean(:value),
        lower = lowerboot(:value, alpha = 0.05),
        upper = upperboot(:value, alpha = 0.05),
    ) |>
    (
        @vlplot(
            width = 145,
            height = 75,
            config = {
                legend = {disable = true},
                bar = {discreteBandSize = barwidth}
            },
            x = {:condition,
                axis = {title = "", labelAngle = 0,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}},
            color = {:measure, scale = {range = "#".*hex.(pcolors)}}) +
        @vlplot({:bar, xOffset = barwidth/2},
            transform = [{filter = "datum.measure == 'var'"}],
            y = {:score, stack = nothing, title = "Decoding Variance"}) +
        @vlplot({:bar, xOffset = -barwidth/2},
            transform = [{filter = "datum.measure != 'var'"}],
            y = {:score, stack = nothing, title = "Decoding Variance"}) +
        @vlplot({:rule, xOffset = barwidth/2},
            transform = [{filter = "datum.measure == 'var'"}],
            y = {:lower, title = ""}, y2 = :upper, color = {value = "black"}) +
        @vlplot({:rule, xOffset = -barwidth/2},
            transform = [{filter = "datum.measure != 'var'"}],
            y = {:lower, title = ""}, y2 = :upper, color = {value = "black"}) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "bottom", dx = 0, dy = -barwidth-2},
            transform = [{filter = "datum.condition == 'global' && datum.measure != 'var'"}],
            # x = {datum = "spatial"}, y = {datum = 0.},
            y = {datum = 0},
            color = {value = "black"},
            text = {value = "Global Var."},
        ) +
        @vlplot({:text, angle = -90, fontSize = 9, align = "left", baseline = "top", dx = 0, dy = barwidth+2},
            transform = [{filter = "datum.condition == 'global' && datum.measure == 'var'"}],
            # x = {datum = "spatial"}, y = {datum = },
            y = {datum = 0},
            color = {value = "black"},
            text = {value = "Local Var."})
    );
pl |> save(joinpath(dir, "decode_pretarget_flicker.svg"))

pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid]) |>
    @combine(__, var = mean(:var .- :timevar)) |>
    groupby(__, [:condition]) |>
    @combine(__,
        score = mean(:var),
        lower = lowerboot(:var, alpha = 0.318),
        upper = upperboot(:var, alpha = 0.318),
    ) |>
    (
        @vlplot(
            config = {legend = {disable = true}},
            x = :condition, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:bar, y = {:score, stack = nothing, title = "Variance difference"}) +
        @vlplot(:errorbar, y = {:lower, title = ""}, y2 = :upper, color = {value = "black"})
    );
pl |> save(joinpath(dir, "decode_pretarget_flicker_diff.svg"))

pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid, :trial]) |>
    @combine(__, var = mean(:var .- :timevar)) |>
    # groupby(__, [:condition, :trial]) |>
    # @combine(__,
    #     score = mean(:var),
    #     lower = lowerboot(:var, alpha = 0.05),
    #     upper = upperboot(:var, alpha = 0.05),
    # ) |>
    sort!(__, :trial) |>
    (
        @vlplot(:point, x = :trial, y = {:var, title = "Variance difference", type = :quantitative})
        # @vlplot(:errobar, y = {:lower, title = ""}, y2 = :upper, color = {value = "black"})
    );
pl |> save(joinpath(dir, "decode_pretarget_flicker_diff_time.svg"))

pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid]) |>
    @combine(__, score = mean(:scoreother)) |>
    groupby(__, [:condition]) |>
    @combine(__,
        score = mean(:score),
        lower = lowerboot(:score, alpha = 0.318),
        upper = upperboot(:score, alpha = 0.318),
    ) |>
    (
        @vlplot(
            config = {legend = {disable = true}},
            x = :condition, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:point, y = {:score, title = "Rectified correlation of other sources"}) +
        @vlplot(:errorbar, y = {:lower, title = ""}, y2 = :upper, color = {value = "black"})
    );
pl |> save(joinpath(dir, "decode_pretarget_attend_other.svg"))

pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid]) |>
    @combine(__, score = mean(:timecor)) |>
    groupby(__, [:condition]) |>
    @combine(__,
        score = mean(:score),
        lower = lowerboot(:score, alpha = 0.318),
        upper = upperboot(:score, alpha = 0.318),
    ) |>
    (
        @vlplot(
            config = {legend = {disable = true}},
            x = :condition, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:point, y = {:score, title = "Autocorrelation"}) +
        @vlplot(:errorbar, y = {:lower, title = ""}, y2 = :upper, color = {value = "black"})
    );
pl |> save(joinpath(dir, "decode_pretarget_attend_cor.svg"))

pl = @_ pretarget_df |>
    groupby(__, [:condition, :sid]) |>
    @combine(__, score = mean(:var)) |>
    groupby(__, [:condition]) |>
    @combine(__,
        score = mean(:score),
        lower = lowerboot(:score, alpha = 0.318),
        upper = upperboot(:score, alpha = 0.318),
    ) |>
    (
        @vlplot(
            config = {legend = {disable = true}},
            x = :condition, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:point, y = {:score, title = "Variance of derivative"}) +
        @vlplot(:errorbar, y = {:lower, title = ""}, y2 = :upper, color = {value = "black"})
    );
pl |> save(joinpath(dir, "decode_pretarget_attend_var.svg"))

tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
pldata = @_ plotdf |>
    unstack(__, [:condition, :time, :sid, :trial], :train_type, :score) |>
    @transform(__, scorediff = :var"athit-target" .- :var"athit-other") |> #"
    groupby(__, [:condition, :time, :sid]) |>
    @combine(__,
        score = mean(:scorediff),
        lower = lowerboot(:scorediff, alpha = 0.318),
        upper = upperboot(:scorediff, alpha = 0.318)
    )

pl = pldata |>
    @vlplot(
        facet = {field = :sid, columns = 4, type = :ordinal},
        config = {facet = {columns = 4}},
    ) +
    (
        @vlplot(x = {:time, type = :quantitative}, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:line, y = :score) +
        @vlplot(:errorband, y = :lower, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_timeline_diff_ind.svg"))

tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
pl = @_ plotdf |>
    @where(__, :sid .== 24) |>
    # @where(__, :sound_index .< 20) |>
    unstack(__, [:condition, :time, :sid, :sound_index], :train_type, :score) |>
    @transform(__, scorediff = :var"athit-target" .- :var"athit-other") |> #"
    @vlplot(
        facet = {field = :sound_index, columns = 4, type = :ordinal},
        config = {facet = {columns = 6}},
    ) +
    (
        @vlplot(width = 50, height = 50,
            x = {:time, type = :quantitative}, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot(:line, y = :scorediff)
    );
pl |> save(joinpath(dir, "decode_timeline_diff_trials.svg"))

# spotlight hypothesis predicts greater power in global signal somewhere between 200ms to 2s
# wavelength, before the target occurs e.g. 5 Hz to 0.5 Hz
flen = 32
fftdata = Array{Float64}(undef, flen)
function freqpower(x,times)
    n = findfirst(>=(first(times) + 0.25), times)
    stop = argmin(abs.(times .- 0))
    len = stop - n + 1
    off = max(0, len - flen)
    start = (stop - len) - off + 1
    slice = view(x,start:stop)
    fftdata[1:length(slice)] = slice
    fftdata[(length(slice)+1):end] .= 0
    result = abs.(rfft(fftdata))
    DataFrame(
        power = result,
        freq = range(0, decode_sr, length = length(result))
    )
end

powerdf = @_ timelines |>
    groupby(__, [:source, :time, :sid, :trial, :train_type, :condition]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:source, :sid, :trial, :train_type, :condition]) |>
    combine(freqpower(_1.score, _1.time), __) |>
    groupby(__, [:condition, :sid, :freq]) |>
    @combine(__, mean = mean(log.(:power)))

pl = @_ powerdf |> groupby(__, [:condition, :freq]) |>
    @combine(__,
        mean = mean(:mean),
        lower = lowerboot(:mean),
        upper = upperboot(:mean),
    ) |>
    @vlplot(x = :freq, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
    @vlplot(:line, y = :mean) +
    @vlplot(:point, y = :mean) +
    @vlplot(:errorbar, y = :lower, y2 = :upper);
pl |> save(joinpath(dir, "decode_timeline_freqpower.svg"))


