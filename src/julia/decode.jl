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
    JLD

dir = mkpath(joinpath(plotsdir(), "figure6_parts"))

using GermanTrack: colors

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
subjects, events = load_all_subjects(processed_datadir("eeg"), "h5",
    encoding = eeg_encoding, framerate = sr)
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

x .-= mean(x, dims = 2)
x ./= std(x, dims = 2)

# Setup stimulus data
# -----------------------------------------------------------------

stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
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

# Train Model
# =================================================================

eegindices(row::DataFrameRow) = (row.offset):(row.offset + row.len - 1)
function eegindices(df::AbstractDataFrame)
    mapreduce(eegindices, vcat, eachrow(df))
end

function zscoremany(xs)
    μ = mean(reduce(vcat, xs))
    for x in xs
        x .-= μ
    end
    σ = std(reduce(vcat, xs))
    for x in xs
        x ./= σ
    end

    xs
end


filename = processed_datadir("analyses", "decode-predict-freqbin.jld")
if !isfile(filename)
    nfolds = 5

    @info "Generating cross-validated predictions, this could take a bit..."

    groupings = [:encoding]
    groups = @_ DataFrame(stimuli) |>
        @where(__, :condition .== "global") |>
        # @where(__, :is_target_source) |>
        # @where(__, :windowing .== "target") |>
        # train on quarter of subjects
        # @where(__, :sid .<= sort!(unique(:sid))[div(end,4)]) |>
        addfold!(__, nfolds, :sid, rng = stableRNG(2019_11_18, :decoding)) |>
        insertcols!(__, :predict => Ref(Float32[])) |>
        groupby(__, [:encoding]) |>
        transform!(__, :data => zscoremany => :data) |>
        groupby(__, groupings)

    max_steps = 50
    nλ = 12
    batchsize = 2048
    train_types = vcat(
        string.("athit-other-",["male","fem1","fem2"]),
        string.("athit-target-",["male","fem1","fem2"]),
        "pre-miss-target",
        # string.("athit-mix-",["male+fem1","male+fem2","fem1+fem2"]),
    )
    progress = Progress(max_steps * length(groups) * nfolds * nλ * length(train_types))
    validate_fraction = 0.2

    predictions, coefs = filteringmap(groups, folder = foldl, streams = 2, desc = nothing,
        :fold => 1:nfolds,
        :λ => exp.(range(log(1e-4),log(1e-1),length=nλ)),
        :train_type => train_types,
        function(sdf, fold, λ, train_type)
            hittype, windowing, source, is_target =
                train_type == "pre-miss-target" ? ("miss", "pre-target", sdf.target_source, true) :
                startswith(train_type, "athit-target-") ?
                    ("hit", "target", split(train_type, "-")[end], true) :
                startswith(train_type, "athit-other-") ?
                    ("hit", "target", split(train_type, "-")[end], false) :
                error("Unexpected `train_type` value of $train_type.")

            sdf = view(sdf, (sdf.source .== source) .&
                (sdf.is_target_source .== is_target), :)
            isempty(sdf) && return Empty(DataFrame)

            nontest = @_ filter((_1.fold != fold) &&
                            (_1.hittype == hittype) &&
                            (_1.windowing == windowing), sdf)
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

            model, taken_steps = lassoflux(xᵢ, yᵢ, λ, Flux.Optimise.RADAM(),
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
            test.steps = taken_steps
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

            test, coefs
        end)

    ProgressMeter.finish!(progress)
    alert("Completed model training!")

    save(filename, "predictions", predictions, "coefs", coefs)
else
    data = load(filename)
    predictions = data["predictions"]
    coefs = data["coefs"]
end

# Plotting
# -----------------------------------------------------------------

function zscoresafe(x)
    x = zscore(x)
    any(isnan, x) ? zero(x) : x
end

# score(x,y) = -sqrt(mean(abs2, xi - yi for (xi,yi) in zip(x,y)))
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
        :stim_id => (x -> cut(meta.target_salience[x], 2)) => :target_salience,
        :stim_id => (x -> meta.target_salience[x]) => :target_salience_level,
        [:hittype, :windowing] => ByRow((x,y) -> string(x, "-", y)) => :test_type
    )

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
    @combine(__, score = mean(:score))

best_λs = @_ scores |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :train_type, :test_type, :source, :λ, :fold]) |>
    @combine(__, score = nanmean(:score)) |>
    groupby(__, [:condition, :train_type, :test_type, :λ, :fold]) |>
    @combine(__, score = mean(:score)) |>
    @where(__, (:train_type .== "athit-target") .& (:test_type .== "hit-target")) |>
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

example = @_ predictions |>
    @where(__, (:λ .== first(best_λs.λ)) .& (:sid .== 33) .&
              (:windowing .== "target") .&
              (:hittype.== "hit") .&
              (:train_type .== "athit-target") .&
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
    filter(_.λ == best_λ[_.fold], __) |>
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
            x = {:train_type, axis = {title = "Source", labelAngle = -45,
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
    @where(__, (:train_type .∈ Ref(["athit", "atmiss"])) .& (:test_type .== "hit-target")) |>
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
    groupby(__, [:sid, :condition, :target_window, :source, :target_switch_label]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :target_window, :target_switch_label]) |>
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
            color = {:target_window, scale = {range = "#".*hex.(tcolors)}},
            x = {:target_window, axis = {title = "Source", labelAngle = -45,
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
    groupby(__, [:sid, :condition, :target_window, :source, :target_salience]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :target_window, :target_salience]) |>
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
            color = {:target_window, scale = {range = "#".*hex.(tcolors)}},
            x = {:target_window, axis = {title = "Source", labelAngle = -45,
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
    @where(__, :target_window .∈ Ref(["athit-hit", "pre-miss-hit"])) |>
    @transform(__,
        condition = string.(:condition),
        target_window = recode(:target_window,
            "athit-hit" => "target", "pre-miss-hit" => "nontarget"),
        target_salience = string.(recode(:target_salience, (levels(:target_salience) .=> ["Low", "High"])...)),
    ) |>
    groupby(__, [:sid, :condition, :trialnum, :target_salience, :target_time_label, :target_switch_label, :target_window]) |>
    @combine(__, score = maximum(:score)) |>
    unstack(__, [:sid, :condition, :trialnum, :target_salience, :target_time_label, :target_switch_label], :target_window, :score) |>
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

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
        is_target_source = recode(:is_target_source, true => "Target", false => "Non-target")
    ) |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :target_salience_level]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:sid, :condition, :is_target_source, :target_salience_level]) |>
    @combine(__, score = mean(:score)) |>
    @vlplot(
        facet = {
            column = {field = :condition, type = :ordinal},
            # row = {field = :is_target_source, type = :ordinal}
        }
    ) + (
        @vlplot({:point},
            x     = :target_salience_level,
            y     = {:score, type = :quantitative, aggregate = :mean},
            color = {:is_target_source, scale = {range = "#".*hex.(colors[[1,3]])}}
        )
    );
pl |> save(joinpath(dir, "decode_salience_continuous.svg"))

# TODO: plot decoding scores vs. hit-rate
