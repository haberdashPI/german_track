export prepare_decode_data, prepare_decode_stimuli, train_decoder, plot_decode_lambdas,
    EEGFeatureSelector, StimSelector, MultiSelector
using ColorSchemes

function prepare_decode_stimuli(params, windows, prefix)
    # Setup stimulus data
    # -----------------------------------------------------------------

    meta = load_stimulus_metadata()
    stimuli = Empty(Vector)

    starts = vcat(1,1 .+ cumsum(windows.len))
    progress = Progress(size(windows, 1), desc = "Organizing stimulus data...")
    for (i, trial) in enumerate(eachrow(windows))
        for (encoding, encoding_code) in pairs(params.stimulus.encodings)
            for source in params.stimulus.sources
                stim, stim_id = load_stimulus(source, trial, encoding_code,
                    params.stimulus.samplerate, meta)
                start = trial.start
                stop = min(size(stim,1), trial.start + trial.len - 1)
                fullrange = starts[i] : (starts[i+1] - 1)

                stimulus = if stop >= start
                    stimulus = Float32.(@view(stim[start:stop, :]))
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
        addfold!(__, params.train.nfolds, :sid, rng = stableRNG(2019_11_18, :decoding)) |>
        # insertcols!(__, :prediction => Ref(Float32[])) |>
        groupby(__, [:encoding]) |>
        transform!(__, :data => (x -> mean(reduce(vcat, x))) => :datamean, ungroup = false) |>
        transform!(__, [:data, :datamean] => ByRow((x, μ) -> x .-= μ) => :data, ungroup = false) |>
        transform!(__, :data => (x -> std(reduce(vcat, x))) => :datastd, ungroup = false) |>
        transform!(__, [:data, :datastd] => ByRow((x, σ) -> x .-= σ) => :data)

    GermanTrack.@save_cache prefix stimulidf

    stimulidf
end

function prepare_decode_data(params, prefix)
    data_prefix = joinpath(processed_datadir("analyses", "decode-data", "freqbinpower-sr$(params.stimulus.samplerate)"))
    GermanTrack.@load_cache data_prefix (subjects, :bson)
    events = load_all_subject_events(processed_datadir("eeg"), "h5")

    meta = GermanTrack.load_stimulus_metadata()

    function windowing_start_time(event, triallen)
        event.windowing == "target" ? meta.target_times[event.sound_index] :
        event.windowing == "pre-target" ?
            max(0.0, meta.target_times[event.sound_index]-1.5) *
                rand(GermanTrack.trialrng((:decode_windowing, seed), event)) :
        event.windowing == "random1" ?
            (max(triallen-1, 0.0) / params.stimulus.samplerate) *
            rand(GermanTrack.trialrng((:decode_windowing, seed, :first), event)) :
        event.windowing == "random2" ?
            (max(triallen-1, 0.0) / params.stimulus.samplerate) *
            rand(GermanTrack.trialrng((:decode_windowing, seed, :second), event)) :
        error("Unexpected windowing `$(event.windowing)`")
    end

    target_length = 1.0

    seed = 2019_11_18
    target_samples = round(Int, params.stimulus.samplerate*target_length)
    function event2window(event)
        triallen   = size(subjects[event.sid].eeg[event.trial], 2)
        start_time = windowing_start_time(event, triallen)

        start      = clamp(round(Int, params.stimulus.samplerate*start_time), 1, triallen)
        len        = clamp(target_samples, 1, triallen-start)
        (
            start     = start,
            len       = len,
            trialnum  = event.trial,
            event...
        )
    end

    windows = @_ events |>
        transform!(__, AsTable(:) => ByRow(findresponse) => :hittype) |>
        filter(_.hittype ∈ ["hit", "miss"], __) |>
        repeatby(__, :windowing => ["random1", "random2"]) |>
        combine(__, AsTable(:) => ByRow(event2window) => AsTable) |>
        transform!(__, :len => (x -> lag(cumsum(x), default = 1)) => :offset)

    nobs = sum(windows.len)
    nfeatures = size(first(subjects)[2].eeg[1],1)
    x = Array{Float32}(undef, nfeatures*params.stimulus.nlags, nobs)

    progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
    Threads.@threads for (i, trial) in collect(enumerate(eachrow(windows)))
        tstart = trial.start
        tstop = trial.start + trial.len - 1
        xstart = trial.offset
        xstop = trial.offset + trial.len - 1

        trialdata = withlags(subjects[trial.sid].eeg[trial.trialnum]', params.stimulus.lags)
        x[:, xstart:xstop] = view(trialdata, tstart:tstop, :)'
        next!(progress)
    end
    x_μ = mean(x, dims = 2)
    x .-= x_μ
    x_σ = std(x, dims = 2)
    x ./= x_σ
    x_scores = DataFrame(μ = vec(x_μ), σ = vec(x_σ))

    GermanTrack.@save_cache prefix x_scores

    x, windows, nfeatures
end

struct StimSelector
    fn
end
struct EEGFeatureSelector
    featfn
end
struct MultiSelector
    selectors
    MultiSelector(sels...) = new(sels)
end

selectstim(x::StimSelector, df, kind) = x.fn(df, kind)
selectstim(x::EEGFeatureSelector, df, kind) = df
function selectstim(x::MultiSelector, df, kind)
    helper(sels, df) = isempty(sels) ? df :
        helper(sels[2:end], selectstim(sels[1], df, kind))
    helper(x.selectors, df)
end

selectdata(x::StimSelector, data, row) = (:, :)
function selectdata(x::EEGFeatureSelector, data, row::DataFrameRow)
    (x.featfn(data, row), (row.offset):(row.offset + row.len - 1))
function selectdata(x, data, rows::AbstractDataFrame)
    local sel
    local rowsel
    local colsel
    try
        sel = @_ map(selectdata(x, data, _), eachrow(rows))
        rowsel = @_ map(_[1], sel)
        colsel = @_ mapreduce(_[2], vcat, sel)
        all(==(first(rowsel)), rowsel) || error("Non-uniform feature selection.")
        first(rowsel), colsel
    catch e
        @infiltrate
        rethrow(e)
    end
end
function selectdata(x::MultiSelector, data, rows)
    helper(sels, data) = isempty(sels) ? data :
        helper(sels[2:end], selectdata(sels[1], data, rows))
    helper(x.selectors, data)
end

function train_decoder(params, x, modelsetup, train_types)
    toxy(df, selector) = isempty(df) ? ([], []) : (
        x[selectdata(selector, x, df)...],
        reshape(reduce(vcat, row.data for row in eachrow(df)),1,:)
    )

    progress = Progress(params.train.max_steps * ngroups(modelsetup))

    modelrun = combine(modelsetup) do fold
        selector = train_types[fold.train_type[1]]
        train = @_ fold |> selectstim(selector, __, :train) |>
            @where(__, :split .== "train") |> toxy(__, selector)
        val   = @_ fold |> selectstim(selector, __, :train) |>
            @where(__, :split .== "validate")
        test  = @_ fold |> selectstim(selector, __, :test)  |>
            @where(__, :split .== "test")

        (isempty(train[1]) || isempty(test) || isempty(val)) && return DataFrame()

        model = GermanTrack.decoder(train[1], train[2], fold.λ[1], Flux.Optimise.RADAM(),
            progress = progress,
            batch = params.train.batchsize,
            max_steps = params.train.max_steps,
            min_steps = params.train.min_steps,
            patience = params.train.patience,
            inner = params.train.hidden_units,
            validate = @_ val |> toxy(__, selector)
        )

        test[!, :prediction] = [model(x[selectdata(selector, x, row)...]) for row in eachrow(test)]
        val[!, :prediction] = [model(x[selectdata(selector, x, row)...]) for row in eachrow(val)]
        test[!, :steps] .= GermanTrack.nsteps(model)

        DataFrame(model = model, result = test, validate = val)
    end

    ProgressMeter.finish!(progress)
    # alert("Completed model training!")

    predictions = @_ modelrun |> groupby(__, Not([:result, :model, :validate])) |>
        combine(only(_.result), __)
    valpredictions = @_ modelrun |> groupby(__, Not([:result, :model, :validate])) |>
        combine(only(_.validate), __)
    models = select(modelrun, Not([:result, :validate]))

    predictions, valpredictions, models
end

function plot_decode_lambdas(params, predictions, valpredictions, dir)
    score(x,y) = isempty(x) ? zero(eltype(x)) : cor(vec(x), vec(y))
    meta = GermanTrack.load_stimulus_metadata()
    function compute_scores(p)
        @_ p |>
            @transform(__, score = score.(:prediction, :data)) |>
            groupby(__, Not(:encoding)) |> @combine(__, score = mean(:score)) |>
            transform(__,
                :stim_id => ByRow(id -> (;(kw => meta[kw][id] for kw in [
                    :target_time_label,
                    :target_switch_label,
                    :target_times,
                    :salience_label,
                    :target_salience,
                ])...)) => AsTable,
                [:hittype, :windowing] => ByRow((x,y) -> string(x, "-", y)) => :test_type
            )
    end
    scores, valscores = compute_scores.((predictions, valpredictions))

    # score(x,y) = -sqrt(mean(abs2, xi - yi for (xi,yi) in zip(x,y)))
    tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 3)]

    function nanmean(xs)
        xs_ = (x for x in xs if !isnan(x))
        isempty(xs_) ? 0.0 : mean(xs_)
    end
    pldata = @_ scores |>
        # @where(__, (:windowing .== "target")) |>
        groupby(__, [:condition, :is_target_source, :λ, :sid]) |>
        @combine(__, score = nanmean(:score)) |>
        groupby(__, [:condition, :is_target_source, :λ]) |>
        combine(__, :score => boot(alpha = sqrt(0.05)) => AsTable)

    best_λs = @_ valscores |>
        groupby(__, [:condition, :is_target_source, :fold, :λ, :sid]) |> @combine(__, score = nanmean(:score)) |>
        groupby(__, [:fold, :condition, :λ]) |> @combine(__, score = median(:score)) |>
        groupby(__, Not(:condition)) |> @combine(__, score = minimum(:score)) |>
        repeatby(__, :cross_fold => 1:params.train.nfolds) |>
        @where(__, :cross_fold .!= :fold) |>
        combine(__, DataFrame(score = maximum(_1.score), λ = _1.λ[argmax(_1.score)]))

    best_λ = Dict(row.cross_fold => row.λ for row in eachrow(best_λs))
    # best_λ = lambdas[argmin(abs.(lambdas .- 0.002))]

    tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 3)]
    pl = @_ pldata |>
        # @where(__, :test_type .== "hit-target") |>
        @vlplot(
            facet = {column = {field = :condition, type = :nominal}}
        ) +
        (
            @vlplot(x = {:λ, scale = {type = :log}}) +
            @vlplot({:line, strokeCap = :round}, y = :value,
                color = {:is_target_source, type = :nominal, scale = {range = "#".*hex.(tcolors)}}) +
            @vlplot({:errorbar, ticks = {size = 5}}, y = :lower, y2 = :upper, color = "is_target_source:n") +
            @vlplot({:point, filled = true}, y = :value, color = "is_target_source:n") +
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

    best_λ
end
