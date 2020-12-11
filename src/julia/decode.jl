# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf

dir = mkpath(joinpath(plotsdir(), "figure6_parts"))

using GermanTrack: colors

# Setup EEG Data
# -----------------------------------------------------------------

# eeg_encoding = FFTFilteredPower("freqbins", Float64[1, 3, 7, 15, 30, 100])
# eeg_encoding = JointEncoding(
#     FilteredPower("delta", 1, 3),
#     FilteredPower("theta", 3, 7),
#     FilteredPower("alpha", 7, 15),
#     FilteredPower("beta", 15, 30),
#     FilteredPower("gamma", 30, 100),
# )
eeg_encoding = RawEncoding()

sr = 32
subjects, events = load_all_subjects(processed_datadir("eeg"), "h5",
    encoding = eeg_encoding, framerate = sr)
meta = GermanTrack.load_stimulus_metadata()

target_length = 1.0
max_lag = 2.0

target_samples = round(Int, sr*target_length)
windows = @_ events |>
    filter(ishit(_) == "hit", __) |> eachrow |>
    map(function(event)
        triallen     = size(subjects[event.sid].eeg[event.trial], 2)
        start        = clamp(round(Int, sr*meta.target_times[event.sound_index]), 1,
                            triallen)
        len          = clamp(target_samples, 1, triallen-start)
        (
            start    = start,
            len      = len,
            trialnum = event.trial,
            event[[:condition, :sid, :target_source, :sound_index]]...
        )
        end, __) |>
    DataFrame

nobs = sum(windows.len)
starts = vcat(1,1 .+ cumsum(windows.len))
nfeatures = size(first(subjects)[2].eeg[1],1)
nlags = round(Int,sr*max_lag)
x = Array{Float64}(undef, nfeatures*nlags, nobs)

progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
Threads.@threads for (i, trial) in collect(enumerate(eachrow(windows)))
    start = trial.start
    stop = trial.start + trial.len - 1
    trialdata = withlags(subjects[trial.sid].eeg[trial.trialnum]', -(nlags-1):0)
    x[:, starts[i] : (starts[i+1]-1)] = @view(trialdata[start:stop, :])'
    next!(progress)
end

# Setup stimulus data
# -----------------------------------------------------------------

stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
encodings = ["pitch", "envelope"]
source_names = ["male", "fem1", "fem2"]
sources = [male_source, fem1_source, fem2_source]
dims = (nobs,length(encodings),length(source_names))
dimindex(dims, n) = getindex.(CartesianIndices(dims), n) |> vec
stimuli = DataFrame(
    value            = Array{Float64}(undef, prod(dims)),
    observation      = dimindex(dims, 1),
    encoding         = categorical(encodings[dimindex(dims, 2)]),
    source           = categorical(source_names[dimindex(dims,3)]),
    is_target_source = BitArray(undef, prod(dims)),
    sid              = Array{Int}(undef, prod(dims)),
    condition        = CategoricalArray{String}(undef, prod(dims),
                                                levels = levels(windows.condition)),
    trial            = Array{Int}(undef, prod(dims))
)

progress = Progress(size(windows, 1), desc = "Organizing stimulus data...")
for (i, trial) in enumerate(eachrow(windows))
    for (j, encoding) in enumerate(encodings)
        for (source_name, source) in zip(source_names, sources)
            stim, stim_id = load_stimulus(source, trial, stim_encoding, sr, meta)
            start = trial.start
            stop = min(size(stim,1), trial.start + trial.len - 1)
            fullrange = starts[i] : (starts[i+1] - 1)

            if stop >= start
                indices = @with(stimuli,
                    findall((:encoding .== encoding) .& (:source .== source_name)))

                len = stop - start + 1
                fillrange =  starts[i]        : (starts[i] + len - 1)
                zerorange = (starts[i] + len) : (starts[i+1]     - 1)

                stimuli[indices[fillrange], :value]  = @view(stim[start:stop, j])
                stimuli[indices[zerorange], :value] .= zero(eltype(stimuli.value))
            else
                stimuli[indices[fullrange], :value] .= zero(eltype(stimuli.value))
            end

            stimuli[indices[fullrange], :is_target_source] .= trial.target_source == source_name
            stimuli[indices[fullrange], :sid]              .= trial.sid
            stimuli[indices[fullrange], :condition]        .= trial.condition
            stimuli[indices[fullrange], :trial]            .= trial.trialnum
        end
    end
    next!(progress)
end

# Train Model
# -----------------------------------------------------------------

@_ stimuli |>
    addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :decoding))

sdf = @where(stimuli, (:fold .!= 1) .& (:encoding .== "pitch") .& :is_target_source)

model = fit(LassoRegression, @views(x[:, sdf.observation])', sdf.value)

@info "Generating cross-validated predictions, this could take a bit... (~15 minutes)"
predictions, coefs = @_ stimuli |>
    addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :decoding)) |>
    @transform(__, predict = 0.0) |>
    groupby(__, [:encoding, :is_target_source]) |>
    filteringmap(__, folder = foldxt, streams = 2, desc = "Gerating predictions...",
        :fold => 1:10,
        function(sdf, fold)
            train = filter(x -> x.fold != fold, sdf)
            test  = filter(x -> x.fold == fold, sdf)

            model = fit(LassoPath, @view(x[:, train.observation])', train.value)
            test.predict = predict(model, @view(x[:, test.observation])',
                select = MinAICc())

            coefs = DataFrame(coef(model, MinAICc())',
                map(x -> @sprintf("coef%02d", x), 0:size(x,1)))

            test, coefs
        end
    )

scores = @_ predictions |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :trial]) |>
    @combine(__,
        cor = cor(:value, :predict),
        l1  = mean(abs.(:value .- :predict))
    ) |>
    stack(__, [:cor, :l1], [:sid, :condition, :is_target_source, :source, :trial],
        variable_name = :measure)

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @where(__, :measure .!= "l1") |> # l1 is no good
    @transform(__,
        condition = string.(:condition),
        measure   = string.(:measure),
    ) |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :measure]) |>
    @combine(__, value = mean(:value)) |>
    groupby(__, [:sid, :condition, :is_target_source, :measure]) |>
    @combine(__, value = mean(:value)) |>
    @vlplot(
        width = 242, autosize = "fit",
        config = {legend = {disable = true}},
        color = {"is_target_source:o", scale = {range = "#".*hex.(colors[[1,3]])}},
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:value, title = ["Decoder Correlation", "(Envelope & Pitch Surprisal)"],
            scale = {zero = false}},
    ) +
    @vlplot({:point, xOffset = -mean_offset},
        transform = [{filter = "datum.is_target_source"}],
        y = "mean(value)",
    ) +
    @vlplot({:point, filled = true, xOffset = -ind_offset},
        transform = [{filter = "datum.is_target_source"}],
    ) +
    @vlplot({:point, xOffset = mean_offset},
        transform = [{filter = "!datum.is_target_source" }],
        y = "mean(value)",
    ) +
    @vlplot({:point, filled = true, xOffset = ind_offset},
        transform = [{filter = "!datum.is_target_source" }],
    ) +
    @vlplot({:rule, xOffset = -mean_offset},
        transform = [{filter = "datum.is_target_source"}],
        color = {value = "black"},
        y = "ci0(value)",
        y2 = "ci1(value)",  # {"value:q", aggregate = :ci1}
    ) +
    @vlplot({:rule, xOffset = mean_offset},
        transform = [{filter = "!datum.is_target_source"}],
        color = {value = "black"},
        y = "ci0(value)",
        y2 = "ci1(value)",  # {"value:q", aggregate = :ci1}
    ) +
    @vlplot({:text, align = "center", dx = -ind_offset, dy = -10},
        transform = [{filter = "datum.is_target_source && datum.condition == 'global'"}],
        y = "max(value)",
        text = {value = "Target"}
    ) +
    @vlplot({:text , align = "center", dx = ind_offset, dy = -10},
        transform = [{filter = "!datum.is_target_source && datum.condition == 'global'"}],
        y = "max(value)",
        text = {value = "Non-target"}
    );
pl |> save(joinpath(dir, "decode.svg"))


# start with something basic: decoding accuracry (e.g. correlation or L1)
# for target vs. the two non-target stimuli
