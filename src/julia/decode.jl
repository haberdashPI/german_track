# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn

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
    trial            = Array{Int}(undef, prod(dims)),
    stim_id          = Array{Int}(undef, prod(dims))
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
            stimuli[indices[fullrange], :stim_id]          .= stim_id
        end
    end
    next!(progress)
end

# Train Model
# -----------------------------------------------------------------

# TODO: try a decoder per

file = processed_datadir("analyses", "decode-predict.json")
GermanTrack.@store_cache file predictions coefs begin
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
end

meta = GermanTrack.load_stimulus_metadata()
scores = @_ predictions |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :trial, :stim_id]) |>
    @combine(__, cor = cor(:value, :predict)) |>
    transform!(__,
        :stim_id => (x -> meta.target_time_label[x]) => :target_time_label,
        :stim_id => (x -> meta.target_switch_label[x]) => :target_switch_label,
        :stim_id => (x -> cut(meta.target_salience[x], 2)) => :target_salience,
        :stim_id => (x -> meta.target_salience[x]) => :target_salience_level
    )

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :is_target_source, :source]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :is_target_source]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        width = 242, autosize = "fit",
        config = {legend = {disable = true}},
        color = {"is_target_source:o", scale = {range = "#".*hex.(colors[[1,3]])}},
        x = {:condition, axis = {title = "", labelAngle = 0,
            labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
        y = {:cor, title = ["Decoder Correlation", "(Envelope & Pitch Surprisal)"],
            scale = {zero = false}},
    ) +
    @vlplot({:point, xOffset = -mean_offset},
        transform = [{filter = "datum.is_target_source"}],
        y = "mean(cor)",
    ) +
    @vlplot({:point, filled = true, xOffset = -ind_offset},
        transform = [{filter = "datum.is_target_source"}],
    ) +
    @vlplot({:point, xOffset = mean_offset},
        transform = [{filter = "!datum.is_target_source" }],
        y = "mean(cor)",
    ) +
    @vlplot({:point, filled = true, xOffset = ind_offset},
        transform = [{filter = "!datum.is_target_source" }],
    ) +
    @vlplot({:rule, xOffset = -mean_offset},
        transform = [{filter = "datum.is_target_source"}],
        color = {value = "black"},
        y = "ci0(cor)",
        y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
    ) +
    @vlplot({:rule, xOffset = mean_offset},
        transform = [{filter = "!datum.is_target_source"}],
        color = {value = "black"},
        y = "ci0(cor)",
        y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
    ) +
    @vlplot({:text, align = "center", dx = -ind_offset, dy = -10},
        transform = [{filter = "datum.is_target_source && datum.condition == 'global'"}],
        y = "max(cor)",
        text = {value = "Target"}
    ) +
    @vlplot({:text , align = "center", dx = ind_offset, dy = -10},
        transform = [{filter = "!datum.is_target_source && datum.condition == 'global'"}],
        y = "max(cor)",
        text = {value = "Non-target"}
    );
pl |> save(joinpath(dir, "decode.svg"))

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
        target_salience = string.(recode(:target_salience, (levels(:target_salience) .=> ["Low", "High"])...)),
        is_target_source = recode(:is_target_source, true => "Target", false => "Non-target")
    ) |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :target_salience]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :is_target_source, :target_salience]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet  = {column = {field = :is_target_source, type = :nominal}}
    ) + (
        @vlplot(
            width = 175,
            color = {"target_salience:o", scale = {range = "#".*hex.(colors[[1,3]])}},
            x = {:condition, axis = {title = "", labelAngle = 0,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
            y = {:cor, title = ["Decoder Correlation", "(Envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset},
            transform = [{filter = "(datum.target_salience == 'Low')"}],
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = -ind_offset},
            transform = [{filter = "(datum.target_salience == 'Low')"}],
        ) +
        @vlplot({:point, xOffset = mean_offset},
            transform = [{filter = "!(datum.target_salience == 'Low')" }],
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = ind_offset},
            transform = [{filter = "!(datum.target_salience == 'Low')" }],
        ) +
        @vlplot({:rule, xOffset = -mean_offset},
            transform = [{filter = "(datum.target_salience == 'Low')"}],
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        ) +
        @vlplot({:rule, xOffset = mean_offset},
            transform = [{filter = "!(datum.target_salience == 'Low')"}],
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = -ind_offset, dy = 10},
            transform = [{filter = "(datum.target_salience == 'Low') && datum.condition == 'global'"}],
            y = "min(cor)",
            text = {value = "Low"}
        ) +
        @vlplot({:text , align = "left", dx = ind_offset, dy = -10},
            transform = [{filter = "!(datum.target_salience == 'Low') && datum.condition == 'global'"}],
            y = "max(cor)",
            text = {value = "High"}
        )
    )
pl |> save(joinpath(dir, "decode_salience.svg"))

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
        is_target_source = recode(:is_target_source, true => "Target", false => "Non-target")
    ) |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :target_switch_label]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :is_target_source, :target_switch_label]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet  = {column = {field = :is_target_source, type = :nominal}}
    ) + (
        @vlplot(
            width = 175,
            color = {"target_switch_label:o", scale = {range = "#".*hex.(colors[[1,3]])}},
            x = {:condition, axis = {title = "", labelAngle = 0,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
            y = {:cor, title = ["Decoder Correlation", "(Envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset},
            transform = [{filter = "(datum.target_switch_label == 'near')"}],
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = -ind_offset},
            transform = [{filter = "(datum.target_switch_label == 'near')"}],
        ) +
        @vlplot({:point, xOffset = mean_offset},
            transform = [{filter = "!(datum.target_switch_label == 'near')" }],
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = ind_offset},
            transform = [{filter = "!(datum.target_switch_label == 'near')" }],
        ) +
        @vlplot({:rule, xOffset = -mean_offset},
            transform = [{filter = "(datum.target_switch_label == 'near')"}],
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        ) +
        @vlplot({:rule, xOffset = mean_offset},
            transform = [{filter = "!(datum.target_switch_label == 'near')"}],
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = -ind_offset, dy = 10},
            transform = [{filter = "(datum.target_switch_label == 'near') && datum.condition == 'global'"}],
            y = "min(cor)",
            text = {value = "Near"}
        ) +
        @vlplot({:text , align = "left", dx = ind_offset, dy = -10},
            transform = [{filter = "!(datum.target_switch_label == 'near') && datum.condition == 'global'"}],
            y = "max(cor)",
            text = {value = "Far"}
        )
    )
pl |> save(joinpath(dir, "decode_switch.svg"))


mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
        is_target_source = recode(:is_target_source, true => "Target", false => "Non-target")
    ) |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :target_time_label]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :is_target_source, :target_time_label]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet  = {column = {field = :is_target_source, type = :nominal}}
    ) + (
        @vlplot(
            width = 175,
            color = {"target_time_label:o", scale = {range = "#".*hex.(colors[[1,3]])}},
            x = {:condition, axis = {title = "", labelAngle = 0,
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}, },
            y = {:cor, title = ["Decoder Correlation", "(Envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset},
            transform = [{filter = "(datum.target_time_label == 'early')"}],
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = -ind_offset},
            transform = [{filter = "(datum.target_time_label == 'early')"}],
        ) +
        @vlplot({:point, xOffset = mean_offset},
            transform = [{filter = "!(datum.target_time_label == 'early')" }],
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = ind_offset},
            transform = [{filter = "!(datum.target_time_label == 'early')" }],
        ) +
        @vlplot({:rule, xOffset = -mean_offset},
            transform = [{filter = "(datum.target_time_label == 'early')"}],
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        ) +
        @vlplot({:rule, xOffset = mean_offset},
            transform = [{filter = "!(datum.target_time_label == 'early')"}],
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = -ind_offset, dy = 10},
            transform = [{filter = "(datum.target_time_label == 'early') && datum.condition == 'global'"}],
            y = "min(cor)",
            text = {value = "Early"}
        ) +
        @vlplot({:text , align = "left", dx = ind_offset, dy = -10},
            transform = [{filter = "!(datum.target_time_label == 'early') && datum.condition == 'global'"}],
            y = "max(cor)",
            text = {value = "Late"}
        )
    )
pl |> save(joinpath(dir, "decode_timing.svg"))

mean_offset = 15
ind_offset = 6
pl = @_ scores |>
    @transform(__,
        condition = string.(:condition),
        is_target_source = recode(:is_target_source, true => "Target", false => "Non-target")
    ) |>
    groupby(__, [:sid, :condition, :is_target_source, :source, :target_salience_level]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :is_target_source, :target_salience_level]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        facet = {
            column = {field = :condition, type = :ordinal},
            # row = {field = :is_target_source, type = :ordinal}
        }
    ) + (
        @vlplot({:point, filled = true},
            x     = :target_salience_level,
            y     = :cor,
            color = {:is_target_source, scale = {range = "#".*hex.(colors[[1,3]])}}
        )
    );
pl |> save(joinpath(dir, "decode_salience_continuous.svg"))


# start with something basic: decoding accuracry (e.g. correlation or L1)
# for target vs. the two non-target stimuli
