# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn, ShiftedArrays

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
        event[[:condition, :sid, :target_source, :sound_index]]...
    )
end

windows = @_ events |>
    filter(ishit(_) == "hit", __) |> eachrow |>
    Iterators.product(__, ["target", "pre-target"]) |>
    map(event2window(_...), __) |> vec |>
    DataFrame |>
    transform!(__, :len => (x -> lag(cumsum(x), default = 1)) => :offset)

nobs = sum(windows.len)
starts = vcat(1,1 .+ cumsum(windows.len))
nfeatures = size(first(subjects)[2].eeg[1],1)
nlags = round(Int,sr*max_lag)
x = Array{Float64}(undef, nfeatures*nlags, nobs)

progress = Progress(size(windows, 1), desc = "Organizing EEG data...")
Threads.@threads for (i, trial) in collect(enumerate(eachrow(windows)))
    tstart = trial.start
    tstop = trial.start + trial.len - 1
    xstart = trial.offset
    xstop = trial.offset + trial.len - 1

    trialdata = withlags(subjects[trial.sid].eeg[trial.trialnum]', -(nlags-1):0)
    x[:, xstart:xstop] = @view(trialdata[tstart:tstop, :])'
    next!(progress)
end

# Setup stimulus data
# -----------------------------------------------------------------

stim_encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())
encodings = ["pitch", "envelope"]
source_names = ["male", "fem1", "fem2"]
sources = [male_source, fem1_source, fem2_source]

stimuli = Empty(Vector)

progress = Progress(size(windows, 1), desc = "Organizing stimulus data...")
for (i, trial) in enumerate(eachrow(windows))
    for (j, encoding) in enumerate(encodings)
        for (source_name, source) in zip(source_names, sources)
            stim, stim_id = load_stimulus(source, trial, stim_encoding, sr, meta)
            start = trial.start
            stop = min(size(stim,1), trial.start + trial.len - 1)
            fullrange = starts[i] : (starts[i+1] - 1)

            stimulus = if stop >= start
                stimulus = @view(stim[start:stop, j])
            else
                Float64[]
            end

            stimuli = push!!(stimuli, (
                trial...,
                source           = source_name,
                encoding         = encoding,
                start            = start,
                stop             = stop,
                len              = stop - start + 1,
                data             = stimulus,
                is_target_source = trial.target_source == source_name,
                stim_id          = stim_id,

            ))
        end
    end
    next!(progress)
end

# Train Model
# -----------------------------------------------------------------

eegindices(row::DataFrameRow) = (row.offset):(row.offset + row.len - 1)
function eegindices(df::DataFrame)
    mapreduce(eegindices, vcat, eachrow(df))
end

# try alternative controll model training
# where we have a different model for each type of non-target
# and another where we have a different model for each type of target *and*
# non-target
file = processed_datadir("analyses", "decode-predict.json")
GermanTrack.@cache_results file predictions coefs begin
    @info "Generating cross-validated predictions, this could take a bit... (~15 minutes)"
    predictions, coefs = @_ DataFrame(stimuli) |>
        addfold!(__, 10, :sid, rng = stableRNG(2019_11_18, :decoding)) |>
        insertcols!(__, :predict => Ref(Float64[])) |>
        groupby(__, [:encoding, :is_target_source, :windowing]) |>
        filteringmap(__, folder = foldxt, streams = 2, desc = "Gerating predictions...",
            :fold => 1:10,
            function(sdf, fold)
                train = filter(x -> x.fold != fold, sdf)
                test  = filter(x -> x.fold == fold, sdf)

                model = fit(LassoPath, copy(@view(x[:, eegindices(train)])'),
                    reduce(vcat, train.data))
                test.predict = map(eachrow(test)) do testrow
                    predict(model, @view(x[:, eegindices(testrow)])', select = MinAICc())
                end

                coefs = DataFrame(coef(model, MinAICc())',
                    map(x -> @sprintf("coef%02d", x), 0:size(x,1)))

                test, coefs
            end
        )
end

meta = GermanTrack.load_stimulus_metadata()
score(x,y) = cor(x,y)
scores = @_ predictions |>
    groupby(__, [:sid, :condition, :source, :is_target_source, :trialnum, :stim_id, :windowing]) |>
    @combine(__, cor = mean(score.(:data, :predict))) |>
    transform!(__,
        :stim_id => (x -> meta.target_time_label[x]) => :target_time_label,
        :stim_id => (x -> meta.target_switch_label[x]) => :target_switch_label,
        :stim_id => (x -> cut(meta.target_salience[x], 2)) => :target_salience,
        :stim_id => (x -> meta.target_salience[x]) => :target_salience_level,
        [:windowing, :is_target_source] =>
            ByRow((w,t) ->
                (w == "target"     &&  t) ? "Target" :
                (w == "target"     && !t) ? "Non-target" :
                (w == "pre-target" &&  t) ? "Before target" :
              #=(w == "non-target" && !t)=# "Before non-target"
            ) => :target_window
    )

tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
mean_offset = 6
pl = @_ scores |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :target_window, :source]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :target_window]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {column = {field = :condition, type = :nominal}},
    ) + (
        @vlplot(
            width = 75, autosize = "fit",
            color = {:target_window, scale = {range = "#".*hex.(tcolors)}},
            x = {:target_window, axis = {title = "Source", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, },
            y = {:cor, title = ["Decoder Correlation", "(Envelope & Pitch Surprisal)"],
                scale = {zero = false}},
        ) +
        @vlplot({:point, xOffset = -mean_offset/2},
            y = "mean(cor)",
        ) +
        @vlplot({:point, filled = true, xOffset = mean_offset/2},
        ) +
        @vlplot({:rule, xOffset = -mean_offset/2},
            color = {value = "black"},
            y = "ci0(cor)",
            y2 = "ci1(cor)",  # {"cor:q", aggregate = :ci1}
        )
    );
pl |> save(joinpath(dir, "decode.svg"))

tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
mean_offset = 6
pldata = @_ scores |>
    @transform(__, condition = string.(:condition)) |>
    groupby(__, [:sid, :condition, :target_window, :source]) |>
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :target_window]) |>
    @combine(__, cor = mean(:cor)) |>
    unstack(__, [:sid, :condition], :target_window, :cor) |>
    stack(__, Between(Symbol("Before non-target"), Symbol("Non-target")),
        [:sid, :condition, :Target], variable_name = :baseline, value_name = :basevalue)
pl = pldata |>
    @vlplot(
        config = {legend = {disable = true}},
        facet = {
            row = {field = :condition, type = :nominal},
            column = {field = :baseline, type = :nominal}
        },
    ) + (
        @vlplot() +
        @vlplot({:point, filled = true},
            width = 100, height = 100, autosize = "fit",
            color = {:condition, scale = {range = "#".*hex.(colors)}},
            x = {:basevalue, axis = {title = "Baseline Corrleation", labelAngle = -45,
                labelExpr = "split(datum.label,'\\n')"}, scale = {zero = false}},
            y = {:Target, title = "Target Correlation",
                scale = {zero = false}},
        ) +
        (
            @vlplot(data = {values = [
                {x = minimum(pldata.basevalue)*0.95, y = minimum(pldata.Target)*0.95},
                {x = maximum(pldata.basevalue)*1.05, y = maximum(pldata.Target)*1.05}]}) +
            @vlplot({:line, strokeDash = [2, 2], strokeWidth = 1}, x = :x, y = :y)
        )
    )
pl |> save(joinpath(dir, "decode_ind.svg"))

scolors = ColorSchemes.bamako[[0.2,0.8]]
mean_offset = 6
pldata = @_ scores |>
    @where(__, :target_window .∈ Ref(["Target", "Before non-target"])) |>
    @transform(__,
        condition = string.(:condition),
        target_window = recode(:target_window,
            "Target" => "target", "Before non-target" => "nontarget"),
        target_salience = string.(recode(:target_salience, (levels(:target_salience) .=> ["Low", "High"])...)),
    ) |>
    groupby(__, [:sid, :condition, :trialnum, :target_salience, :target_time_label, :target_switch_label, :target_window]) |>
    @combine(__, cor = maximum(:cor)) |>
    unstack(__, [:sid, :condition, :trialnum, :target_salience, :target_time_label, :target_switch_label], :target_window, :cor) |>
    @transform(__, cordiff = :nontarget .- :target)

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
            y = {:cordiff, title = ["Target - Non-target Correlation"],
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
            y = {:cordiff, title = ["Target - Non-target Correlation"],
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
                scale = {range = "#".*hex.(colors)}},
            x = {:target_switch_label, type = :nominal,
                sort = ["Low", "High"],
                type = :ordinal,
                axis = {title = "Salience", labelAngle = -45,
                    labelExpr = "slice(datum.label,'\\n')"}, },
            y = {:cordiff, title = ["Target - Non-target Correlation"],
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
                axis = {title = "Salience", labelAngle = -45,
                    labelExpr = "slice(datum.label,'\\n')"}, },
            y = {:cordiff, title = ["Target - Non-target Correlation"],
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
    @combine(__, cor = mean(:cor)) |>
    groupby(__, [:sid, :condition, :is_target_source, :target_salience_level]) |>
    @combine(__, cor = mean(:cor)) |>
    @vlplot(
        facet = {
            column = {field = :condition, type = :ordinal},
            # row = {field = :is_target_source, type = :ordinal}
        }
    ) + (
        @vlplot({:line},
            x     = :target_salience_level,
            y     = {:cor, type = :quantitative, aggregate = :mean},
            color = {:is_target_source, scale = {range = "#".*hex.(colors[[1,3]])}}
        )
    );
pl |> save(joinpath(dir, "decode_salience_continuous.svg"))

# TODO: plot decoding scores vs. hit-rate

# start with something basic: decoding accuracry (e.g. correlation or L1)
# for target vs. the two non-target stimuli
