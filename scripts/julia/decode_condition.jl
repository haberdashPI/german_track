# Setup
# =================================================================

using DrWatson #; @quickactivate("german_track")
using EEGCoding, GermanTrack, DataFrames, StatsBase, Underscores, Transducers,
    BangBang, ProgressMeter, HDF5, DataFramesMeta, Lasso, VegaLite, Colors,
    Printf, LambdaFn, ShiftedArrays, ColorSchemes, Flux, CUDA, GLM, SparseArrays,
    JLD, Arrow, FFTW, GLM, CategoricalArrays, Tables, DataStructures, Indexing,
    LazyArrays

dir = mkpath(joinpath(plotsdir(), "figure2_parts"))
meta = GermanTrack.load_stimulus_metadata()

using GermanTrack: colors

include(joinpath(scriptsdir(), "julia", "setup_decode_params.jl"))

# Main figure
# =================================================================

# variable setup
# -----------------------------------------------------------------

prefix = joinpath(processed_datadir("analyses", "decode-timeline"), "testing")
GermanTrack.@load_cache prefix timelines

# setup plot data
plotdf = @_ timelines |>
    # @where(__, :train_type .!= "atmiss-target") |>
    # @where(__, :train_condition .== :condition) |>
    groupby(__, [:condition, :time, :sid, :is_target_source, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

# labels = OrderedDict(
#     "athit-target" => "Target Source",
#     "athit-other" => "Other Sources",
#     "athit-pre-target" => "Baseline"
# )
# tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

# Paper plots
# -----------------------------------------------------------------

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12,14].+1)]]
pcolors[[1,end]] = GermanTrack.grayify.(pcolors[[1,end]])

target_len_y = -0.075
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :is_target_source, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type =
            getindices(Dict(true => "Target Source", false => "Other Sources"),
            :is_target_source)
    ) |>
    # @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    combine(__, :score => boot(alpha = sqrt(0.05)) => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" :
            type == "Baseline" ? "before" : cond) => :train_label) |>
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
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "global", "spatial", "object", "before"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            strokeDash = {:train_type, range = [[1,0], [4,1], [2,1]], sort = ["Target Source", "Other Sources", "Baseline"]},
            y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "right", dx = -3, dy = -20},
            transform = [
                {filter = "datum.time > 2.1 && datum.time < 2.2 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "left", baseline = "top", dx = 0, dy = 9},
            transform = [
                {filter = "datum.time > 2 && datum.time < 2.1 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.1, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "center", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.8 && datum.time < 1.9 && datum.train_type == 'Baseline'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.1, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "fig2c.svg"))

# Supplement: decoding difference
# =================================================================

target_len_y = -0.075
pl = @_ plotdf |>
    @transform(__,
        train_type = train_type = getindices(Dict(true => "target", false => "other"),
            :is_target_source)
    ) |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    unstack(__, [:condition, :time, :sid], :train_type, :score) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        diff = :target .- :other,
    ) |>
    # @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time]) |>
    combine(__, :diff => boot(alpha = sqrt(0.05)) => AsTable) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    ) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:condition, sort = ["global", "spatial", "object", "before"],
                title = "Source", scale = { range = "#".*hex.(GermanTrack.colors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            y = {:value, title = "Target - Other Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    ));
pl |> save(joinpath(dir, "decode_diff.svg"))

# Supplement: decoding by source aligned to switches
# =================================================================

prefix = joinpath(processed_datadir("analyses", "decode-timeline-switch"), "testing")
GermanTrack.@load_cache prefix timelines
switch_offset(sound_index, switch_index) = meta.switch_regions[sound_index][switch_index][2]

# Switch by source accuracy
# -----------------------------------------------------------------

pcolors = GermanTrack.colors

target_wins(row) =
    row[Symbol(row.target_source)] == max(row.male, row.fem1, row.fem2)
function dir_wins(row)
    cols = [:male, :fem1, :fem2]
    maxcol = cols[argmax(getindices(row,cols))]
    row[Symbol(maxcol, :dir)] ≥ 0
end

switch_end(stim, index) = meta.switch_regions[stim][index][2]
sourcename(str) = match(r"\w+", str).match

trained_speaker(si) = get(["male", "fem1", "fem2"], Int(meta.speakers[si]), missing)
azimuths = @_ timelines |> groupby(__, [:sound_index, :switch_index]) |>
    combine(__, [:switch_index, :sound_index, :time] =>
        ((switch, sound, time) -> azimuthdf(first(switch), first(sound), unique(time) |> sort!, meta)) => AsTable)
# setup plot data
plotdf_base = @_ timelines |>
    @where(__, :source .== :trained_source) |>
    @where(__, :time .+ switch_offset.(:sound_index, :switch_index) .<
        getindices(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :lagcut .== 0) |>
    @transform(__,
        is_trained_target = trained_speaker.(:sound_index) .== :trained_source,
        source_name = sourcename.(:source)
    ) |>
    groupby(__, [:sid, :trial, :encoding, :switch_index, :lagcut,
        :time, :condition, :hittype]) |>
    @transform(__, target_source = first(:trained_source[:is_trained_target])) |>
    select(__, Not([:is_target_source, :is_trained_target])) |>
    unstack(__, Not([:source, :trained_source, :source_name, :score]), :trained_source, :score) |>
    leftjoin(__, azimuths, on = [:time, :sound_index, :switch_index]) |>
    subset(__, r"dir$" => ByRow((cols...) -> any(!ismissing, cols)))

function intarget(meta, id, time, switch_index)
    if meta.target_times[id] > 0
        switch_time = meta.switch_regions[id][switch_index][2] + time
        return meta.target_times[id] < switch_time < (meta.target_times[id] + 1)
    else
        return false
    end
end

target_counts = @_ timelines |>
    @where(__, :is_target_source .& (:source .== :trained_source) .& (:lagcut .== 0)) |>
    @transform(__, target_count = intarget.(Ref(meta), :sound_index, :time, :switch_index))

plotdf_base_source = @_ plotdf_base |>
    @where(__, :condition .!= "spatial") |>
    groupby(__, Not(:encoding)) |>
    combine(__, [:male, :fem1, :fem2] .=> mean, renamecols = false) |>
    transform(__, AsTable(:) => ByRow(target_wins) => :correct) |>
    insertcols!(__, :spatial_source => false, :chance => 1/3)

plotdf_base_dir = @_ plotdf_base |>
    @where(__, :condition .!= "object") |>
    groupby(__, Not(:encoding)) |>
    combine(__, [:male, :fem1, :fem2] .=> mean, renamecols = false) |>
    transform(__, AsTable(:) => ByRow(dir_wins) => :correct) |>
    insertcols!(__, :spatial_source => true, :chance => 0.5)

plotdf = @_ vcat(plotdf_base_source, plotdf_base_dir) |>
    @transform(__, switch_timing = cut(switch_end.(:sound_index, :switch_index), 2)) |>
    groupby(__, [:condition, :time, :lagcut, :sid, :switch_timing, :hittype, :spatial_source]) |>
    @combine(__, correct = mean(:correct), chance = mean(:chance)) |>
    groupby(__, [:condition, :time, :lagcut, :switch_timing, :hittype, :spatial_source]) |>
    combine(__, :correct => boot(alpha = 0.05) => AsTable, :chance => mean => :chance)

plotdf = innerjoin(plotdf, target_counts, on = [:condition, :time, :lagcut, :hittype])

pl = @_ plotdf |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
    ) |>
    @where(__, :hittype .== "hit") |>
    # @where(__, -1 .< :time .< 2.5) |>
    @vlplot(
        spacing = 5,
        facet = {column = {field = :switch_timing}, row = {field = :spatial_source}}
        # config = {legend = {disable = true}},
    ) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            color = {:condition, sort = ["global", "spatial", "object", "before"],
                title = "Condition", scale = { range = "#".*hex.(GermanTrack.colors) }}
        ) +
        (@vlplot(
            x = {:time, type = :quantitative, title = "Time from Switch offset (s)"}) +
            @vlplot({:line, strokeJoin = :round},
                y = {:value, title = "P(Correct)", scale = {zero = false}}) +
            @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
        )+
        @vlplot({:rule, clip = true, strokeDash = [2 2], size = 1},
            y = "mean(chance)", color = {value = "black"})
    ) +  (
        @vlplot({:area, opacity = 0.3},
        y = {:target_count, title = "P(target)", scale = {domain = [0, 1]}},
        x = :time,
        color = {value = "black"}
    )));
pl |> save(joinpath(dir, "decode_source_near_switch_by_source.svg"))

# Plot by salience
# -----------------------------------------------------------------

plotdf = @_ plotdf_base |>
    @transform(__, salience_label = getindices(meta.salience_label, :sound_index)) |>
    groupby(__, [:condition, :time, :lagcut, :sid, :switch_timing, :hittype,
        :spatial_source, :salience_label]) |>
    @combine(__, correct = mean(:correct)) |>
    groupby(__, [:condition, :time, :lagcut, :switch_timing, :hittype,
        :spatial_source, :salience_label]) |>
    combine(__, :correct => boot(alpha = 0.05) => AsTable) |>
    @transform(__, chance = ifelse.(:spatial_source, 0.5, 1/3))

pl = @_ plotdf |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
    ) |>
    @where(__, :hittype .== "hit") |>
    @where(__, .!(:spatial_source)) |>
    # @where(__, -1 .< :time .< 2.5) |>
    @vlplot(
        spacing = 5,
        facet = {column = {field = :switch_timing}, row = {field = :condition}}
        # config = {legend = {disable = true}},
    ) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            color = {:salience_label, #sort = ["global", "spatial", "object", "before"],
                title = "Condition"}#, scale = { range = "#".*hex.(GermanTrack.colors) }}
        ) +
        (@vlplot(
            x = {:time, type = :quantitative, title = "Time from Switch offset (s)"}) +
            @vlplot({:line, strokeJoin = :round},
                y = {:value, title = "P(Correct)", scale = {zero = false}}) +
            @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
        )+
        @vlplot({:rule, clip = true, strokeDash = [2 2], size = 1},
            y = "mean(chance)", color = {value = "black"})
    ));
pl |> save(joinpath(dir, "decode_source_near_switch_by_source_salience.svg"))

# TODO: plot by individual subject, at a few time points
alltimes = plotdf_base.time |> unique |> sort!
refs = [1.75, 3.5]
time_points = alltimes[getindex.(argmin(abs.(alltimes .- refs'), dims = 1), 1)]
plotdf_slice = @_ plotdf_base |>
    @transform(__, salience_label = getindices(meta.salience_label, :sound_index)) |>
    @where(__, :time .∈ Ref(time_points)) |>
    @where(__, (:hittype .== "hit") .& .!(:spatial_source)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
    ) |>
    groupby(__, [:condition, :time, :sid, :switch_timing, :salience_label]) |>
    @combine(__, correct = mean(:correct)) |>
    @where(__, contains.(string.(:switch_timing),"Q2")) |>
    insertcols!(__, :chance => 1/3)

pl = plotdf_slice |> @vlplot(
    spacing = 5,
    facet = {column = {field = :time, type = :nominal}, row = {field = :salience_label}}
) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            color = {:condition, sort = ["global", "spatial", "object", "before"],
                title = "Condition", scale = { range = "#".*hex.(GermanTrack.colors) }}
        ) +
        @vlplot(:point, x = :condition, y = :correct) +
        @vlplot({:rule, clip = true, strokeDash = [2 2], size = 1},
            y = "mean(chance)", color = {value = "black"})
    )
);
pl |> save(joinpath(dir, "decode_source_near_switch_by_source_salience_indsbj.svg"))

# TODO: plot by individual trial/subject at these time points

# Plot by source type
# -----------------------------------------------------------------

plotdf = @_ plotdf_base |>
    groupby(__, [:condition, :time, :lagcut, :sid, :switch_timing, :hittype, :spatial_source, :trained_source_name]) |>
    @combine(__, correct = mean(:correct)) |>
    groupby(__, [:condition, :time, :lagcut, :switch_timing, :hittype, :spatial_source, :trained_source_name]) |>
    combine(__, :correct => boot(alpha = 0.05) => AsTable) |>
    @transform(__, chance = ifelse.(:spatial_source, 0.5, 1/3))

pl = @_ plotdf |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
    ) |>
    @where(__, :hittype .== "hit") |>
	@where(__, :spatial_source) |>
    # @where(__, -1 .< :time .< 2.5) |>
    @vlplot(
        spacing = 5,
        facet = {column = {field = :switch_timing}, row = {field = :trained_source_name}}
        # config = {legend = {disable = true}},
    ) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            color = {:condition, sort = ["global", "spatial", "object", "before"],
                title = "Condition", scale = { range = "#".*hex.(GermanTrack.colors) }}
        ) +
        (@vlplot(
            x = {:time, type = :quantitative, title = "Time from Switch offset (s)"}) +
            @vlplot({:line, strokeJoin = :round},
                y = {:value, title = "P(Correct)", scale = {zero = false}}) +
            @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
        )+
        @vlplot({:rule, clip = true, strokeDash = [2 2], size = 1},
            y = "mean(chance)", color = {value = "black"})
    ));
pl |> save(joinpath(dir, "decode_source_near_switch_by_source_name_class.svg"))

# Switch by source accuracy, across different amonuts of lag
# -----------------------------------------------------------------

pcolors = ColorSchemes.imola[range(0.2,0.8,length=3)]

target_wins(row) = row[Symbol(row.trained_source)] == max(row.male, row.fem1, row.fem2)
switch_end(stim, index) = meta.switch_regions[stim][index][2]

# setup plot data
plotdf = @_ timelines |>
    @where(__, :time .+ switch_offset.(:sound_index, :switch_index) .<
        getindices(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :condition .== "global") |>
    groupby(__, Not([:source, :trained_source, :is_target_source, :score, :lagcut])) |>
    @transform(__, target_source = first(:source[:is_target_source])) |>
    @where(__, :trained_source .== :target_source) |>
    select(__, Not(:is_target_source)) |>
    unstack(__, Not([:source, :score]), :source, :score) |>
    insertcols!(__, :correct => target_wins.(eachrow(__))) |>
    @transform(__, switch_timing = cut(switch_end.(:sound_index, :switch_index), 2)) |>
    groupby(__, [:condition, :time, :lagcut, :sid, :switch_timing, :hittype]) |>
    @combine(__, correct = mean(:correct)) |>
    groupby(__, [:condition, :time, :lagcut, :switch_timing, :hittype]) |>
    combine(__, :correct => boot(alpha = 0.05) => AsTable)

pl = @_ plotdf |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
    ) |>
    # @where(__, -1 .< :time .< 2.5) |>
    @vlplot(
        spacing = 5,
        facet = {column = {field = :switch_timing}, row = {field = :hittype}}
        # config = {legend = {disable = true}},
    ) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            color = {:lagcut, type = :nominal, sort = ["global", "spatial", "object", "before"],
                title = "Lags", scale = { range = "#".*hex.(pcolors) }}
        ) +
        (@vlplot(
            x = {:time, type = :quantitative, title = "Time from Switch offset (s)"}) +
            @vlplot({:line, strokeJoin = :round},
                y = {:value, title = "P(Correct)", scale = {zero = false}}) +
            @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
        )+
        @vlplot({:rule, clip = true, strokeDash = [2 2], size = 1},
            y = {datum = 1/3}, color = {value = "black"})
    ));
pl |> save(joinpath(dir, "decode_source_near_switch_lag_class.svg"))

# source switching rate over time
# -----------------------------------------------------------------

pcolors = GermanTrack.colors

# setup plot data
decode_scores = @_ timelines |>
    @where(__, :source .== :trained_source) |>
    @where(__, :time .+ switch_offset.(:sound_index, :switch_index) .<
        getindices(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :lagcut .== 0) |>
    @where(__, :hittype .== "hit") |>
    @transform(__,
        is_trained_target = trained_speaker.(:sound_index) .== :trained_source,
    ) |>
    groupby(__, [:sid, :trial, :encoding, :lagcut, :time, :condition, :hittype, :switch_index]) |>
    @transform(__, target_source = first(:trained_source[:is_trained_target])) |>
    select(__, Not([:is_target_source, :is_trained_target])) |>
    unstack(__, Not([:source, :trained_source, :score]), :trained_source, :score)

switch_band = 1
timeΔ = mean(diff(unique(timelines.time)))
decode_switches = @_ decode_scores |>
    groupby(__, Not(:encoding)) |>
    combine(__, [:male, :fem1, :fem2] .=> mean, renamecols = false) |>
    groupby(__, [:sid, :trial, :condition]) |>
    @repeatby(__,
        window = range(extrema(parent(__).time)..., step = 0.25),
        band = [0.5, 1, 2, 3]
    ) |>
    @where(__, abs.(:time .- :window) .< (:band./2)) |>
    @combine(__,
        switch_mass = GermanTrack.dominant_mass(Hcat(:male, :fem1, :fem2)),
        switch_length = timeΔ.*GermanTrack.streak_length(Hcat(:male, :fem1, :fem2), 1),
    )

pcolors = GermanTrack.colors
plotdf = @_ decode_switches |>
    groupby(__, [:condition, :window, :sid, :band]) |>
    @combine(__, switch_mass = mean(:switch_mass)) |>
    groupby(__, [:condition, :window, :band]) |>
    combine(__, :switch_mass => boot(alpha = 0.05) => AsTable)

pl = plotdf |>
    @vlplot(facet = {column = {field = :band, title = "Window Width (s)"}}) +
    (@vlplot() +
        (@vlplot(x = {:window, title = "Window Center (s)"},
            color = {:condition, scale = { range = "#".*hex.(pcolors) }}) +
         @vlplot(:line, y = {:value, title = "Prop. Winning Source > Lossing Sources"}) +
         @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) )
    )
pl |> save(joinpath(dir, "decode_switch_rate_near_switch.svg"))

plotdf = @_ decode_switches |>
    groupby(__, [:condition, :window, :sid, :band]) |>
    @combine(__, switch_length = mean(:switch_length)) |>
    groupby(__, [:condition, :window, :band]) |>
    combine(__, :switch_length => boot(alpha = 0.05) => AsTable)

pl = plotdf |>
    @vlplot(facet = {column = {field = :band, title = "Window Width (s)"}}) +
    (@vlplot() +
        (@vlplot(x = {:window, title = "Window Center (s)"}, color = :condition) +
         @vlplot(:line, y = {:value, title = "Switch Duration for Dominant Source (s)"}) +
         @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) )
    )
pl |> save(joinpath(dir, "decode_switch_length_near_switch.svg"))

# Supplement: decoding by source
# =================================================================

prefix = joinpath(processed_datadir("analyses", "decode-timeline-source"), "testing")
GermanTrack.@load_cache prefix timelines
trained_speaker(si) = get(["male", "fem1", "fem2"], Int(meta.speakers[si]), missing)

# two figures:
# - trained sources when they're the target, across the different lags
# - trained sources for target - non-target cases (for each lag at first, but probalby just for the shortest lag)

# Timeline for decoding by source, when the soure is the target
# -----------------------------------------------------------------

pcolors = vcat(ColorSchemes.imola[range(0.2,0.7,length=3)], ColorSchemes.lajolla[0.8])

# setup plot data
plotdf = @_ timelines |>
    @where(__, :time .< getindex(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :is_target_source .& (:source .== :trained_source)) |>
    groupby(__, [:condition, :time, :sid, :trial, :sound_index, :fold, :lagcut]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, [:condition, :time, :lagcut, :sid]) |>
    @combine(__, score = mean(:score)) |>
    # @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :lagcut]) |>
    combine(__,
        :sid => length ∘ unique => :N,
        :score => boot(alpha = sqrt(0.05)) => AsTable,
    ) |> @where(__, :N .>= 24)


laglabels = Dict(
    0 => "3 sec",
    32 => "2 sec",
    64 => "1 sec"
)

target_len_y = -0.075
pl = @_ plotdf |>
    @transform(__, laglabel = getindices(laglabels, :lagcut)) |>
    @vlplot(
        spacing = 5,
        # config = {legend = {disable = true}},
    facet = {
        column = {field = :condition, title = "",
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:laglabel, type = "ordinal",
                title = "Lags", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round},
            # strokeDash = {:test_type, range = [[1,0], [4,1], [2,1]], sort = ["Trained Source", "Other Sources", "Baseline"]},
            y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    ));
pl |> save(joinpath(dir, "decode_by_source_trained_target.svg"))

# Decode by source, difference from target vs non-target trials
# -----------------------------------------------------------------

pcolors = GermanTrack.colors

# setup plot data
plotdf = @_ timelines |>
    @where(__, :time .< getindex(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, (:source .== :trained_source) .& (:lagcut .== 0)) |>
    groupby(__, [:condition, :time, :sid, :trial, :sound_index, :fold, :lagcut, :is_target_source]) |>
    @combine(__, score = mean(:score)) |>
    groupby(__, Not([:trial])) |>
    @combine(__, score = mean(:score)) |>
    @transform(__, source_kind = ifelse.(:is_target_source, "target", "other")) |>
    select(__, Not(:is_target_source)) |>
    unstack(__, Not([:source_kind, :score]), :source_kind, :score) |>
    @transform(__, diff = :target .- :other) |>
    groupby(__, [:condition, :time, :lagcut, :sid]) |>
    @combine(__, diff = mean(:diff)) |>
    groupby(__, [:condition, :time, :lagcut]) |>
    combine(__, :diff => boot(alpha = 0.05) => AsTable) |>
    @transform(__, laglabel = getindices(laglabels, :lagcut))


function inswitches(meta, id, time)
    switches = meta.switch_regions[id]
    @_ any(inswitch(_, time), switches)
end
inswitch((start, stop), time) = start ≤ time ≤ stop

switch_counts = @_ timelines |>
    @where(__, :is_target_source .& (:source .== :trained_source) .& (:lagcut .== 0)) |>
    @transform(__, switch_count = inswitches.(Ref(meta), :sound_index, :time)) |>
    groupby(__, [:condition, :time]) |>
    @combine(__, switch_count = mean(:switch_count))

plotdf = innerjoin(plotdf, switch_counts, on = [:condition, :time])

target_len_y = -0.075
pl = @_ plotdf |>
    @transform(__, laglabel = getindices(laglabels, :lagcut)) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
        resolve = {scale = {y = :independent}},
    facet = {
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            resolve = {scale = {y = "independent"}},
            color = {:condition, type = "ordinal",
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        ( @vlplot() +
            ( @vlplot(
                y = {:value, title = "Target - Non-target", type = :quantitative,
                    scale = {domain = [-0.1, 0.3]}},
                x = {:time, type = :quantitative, title = "Time (s)"}) +
              @vlplot({:line, clip = true, strokeJoin = :round}) +
              @vlplot({:errorband, clip = true},
                  y = {:lower, title = "Target - Non-target"}, y2 = :upper)) +
            @vlplot({:rule, clip = true, strokeDash = [4 4], size = 1}, y = {datum = 0}, color = {value = "black"})
        ) +
        @vlplot({:area, opacity = 0.3},
            y = {:switch_count, title = "P(switch)", scale = {domain = [0, 1]}},
            x = :time,
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "decode_by_source_target_diff.svg"))

# Decode sources by classification accuracy
# -----------------------------------------------------------------

pcolors = GermanTrack.colors

target_wins(row) = row[Symbol(row.target_source)] == max(row.male, row.fem1, row.fem2)

# setup plot data
plotdf_base = @_ timelines |>
    @where(__, :source .== :trained_source) |>
    @where(__, :condition .!= "spatial") |>
    @where(__, :time .< getindex(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :time .< params.train.trial_time_limit) |>
    @where(__, :lagcut .== 0) |>
    @where(__, :hittype .== "hit") |>
    @transform(__,
        is_trained_target = trained_speaker.(:sound_index) .== :trained_source,
    ) |>
    groupby(__, [:sid, :trial, :encoding, :lagcut, :time, :condition, :hittype]) |>
    @transform(__, target_source = first(:trained_source[:is_trained_target])) |>
    select(__, Not([:is_target_source, :is_trained_target])) |>
    unstack(__, Not([:source, :trained_source, :score]), :trained_source, :score) |>
    insertcols!(__, :correct => target_wins.(eachrow(__)))

plotdf = @_ plotdf_base |>
    groupby(__, [:condition, :time, :lagcut, :sid]) |>
    @combine(__, correct = mean(:correct)) |>
    groupby(__, [:condition, :time, :lagcut]) |>
    combine(__, :correct => boot(alpha = 0.05) => AsTable)

function inswitches(meta, id, time)
    switches = meta.switch_regions[id]
    @_ any(inswitch(_, time), switches)
end
inswitch((start, stop), time) = start ≤ time ≤ stop

switch_counts = @_ timelines |>
    @where(__, :is_target_source .& (:source .== :trained_source) .& (:lagcut .== 0)) |>
    @transform(__, switch_count = inswitches.(Ref(meta), :sound_index, :time)) |>
    groupby(__, [:condition, :time]) |>
    @combine(__, switch_count = mean(:switch_count))


plotdf = innerjoin(plotdf, switch_counts, on = [:condition, :time])

target_len_y = -0.075
pl = @_ plotdf |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
        resolve = {scale = {y = :independent}},
    facet = {
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            resolve = {scale = {y = "independent"}},
            color = {:condition, type = "ordinal",
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        ( @vlplot() +
            ( @vlplot(
                y = {:value, title = "P(Correct)", type = :quantitative,
                    scale = {domain = [0.2 ,0.5]}},
                x = {:time, type = :quantitative, title = "Time (s)"}) +
              @vlplot({:line, clip = true, strokeJoin = :round}) +
              @vlplot({:errorband, clip = true},
                  y = {:lower, title = "P(Correct)"}, y2 = :upper)) +
            @vlplot({:rule, clip = true, strokeDash = [4 4], size = 1}, y = {datum = 1/3}, color = {value = "black"})
        ) +
        @vlplot({:area, opacity = 0.3},
            y = {:switch_count, title = "P(switch)", scale = {domain = [0, 1]}},
            x = :time,
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "decode_by_source_class.svg"))

target_len_y = -0.075
pl = @_ plotdf |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
        resolve = {scale = {y = :independent}},
    facet = {
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 64, height = 65,
            resolve = {scale = {y = "independent"}},
            color = {:condition, type = "ordinal",
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        ( @vlplot() +
            ( @vlplot(
                y = {:value, title = "P(Correct)", type = :quantitative,
                    scale = {domain = [0.25 ,0.45]}},
                x = {:time, type = :quantitative, title = "Time (s)"}) +
              @vlplot({:line, clip = true, strokeJoin = :round}) +
              @vlplot({:errorband, clip = true},
                  y = {:lower, title = "P(Correct)"}, y2 = :upper)) +
            @vlplot({:rule, clip = true, strokeDash = [4 4], size = 1}, y = {datum = 1/3}, color = {value = "black"})
        ) +
        @vlplot({:area, opacity = 0.3},
            y = {:switch_count, title = "P(switch)", scale = {domain = [0, 1]}},
            x = :time,
            color = {value = "black"}
        )
    ));
pl |> save(joinpath(dir, "present", "decode_by_source_class.svg"))


# attention switching rate over time
# -----------------------------------------------------------------

pcolors = GermanTrack.colors

# setup plot data
decode_scores = @_ timelines |>
    @where(__, :source .== :trained_source) |>
    @where(__, :time .< getindex(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :time .< params.train.trial_time_limit) |>
    @where(__, :lagcut .== 0) |>
    @where(__, :hittype .== "hit") |>
    @transform(__,
        is_trained_target = trained_speaker.(:sound_index) .== :trained_source,
    ) |>
    groupby(__, [:sid, :trial, :encoding, :lagcut, :time, :condition, :hittype]) |>
    @transform(__, target_source = first(:trained_source[:is_trained_target])) |>
    select(__, Not([:is_target_source, :is_trained_target])) |>
    unstack(__, Not([:source, :trained_source, :score]), :trained_source, :score)

switch_band = 1
timeΔ = mean(diff(unique(timelines.time)))
times = range(extrema(decode_scores.time)..., step = 0.25)
decode_switches = @_ decode_scores |>
    groupby(__, Not(:encoding)) |>
    combine(__, [:male, :fem1, :fem2] .=> mean, renamecols = false) |>
    groupby(__, [:sid, :trial, :condition]) |>
    @repeatby(__,
        window = times,
        band = [0.5, 1, 2, 3]
    ) |>
    @where(__, abs.(:time .- :window) .< (:band./2))

plotdf = @_ decode_switches |>
    @combine(__, switch_mass = GermanTrack.dominant_mass(Hcat(:male, :fem1, :fem2))) |>
    groupby(__, [:condition, :window, :sid, :band]) |>
    @combine(__, switch_mass = mean(:switch_mass)) |>
    groupby(__, [:condition, :window, :band]) |>
    combine(__, :switch_mass => boot(alpha = 0.05) => AsTable)

pl = plotdf |>
    @vlplot(facet = {column = {field = :band, title = "Window Width (s)"}}) +
    (@vlplot() +
        (@vlplot(x = {:window, title = "Window Center (s)"}, color = :condition) +
         @vlplot(:line, y = {:value, title = "Prop. Winning Source > Lossing Sources"}) +
         @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) )
    )
pl |> save(joinpath(dir, "decode_switch_rate.svg"))

plotdf = @_ decode_switches |>
    @combine(__, switch_length = timeΔ.*GermanTrack.streak_length(Hcat(:male, :fem1, :fem2), 1)) |>
    groupby(__, [:condition, :window, :sid, :band]) |>
    @combine(__, switch_length = mean(:switch_length)) |>
    groupby(__, [:condition, :window, :band]) |>
    combine(__, :switch_length => boot(alpha = 0.05) => AsTable)

pl = plotdf |>
    @vlplot(facet = {column = {field = :band, title = "Window Width (s)"}}) +
    (@vlplot() +
        (@vlplot(x = {:window, title = "Window Center (s)"}, color = :condition) +
         @vlplot(:line, y = {:value, title = "Switch Duration for Dominant Source (s)"}) +
         @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) )
    )
pl |> save(joinpath(dir, "decode_switch_length.svg"))

function bin(x, nbins)
    _min, _max = extrema(x)
    step = (_max - _min) / nbins
    min.(nbins-1, floor.(x ./ step)).*step .+ step./2
end

# what are all the extra columns for????
plotdf_base = @_ decode_switches |>
    @where(__, 0.5 .< :window .< 5.5) |>
    @where(__, :band .== 1.0) |>
    combine(__, [:male, :fem1, :fem2] =>
        ((m,f1,f2) -> streak_stats(Hcat(m,f1,f2), 2)) => AsTable) |>
    @transform(__, streak_length_bin = bin(:streak_length.*timeΔ, 12)) |>
    groupby(__, [:condition, :window, :sid, :band, :streak_length_bin]) |>
    @combine(__, bin_time = sum(:count) .* first(:streak_length_bin)) |>
    groupby(__, [:condition, :window, :sid, :band]) |>
    @transform(__, bin_prop = :bin_time ./ sum(:bin_time)) |>
    groupby(__, Not([:window, :bin_prop])) |>
    @combine(__, bin_prop = mean(:bin_prop))

plotdf = @_ plotdf_base |>
    groupby(__, Not([:sid, :bin_prop, :bin_time])) |>
    combine(__, :bin_prop => boot(stat = mean, alpha = 0.05) => AsTable)

pl = plotdf |>
    # @vlplot(facet = {column = {field = :window, type = :nominal}}) +
    (@vlplot() +
        (@vlplot(x = {:streak_length_bin, type = :quantitative, title = "Focus Length (binned)"},
            color = {:condition, type = "ordinal", scale = {range = "#".*hex.(pcolors)}}) +
         @vlplot(:line, y = {:value, title = "Prop. of Time"}) +
         @vlplot({:errorband, ticks = {width = 5, color = "black"}}, y = {:lower, title = ""}, y2 = :upper)));
pl |> save(joinpath(dir, "decode_switch_stats.svg"))

pl = plotdf |>
    # @vlplot(facet = {column = {field = :window, type = :nominal}}) +
    (@vlplot(width = 70, height = 60) +
        (@vlplot(x = {:streak_length_bin, type = :quantitative, title = "Focus Length (binned)"},
            color = {:condition, type = "ordinal", scale = {range = "#".*hex.(pcolors)}}) +
         @vlplot(:line, y = {:value, title = "Prop. of Time"}) +
         @vlplot({:errorband, ticks = {width = 5, color = "black"}}, y = {:lower, title = ""}, y2 = :upper)));
pl |> save(joinpath(dir, "present", "decode_switch_stats.svg"))

function weighted_mean_above(x, count, thresh)
    ixs = findall(>(thresh), x)
    if isempty(ixs)
        0.0
    else
        sum(x[ixs].^2 .* count[ixs]) / sum(count[ixs] .* x[ixs])
    end
end

plotdf_base = @_ decode_switches |>
    # @where(__, :window .∈ Ref(times[1:5:end])) |>
    @where(__, 0.5 .< :window .< 5.5) |>
    @where(__, :band .== 1.0) |>
    combine(__, [:male, :fem1, :fem2] =>
        ((m,f1,f2) -> streak_stats(Hcat(m,f1,f2), 2)) => AsTable) |>
    groupby(__, [:condition, :window, :sid, :band]) |>
    # computes the mean time for a given time point
    # (since the longer streaks encompass more time points, they
    # are count as more data points)
    @combine(__, mean_length = weighted_mean_above(:streak_length, :count,
        quantile(mapreduce(fill,vcat,:streak_length,:count), 0.95))) |>
    groupby(__, [:condition, :sid, :band]) |>
    @combine(__, mean_length = timeΔ.*mean(:mean_length))

plotdf = @_ plotdf_base |>
    groupby(__, Not([:mean_length, :sid])) |>
    combine(__, :mean_length => boot(stat = mean, alpha = 0.05) => AsTable)

pl = plotdf |>
    # @vlplot(facet = {column = {field = :window, type = :nominal}}) +
    (@vlplot(config = {legend = {disable = true}}) +
        (@vlplot(x = {:condition, type = :nominal, title = "Condition"},
            color = {:condition, type = "ordinal", scale = {range = "#".*hex.(pcolors)}}) +
         @vlplot({:point, filled = true}, y = {:value, title = "Prop. of Time with > 95th quantile focus length", scale = {zero = false}}) +
         @vlplot({:errorbar, ticks = {width = 5, color = "black"}}, y = {:lower, title = ""}, y2 = :upper)));
pl |> save(joinpath(dir, "decode_switch_cleaned_mean.svg"))

pl = plotdf |>
    # @vlplot(facet = {column = {field = :window, type = :nominal}}) +
    (@vlplot(width = 50, height = 50, config = {legend = {disable = true}}) +
        (@vlplot(x = {:condition, type = :nominal, axis = {title = "", labelAngle = -45}},
            color = {:condition, type = "ordinal", scale = {range = "#".*hex.(pcolors)}}) +
         @vlplot({:point, filled = true}, y = {:value, title = "Prop. of Time", scale = {zero = false}}) +
         @vlplot({:errorbar, ticks = {width = 5, color = "black"}}, y = {:lower, title = ""}, y2 = :upper)));
pl |> save(joinpath(dir, "present", "decode_switch_cleaned_mean.svg"))


# Decoding by sources across different target times
# -----------------------------------------------------------------

# setup plot data
plotdf = @_ timelines |>
    @where(__, :time .< getindex(meta.trial_lengths, :sound_index) .- 1) |>
    @where(__, :time .< params.train.trial_time_limit) |>
    @where(__, :lagcut .== 0) |>
    @transform(__, target_time_label = getindices(meta.target_time_label, :sound_index)) |>
    select(__, Not(:is_target_source)) |>
    unstack(__, Not([:source, :score]), :source, :score) |>
    insertcols!(__, :correct => target_wins.(eachrow(__))) |>
    groupby(__, [:condition, :time, :lagcut, :sid, :target_time_label]) |>
    @combine(__, correct = mean(:correct)) |>
    groupby(__, [:condition, :time, :lagcut, :target_time_label]) |>
    combine(__, :correct => boot(alpha = 0.05) => AsTable)


switch_counts = @_ timelines |>
    @where(__, :is_target_source .& (:source .== :trained_source) .& (:lagcut .== 0)) |>
    @transform(__, switch_count = inswitches.(Ref(meta), :sound_index, :time)) |>
    groupby(__, [:condition, :time]) |>
    @combine(__, switch_count = mean(:switch_count))

plotdf = innerjoin(plotdf, switch_counts, on = [:condition, :time])

target_len_y = -0.075
pl = @_ plotdf |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
        resolve = {scale = {y = :independent}},
    facet = {
        row = {field = :target_time_label, title = "Target Timing"},
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 64, height = 65,
            resolve = {scale = {y = "independent"}},
            color = {:condition, type = "ordinal",
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        ( @vlplot() +
            ( @vlplot(
                y = {:value, title = "P(Correct)", type = :quantitative,
                    scale = {domain = [0.2 ,0.5]}},
                x = {:time, type = :quantitative, title = "Time (s)"}) +
              @vlplot({:line, clip = true, strokeJoin = :round}) +
              @vlplot({:errorband, clip = true},
                  y = {:lower, title = "P(Correct)"}, y2 = :upper)) +
            @vlplot({:rule, clip = true, strokeDash = [4 4], size = 1}, y = {datum = 1/3}, color = {value = "black"})
        )
    ));
pl |> save(joinpath(dir, "decode_by_source_class_target_time.svg"))

# Supplement 0: decoding broken down by early/late near/far
# and their interactions
# =================================================================

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12].+1)]]
pcolors[1] = GermanTrack.grayify(pcolors[1])

# setup plot data
plotdf = @_ timelines |>
    @where(__, :train_type .!= "atmiss-target") |>
    groupby(__, Not([:encoding, :source])) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        target_time_label = getindex.(Ref(meta.target_time_label), :sound_index),
        target_switch_label = getindex.(Ref(meta.target_switch_label), :sound_index)
    )

target_len_y = 0.15
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid, :target_switch_label]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type, :target_switch_label]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {field = :target_switch_label, sort = ["near", "far"]},
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.1, 0.2]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "decode_nearfar_condition.svg"))

target_len_y = 0.2
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid, :target_time_label]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type, :target_time_label]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {field = :target_time_label, sort = ["early", "late"]},
        column = {field = :condition, title = "",
            sort = ["global", "spatial", "object"],
            header = {
                title = "",
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
                labelFontWeight = "bold",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.15, 0.25]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "decode_earlylate_condition.svg"))

target_len_y = 0.2
pl = @_ plotdf |>
    @where(__, :condition .== "object") |>
    groupby(__, [:condition, :time, :train_type, :sid, :target_switch_label, :target_time_label]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type, :target_switch_label, :target_time_label]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {field = :target_time_label, sort = ["early", "late"]},
        column = {field = :target_switch_label, sort = ["near", "far"]}
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors[[1,4]]) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.15, 0.25]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :center},
            x = {datum = 0.5}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(dir, "decode_earlylate_nearfar.svg"))

# Supplement 1: decoding generalization across conditions
# =================================================================

# setup plot data
plotdf = @_ timelines |>
    groupby(__, [:condition, :train_condition, :time, :sid, :train_type, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

labels = OrderedDict(
    "athit-target" => "Target Source",
    "athit-other" => "Other Sources",
)
tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

supdir = mkpath(joinpath(dir, "supplement"))

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12].+1)]]
pcolors[1] = GermanTrack.grayify(pcolors[1])

target_len_y = 0.135
pl = @_ plotdf |>
    groupby(__, [:condition, :train_condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :train_condition, :time, :train_type]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        row = {
            field = :train_condition, title = "Decoder Training",
            sort = ["global", "spatial", "object"],
                header = {labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"},
        },
        column = {field = :condition, title = "Decoder Testing",
            sort = ["global", "spatial", "object"],
            header = {
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["other", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Target Source'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Other Sources'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.05, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(supdir, "decode_condition_generalize.svg"))

# Supplement 2: decoding by miss
# =================================================================

# setup plot data
plotdf = @_ timelines |>
    @where(__, :train_type .!= "athit-other") |>
    @where(__, :condition .== :train_condition) |>
    groupby(__, [:condition, :time, :sid, :train_type, :trial, :sound_index, :fold]) |>
    @combine(__, score = mean(:score))

labels = OrderedDict(
    "athit-target" => "Hit",
    "atmiss-target" => "Miss",
)
tolabel(x) = labels[x]
tcolors = ColorSchemes.imola[[0.3, 0.8]]

supdir = mkpath(joinpath(dir, "supplement"))

steps = range(0.1, 0.9, length = 15)
steps = vcat(steps[1] - step(steps), steps)
pcolors = ColorSchemes.batlow[steps[vcat(1,[1,7,12].+1)]]
pcolors[1] = GermanTrack.grayify(pcolors[1])

target_len_y = 0.135
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Miss" ? "miss" : cond) => :train_label) |>
    @vlplot(
        spacing = 5,
        config = {legend = {disable = true}},
    facet = {
        column = {field = :condition, title = "Decoder Testing",
            sort = ["global", "spatial", "object"],
            header = {
                labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)",
            }
        }
    }) + (@vlplot() +
    (
        @vlplot(
            width = 128, height = 130,
            x = {:time, type = :quantitative, title = "Time (s)"},
            color = {:train_label, sort = ["miss", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Decoding Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3, dy = -9},
            transform = [
                {filter = "datum.time > 1.25 && datum.time < 1.5 && datum.train_type == 'Hit'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:upper, aggregate = :mean},
            text = :train_type_lbl
            # color = {value = "black"}
        ) +
        @vlplot({:text, align = "right", baseline = "top", dx = 3, dy = 3},
            transform = [
                {filter = "datum.time > 1.3 && datum.time < 1.4 && datum.train_type == 'Miss'"},
                {calculate = "split(datum.train_type,' ')", as = "train_type_lbl"}
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:lower, aggregate = :mean, scale = {domain = [-0.05, 0.15]}},
            text = :train_type_lbl
            # color = {value = "black"}
        )
    ) +
    # Target annotation
    (
        @vlplot(data = {values = [{}]}) +
        @vlplot(mark = {:text, size = 11, baseline = "bottom", align = :left},
            x = {datum = 0}, y = {datum = target_len_y},
            text = {value = ["Target", "Extent"]},
            color = {value = "black"}
        ) +
        @vlplot({:rect, opacity = 0.25},
            x = {datum = 0}, x2 = {datum = 1},
            color = {value = "gray"},
        )
    ));
pl |> save(joinpath(supdir, "decode_condition_miss.svg"))


# Supplement 3: decoding fall outside CI of baseline
# =================================================================



# Presentation plots
# =================================================================

mkpath(joinpath(dir, "present"))
thresh = 0.1

pldata = @_ timelines |>
    groupby(__, Not([:encoding, :source])) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type),
    ) |>
    @where(__, (-2 .< :time .< -1) .| (0.5 .< :time .< 1.5)) |>
    @transform(__,
        region_label = ifelse.(:time .< 0, "before-target", "near-target"),
    ) |>
    unstack(__, Not([:train_type, :score]), :train_type, :score, allowduplicates = true) |>
    groupby(__, [:condition, :region_label, :sid]) |>
    @combine(__,
        favor_target = mean((:var"Target Source" .- :var"Other Sources") .> thresh),
        favor_other = mean((:var"Target Source" .- :var"Other Sources") .< -thresh)
    ) |>
    stack(__, r"favor", variable_name = :favor) |>
    @transform(__, favor = replace.(:favor, r"favor_(.*)" => s"\1")) |>
    groupby(__, [:condition, :region_label, :favor]) |>
    combine(__, :value => boot(alpha = sqrt(0.05)) => AsTable)

barwidth = 18
pl = @_ pldata |>
    @transform(__,
        region_label = replace(:region_label,
            "near-target" => "Near (0.5 s, 1.5 s)",
            "before-target" => "Before (-2 s, -1 s)"
        )
    ) |>
    @vlplot(spacing = 5,
        config = {
            legend = {disable = true},
            bar = {discreteBandSize = barwidth},
        },
        facet = {
            row = {field = :region_label, title = "", sort = ["Near (0.5 s, 1.5 s)", "Before (-2 s, -1 s)"]},
            column = {field = :condition, title = "", sort = ["global", "spatial", "object"],
                header = {labelFontWeight = "bold",
                    labelExpr = "upper(slice(datum.label,0,1)) + slice(datum.label,1)"}
            }
        }) +
    (@vlplot(x = {:favor, axis = {title = "Source", labelAngle = -45}},
        height = 90, width = 60) +
    @vlplot({:bar}, y = {:value, title = "Time Favoring (s)"}, color = :condition) +
    @vlplot(:rule, y = :lower, y2 = :upper));
pl |> save(joinpath(dir, "present", "fig2d.svg"))

pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__,
        time = :time .+ params.test.winlen_s,
        train_type = tolabel.(:train_type)
    ) |>
    @where(__, -1 .< :time .< 2.5) |>
    groupby(__, [:condition, :time, :train_type]) |>
    combine(__, :score => boot => AsTable) |>
    transform(__, [:condition, :train_type] =>
        ByRow((cond, type) -> type == "Other Sources" ? "other" : cond) => :train_label) |>
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
            color = {:train_label, sort = ["other", "global", "spatial", "object"],
                title = "Source", scale = { range = "#".*hex.(pcolors) }}
        ) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = "Mean Correlation"}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper)
    );
pl |> save(joinpath(dir, "decode_timeline.svg"))


tcolors = ColorSchemes.lajolla[range(0.3,0.9, length = 4)]
pl = @_ plotdf |>
    groupby(__, [:condition, :time, :train_type, :sid]) |>
    @combine(__, score = mean(:score)) |>
    @transform(__, time = :time .+ params.test.winlen_s,
        condition = uppercasefirst.(:condition)) |>
    @where(__, -1 .< :time .< 2.5) |>
    unstack(__, [:condition, :time, :sid], :train_type, :score) |>
    @transform(__, scorediff = :var"athit-target" .- :var"athit-other") |> #"
    groupby(__, [:condition, :time]) |>
    combine(__, :scorediff => boot(alpha = sqrt(0.05)) => AsTable) |>
    @vlplot(
        config = {legend = {disable = true}},
        height = 90, width = 100,
    ) +
    (
        @vlplot(x = {:time, type = :quantitative, title = "Time"}, color = {:condition, scale = {range = "#".*hex.(colors)}}) +
        @vlplot({:line, strokeJoin = :round}, y = {:value, title = ["Target - Other", "Correlation"]}) +
        @vlplot(:errorband, y = {:lower, title = ""}, y2 = :upper) +
        @vlplot({:text, align = "left", dx = 3},
            transform = [
                {filter = "datum.time > 2.25 && datum.time < 2.5"},
            ],
            x = {:time, aggregate = :max, title = ""},
            y = {:value, aggregate = :mean},
            text = {field = :condition}
            # color = {value = "black"}
        )
    );
pl |> save(joinpath(dir, "decode_timeline_diff.svg"))
