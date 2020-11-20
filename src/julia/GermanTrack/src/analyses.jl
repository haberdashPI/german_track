export select_windows, shrinktowards, ishit, lowerboot, boot, upperboot, pick_λ_winlen,
    addfold!, splayby, mapgroups, filteringmap

"""
    lowerboot(x; alpha = 0.05, n = 10_000)

Lower bound of the bootstrapped confidnece interval.
"""
lowerboot(x; alpha = 0.05, n = 10_000) =
    confint(bootstrap(mean, x, BasicSampling(n)), BasicConfInt(1 - alpha))[1][2]
"""
    boot(x; alpha = 0.05, n = 10_000)

Bootstrapped estimate
"""
boot(x; alpha = 0.05, n = 10_000) =
    confint(bootstrap(mean, x, BasicSampling(n)), BasicConfInt(1 - alpha))[1][1]

"""
    uppperboot(x; alpha = 0.05, n = 10_000)

Upper bound of the bootstrapped confidnece interval.
"""
upperboot(x; alpha = 0.05, n = 10_000) =
    confint(bootstrap(mean, x, BasicSampling(n)), BasicConfInt(1 - alpha))[1][3]


"""
    wmean(vals, weights, [default = one(eltype(vals))/2])

Compute a weighted mean, treating missing values as `default`.
"""
wmean(x, w, default = one(eltype(x))/2) =
    iszero(sum(w)) ? default : mean(coalesce.(x, default), weights(w))

"""
    shrinktowards([x],mu;by=0.01)

Reduce the value of `x` towards `mu` by (the presumably small value) `by*(x-mu)`.
You can exclude the `x` argument to curry the function.
"""
shrinktowards(mu;by=0.01) = x -> shrinktowards(x,mu,by=by)
shrinktowards(x,mu;by=0.01) = (1-by)*(x-mu) + mu

"""
    spread([x,]scale,npoints,[indices=Colon()])

Create a spread of `npoints` values placed evenly along the Normal distribution
with a standard deviation of scale/2. Leaving out `x` returns curried function.
To select only a subset of points use the `indices` keyword argument.
"""
spread(scale,npoints;indices=Colon()) = x -> spread(x,scale,npoints,indices=indices)
spread(x,scale,npoints;indices=Colon()) =
    quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints)[indices])

"""
    select_windows(conditions, subjects)

Select windows of eeg data defined by a bounding function.

## Arugments

- conditions: a dictionary where each key is a named tuple describing the
specific condition windows will be associated with and each entry is a
function. When applied to the rows of the event data this function will
return the range of times in seconds (as a tuple) to be included or
no_indices (if no data applies for the given case)

- subjects: a dictionary containing the file => data pairs, each data entry
should contain a named tuple of eeg (for EEGData), event (for data frame of
events) and sid (a String) entries.

"""
function select_windows(conditions, subjects)
    reduce(vcat, select_windows_helper.(collect(conditions), Ref(subjects))) |>
        categorical!
end

function select_windows_helper((condition, boundfn), subjects)
    bounds = Dict((file, i) => bounds
        for file in keys(subjects)
        for (i, bounds) in enumerate(boundfn.(eachrow(subjects[file].events))))
    indices = @_ filter(!isempty(bounds[_]), keys(bounds)) |> collect |> sort!

    if !isempty(indices)
        mapreduce(vcat, indices) do (file, i)
            eeg, events = subjects[file]
            start = bounds[(file, i)][1]
            ixs = bound_indices(bounds[(file, i)], 256, size(eeg[i], 2))

            # power in relevant frequency bins across all channels and times
            DataFrame(
                sid = sidfor(file),
                trial = i;
                condition...,
                window = [view(eeg[i], :, ixs)]
            )
        end
    else
        DataFrame()
    end
end


"""
    ishit(row; kwds...)

Correctly interprets a given row of the data as a hit, correct rejection, false positive
or miss. Since the directions are different for each condition, how we interpret a an
indication of a detected target depends on the condition.
"""
function ishit(row; mark_false_targets = false, kwds...)
    vals = merge(row,kwds)
    if vals.target_present
        if vals.condition == "global"
            vals.reported_target ? "hit" : "miss"
        elseif vals.condition == "object"
            vals.target_source == "male" ?
                (vals.reported_target ? "hit" : "miss") :
                (!vals.reported_target ? "reject" :
                    (mark_false_targets ? "falsep-target" : "falsep"))
        else
            @assert vals.condition == "spatial"
            vals.direction == "right" ?
                (vals.reported_target ? "hit" : "miss") :
                (!vals.reported_target ? "reject" :
                    (mark_false_targets ? "falsep-target" : "falsep"))
        end
    else
        vals.reported_target ? "reject" : "falsep"
    end
end

myfirst(x) = first(x)
myfirst(x::Symbol) = x

"""
    pick_λ_winlen(df, n_folds, factors, maximize_across; fold_col = fold_col,
        λ_col = :λ, winlen_col = :winlen, smoothing, slope_thresh, flat_thresh, dir,
        grand_mean_plot = "grandmean", lambda_plot = "lambdas")

Pick the best λ and window length for classification. The classification accuracy is assumed
to be encoded in a `:correct` column (and optionally a `:weight`) column. Any additional
factors in the anlaysis should be included in `factors`. The folds can be added to a data
set using [`addfold!`](#).

To pick λ, the analysis looks at the maximum performance for each value of `maximize_across`
factors and then averages these performance values. The λ is selected assuming there is a
peak in the derivative (`slope_thresh`) that then levels off (`flat_thresh`) as we move from
larger to smaller λ values (a reverse sigmoid shape), and it tries to find the point where
performance levels off. The derivative is computed from a smoothed version of the
performance, where `smoothing` ranges from 0 to 1.

To pick winlen, the analysis uses the selected λ for the given fold, and picks the best
performing window length across all `maximize_acorss` factors. This selection uses the same
subjects that were seen during λ selection.

The returned map of λ and winlen is assigned to the left out subjects of a fold, so the
resulting λ and winlen have been selected by a process that never employes these subjects.

You can view the generated plots to verify that the assumptions of the analysis are
reasonable and a sensible λ value for each fold is selected.

# Arguments
- `df` The data to analyze

"""
function pick_λ_winlen(df, factors, maximize_across; λ_col = :λ, fold_col = :fold,
    winlen_col = :winlen, sid_col = :sid, smoothing, slope_thresh, flat_thresh,
    dir, windows_plot = "windows", grand_mean_plot = "grandmean", lambda_plot = "lambdas")

    means = @_ df |>
        groupby(__, vcat(factors, [λ_col, fold_col, :winlen])) |>
        combine(__,
            :nzcoef => mean => :nzcoef,
            [:correct, :weight] => GermanTrack.wmean => :mean)

    bestmeans = @_ means |>
        groupby(__, vcat(maximize_across, [sid_col, λ_col, fold_col])) |>
        combine(__, :nzcoef => mean => :nzcoef,
                    :mean => maximum => :mean,
                    :mean => logit ∘ shrinktowards(0.5, by = 0.01) ∘ maximum => :logitmean)

    logitmeandiff = @_ filter(_.λ == 1.0, bestmeans) |>
        deletecols!(__, [λ_col, :nzcoef, :mean]) |>
        rename!(__, :logitmean => :logitnullmean) |>
        innerjoin(__, bestmeans, on = vcat(maximize_across, sid_col, fold_col)) |>
        transform!(__, [:logitmean,:logitnullmean] => (-) => :logitmeandiff)

    diff0(x) = vcat(0,diff(x))
    filtfn(x) = filtfilt(digitalfilter(Lowpass(1 - smoothing), Butterworth(5)), x)
    grandlogitmeandiff = @_ logitmeandiff |>
        groupby(__, λ_col) |>
        filteringmap(__, folder=foldl,
            :train_fold => map(fold -> fold => (sdf -> sdf.fold != fold), unique(df.fold)),
            (sdf, fold) -> DataFrame(logitmeandiff = mean(sdf.logitmeandiff))) |>
        sort!(__, [λ_col]) |>
        groupby(__, [:train_fold]) |>
        transform!(__, :logitmeandiff => (x -> abs.(diff0(filtfn(x)))) => :logitmeandiff)

    pl = grandlogitmeandiff |> @vlplot() +
    @vlplot(:line,
        config = {},
        color = {:train_fold, type = :nominal,
            legend = {orient = :none, legendX = 175, legendY = 0.5}},
        x     = {λ_col, scale = {type = :log, domain = [0.01, 0.35]},
        title = "Regularization Parameter (λ)"},
        y     = {:logitmeandiff, aggregate = :mean, type = :quantitative,
                title = "Model - Null Model Accuracy (logit scale)"}) |>
    save(joinpath(dir, string(grand_mean_plot,".svg")))

    function pickλ(df)
        peaks = @_ maxima(df.logitmeandiff) |>
            filter(df.logitmeandiff[_] > slope_thresh, __)
        first_near_zero = @_ findlast(_ < flat_thresh, df[1:peaks[end],:logitmeandiff])
        df[(1:peaks[end])[first_near_zero],[λ_col]]
    end
    λs = @_ grandlogitmeandiff |> groupby(__,:train_fold) |> combine(pickλ,__)

    λs[!,:fold_text] .= string.("Fold: ",λs.train_fold)
    λs[!,:yoff] = [0.1,0.15]
    λ_map = Dict(row.train_fold => row.λ for row in eachrow(λs))

    windowmeans = @_ means |>
        groupby(__, vcat(maximize_across, [λ_col, fold_col, winlen_col])) |>
        combine(__, :mean => logit ∘ shrinktowards(0.5, by = 0.01) ∘ mean => :logitmean)

    winmeandiff = @_ windowmeans |>
        filter(_.λ == 1.0, __) |>
        deletecols!(__, [λ_col]) |>
        rename!(__, :logitmean => :logitnullmean) |>
        innerjoin(__, filter(x -> x.λ ∈ values(λ_map), windowmeans),
            on = vcat(maximize_across, fold_col, winlen_col)) |>
        transform!(__, [:logitmean, :logitnullmean] => (-) => :logitmeandiff)

    @_ means |>
        groupby(__, vcat(maximize_across, [λ_col, fold_col, winlen_col, :winstart])) |>
        combine(__, :mean => logit ∘ shrinktowards(0.5, by = 0.01) ∘ mean => :logitmean) |>
        filter(_.λ ∈ values(λ_map), __) |>
    @vlplot(:line,
        x = :winlen,
        y = {:logitmean, aggregate = :mean, type = :quantitative},
        column = {:fold, type = :nominal},
        color = :winstart
    ) |> save(joinpath(dir, string(windows_plot, ".svg")))

    bestlens = @_ winmeandiff |>
        groupby(__, [winlen_col]) |>
        filteringmap(__, folder=foldl,
            :train_fold => map(fold -> fold => (sdf -> sdf.fold != fold), unique(df.fold)),
            function(sdf, fold)
                sdf = filter(x -> x.λ == λ_map[fold], sdf)
                DataFrame(logitmeandiff = maximum(sdf.logitmeandiff))
            end) |>
        groupby(__, :train_fold) |>
        combine(__, [:winlen, :logitmeandiff] =>
            ((len, diff) -> len[argmax(diff)]) => :bestlen)

    winlen_map = Dict(row.train_fold => row.bestlen for row in eachrow(bestlens))

    @vlplot() +
    vcat(
        logitmeandiff |> @vlplot(
        :line, width = 750, height = 100,
            color = {field = myfirst(maximize_across), type = :nominal},
            x     = {λ_col, scale = {type = :log, domain = [0.01, 0.35]},
                    title = "Regularization Parameter (λ)"},
            y     = {:nzcoef, aggregate = :max, type = :quantitative,
                    title = "# of non-zero coefficients (max)"}
        ),
        (
            @_ bestmeans |> DataFrames.transform(__, :mean => ByRow(x -> 100x) => :mean) |>
            @vlplot(
                width = 750, height = 200,
                x = {λ_col, scale = {type = :log}},
                color = {field = myfirst(maximize_across), type = :nominal},
            ) +
            @vlplot(
                :line,
                y = {:mean, aggregate = :mean, type = :quantitative,
                    title = "% Correct", scale = {domain = [50, 100]}},
            ) +
            @vlplot(
                :errorband,
                y = {:mean, aggregate = :ci, type = :quantitative}
            )
        ),
        (
            @vlplot() +
            (
                logitmeandiff |> @vlplot(
                    width = 750, height = 200,
                    x     = {λ_col, scale = {type = :log},
                            title = "Regularization Parameter (λ)"},
                    color = {field = myfirst(maximize_across), type = :nominal}) +
                @vlplot(:errorband,
                    y = {:logitmeandiff, aggregate = :ci,   type = :quantitative,
                        title = "Model - Null Model Accuracy (logit scale)"}) +
                @vlplot(:line,
                    y = {:logitmeandiff, aggregate = :mean, type = :quantitative})
            ) +
            (
                @vlplot(data = {values = [{}]}, encoding = {y = {datum = 0}}) +
                @vlplot(mark = {type = :rule, strokeDash = [2, 2], size = 2})
            ) +
            (
                @vlplot(data = λs) +
                @vlplot({:rule, strokeDash = [4, 4], size = 3}, x = λ_col,
                    color = {value = "green"}) +
                @vlplot({:text, align = :left, dy = -8, size =  12, angle = 90},
                    text = :fold_text, x = λ_col, y = :yoff)
            )
        )
    ) |> save(joinpath(dir, string(lambda_plot,".svg")))

    return λ_map, winlen_map
end

"""
    mapgroups(df, groups, fn;folder = foldxt, desc = "Progress")

Apply `fn` to each group in a grouped data frame, in parallel (by default), `append!!`ing
the returned values together. Set `folder = foldl` if you want to run the process in
serial.

Since this assumes a long running process, it creates a progress bar. You can change
the description for the progress bar using `desc`.

"""
function mapgroups(df, vars, fn, ;folder = foldxt, desc = "Progress")
    groups = groupby(df, vars)
    progress = Progress(length(groups), desc = desc)
    function fn_((key,sdf))
        result = fn(sdf)
        if !isempty(result)
            result[!, keys(key)] .= permutedims(collect(values(key)))
        end
        next!(progress)

        result
    end

    folder(append!!, Map(fn_), collect(pairs(groups)))
end

"""
    addfold!(df, n, col; rng = Random.GLOBAL_RNG)

Insert a new column in dataframe `df` for the fold, and a column for the train_folds. There
are `n` folds. The fold of a row is determined by the identity of the column `col`.
"""
function addfold!(df, n, col; rng = Random.GLOBAL_RNG)
    train,test = folds(n, unique(df[:,col]), rng = rng)
    train,test = Set.(train), Set.(test)
    df[!, :fold] = map(colval -> findfirst(fold -> colval ∈ fold, test), df[:,col])
    df
end

"""
    filteringmap(df,filtering1 => (value1 => filterfn1, etc...), filtering2 => etc..., fn,
        folder = foldxt, desc = "Progres...")

Repeatedly map a function `fn` over a data frame or a grouped data frame, applying the
function for each group and each set of filtersings. These filterings behave like the groups
of a grouped data frame, but they can include rows are not mutually exlusive to one another.
The `filtering` arguments specify the overlapping groups: group N's name is the value of
`mapN`, and the values of the group variable are `valueK`; group `valueK` contains all rows
that match the filtering function `filterfnK`

"""
function filteringmap(df, filterings_fn...; folder = foldxt, desc = "Progress...")
    fn = filterings_fn[end]
    filterings = filterings_fn[1:(end-1)]

    flattened = @_ map(f -> collect(map(pair -> (f[1], pair...), f[2])), filterings) |>
        Iterators.product(__...) |> collect
    groupings = filtermap_groupings(df, flattened)
    progress = Progress(length(groupings), desc = desc)

    function filtermap(((key, group), filterings))
        filtered = group
        for (name, val, filterfn) in filterings
            filtered = filter(filterfn,filtered)
        end

        local result
        if !isempty(filtered)
            result = fn(filtered, getindex.(filterings, 2)...)

            if !isempty(result)
                for (name, val, filterfn) in filterings
                    result[!, name] .= val
                end
                if !isempty(key)
                    for (k,v) in pairs(key)
                        result[!, k] .= v
                    end
                end
            end
        else
            result = Empty(DataFrame)
        end
        next!(progress)

        result
    end

    folder(append!!, Map(filtermap), collect(groupings),
        init = Empty(DataFrame))
end

filtermap_groupings(df::GroupedDataFrame, flattened) =
    Iterators.product(pairs(df), flattened) |> collect

struct EmptyKey; end
Base.isempty(x::EmptyKey) = true
filtermap_groupings(df::DataFrame, flattened) =
    Iterators.product([(EmptyKey(), df)], flattened) |> collect

macro cache_results(file, args...)

    body = args[end]
    symbols = args[1:end-1]
    if !@_ all(_ isa Symbol, symbols)
        error("Expected variable names and a code body")
    end

    # check for all symbols before running the code
    # (avoids getting the error after running some long-running piece of code)
    found_symbols = Set{Symbol}()
    MacroTools.postwalk(body) do expr
        if expr isa Symbol && expr ∈ symbols
            push!(found_symbols, expr)
        end
        expr
    end
    missing_index = @_ findfirst(_ ∉ found_symbols, symbols)

    if !isnothing(missing_index)
        error("Could not find symbol `$(symbols[missing_index])` in cache body, check spelling.")
    end

    quote
        begin
            function run(ignore)
                $(esc(body))
                Dict($((map(x -> :($(QuoteNode(x)) => jsonout($(esc(x)))), symbols))...))
            end
            results = if isfile($(esc(file)))
                jsonin(open(io -> JSON3.read(io), $(esc(file))))
            else
                let (dir, prefix, suffix) = cache_results_parser($(esc(file)))
                    produce_or_load(dir, (;), run, prefix = prefix, suffix = suffix)[1] |>
                        jsonin
                end
            end

            $((map(x -> :($(esc(x)) = results[$(QuoteNode(x))]), symbols))...)

            nothing
        end
    end
end

macro load_cache(file, args...)
    symbols = args[1:end]
    if !@_ all(_ isa Symbol, symbols)
        error("Expected variable names")
    end

    quote
        begin
            results = if isfile($(esc(file)))
                jsonin(open(io -> JSON3.read(io), $(esc(file))))
            end

            $((map(x -> :($(esc(x)) = results[$(QuoteNode(x))]), symbols))...)

            nothing
        end
    end
end

function jsonout(x::DataFrame)
    Dict(:isdataframe => true, :columns => JSONTables.ObjectTable(Tables.columns(x)))
end
jsonout(x) = x

jsonin(x) = x
function jsonin(data::AbstractDict)
    if haskey(data, :isdataframe) && data[:isdataframe]
        if data[:columns] isa JSONTables.ObjectTable
            data[:columns].x
        else
            jsontable(data[:columns]) |> DataFrame
        end
    else
        Dict(cleanup_key(k) => jsonin(v) for (k,v) in pairs(data))
    end
end
function cleanup_key(x::Symbol)
    if occursin(r"^[0-9-]+$", string(x))
        parse(Int, string(x))
    elseif occursin(r"^[0-9.-]+$", string(x))
        parse(Float64, string(x))
    else
        x
    end
end
cleanup_key(x) = x

function cache_results_parser(filename)
    @show filename
    parts = splitpath(filename)
    dir = joinpath(parts[1:(end-1)]...)
    prefixmatch = match(r"^(.+)\.([a-z]+)$", parts[end])
    if isnothing(prefixmatch)
        error("Could not find filetype in file name.")
    end
    prefix = prefixmatch[1]
    suffix = prefixmatch[2]

    string.((dir, prefix, suffix))
end
