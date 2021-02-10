export select_windows, shrinktowards, ishit, lowerboot, boot, upperboot,
    addfold!, splayby, mapgroups, filteringmap, compute_powerbin_features, cross_folds,
    shrink, wsem

"""
    lowerboot(x; alpha = 0.05, n = 10_000)

Lower bound of the bootstrapped confidnece interval.
"""
lowerboot(x; stat = mean, alpha = 0.05, n = 10_000) =
    confint(bootstrap(stat, x, BasicSampling(n)), BasicConfInt(1 - alpha))[1][2]
"""
    boot(x; alpha = 0.05, n = 10_000)

Bootstrapped estimate
"""
boot(x; stat = mean, alpha = 0.05, n = 10_000) =
    confint(bootstrap(stat, x, BasicSampling(n)), BasicConfInt(1 - alpha))[1][1]

"""
    uppperboot(x; alpha = 0.05, n = 10_000)

Upper bound of the bootstrapped confidnece interval.
"""
upperboot(x; stat = mean, alpha = 0.05, n = 10_000) =
    confint(bootstrap(stat, x, BasicSampling(n)), BasicConfInt(1 - alpha))[1][3]


"""
    wmean(vals, weights, [default = one(eltype(vals))/2])

Compute a weighted mean, treating missing values as `default`.
"""
wmean(x, w, default = one(eltype(x))/2) =
    iszero(sum(w)) ? default : mean(coalesce.(x, default), weights(w))

# CHEAP ESTIMATE: not perfect; there are better formulas, but this simple approach fine for
# my purposes
function wsem(x, w, default = one(eltype(x))/2)
    keep = .!(ismissing.(x) .| iszero.(w))
    x = x[keep]; w = w[keep]

    sum(w.*(x .- GermanTrack.wmean(x, w, default)).^2) / (sum(w) - 1) / sqrt(length(x))
end

"""
    shrinktowards([x],mu;by=0.01)

Reduce the value of `x` towards `mu` by (the presumably small value) `by*(x-mu)`.
You can exclude the `x` argument to curry the function.
"""
shrinktowards(mu;by=0.01) = x -> shrinktowards(x,mu,by=by)
shrinktowards(x,mu;by=0.01) = (1-by)*(x-mu) + mu
shrink(x) = shrinktowards(x,0.5)

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
    progress = setupprogress(length(groups), desc)
    function fn_((key,sdf))
        result = fn(sdf)
        if !isempty(result)
            for (k,v) in pairs(key)
                result[!, k] .= v
            end
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
    foldvals = folds(n, unique(df[:,col]), rng = rng)
    test = Set.(getindex.(foldvals, 2))
    df[!, :fold] = map(colval -> findfirst(fold -> colval ∈ fold, test), df[:,col])
    df
end

"""
    cross_folds(folds)

Supply the fold specification for `filteringmap` to do a cross-validated parameter selection
across the given folds. For each fold, the mapping function of `filteringmap` will get all
data not belonging to that fold.
"""
cross_folds(folds) = map(fold -> fold => @_(__.fold != fold), folds)

struct NoProgress; end
ProgressMeter.next!(::NoProgress) = nothing
setupprogress(n, ::Nothing) = NoProgress()
setupprogress(n, str::String) = Progress(n, desc = str)

"""
    filteringmap(df,filtering1 => (value1 => filterfn1 | value1, etc...),
        filtering2 => etc..., fn, streams = 1
        folder = foldxt, desc = "Progres...")

Repeatedly map a function `fn` over a data frame or a grouped data frame, applying the
function for each group and each set of filterings.

## Details

The `fn` takes each group of `df` as its first argument, and the current value for each
filtering as its remaining arguments. These filterings behave like the groups of a grouped
data frame, but they can include rows that are not mutually exlusive to one another. The
`filtering` arguments specify the overlapping groups: group N's name is the value of `mapN`,
and the values of the group variable are `valueK`; group `valueK` contains all rows that
match the filtering function `filterfnK`. If all filtering functions for a given filtering
are `(x -> true)` (include all rows), you need not specify the `filteringfn`, using
`filtering1 => (value1, value2, etc...)` instead.

If desc is set to `nothing`, no progress bar will be shown.

## Multiple streams

If streams > 1 then `fn` should return a tuple of size `streams`, and filterings
will return the result of each tuple as separate data frames.

## Example

Here's an example using multiple streams

    df = DataFrame(x = 1:10, group = rand(["joe", "bob"], 10))
    result = filteringmap(df, streams = 2,
        :region => (:lower => x -> x.x < 7, :upper => x -> x.x > 4),
        function(sdf, region)
            DataFrame(mean = mean(sdf.x), bobs = sum(sdf.group .== "bob")),
                DataFrame(xnorm = sdf.x ./ mean(sdf.x))
        end
    )

"""
function filteringmap(df, filterings_fn...; folder = foldxt,
    desc = "Progress...", addlabels = true, streams = 1)
    fn = filterings_fn[end]
    filterings = filterings_fn[1:(end-1)]

    defaultpair(x::Pair) = x
    defaultpair(x) = x => (x -> true)

    flattened = @_ filterings |>
        map(f -> collect(map(pair -> (f[1], defaultpair(pair)...), f[2])), __) |>
        Iterators.product(__...) |> collect
    groupings = filtermap_groupings(df, flattened)
    progress = setupprogress(length(groupings), desc)

    function addcolumns!(result, key, filterings)
        if !isempty(result)
            if addlabels
                for (name, val, filterfn) in filterings
                    result[!, name] .= Ref(val)
                end
            end
            if !isempty(key)
                for (k,v) in pairs(key)
                    result[!, k] .= v
                end
            end

            return result
        else
            return Empty(DataFrame)
        end
    end

    function filtermap(((key, group), filterings))
        filtered = group
        for (name, val, filterfn) in filterings
            filtered = filter(filterfn,filtered)
        end

        local result
        if !isempty(filtered)
            result = fn(filtered, getindex.(filterings, 2)...)
            if streams > 1
                result = @_ map(addcolumns!(_, key, filterings), result)
            else
                result = addcolumns!(result, key, filterings)
            end
        else
            if streams > 1
                result = Tuple(Empty(DataFrame) for _ in 1:streams)
            else
                result = Empty(DataFrame)
            end
        end
        next!(progress)

        result
    end

    if streams > 1
        init_result = Tuple(Empty(DataFrame) for _ in 1:streams)
        function append_streams!!(results, streams)
            if length(streams) != length(results)
                error("Cannot append $(length(streams)) streams to "*
                    "$(length(results)) streams.")
            end
            append!!.(results, streams)
        end

        folder(append_streams!!, Map(filtermap), collect(groupings),
            init = init_result)
    else
        folder(append!!, Map(filtermap), collect(groupings), init = Empty(DataFrame))
    end
end

filtermap_groupings(df::GroupedDataFrame, flattened) =
    Iterators.product(pairs(df), flattened) |> collect

struct EmptyKey; end
Base.isempty(x::EmptyKey) = true
filtermap_groupings(df::DataFrame, flattened) =
    Iterators.product([(EmptyKey(), df)], flattened) |> collect

macro cache_results(prefix, args...)
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

macro store_cache(file, args...)
    symbols = args[1:end]
    if !@_ all(_ isa Symbol, symbols)
        error("Expected variable names")
    end

    quote
        begin
            function run(ignore)
                Dict($((map(x -> :($(QuoteNode(x)) => jsonout($(esc(x)))), symbols))...))
            end
            let (dir, prefix, suffix) = cache_results_parser($(esc(file)))
                produce_or_load(dir, (;), run, prefix = prefix, suffix = suffix, force = true)[1] |>
                    jsonin
            end

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
