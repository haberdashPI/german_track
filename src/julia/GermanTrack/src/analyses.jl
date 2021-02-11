export select_windows, shrinktowards, findresponse, boot,
    addfold!, splayby, mapgroups, repeatby, compute_powerbin_features, cross_folds,
    shrink, wsem, repeatby

"""
    boot(x; alpha = 0.05, n = 10_000)

Bootstrapped estimate. Argument `x` can be curried: `x |> boot(n = 1_000)`
"""
function boot(x; stat = mean, alpha = 0.05, n = 10_000,
    sampling = BasicSampling(n),
    confint = BasicConfInt(1 - alpha))

    vals = Bootstrap.confint(bootstrap(stat, x, sampling), confint)[1]
    NamedTuple{(:value, :lower, :upper)}(vals)
end
boot(;kwds...) = x -> boot(x; kwds...)

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
    findresponse(row; kwds...)

Correctly interprets a given row of the data as a hit, correct rejection, false positive
or miss. Since the directions are different for each condition, how we interpret a an
indication of a detected target depends on the condition.
"""
function findresponse(row; mark_false_targets = false, kwds...)
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

struct NoProgress; end
ProgressMeter.next!(::NoProgress) = nothing

"""
    tcombine(df, fn; showprogress = true, desc = "Progress")

Apply `fn` to each group in a grouped data frame, in parallel (by default), `append!!`ing
the returned values together.

Since this assumes a long running process, it creates a progress bar by default. You can
change the description for the progress bar using `desc`.

"""
function tcombine(df::GroupedDataFrame, fn; showprogress = true, desc = "Processing...",
    progress = showprogress ? ProgressMeter(length(df)) : NoProgress())

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

    foldxt(append!!, Map(fn_), collect(pairs(groups)))
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

struct RepeatedDataFrame
    df::AbstractDataFrame
    repeaters
end

"""
    repeatby(df, col => vals...)

Lazily repeat the rows in df, one repeat for each possible combination of the new columns'
values specified as `col => vals`. These repeats are realized upon a call to `combine`. This
avoids the actual memory cost of repeating df if it is particularly large and the output of
a call to combine is relatively small.

The key-value pairs are inserted as additional columns to the data frame group passed to
`combine`

"""
repeateby(df, repeaters...) = RepeatedDataFrame(df, repeaters)

addprogress(f, rd::RepeatedDataFrame{<:Nothing}) = f
function addprogress(f::Base.Callable, rd::RepeatedDataFrame{<:ProgressMeter})
    function newf(xs...)
        result = f(xs...)
        next!(rd.progress)
        result
    end
end
addprogress(f::Pair, rd::RepeatedDataFrame{<:ProgressMeter}) =
    f[1] => addprogress(f[2], fd)
addprogress(f::Pair{<:Any, <:Pair}, rd::RepeatedDataFrame{<:ProgressMeter}) =
    f[1] => addprogress(f[2][1], fd) => f[2][2]
addprogress(f::Tuple, rd::RepeatedDataFrame{<:ProgressMeter}) =
    @_ map(addprogress(_, rd), f)

# these are the methods to override to support the interface for combine
# c.f. DataFrames/src/groupeddataframe/splitapplycombine.jl
function combine(f::Union{Base.Callable, Pair}, rd::RepeateBy; kwds...)
    f = addprogress(f, )
    combine_repeat(rd, df -> combine(f, df; kwds...))
end

function combine(rd::RepeateBy,
        cs::Union{Pair, Base.Callable, ColumnIndex, MultiColumnIndex}...;
        kwds...)
    cs = addprogress(cs, rd)
    combine_repeat(rd, df -> combine(rd, cs...; kwds...))
end

function tcombine(df::RepeateBy, fn; showprogress = true, desc = "Processing...")
    progress = !showprogress ? nothing :
        ProgressMeter(glength(df) * @_ prod(length(_[2]), repeaters), desc = desc)
    rd = RepeatedDataFrame(df, repeaters, progress, foldxt)

    combine_repeat(rd, df -> tcombine(df, fn, progress = progress))
end

function combine_repeat(rd::RepeatedDataFrame, combinefn::Function)
    function filtermap(repeat)
        df = copy(rd.df, copycols = false)
        for (k,v) in pairs(repeat)
            df[!, k] .= v
        end
        combinefn(df)
    end

    @_ rd.repeaters |>
        map(_1[1] .=> _1[2], __)
        Iterators.product(__...) |>
        rd.folder(append!!, Map(filtermap), __, init = Empty(DataFrame))
end

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
