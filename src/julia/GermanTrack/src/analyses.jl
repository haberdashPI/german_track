export select_windows, shrinktowards, findresponse, boot,
    addfold!, repeatby, compute_powerbin_features,
    shrink, wsem, tcombine, testsplit

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
    tcombine(fn, df; showprogress = true, desc = "Progress")

Apply `fn` to each group in a grouped data frame, in parallel, `append!!`ing the returned
values together.

Since this assumes a long running process, it creates a progress bar by default. You can
change the description for the progress bar using `desc`.

"""
tcombine(fn::Base.Callable, df; kwds...) = tcombine(df, fn; kwds...)
function tcombine(df::GroupedDataFrame, fn;
    showprogress = true, desc = "Processing...",
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

    foldxt(append!!, Map(fn_), collect(pairs(df)))
end

function tcombine(df::DataFrame, fn; showprogress = true, desc = "Processing...",
    progress = NoProgress())

    fn(df)
end

"""
    addfold!(df, n, col; rng = Random.GLOBAL_RNG)

Insert a new column in dataframe `df` for the fold. There
are `n` folds. The fold of a row is determined by the identity of the column `col`.
"""
function addfold!(df, n, col; rng = Random.GLOBAL_RNG)
    foldvals = folds(n, unique(df[:,col]), rng = rng)
    test = Set.(getindex.(foldvals, 2))
    df[!, :fold] = map(colval -> findfirst(fold -> colval ∈ fold, test), df[:,col])
    df
end

struct RepeatedDataFrame{D}
    df::D
    repeaters
    applyers
end
function Base.show(io::IO, ::MIME"text/plain", x::RepeatedDataFrame)
    println(io, "Lazy repeating data frame:")
    println(io, "---------------------------")
    println(io, "With repeaters: ")
    for rep in x.repeaters
        show(io, MIME"text/plain"(), rep)
        println()
    end
    println(io)
    println(io, "Of data:")
    show(io, MIME"text/plain"(), x.df)
    println(io)
    if !isempty(x.applyers)
        println(io, "NOTE: data has unapplied modifications stored in "*
            "$(length(x.applyers)) functors")
    end
end

"""
    repeatby(df, col => vals...)

Lazily repeat the rows in df, one repeat for each possible combination of the new columns'
values specified as `col => vals`. The key-value pairs are inserted as additional columns to
the data frame group passed to `combine`. These repeats are realized upon a call to
`combine` or `tcombine`.

Any other operations (filtering, transforming, selecting) that are applied to the repeated
data frame are stored lazily as functors and applied only once the call to `combine` is
realized. This approach avoids the actual memory cost of repeating df if it is particularly
large and the output of a call to combine is relatively small. The mutating versions of
operators (e.g. `transform!`) are not supported (as this would lead to unpredictable
behavior when lazily repeating).

"""
repeatby(df, repeaters...) = RepeatedDataFrame(df, repeaters, [])

# these are the methods to override to support the interface for combine
# c.f. DataFrames/src/groupeddataframe/splitapplycombine.jl
DataFrames.combine(f::Union{Base.Callable, Pair}, rd::RepeatedDataFrame; kwds...) =
    combine_repeat(rd, df -> combine(f, df; kwds...), foldl)

DataFrames.combine(rd::RepeatedDataFrame,
        cs::Union{Pair, Base.Callable, DataFrames.ColumnIndex, DataFrames.MultiColumnIndex}...;
        kwds...) =
    combine_repeat(rd, df -> combine(df, cs...; kwds...), foldl)

DataFrames.select(f::Union{Base.Callable, Pair}, rd::RepeatedDataFrame; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> select(f, df)))

DataFrames.select(rd::RepeatedDataFrame, @nospecialize(args...); kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> select(df, args...; kwds...)))

DataFrames.transform(f::Union{Base.Callable, Pair}, rd::RepeatedDataFrame; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> transform(f, df)))

DataFrames.transform(rd::RepeatedDataFrame, @nospecialize(args...); kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> transform(df, args...; kwds...)))

Base.filter(f, rd::RepeatedDataFrame; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> filter(f, df)))

DataFramesMeta.where(rd::RepeatedDataFrame{<:AbstractDataFrame}, f) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> DataFramesMeta.where(df, f)))

DataFramesMeta.where(rd::RepeatedDataFrame{<:GroupedDataFrame}, f) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> groupby(DataFramesMeta.where(df, f), groupcols(rd.df))))

glength(x::GroupedDataFrame) = length(x)
glength(x::DataFrame) = 1
function tcombine(rd::RepeatedDataFrame, fn; showprogress = true, desc = "Processing...",
    progress = !showprogress ? NoProgress() :
        Progress(glength(rd.df) * prod(x -> length(x[2]), rd.repeaters), desc = desc))

    combine_repeat(rd, df -> tcombine(df, fn, progress = progress), foldxt)
end

function addcols(df::AbstractDataFrame, repeat)
    df = copy(df, copycols = false)
    for (k,v) in repeat
        df[!, k] .= Ref(v)
    end
    df
end

function addcols(df::GroupedDataFrame, repeat)
    groupby(addcols(parent(df), repeat), groupcols(df))
end

maybe_addcols(df::AbstractDataFrame, repeat) = addcols(df, repeat)
maybe_addcols(x, repeat) = x

function combine_repeat(rd::RepeatedDataFrame, combinefn::Function, folder)
    function apply(repeat)
        input = addcols(rd.df, repeat)
        for ap in rd.applyers
            input = ap(input)
        end
        maybe_addcols(combinefn(input), repeat)
    end
    @_ rd.repeaters |>
        map(_1[1] .=> _1[2], __) |>
        Iterators.product(__...) |>
        folder(append!!, Map(apply), __, init = Empty(DataFrame))
end

function _parse_cache_args(prefix, args)
    # 2. symbols to cache
    cache_symbols(x) = error("Unexpected expression `$(x)`")

    # 2a. a bare variable name
    cache_symbols(x::Symbol) = x
    cache_filetypes(x::Symbol) = :arrow

    # 2b. tuple of (varname, :filetype)
    function cache_symbols(ex::Expr)
        if isexpr(ex, :tuple)
            @capture(ex, (var_, type_))
            return var
        else
            error("Unexpected expression `$(ex)`.")
        end
    end
    function cache_filetypes(ex::Expr)
        @capture(ex, (var_, type_))
        if type isa QuoteNode
            return type.value
        else
            error("Unexpected expression `$(type)`.")
        end
    end
    symbols = cache_symbols.(args[1:end])
    file_types = cache_filetypes.(args[1:end])

    checktypes(x::Symbol) = x ∈ [:arrow, :bson, :jld] || error("Unexpected filetype `$(x)``.")
    checktypes.(file_types)

    symbols, file_types
end

_fnames(prefix, symbols, types) =
    :(string.($(esc(prefix)), "-", $(string.(symbols)), ".", $(string.(types))))
_fname(prefix, symbols, types, i) =
    :(string($(esc(prefix)), "-", $(string(symbols[i])), ".", $(string(types[i]))))

macro use_cache(prefix, args...)
    body = args[end]
    symbols, types = _parse_cache_args(prefix, args[1:(end-1)])

    #### Verify that each variable listed in cache header exists in the body
    # of the macro

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

    #### code generation
    quote
        begin
            # run body
            if all(isfile, $(_fnames(prefix, symbols, types)))
                $(_load_cache(prefix, symbols, types))
            else # create the values
                $(esc(body))
                $(_save_cache(prefix, symbols, types))
            end
            nothing
        end
    end
end

macro save_cache(prefix, args...)
    symbols, types = _parse_cache_args(prefix, args)
    _save_cache(prefix, symbols, types)
end

macro load_cache(prefix, args...)
    symbols, types = _parse_cache_args(prefix, args)
    _load_cache(prefix, symbols, types)
end

function _load_cache(prefix, symbols, types)
    quote
        $(map(enumerate(zip(symbols, types))) do (i, (var, type))
            if type == :arrow
                :($(esc(var)) = DataFrame(Arrow.Table($(_fname(prefix, symbols, types, i)))))
            elseif type == :bson
                :($(esc(var)) = load($(_fname(prefix, symbols, types, i)))[:data])
            elseif type == :jld
                :($(esc(var)) = load($(_fname(prefix, symbols, types, i)), "data"))
            else
                errror("Unexpected error: report a bug.")
            end
        end...)
    end
end

function _save_cache(prefix, symbols, types)
    quote
        # store code state
        state = convert(Dict{String, String}, tag!(Dict()))

        # store variables in files
        $(map(enumerate(zip(symbols, types))) do (i, (var, type))
            if type == :arrow
                quote
                    Arrow.setmetadata!($(esc(var)), state)
                    Arrow.write($(_fname(prefix, symbols, types, i)), $(esc(var)), compress = :lz4)
                    @info string("Saved ", $(string(var))," to ", $(_fname(prefix, symbols, types, i)))
                end
            elseif type == :bson
                quote
                    data = deepcopy(state)
                    data[:data] = $(esc(var))
                    save($(_fname(prefix, symbols, types, i)), data)
                    @info string("Saved ", $(string(var))," to ", $(_fname(prefix, symbols, types, i)))
                end
            elseif type == :jld
                quote
                    save($(_fname(prefix, symbols, types, i)), "data", $(esc(var)), "state", state)
                    @info string("Saved ", $(string(var))," to ", $(_fname(prefix, symbols, types, i)))
                end
            else
                errror("Unexpected error: report a bug.")
            end
        end...)
    end
end
