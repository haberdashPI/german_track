export select_windows, shrinktowards, findresponse, boot,
    addfold!, repeatby, compute_powerbin_features,
    shrink, wsem, tcombine, testsplit, ngroups, prcombine, @repeatby,
    dominant_mass, streak_length

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
    shrink(x)

Reduce the value of `x` towards `mu` by `by*(x-mu)`, where `by` is a presumably small value.
You can exclude the `x` argument to curry the function. When using `shrink`, `mu` is set to
0.5 and `by` is 0.01.
"""
shrinktowards(mu;by=0.01) = x -> shrinktowards(x,mu,by=by)
shrinktowards(x,mu;by=0.01) = (1-by)*(x-mu) + mu
shrink(x) = shrinktowards(x,0.5)


function dominant_index(x)
    indices = argmax.(eachrow(x))
    c = counts(indices, size(x, 2))
    indices, (1:size(x,2))[argmax(c)]
end

"""
    dominant_mass(x)

Given a matrix of decoding correlations, witha source per column and a time point per row,
report the maximum period of time a given source has the largest correlation.
"""
function dominant_mass(x)
    indices, dominant = dominant_index(x)
    mean(i == dominant for i in indices)
end

function streak_stats(x, skip_blips)
    stats = zeros(Int, size(x))
    old_streak = 0
    streak = 0
    misses = 0
    max_streak = 0
    for val in x
        if val
            if misses <= skip_blips && old_streak > 0
                stats[old_streak] -= 1
                streak = old_streak + 1
                old_streak = 0
                misses = 0
            else
                streak += 1
            end
        else
            if streak > 0
                stats[streak] += 1
                max_streak = max(max_streak, streak)
                old_streak = streak
                streak = 0
                misses = 0
            end
            misses += 1
        end
    end
    if streak > 0
        stats[streak] += 1
        max_streak = max(max_streak, streak)
    end
    view(stats,1:max_streak)
end

function streak_length(x, skip_blips)
    indices, dominant = dominant_index(x)
    streaks = streak_stats(indices .== dominant, skip_blips)
    wmean(1:length(streaks), streaks)
end

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
    prcombine(df, fn; showprogress = true, desc = "Progress")
    prcombine(fn, df; showprogress = true, desc = "Progress")

Like `combine`, but also shows a progress bar.
"""
function prcombine(fn::Base.Callable, df; kwds...)
    @assert !(df isa Base.Callable)
    prcombine(df, fn; kwds...)
end
prcombine(df, fn; kwds...) = __combine__(df, fn, foldl; kwds...)

"""
    tcombine(df, fn; showprogress = true, desc = "Progress")
    tcombine(fn, df; showprogress = true, desc = "Progress")

Apply `fn` to each group in a grouped data frame, in parallel, `append!!`ing the returned
values together.

Since this assumes a long running process, it creates a progress bar by default. You can
change the description for the progress bar using `desc`.

"""
function tcombine(fn::Base.Callable, df; kwds...)
    @assert !(df isa Base.Callable)
    tcombine(df, fn; kwds...)
end
tcombine(df, fn; kwds...) = __combine__(df, fn, foldxt; kwds...)
function __combine__(df::GroupedDataFrame, fn, folder;
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

    folder(append!!, Map(fn_), collect(pairs(df)))
end

function __combine__(df::DataFrame, fn, folder; showprogress = true, desc = "Processing...",
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

function validationset(df, bycol, foldcol, crossfoldcol, validate, rng)
    use(rng, df) = rng
    use(rng::Base.Callable, df) = rng(df)

    if rng !== nothing
        valby = @_ df |>
            @where(__, cols(foldcol) .!= cols(crossfoldcol)) |>
            @with(__, unique(cols(bycol)))
        valcount = round(Int, validate*length(valby))
        sample(use(rng, df), valby, valcount, replace = false)
    else
        empty(view(df, :, bycol))
    end
end

"""
    testsplit(df, bycol, fold, foldcol = :fold, splitcol = :split; validate = 0.2, rng = nothing)

Add a column that splits data into train, test and (possibly) validation sets for a given
fold. The test set are all `df[:, :fold] == df[:, :cross_fold]`, and the train set are all
`df[:, :fold] != df[:, :cross_fold]`. This method most effect when applied after a call
to `repeatby`.

If `rng` is `nothing` no validate set is defined, if it is defined, it selects rows for
validation according to the identity of `bycol` (e.g. :subjectid).

Note that `rng` may be a function of one argument: in this case it is passed
the data frame and it should return a random number generator.
"""
function testsplit(df::Union{AbstractDataFrame, GroupedDataFrame}, bycol, foldcol = :fold,
    crossfoldcol = :cross_fold, splitcol = :split;
    validate = 0.2, rng = nothing)

    combine(df, ungroup = false) do sdf
        valset = validationset(sdf, bycol, foldcol, crossfoldcol, validate, rng)
        transform(sdf, [foldcol, crossfoldcol, bycol] => ByRow((fold, cross, by) ->
            fold == cross ? "test" : by ∈ valset ? "validate" : "train") => splitcol)
    end
end

testsplit(rd::RepeatedDataFrame, args...; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> testsplit(df, args...; kwds...)))

"""
    ngroups(df)

Report the number of groupings in the data frame. For a data frame, this is 1, for a grouped
data frame this is its length, and the repeated data frame is the number of repeats times
the number of groupings for the data frame.
"""
ngroups(df::AbstractDataFrame) = 1
ngroups(df::GroupedDataFrame) = length(df)
ngroups(df::RepeatedDataFrame) = @_(prod(length(_[2]), df.repeaters)) * ngroups(df.df)

"""
    @repeatby(df, col = [computed value], ...)

Lazily repeat the rows in df, one repeat for each possible combination of the newly
defined columns. (See [`repeatby`](#) for more details).

The column values behave as if they were wrapped inside `@with` from DataFramesMeta,
allowing you to reference existing columns of `df` using symbols.

## Example

    @repeatby(df, cross_fold = unique(:fold))
"""
macro repeatby(df, repeaters...)
    if length(repeaters) < 1
        error("Expected at least one repeater")
    end
    tempdf = gensym(:df)
    quote
        let
            $tempdf = $(esc(df))
            repeatby($tempdf, $((parse_repeater.(Ref(tempdf), repeaters))...))
        end
    end
end

function parse_repeater(df, repeater)
    @capture(repeater, repeat_ = expression_) || error("Expected keyword argument")

    # NOTE: we can't escape the entire expression below, because this interacts poorly with
    # variables defined in the `@with` macro (leading to values that should be unescaped in
    # an being in escaped form, which results in cryptic erros about a gensym variable being
    # undefined). Instead, we walk through the expression and escape any individual symbols
    # we find.
    expr = MacroTools.postwalk(x -> x isa Symbol ? esc(x) : x, expression)
    :($(QuoteNode(repeat)) => @with($df, $expr))
end

"""
    repeatby(df, col => vals...)

Lazily repeat the rows in df, one repeat for each possible combination of the new columns'
values specified as `col => vals`. The key-value pairs are inserted as additional columns to
the data frame group passed to `combine`. These repeats are realized upon a call to
`combine` or `tcombine`.

Any other operations (filtering, transforming, selecting and joins) that are applied to the
repeated data frame are stored lazily as functors and applied only once the call to
`combine` is realized. This approach avoids the actual memory cost of repeating df if it is
particularly large and the output of a call to combine is relatively small. The mutating
versions of operators (e.g. `transform!`) are not supported (as this would lead to
unpredictable behavior when lazily repeating).

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

DataFrames.groupby(rd::RepeatedDataFrame, args...; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> groupby(df, args...; kwds...)))

DataFrames.innerjoin(rd::RepeatedDataFrame, args...; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> myjoin(innerjoin, df, args...; kwds...)))

DataFrames.outerjoin(rd::RepeatedDataFrame, args...; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> myjoin(outerjoin, df, args...; kwds...)))

DataFrames.rightjoin(rd::RepeatedDataFrame, args...; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> myjoin(rightjoin, df, args...; kwds...)))

DataFrames.leftjoin(rd::RepeatedDataFrame, args...; kwds...) =
    RepeatedDataFrame(rd.df, rd.repeaters, vcat(rd.applyers,
        df -> myjoin(leftjoin, df, args...; kwds...)))

myjoin(joinfn, df::AbstractDataFrame, args...; kwds...) = joinfn(df, args...; kwds...)
function myjoin(joinfn, gd::GroupedDataFrame, args...; kwds...)
    groupby(myjoin(joinfn, parent(gd), args...; kwds...), groupcols(gd))
end

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
function __combine__(rd::RepeatedDataFrame, fn, folder; showprogress = true, desc = "Processing...",
    progress = !showprogress ? NoProgress() :
        Progress(glength(rd.df) * prod(x -> length(x[2]), rd.repeaters), desc = desc))

    combine_repeat(rd, df -> __combine__(df, fn, folder, progress = progress), folder)
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

maybe_addcols(df::AbstractDataFrame, repeat) = isempty(df) ? df : addcols(df, repeat)
maybe_addcols(x, repeat) = x

function combine_repeat(rd::RepeatedDataFrame, combinefn::Function, folder)
    function apply(repeat)
        input = addcols(rd.df, repeat)
        for ap in rd.applyers
            input = ap(input)
        end
        if isempty(input)
            Empty(DataFrame)
        else
            maybe_addcols(combinefn(input), repeat)
        end
    end
    @_ rd.repeaters |>
        map(_1[1] .=> _1[2], __) |>
        Iterators.product(__...) |>
        folder(append!!, Map(apply), __, init = Empty(DataFrame))
end
