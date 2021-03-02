export cachefn, cache_dir, JointEncoding, encode, folds, randmix, withlags
using BSON: @save, @load
using BSON
using ProgressMeter

cache_dir_ = Ref("")
set_cache_dir!(str) = cache_dir_[] = str
function cache_dir(args...)
    dir = joinpath(cache_dir_[], args...)
    isdir(dir) || mkdir(dir)
    dir
end

function withlags(x,lags)
    if lags == 0:0
        x
    end

    nl = length(lags)
    n,m = size(x)
    y = similar(x,size(x,1),m*nl)
    z = zero(eltype(y))
    for I in CartesianIndices(x)
        for (l,lag) in enumerate(lags)
            r,c = I[1],I[2]
            r_ = r - lag
            y[r,(l-1)*m+c] = 0 < r_ <= n ? x[r_,c] : z
        end
    end
    y
end

"""
    folds(k, indices, [test_indices];
        on_all_empty_test = :error,
        filter_empty_test = on_all_empty_test == :error,
        filter_empty_train = true)

Robust generation of k-folds for cross-validation.

Gracefully handles situations with too few data points for a given fold count by
creating some empty folds (these can be filtered out).

## Parameters
- `k`: the desired number of folds to produce
- `indices`: the indices over which to generate folds (e.g. 1:N)
- `test_indices`: optional argument; handles the case where the test data only partially
    overlap with the training data. Each fold will contain some subset of the train and
    test data, where test data excludes any indices present in the current train data of
    the fold. All test and train data will be covered across the folds.
- `on_all_empty_test`: how to respond if there are too little data to generate even two
    folds (0 or 1 data points); this can be `:error`, `:warn`, `:nothing` or a callback
    function. In the case of `:warn` and `:nothing` a single "fold" is returned
    containing all indices as the trian data, and an empty test data set (it may be
    filtered out).
- `filter_empty_test`, `filter_empty_train`: remove any folds that have
    an empty test or train set, respectively.

## Result

An iterable object of folds. Each fold is a tuple of train and then test indices.

"""
function folds(K, indices, test_indices = indices;on_all_empty_test = :error,
        filter_empty_test = on_all_empty_test == :error, filter_empty_train = true,
        rng = GLOBAL_RNG)

    indices = shuffle(rng, indices)
    test_indices = shuffle(rng, test_indices)

    if K == 1
        @warn "Requested a single fold: non cross-validation performed."
        return [(indices, indices)]
    end

    len = length(indices)
    fold_size = len / K

    unshared_indices = setdiff(test_indices, indices)
    k_step = length(unshared_indices) / K
    last_unshared = 0

    if length(indices) ≤ 1
        if on_all_empty_test == :error
            error("≤ 1 data points")
        elseif on_all_empty_test == :warn
            @warn "≤ 1 data points; no test data created"
            filter_empty_test ? Tuple{Vector{Int}, Vector{Int}}[] : [(indices, Int[])]
        elseif on_all_empty_test == :nothing
            filter_empty_test ? Tuple{Vector{Int}, Vector{Int}}[] : [(indices, Int[])]
        else
            on_all_empty_test()
            filter_empty_test ? Tuple{Vector{Int}, Vector{Int}}[] : [(indices, Int[])]
        end
    else
        result = map(1:K) do k
            start = floor(Int, (k-1)fold_size)+1
            stop = (min(len, floor(Int, k*fold_size)))
            shared_test = indices[start:stop]
            if !isempty(shared_test)
                train = setdiff(indices, shared_test)

                from = last_unshared+1
                to = floor(Int, min(length(unshared_indices), k*k_step))
                last_unshared = max(last_unshared, to)
                test = (shared_test ∩ test_indices) ∪ unshared_indices[from:to]
                if k == K && to < length(unshared_indices)
                    @assert to == length(unshared_indices)-1
                    to = length(unshared_indices)
                end

                (train, test)
            else
                shared_test, shared_test # empty, empty
            end
        end
        result = filter_empty_train ? @_(filter(!isempty(_[1]), result)) : result
        result = filter_empty_test ? @_(filter(!isempty(_[2]), result)) : result

        result
    end
end

function loadcache(prefix)
    file = joinpath(cache_dir_[], prefix * ".bson")
    @load file contents
    contents
end

function cachefn(prefix, fn, args...; __oncache__ = () -> nothing, kwds...)
    if cache_dir_[] == ""
        @warn "Using default cache directory `$(abspath(cache_dir_[]))`;"*
            " use EEGCoding.set_cache_dir! to change where results are cached."
    end

    file = joinpath(cache_dir_[], prefix * ".bson")
    if isfile(file)
        __oncache__()
        @load file contents
        contents
    else
        contents = fn(args...;kwds...)
        @save file contents
        contents
    end
end

abstract type Encoding
end

struct JointEncoding <: Encoding
    children::Vector{Encoding}
end
JointEncoding(xs...) = JointEncoding(collect(xs))
Base.string(x::JointEncoding) = join(map(string, x.children), "_")

"""
    encode(x, framerate, method)

Encode data (stimlus or eeg) using the given method, outputing the results
at the given sample rate.
"""
function encode
end
