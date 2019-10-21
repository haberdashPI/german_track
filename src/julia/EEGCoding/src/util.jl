export cachefn, cache_dir, JointEncoding, encode
using BSON: @save, @load
using BSON
using ProgressMeter

cache_dir_ = Ref("")
set_cache_dir!(str) = cache_dir_[] = str
cache_dir() = cache_dir_[]

progress_update!(prog::Progress,n=1) =
    ProgressMeter.update!(prog,prog.counter+n)
progress_update!(prog::Bool,n=1) = @assert !prog

function progress_ammend!(prog::Progress,n)
    prog.n += n
    ProgressMeter.update!(prog,prog.counter)
end
progress_ammend!(prog::Bool,n) = @assert !prog

function folds(K,indices,test_indices=indices)
    len = length(indices)
    fold_size = len / K

    unshared_indices = setdiff(test_indices,indices)
    k_step = length(unshared_indices) / K
    last_unshared = 0

    @assert length(indices) > 1
    map(1:K) do k
        start = floor(Int,(k-1)fold_size)+1
        stop = (min(len,floor(Int,k*fold_size)))
        shared_test = indices[start:stop]
        if !isempty(shared_test)
            train = setdiff(indices,shared_test)

            from = last_unshared+1
            to = floor(Int,min(length(unshared_indices),k*k_step))
            last_unshared = max(last_unshared,to)
            test = (shared_test ∩ test_indices) ∪ unshared_indices[from:to]

            @assert k < K || to == length(unshared_indices)

            (train,test)
        else
            shared_test, shared_test
        end
    end
end

function loadcache(prefix)
    file = joinpath(cache_dir_[],prefix * ".bson")
    @load file contents
    contents
end

function cachefn(prefix,fn,args...;__oncache__=() -> nothing,kwds...)
    if cache_dir_[] == ""
        @warn "Using default cache directory `$(abspath(cache_dir_[]))`;"*
            " use EEGCoding.set_cache_dir! to change where results are cached."
    end

    file = joinpath(cache_dir_[],prefix * ".bson")
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
Base.string(x::JointEncoding) = join(map(string,x.children),"_")

"""
    encode(x,samplerate,method)

Encode data (stimlus or eeg) using the given method, outputing the results
at the given sample rate.
"""
function encode
end
