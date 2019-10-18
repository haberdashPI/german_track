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

function folds(k,indices)
    len = length(indices)
    fold_size = len / k
    @assert length(indices) > 1
    map(1:k) do fold
        start = floor(Int,(fold-1)fold_size)+1
        stop = (min(len,floor(Int,fold*fold_size)))
        test = indices[start:stop]
        if !isempty(test)
            train = setdiff(indices,test)
            (train,test)
        else
            eltype(indices)[], test
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
