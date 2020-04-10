module EEGCoding
using SignalOperators, Requires, Infiltrator, LambdaFn, FFTW, CuArrays

function __init__()
    if CuArrays.functional()
        @eval maybeGPU(x::AbstractArray) = CuArray(x)
    else
        @eval maybeGPU(x::AbstractArray) = x
    end
end

include("parallel.jl")
include("util.jl")
include("eeg.jl")
include("stimuli.jl")
include("static.jl")
include("online.jl")

end # module
