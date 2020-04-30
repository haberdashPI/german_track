module EEGCoding
using SignalOperators, Requires, Infiltrator, LambdaFn, FFTW, CuArrays

const use_gpu = Ref(false)
maybeGPU(x::AbstractArray) = use_gpu[] ? CuArray(x) : x

function __init__()
    use_gpu[] = CuArrays.functional()
end

include("parallel.jl")
include("util.jl")
include("eeg.jl")
include("stimuli.jl")
include("static.jl")
include("online.jl")

end # module
