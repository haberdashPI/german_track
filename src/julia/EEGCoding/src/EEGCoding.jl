module EEGCoding
using SignalOperators, Requires, Infiltrator, LambdaFn, FFTW, CuArrays
using Zygote: @adjoint
using Zygote

const use_gpu = Ref(false)
gpu(x::AbstractArray) = use_gpu[] ? cu(x) : x
unsafe_gpu_free!(x::AbstractArray) = use_gpu[] && CuArrays.unsafe_free!(x)
@adjoint unsafe_gpu_free!(x::AbstractArray) = unsafe_gpu_free!(x), _ -> nothing
# maybeGPU(x::AbstractArray) = x

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
