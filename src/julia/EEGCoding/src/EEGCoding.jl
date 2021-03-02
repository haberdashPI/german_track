module EEGCoding
using SignalOperators, Requires, Infiltrator, LambdaFn, FFTW #, CUDA
# using Zygote: @adjoint
# using Zygote
using Underscores

using CorticalSpectralTemporalResponses
using Unitful
using DataFrames
using Random

# unsafe_gpu_free!(x::AbstractArray) = x
# unsafe_gpu_free!(x::CuArray) = CUDA.unsafe_free!(x)
# @adjoint unsafe_gpu_free!(x::AbstractArray) = unsafe_gpu_free!(x), _ -> nothing

include("util.jl")
include("eeg.jl")
include("stimuli.jl")
# include("static.jl")
# include("online.jl")

end # module
