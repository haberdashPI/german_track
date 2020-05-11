# call JULIA_CUDA_MEMORY_POOL=split /opt/julia/bin/julia --project=.
# to properly run this script
using DrWatson
using BenchmarkTools
@quickactivate("german_track")

using EEGCoding, Distributions, PlotAxes, Flux, DSP, Underscores,
    TensorCast

Uniform = Distributions.Uniform

randenvelope(dur,fr) =
    DSP.filt(digitalfilter(Lowpass(2,fs=fr),Butterworth(2)),randn(dur*fr))
    # Signal(randn,fr) |> Filt(Lowpass,10Hz,order=2) |> Until(dur) |> Array
function randweight(n)
    h = rand(1:n)
    level = rand(Uniform(0.55,1.0))
    off = (1 - level)/(n-1)
    result = fill(off,n)
    result[h] = level
    result
end

# TODO: start with an even simpler version of this regression problem
# to troubleshoot the Flux code

#=
A = vcat(randn(3),zeros(7))
weights = tosimplex(rand(2,1000))

envelopes = Array{Float64}(undef,250,1,3,1000)
for I in CartesianIndices((3,1000))
    envelopes[:,1,I] = randenvelope(5,50)
end

@reduce x[t,f,i] := sum(s) A[f]*envelopes[t,1,s,i]*weights[s,i]
x .+= 1e-8randn(size(x))

# using CuArrays
# CuArrays.allowscalar(false)
Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:200],1:200,
regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)

@info "Timing with GPU:"
@time begin
    Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:200],1:200,
        regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)
end

EEGCoding.use_gpu[] = false

Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:200],1:200,
    regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)
@info "Timing without GPU:"
@time begin
    Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:200],1:200,
        regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)
end
=#

# NOTE: with this basic problem, GPU did not add anything
# the next step is to scale the problem up to what we think we'll
# have for the EEG

# more to the point, is it fast enough withou the GPU???

# how much data do we have...
# steps:
# find a straightforward multiple of the EEG data (e.g.
# two subjects), and figure out timing for this subset

#=

150 trials per subj,
30 channels
~1900 samples per trial at 256 Hz
down sample to 64 Hz 970 samples per trial
17 time lags gives us 30x17 = 510 features
about 3/4 of samples used per trial = 727 samples

300 trials
727 time samples
510 features

per subject
=#

using StatsBase
A = zeros(2,510);
A[StatsBase.sample(1:end,40,replace=false)] .= randn(40);
weights = tosimplex(rand(2,750))

envelopes = Array{Float64}(undef,128,2,3,750)
for I in CartesianIndices((3,750))
    envelopes[:,1,I] = randenvelope(2,64)
    envelopes[:,2,I] = randenvelope(2,64)
end

@reduce x[t,f,i] := sum(s,e) A[e,f]*envelopes[t,e,s,i]*weights[s,i]
x .+= 1e-8randn(size(x))

# EEGCoding.use_gpu[] = false

# Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:150],1:150,
#     regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)
# @info "Timing without GPU:"
# @time begin
#     Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:150],1:150,
#         regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)
# end

# using CuArrays
# CuArrays.allowscalar(false)
# EEGCoding.use_gpu[] = true

Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:150],1:150,
    regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 30)

@info "Timing with GPU:"
@time begin
    Â₂,ŵ₂ = regressSS2(x,envelopes,weights[:,1:150],1:150,
        regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 500)
end

