using DrWatson
@quickactivate("german_track")

using EEGCoding, RCall, Distributions, PlotAxes, Flux, DSP, Underscores,
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

A = vcat(randn(3),zeros(7))
weights = tosimplex(rand(2,1000))

envelopes = Array{Float64}(undef,250,1,3,1000)
for I in CartesianIndices((3,1000))
    envelopes[:,1,I] = randenvelope(5,50)
end

@reduce x[t,f,i] := sum(s) A[f]*envelopes[t,1,s,i]*weights[s,i]
x .+= 1e-8randn(size(x))

Â₂,ŵ₂ = EEGCoding.regressSS2(x,envelopes,weights[:,1:200],1:200,
    regularize=x -> 0.5sum(abs,x),optimizer=AMSGrad(),epochs = 100)

plotaxes(A)
R"quartz()"; plotaxes(vec(Â₂))
R"quartz()"; plotaxes(weights)
R"quartz()"; plotaxes(ŵ₂)

# TODO: try a larger problem, a little more to scale with eeg data
# (use GPU)
