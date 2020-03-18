using DrWatson
@quickactivate("german_track")

using EEGCoding, SignalOperators, SignalOperators.Units, RCall, Distributions,
    PlotAxes, Flux
Uniform = Distributions.Uniform

randenvelope(dur,fr) =
    Signal(randn,fr) |> Filt(Bandpass,1,10) |> Until(dur) |> Array
function randweight(n)
    h = rand(1:n)
    level = rand(Uniform(0.55,1.0))
    off = (1 - level)/(n-1)
    result = fill(off,n)
    result[h] = level

    result
end

T = 5
H = 3
envelopes = [reshape(reduce(hcat,randenvelope(5s,22Hz) for h in 1:H),H,1,:)
    for t in 1:T]
w = reduce(hcat,randweight(3) for t in 1:T)'

eeg = [vcat(0.2randn(20,size(envelopes[1],3)),
            0.2randn(5,size(envelopes[1],3)) .+
            sum(envelopes[t][h,:,:] * w[t,h] for h in 1:H))
       for t in 1:T]

a, ŵ = EEGCoding.regressSS(eeg,envelopes,w[1:2,:],1:2,EEGCoding.CvNorm(0.5,1))

plotaxes(vec(a))
plotaxes(w)
R"quartz()"
plotaxes(ŵ')

# small test, and a "good" hint for the run below
u = randn(3,2)

opt = AMSGrad()
@showprogress for n in 1:10_000
    Flux.train!(x -> sum(Flux.mse(tosimplex(u[i,:]),x[i,:]) for i in 1:3),
        Flux.params(u), [w[3:5,:]], opt)
end


# thought: what if we instead just used the stick breaking, and
# left the variables in the (0,1) range?
a, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:2,:],1:2,EEGCoding.CvNorm(0.5,1),
    batchsize=5,epochs=10_000,status_rate=0.5,optimizer = AMSGrad(),
    testcb = function(decoder)
        diff = mapslices(tosimplex,decoder.u,dims=2) .- w[3:end,:]
        @info "Weight differences: $(sqrt(sum(diff.^2)))."
    end)

plotaxes(vec(a))
R"quartz()"
plotaxes(w)
R"quartz()"
plotaxes(ŵ)

# is it that it can't find the optimal values, or that those
# "optimal" values aren't optimal according to the loss function I wrote?

# a, ŵ = EEGCoding.regressSS(eeg,envelopes,w[1:2,:],1:2,EEGCoding.CvNorm(0.5,1))

a, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:2,:],1:2,EEGCoding.CvNorm(0.5,1),
    batchsize=5,epochs=10,status_rate=0.0,optimizer = AMSGrad(),
    hint = (a,w[3:5,:]),
    testcb = function(decoder)
        diff = mapslices(EEGCoding.zsimplex,decoder.u,dims=2) .- w[3:end,:]
        @info "Weight differences: $(sqrt(sum(diff.^2)))."
    end)



# okay, that looks good, let's try something closer go the size of the actual
# problem

#=
T = 1_000
H = 3

envelopes = [reshape(reduce(hcat,randenvelope(2s,22Hz) for h in 1:H),H,1,:)
    for t in 1:T]
w = reduce(hcat,randweight(3) for t in 1:T)'

eeg = [vcat(0.2randn(490,size(envelopes[1],3)),
            0.2randn(10,size(envelopes[1],3)) .+
            sum(envelopes[t][h,:,:] * w[t,h] for h in 1:H))
       for t in 1:T]

a, ŵ = EEGCoding.regressSS(eeg,envelopes,w[1:10,:],1:10,EEGCoding.CvNorm(0.5,2))

plotaxes(vec(a))
plotaxes(ŵ)
R"quartz()"
plotaxes(w')
=#
