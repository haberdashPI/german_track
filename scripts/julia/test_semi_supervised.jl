using DrWatson
@quickactivate("german_track")

using EEGCoding, RCall, Distributions, PlotAxes, Flux, DSP
Uniform = Distributions.Uniform

# TODO: use something else for the random envelope
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

T = 1_000
H = 3
envelopes = [permutedims(reshape(reduce(hcat,randenvelope(5,50) for h in 1:H),1,:,H),(3,1,2))
    for t in 1:T]
w = reduce(hcat,randweight(3) for t in 1:T)'

eeg = [vcat(1e-1randn(20,size(envelopes[1],3)),
            1e-1randn(5,size(envelopes[1],3)) .+
            sum(envelopes[t][h,:,:] * w[t,h] for h in 1:H))
       for t in 1:T]

#= two things that could be going wrong
1. loss function (does ŵ work better than w?)
2. miss transformation (does w turn into something else?)
=#


a, ŵ = EEGCoding.regressSS(eeg,envelopes,w[1:200,:],1:200,EEGCoding.CvNorm(0.5,1))

plotaxes(vec(a))
p = plotaxes(w);
R"$p + scale_fill_continuous(limits=c(0,1))"
R"quartz()"
p = plotaxes(ŵ');
R"$p + scale_fill_continuous(limits=c(0,1))"

â, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:200,:],1:200,EEGCoding.CvNorm(0.5,1),
    batchsize=300,epochs=2_000,status_rate=2,optimizer = AMSGrad(),
    hint = (â,ŵ'[201:end,:]),
    testcb = function(decoder)
        diff = mapslices(tosimplex,decoder.u,dims=2) .- w[201:end,:]
        @info "Weight differences: $(sqrt(mean(diff.^2)))."
    end)

plotaxes(vec(â))
p = plotaxes(w);
R"$p + scale_fill_continuous(limits=c(0,1))"
R"quartz()"
p = plotaxes(ŵ);
R"$p + scale_fill_continuous(limits=c(0,1))"

# why are some of the weights pushed into an identical pattern? (why does
# this happen even when we seed the initial value with the correct solution?)

â, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:200,:],1:200,EEGCoding.CvNorm(0.5,1),
    batchsize=300,epochs=2_000,status_rate=2,optimizer = AMSGrad(1e-5),
    hint = (â,ŵ'[201:end,:]),
    testcb = function(decoder)
        diff = mapslices(tosimplex,decoder.u,dims=2) .- w[201:end,:]
        @info "Weight differences: $(sqrt(mean(diff.^2)))."
    end)

#  sort of weights from least to most similar

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

# does it work without the unknown weights?
a, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:4,:],1:4,EEGCoding.CvNorm(0.3,1),
    batchsize=5,epochs=20_000,status_rate=0.5,optimizer = ADAM())
plotaxes(vec(a))

plotaxes(ŵ)
R"quartz()"
plotaxes(w)

# small test, and a "good" hint for the run below
u = rand(3,2)

opt = AMSGrad()
@showprogress for n in 1:1_000
    Flux.train!(x -> sum(Flux.mse(zsimplex(u[i,:]),x[i,:]) for i in 1:3),
        Flux.params(u), [w[3:5,:]], opt)
end

# does it work without the unknown weights?
a, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:5,:],1:5,EEGCoding.CvNorm(0.3,1),
    batchsize=5,epochs=20_000,status_rate=0.5,optimizer = ADAM())

a, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:0,:],1:0,EEGCoding.CvNorm(0.3,1),
    batchsize=5,epochs=20_000,status_rate=0.5,optimizer = ADAM())
plotaxes(vec(a))

## eventually

plotaxes(ŵ)
R"quartz()"
plotaxes(w)

# is it different with a larger data set?

# TODO: let's try scaling up again, this time using GPU
# on the cluster to speed things up, I think this just might
# be an issue of need more time to converge, and richer data???

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

a, ŵ = EEGCoding.regressSS2(eeg,envelopes,w[1:10,:],1:10,EEGCoding.CvNorm(0.5,1),
    batchsize=100,epochs=300,status_rate=0.0,optimizer = AMSGrad())

plotaxes(vec(a))
R"quartz()"
plotaxes(ŵ)
R"quartz()"
plotaxes(w)

