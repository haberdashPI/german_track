using DrWatson; @quickactivate("german_track")

using EEGCoding, SignalOperators, SignalOperators.Units, RCall, Distributions
Uniform = Distributions.Uniform

T = 5
H = 3
randenvelope(dur,fr) =
    Signal(randn,fr) |> Filt(Bandpass,1,10) |> Until(dur) |> Array
function randweight(n)
    h = rand(1:n)
    level = rand(Uniform(0.55,1.0))
    off = 1 - level
    result = fill(off,n)
    result[h] = level

    result
end


envelopes = [reshape(reduce(hcat,randenvelope(5s,22Hz) for h in 1:H),H,1,:) for t in 1:T]
w = [randweight(3) for t in 1:T]

eeg = [vcat(0.2randn(20,size(envelopes[1],1)),
            0.2randn(5,size(envelopes[1],1)) .+ envelopes[t][')
       for t in 1:T]



