using DrWatson; quickactivate(@__DIR__,"german_track")
include(joinpath(srcdir(),"julia","setup.jl"))
using PlotAxes

sound,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
    "trial_01_1.wav"))
plotaxes(vec(sum(sound,dims=2)),quantize=(1000,))

x = encode_stimulus(SampleBuf(sound,fs),64,-1)
x = encode_stimulus(SampleBuf(vec(sum(sound,dims=2)),fs),64,-1,
    method=EEGCoding.ASEnvelope())

fbounds = exp.(range(log(90),log(3700),length=5))[2:end-1]
x = encode_stimulus(SampleBuf(vec(sum(sound,dims=2)),fs),64,-1,
    method=EEGCoding.ASBins(fbounds))

encode = EEGCoding.JointEncoding(EEGCoding.ASBins(fbounds),EEGCoding.TargetSuprisal())
x = encode_stimulus(SampleBuf(vec(sum(sound,dims=2)),fs),64,3,
    method=encode)
