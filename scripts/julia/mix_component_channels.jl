using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack, Underscores, WAV, ProgressMeter

dir = joinpath(stimulus_dir(),"mixtures","testing","mixture_components")

wavfiles = @_ readdir(dir) |> filter(endswith(_, "wav"), __)
savedir = mkpath(joinpath(stimulus_dir(), "mixtures", "testing", "mixture_component_channels"))

progress = Progress(length(wavfiles), "Processing Sounds...")
Threads.@threads for file in wavfiles
    data, fs = wavread(joinpath(dir, file))
    wavwrite(sum(data, dims = 2)/2, Fs=fs,
        joinpath(savedir, replace(file, ".wav" => "_mix.wav")))
    wavwrite(data[:, 1],
        joinpath(savedir, replace(file, ".wav" => "_ch1.wav")))
    wavwrite(data[:, 2],
        joinpath(savedir, replace(file, ".wav" => "_ch2.wav")))
    next!(progress)
end
finish!(progress)

