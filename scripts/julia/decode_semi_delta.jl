using DrWatson
@quickactivate("german_track")
include(joinpath(srcdir(), "julia", "setup.jl"))

# write this so that all the code gets setup in each worker

eeg_files = filter(x->occursin(r"^eeg.*_mcca34\.mcca_proj$", x), readdir(data_dir()))
# eeg_files = filter(x->occursin(r"_cleaned\.eeg$", x), readdir(data_dir()))

fs = 32
eeg_encoding = FFTFiltered("delta" => (1.0,3.0),seconds=15,fs=fs,nchannels=34)

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

cachefile = joinpath(cache_dir(),"eeg","delta_subjects$(fs).bson")
isdir(splitdir(cachefile)[1]) || mkpath(splitdir(cachefile)[1])
if isfile(cachefile)
    @load cachefile subjects
else
    subjects = Dict(file =>
        load_subject(joinpath(data_dir(), file),
            stim_info,
            encoding = eeg_encoding,
            framerate=fs)
        for file in eeg_files)
    @save cachefile subjects
end

@static if oncluster()
    @everywhere include("decode_semi_worker.jl")
    @everywhere groups = groupby(df, [:condition,:source])
    models = @distributed (vcat) for i = 1:1 #length(groups)
        findmodel(groups[i])
    end
else
    include("decode_semi_worker.jl")
    models = by(findmodel,df,[:condition,:source])
end

