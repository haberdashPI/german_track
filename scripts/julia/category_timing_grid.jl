# ----------------------------------- Setup ---------------------------------- #

using DrWatson
@quickactivate("german_track")
use_cache = true
seed = 072189
use_slurm = gethostname() == "lcap.cluster"

using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
    Dates, Underscores, Random, Printf, ProgressMeter, VegaLite, FileIO,
    StatsBase, Bootstrap, BangBang, Transducers, PyCall, ScikitLearn, Flux

# local only packages
using Formatting

import GermanTrack: stim_info, speakers, directions, target_times, switch_times

# we parallelize model parameter optimization with multi-process
# computing; I'd love to use multi-threading, but the current model
# is a python implementaiton and PyCall doesn't support multi-threading
# it's not even clear that it is technically feasible given the python GIL
using Distributed
@static if use_slurm
    using ClusterManagers
    if !(nprocs() > 1)
        addprocs(SlurmManager(16), partition="CPU", t="16:00:00", mem="32G",
            exeflags="--project=.")
    end
else
    # if !(nprocs() > 1)
    #     addprocs(4,exeflags="--project=.")
    # end
end

@everywhere begin
    seed = 072189

    using EEGCoding, GermanTrack, DataFrames, Statistics, DataStructures,
        Dates, Underscores, StatsBase, Random, Printf, ProgressMeter, VegaLite,
        FileIO, StatsBase, Bootstrap, BangBang, Transducers, PyCall,
        PyCall, ScikitLearn, Flux

    import GermanTrack: stim_info, speakers, directions, target_times, switch_times
end

@everywhere( @sk_import svm: (NuSVC, SVC) )

if !use_slurm
    dir = joinpath(plotsdir(),string("results_",Date(now())))
    isdir(dir) || mkdir(dir)
end

# ------------------------ Mean Frequency Bin Analysis ----------------------- #

classdf_file = joinpath(cache_dir(),"data","freqmeans_target_time.csv")
if use_cache && isfile(classdf_file)
    classdf = CSV.read(classdf_file)
else
    eeg_files = dfhit = @_ readdir(data_dir()) |> filter(occursin(r".mcca$",_), __)
    subjects = Dict(file => load_subject(joinpath(data_dir(), file), stim_info,
                                        encoding = RawEncoding())
        for file in eeg_files)
    classdf = find_powerdiff(
        subjects,groups=[:target_time],
        hittypes = ["hit"],
        regions = ["target"],
        windows = [(len=len,start=start,before=-len)
            for len in 2.0 .^ range(-1,1,length=10),
                start in [0; 2.0 .^ range(-2,2,length=9)]])
    CSV.write(classdf_file,classdf)
end

# ------------------------------------ End ----------------------------------- #
