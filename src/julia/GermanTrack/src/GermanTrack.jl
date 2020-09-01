module GermanTrack

# # GermanTrack
#
# This module organizes a variety of re-usable functions specific to the current expeirment
# It is not intended to be used independant of the present analysis.

# ## Module dependencies

using DrWatson, WAV, Infiltrator, DataFrames, Printf, ProgressMeter, EEGCoding, Dates,
    Unitful, Distributions, CSV, Random, Statistics, JSON3, HDF5, CRC, RandomNumbers

# ## Files

include("data.jl")       ## functions related to interpreting data
include("features.jl")   ## functions related to computing classification features
include("classifier.jl") ## functions related to classification
include("stimuli.jl")    ## functions related to encoding stimuli
include("files.jl")      ## functions related to loading files
include("random.jl")     ## functions related to pseudo-random number generation

# ## Caching
# Let `EEGCoding` know where it can cache an intermediate results it generates
#
# !!! NOTE
#       We may delete this, since right now we aren't doing any decoding

function __init__()
    cache_dir = mkpath(joinpath(projectdir(), "_research", "cache"))
    EEGCoding.set_cache_dir!(cache_dir)
end

end
