module GermanTrack

# # GermanTrack
#
# This module organizes a variety of re-usable functions specific to the current expeirment
# It is not intended to be used independant of the present analysis.

# ## Module dependencies

using DrWatson, WAV, Infiltrator, DataFrames, Printf, ProgressMeter, EEGCoding, Dates,
    Unitful, Distributions, CSV, Random, Statistics, JSON3, HDF5, CRC, RandomNumbers,
    Random123, Colors, EzXML, Underscores, DataStructures, StatsBase, Lasso, BangBang,
    Transducers, FFTW, Bootstrap, ColorSchemes, VegaLite, StatsFuns, DSP, Peaks,
    JSONTables, MacroTools, DataFramesMeta, Flux, CUDA, Arrow

# ## Files

include("environment.jl") ## for setting up development/analysis environment
include("analyses.jl")    ## for analyzing data
include("plots.jl")       ## for plotting
include("features.jl")    ## for computing classification features
include("classifier.jl")  ## for classification
include("stimuli.jl")     ## for encoding stimuli
include("files.jl")       ## for loading files
include("random.jl")      ## for pseudo-random number generation
include("decoder.jl")

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
