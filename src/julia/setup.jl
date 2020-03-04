using DataFrames, Printf, ProgressMeter, FileIO, EEGCoding, Dates, Distributed,
    Unitful, Distributions, LambdaFn, RCall, VegaLite, CSV, PlotAxes,
    AxisArrays, DataFramesMeta, Random, Statistics, StatsBase, DSP, FFTW,
    DSP.Periodograms, EEGCoding, JSON, DataStructures, ProximalOperators,
    BSON, PooledArrays, Tables, ProgressMeter

using GermanTrack
const cache_dir = GermanTrack.cache_dir
import BSON: @load, @save

oncluster() = gethostname() == "lcap.cluster"
@static if oncluster()
  using ClusterManagers
  addprocs(SlurmManager(1),partition="CPU", t="04:00:00", mem="64G")
  @everywhere using DrWatson
  @everywhere @quickactivate("german_track")
  @everywhere include(joinpath(srcdir(), "julia", "worker_setup.jl"))
end

