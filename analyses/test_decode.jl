include(joinpath(@__DIR__,"..","util","setup.jl"))
using MetaArrays

mf = MatFile(joinpath(data_dir,"test","test.mat"))
Y1 = get_variable(mf,:Y1)
Y2 = get_variable(mf,:Y2)
eeg = get_variable(mf,:cov)
Dec1 = get_variable(mf,:Dec1)
Dec2 = get_variable(mf,:Dec2)
close(mf)

SampledSignals.samplerate(x::MetaArray) = x.samplerate
eegd = meta(eeg,samplerate=200)

m1,m2 = EEGCoding.marker(eegd,Y1,Y2)
# TODO: plot the results to verify them

# TODO: use marker to regenerate m1 and m2 (load from matlab)


# TODO: use attention to regenerate

