include(joinpath(@__DIR__,"..","util","setup.jl"))
using MetaArrays
using Makie

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

scene = Scene()
lines!(scene,axes(m1)[1],clamp.(m1,1,4))
lines!(scene,axes(m2)[1],clamp.(m2,1,4),color=:blue)

y,yl,yu = EEGCoding.attention(m1,m2)

scene = Scene()
band!(scene,axes(y)[1],yl,yu)
lines!(scene,axes(y)[1],y)


# TODO: now that the functions are working test it out on our actual data and
# see if we can generate a good interface
