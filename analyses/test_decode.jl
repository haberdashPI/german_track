include(joinpath(@__DIR__,"..","util","setup.jl"))
using Makie
using BenchmarkTools
using MetaArrays
using Unitful
using Unitful: ms, s

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))
file = eeg_files[1]
eeg, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)

# TODO: start with a single participant, to make sure my pipeline
# is working

# TODO: start by creating a script that generates a plot for each trial. The
# figure should include the swithces for the sources and a marker for the
# target Once I've visually inspected these results I can think about how to
# aggregate the results

# NOTE: it would be worth decoding the left vs. right and the

# NOTE: I also need to think about the smoothness of attention_prob can I
# change this to be less smooth and see more potential detail?

malea, fem1a, fem2a = attention_marker(
    window=250ms,lag=250ms,estimation_length=10s,γ=2e-3,maxit=250,tol=1e-2,
    min_norm=1e-16,samplerate=samplerate(eeg),
    eegtrial(eeg,51)',
    (load_sentence(stim_events,samplerate(eeg),stim_info,51,i) for i in 1:3)...)
μ = mean(mean.((malea,fem1a,fem2a)))
malea ./= μ
fem1a ./= μ
fem2a ./= μ

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)
lines!(scene,t,(malea .+ fem1a .+ fem2a)./3,color=:gray)

scene = Scene()
y,lo,up = attention_prob(max.(1e-4,malea),max.(1e-4,fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:black)

y,lo,up = attention_prob(fem1a,max.(malea,fem2a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:blue)
y,lo,up = attention_prob(fem2a,max.(malea,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:red)

malea, fem1a, fem2a = decodetrial(eeg,stim_events,stim_info,52)

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)

malea, fem1a, fem2a = decodetrial(eeg,stim_events,stim_info,105)

scene = Scene()
y,lo,up = attention_prob(max.(1e-3,malea),max.(1e-3,fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y)
y,lo,up = attention_prob(max.(1e-3,fem1a),max.(1e-3,malea,fem2a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:blue)
y,lo,up = attention_prob(max.(1e-3,fem2a),max.(1e-3,malea,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:red)

malea, fem1a, fem2a = decodetrial(eeg,stim_events,stim_info,1)

scene = Scene()
y,lo,up = attention_prob(max.(1e-3,malea),max.(1e-3,fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y)
y,lo,up = attention_prob(max.(1e-3,fem1a),max.(1e-3,malea,fem2a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:blue)
y,lo,up = attention_prob(max.(1e-3,fem2a),max.(1e-3,malea,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:red)


