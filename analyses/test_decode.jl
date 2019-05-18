include(joinpath(@__DIR__,"..","util","setup.jl"))
using MetaArrays
using Makie
using Unitful
using Unitful: ms, s

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))
file = eeg_files[1]
eeg, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)

# TODO: start by creating a script that generates a plot for each trial. The
# figure should include the swithces for the sources and a marker for the
# target Once I've visually inspected these results I can think about how to
# aggregate the results

# NOTE: it would be worth decoding the left vs. right and the

# NOTE: I also need to think about the smoothness of attention_prob can I
# change this to be less smooth and see more potential detail?

malea, fem1a, fem2a = decodetrial(eeg,stim_events,stim_info,51)

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)

scene = Scene()
y,lo,up = attention_prob(malea,max.(fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y)
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


