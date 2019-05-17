include(joinpath(@__DIR__,"..","util","setup.jl"))
using MetaArrays
using Makie
using Unitful
using Unitful: ms, s

mf = MatFile(joinpath(data_dir,"test","test.mat"))
Y1 = get_variable(mf,:Y1)
Y2 = get_variable(mf,:Y2)
eeg = get_variable(mf,:cov)
Dec1 = get_variable(mf,:Dec1)
Dec2 = get_variable(mf,:Dec2)
close(mf)

m1,m2 = EEGCoding.marker(eegd,Y1,Y2,samplerate=200,maxit=250,tol=2e-3,
    progress=true,lag=0ms)
y,yl,yu = EEGCoding.attention(m1,m2)

scene = Scene()
lines!(scene,axes(m1)[1],clamp.(m1,1,4))
lines!(scene,axes(m2)[1],clamp.(m2,1,4),color=:blue)

scene = Scene()
band!(scene,axes(y)[1],yl,yu)
lines!(scene,axes(y)[1],y)

stim_info = JSON.parsefile(joinpath(stimulus_dir,"config.json"))
eeg_files = filter(x -> occursin(r"_mcca65\.bson$",x),readdir(data_dir))
file = eeg_files[1]
eeg, stim_events, sid = load_subject(joinpath(data_dir,file),stim_info)

function decodetrial(trial)
    male = load_sentence(stim_events,stim_info,trial,1)
    malev = EEGCoding.find_envelope(male,samplerate(eeg))
    fem1 = load_sentence(stim_events,stim_info,trial,2)
    fem1v = EEGCoding.find_envelope(fem1,samplerate(eeg))
    fem2 = load_sentence(stim_events,stim_info,trial,3)
    fem2v = EEGCoding.find_envelope(fem2,samplerate(eeg))

    malea,fem1a,fem2a = EEGCoding.marker(eeg.data[trial]',malev,fem1v,fem2v,
        samplerate=samplerate(eeg),maxit=250,tol=2e-3,progress=true,lag=250ms,
        min_norm=1e-12)

    μ = mean(mean.((malea,fem1a,fem2a)))
    (malea./μ, fem1a./μ, fem2a./μ)
end

malea, fem1a, fem2a = decodetrial(51)

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)

scene = Scene()
y,lo,up = EEGCoding.attention(malea,max.(fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y)
y,lo,up = EEGCoding.attention(fem1a,max.(malea,fem2a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:blue)
y,lo,up = EEGCoding.attention(fem2a,max.(malea,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:red)

malea, fem1a, fem2a = decodetrial(52)

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)

malea, fem1a, fem2a = decodetrial(105)

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)

scene = Scene()
y,lo,up = EEGCoding.attention(malea,max.(fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y)
y,lo,up = EEGCoding.attention(fem1a[2:end],max.(malea[2:end],fem2a[2:end]))
band!(scene,t[2:end],lo,up,color=:lightgray)
lines!(scene,t[2:end],y,color=:blue)
y,lo,up = EEGCoding.attention(fem2a,max.(malea,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:red)

malea, fem1a, fem2a = decodetrial(1)

scene = Scene()
t = ustrip.(uconvert.(s,axes(malea)[1].*250ms))
lines!(scene,t,malea)
lines!(scene,t,fem1a,color=:blue)
lines!(scene,t,fem2a,color=:red)

scene = Scene()
y,lo,up = EEGCoding.attention(malea,max.(fem2a,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y)
y,lo,up = EEGCoding.attention(fem1a,max.(malea,fem2a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:blue)
y,lo,up = EEGCoding.attention(fem2a,max.(malea,fem1a))
band!(scene,t,lo,up,color=:lightgray)
lines!(scene,t,y,color=:red)


