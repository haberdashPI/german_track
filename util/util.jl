using Colors
using Debugger

function mat2bson(file)
    file
end

# function Base.convert(::Type{DataKnot},xs::Vector{OnlineResult})
#     tuples = map(xs) do x
#         NamedTuple{fieldnames(OnlineResult),
#             Tuple{fieldtypes(OnlineResult)...}}(
#             tuple((getfield(x,field) for field in fieldnames(OnlineResult))...)
#         )
#     end
#     convert(DataKnot,tuples)
# end

function clean_eeg!(data)
    EEGData(
        label = convert(Vector{String},data["label"]),
        fs = Int(data["fsample"]),
        data = vec(Array{Matrix{Float64}}(data["trial"]))
    )
end

function load_subject(file,stim_info)
    if !isfile(file)
        error("File '$file' does not exist.")
    end

    stim_events, sid = events_for_eeg(file,stim_info)

    if endswith(file,".mat")
        mf = MatFile(file)
        data = get_mvariable(mf,:dat)
        close(mf)

        data,stim_events,sid
    elseif endswith(file,".bson")
        @load file data
        data,stim_events,sid
    else
        pat = match(r"\.(.+)$",file)
        if pat != nothing
            ext = pat[1]
            error("Unsupported data format '.$ext'.")
        else
            error("Unknown file format for $file")
        end
    end
end

function single(x,message="Expected a single element.")
    it = iterate(x)
    if isnothing(it)
        error(message)
    else
        x_, state = it
        if !isnothing(iterate(x,state))
            error(message)
        end
    end
    x_
end

function plotatten!(scene,method,results;colors=[:black,:red,:blue],raw=false)
    len = minimum(map(x -> length(x.probs),results))
    step = ustrip.(uconvert.(s,method.params.window))
    t = step.*((1:len) .- 1)
    if raw
        for (col,result) in zip(colors,results)
            @views lines!(scene,float(t),result.norms[1:len],color=col)
        end
    else
        for (col,result) in zip(colors,results)
            @views band!(scene,float(t),result.lower[1:len],result.upper[1:len],
                color=RGBA(0.0,0.0,0.0,0.2))
            @views lines!(scene,float(t),result.probs[1:len],color=col,
                linewidth=3.0)
        end
    end

    scene
end

function plottrial(method,results,stim_info,file;
    colors=[:black,:red,:blue],raw=false)

    main = Scene()
    plotresponse!(main,method,results,stim_info,file)
    plottarget!(main,method,results,stim_info,file;colors=colors)
    plotatten!(main,method,results,raw=raw)

    stimulus = Scene()
    plotswitches!(stimulus,method,results,stim_info,file)

    hbox(stimulus,main,sizes=[0.3,0.7])
end

function targetattend(rows,stim_events,stim_info,fs)
    trial = single(unique(map(r->r.trial,rows)),
        "Expected single trial number")
    target_len = stim_info["target_len"]
    stim_index = stim_events.sound_index[trial]
    stim_fs = stim_info["fs"]

    if stim_events.target_present[trial]
        target =
            stim_info["test_block_cfg"]["trial_target_speakers"][stim_index]
        target_time = stim_info["test_block_cfg"]["target_times"][stim_index]
        start = clamp(floor(Int,target_time * fs),1,length(rows[1].probs))
        stop = clamp(ceil(Int,(target_time + target_len) * fs),1,length(rows[1].probs))
        others = setdiff(1:3,target)
        @views begin
            t = rows[target].probs[start:stop]
            o1 = rows[others[1]].probs[start:stop]
            o2 = rows[others[2]].probs[start:stop]
        end
        mean((t .> o1) .& (t .> o2))
    else
        0.0
    end
end

function plotresponse!(scene,method,results,stim_info,file)
    trial = single(unique(map(r->r.trial,results)),
        "Expected single trial number")
    stim_events, = events_for_eeg(file,stim_info)
    step = ustrip.(uconvert.(s,method.params.window))
    len = minimum(map(x -> length(x.probs),results))*step

    if stim_events.target_present[trial] == stim_events.correct[trial]
        poly!(scene,color=RGBA(0,0,0,0.25),Point2f0[
            [0,-3],[0,3], [len,3],[len,-3]
        ])
    end

    scene
end

function plottarget!(scene,method,results,stim_info,file;
    colors=[:black,:red,:blue])

    trial = single(unique(map(r->r.trial,results)),
        "Expected single trial number")
    stim_events, = events_for_eeg(file,stim_info)
    if stim_events.target_present[trial]
        stim_index = stim_events.sound_index[trial]
        start_time = stim_info["test_block_cfg"]["target_times"][stim_index]
        stop_time = stim_info["target_len"]+start_time

        target_speaker =
            stim_info["test_block_cfg"]["trial_target_speakers"][stim_index]
        col = parse(Colorant,colors[target_speaker])
        col = RGBA(col.r,col.g,col.b,0.3)

        poly!(scene,color=col,Point2f0[
            [start_time,-3],
            [start_time,3],
            [stop_time,3],
            [stop_time,-3]
        ])
    end

    scene
end

function plotswitches!(scene,method,results,stim_info,file;
    colors=[:black,:red,:blue])

    trial = single(unique(map(r->r.trial,results)),
        "Expected single trial number")
    stim_events, = events_for_eeg(file,stim_info)
    stim_index = stim_events.sound_index[trial]
    direc_file = joinpath(stimulus_dir,"mixtures","testing",
        @sprintf("trial_%02d.direc",stim_index))
    dirs = load_directions(direc_file)
    len = minimum(length.((dirs.dir1,dirs.dir2,dirs.dir3)))
    t = (1:len)./dirs.samplerate
    lines!(scene,t,dirs.dir1,color=colors[1])
    lines!(scene,t,dirs.dir2,color=colors[2])
    lines!(scene,t,dirs.dir3,color=colors[3])

    scene
end

struct Directions
    dir1::Vector{Float64}
    dir2::Vector{Float64}
    dir3::Vector{Float64}
    samplerate::Float64
end

function load_directions(file)
    open(file,read=true) do stream
        samplerate = read(stream,Float64)
        len1 = read(stream,Int)
        len2 = read(stream,Int)
        len3 = read(stream,Int)

        dir1 = reinterpret(Float64,read(stream,sizeof(Float64)*len1))
        dir2 = reinterpret(Float64,read(stream,sizeof(Float64)*len1))
        dir3 = reinterpret(Float64,read(stream,sizeof(Float64)*len1))

        @assert length(dir1) == len1
        @assert length(dir2) == len2
        @assert length(dir3) == len3

        Directions(dir1,dir2,dir3,samplerate)
    end
end

function events_for_eeg(file,stim_info)
    matched = match(r"eeg_response_([0-9]+)(_[a-z_]+)?([0-9]+)?\.[a-z]+$",file)
    if matched == nothing
        error("Could not find subject id in filename '$file'.")
    end
    sid = parse(Int,matched[1])

    event_file = joinpath(data_dir,@sprintf("sound_events_%03d.csv",sid))
    stim_events = DataFrame(load(event_file))

    target_times = convert(Array{Float64},
        stim_info["test_block_cfg"]["target_times"][stim_events.sound_index])

    # derrived columns
    stim_events[:target_time] = target_times
    stim_events[:target_present] = target_times .> 0
    stim_events[:correct] = stim_events.target_present .==
        (stim_events.response .== 2)
    stim_events[:,:bad_trial] = convert.(Bool,stim_events.bad_trial)

    stim_events, sid
end

const envelopes = Dict{Tuple{Int,Int},Vector{Float64}}()
function load_speaker(events,tofs,info,stim_i,source_i;envelope_method=:rms)
    stim_num = events.sound_index[stim_i]
    load_speaker_(events,tofs,info,stim_num,source_i,envelope_method)
end

function load_speaker_(events,tofs,info,stim_num,source_i,envelope_method)
    if stim_num ∈ keys(envelopes)
        envelopes[stim_num]
    else
        x,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
            @sprintf("trial_%02d_%1d.wav",stim_num,source_i)))
        if size(x,2) > 1
            x = sum(x,dims=2)
        end
        result = find_envelope(SampleBuf(x,fs),tofs,method=envelope_method)
        envelopes[(stim_num,source_i)] = result
        result
    end
end

function load_other_speaker(events,tofs,info,stim_i,source_i;
    envelope_method=:rms)

    stim_num = events.sound_index[stim_i]
    sentences = info["test_block_cfg"]["trial_sentences"][:,source_i]
    # randomly select one of the stimuli != stim_i
    selected = rand(vcat(1:stim_num-1,stim_num+1:length(sentences)))

    load_speaker_(events,tofs,info,selected,source_i,envelope_method)
end

function folds(k,indices)
    len = length(indices)
    fold_size = cld(len,k)
    map(1:k) do fold
        test = indices[((fold-1)fold_size+1) : (min(len,fold*fold_size))]
        train = setdiff(indices,test)

        (train,test)
    end
end

function only_switches(switches,max_time;window=(-0.250,0.250))
    result = Array{Tuple{Float64,Float64}}(undef,length(switches))

    i = 0
    stop = 0
    for switch in switches
        new_stop = min(switch+window[2],max_time)
        if stop < switch+window[1]
            i = i+1
            result[i] = (switch+window[1],new_stop)
        elseif i > 0
            result[i] = (result[i][1], new_stop)
        else
            i = i+1
            result[i] = (0,new_stop)
        end
        stop = new_stop
    end

    view(result,1:i)
end


function remove_switches(switches,max_time;wait_time=0.5)
    result = Array{Tuple{Float64,Float64}}(undef,length(switches)+1)

    start = 0
    i = 0
    for switch in switches
        if start < switch
            i = i+1
            result[i] = (start,switch)
        end
        start = switch+wait_time
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end

    view(result,1:i)
end
