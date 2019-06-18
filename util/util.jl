using Colors
using Debugger
using VegaLite
using LambdaFn
using Debugger

import EEGCoding: AllIndices

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

function plotresponse(method,results,stim_events,trial)
    len = minimum(map(x -> length(x.probs),results))
    step = ustrip.(uconvert.(s,method.params.window))
    if stim_events.target_present[trial] == stim_events.correct[trial]
        DataFrame(start=0,stop=len*step) |> @vlplot() +
            @vlplot(:rect, x=:start, x2=:stop, opacity={value=0.15},
                color={value="#000"})
    end
end

function plottarget(stim_events,trial,stim_info)
    if stim_events.target_present[trial]
        stim_index = stim_events.sound_index[trial]
        start_time = stim_info["test_block_cfg"]["target_times"][stim_index]
        stop_time = stim_info["target_len"]+start_time

        target_speaker_i =
            stim_info["test_block_cfg"]["trial_target_speakers"][stim_index]
        target_speaker = target_speaker_i == 1 ? "male" :
            target_speaker_i == 2 ? "fem1" :
            "unknown"

        DataFrame(start=start_time,stop=stop_time,source=target_speaker) |>
            @vlplot() +
            @vlplot(:rect, x=:start, x2=:stop, color=:source,
                opacity={value=0.2})
    end
end

totimes(::AllIndices,step,len) = step.*((1:len) .- 1)
totimes(x::Tuple,step,len) =
    filter(@λ(inbound(_,x)),range(0,x[2],step=step))
totimes(x::Array{<:Tuple},step,len) =
    filter(@λ(inbounds(_,x)),range(0,maximum(getindex.(x,2)),step=step))

function plotatten(method,results,raw,bounds)
    len = minimum(map(x -> length(x.probs),results))
    step = ustrip.(uconvert.(s,method.params.window))
    t = totimes(bounds,step,len)
    @bp

    df = DataFrame()
    plot = if raw
        for result in results
            df = vcat(df,DataFrame(time=float(t),level=result.norms[1:len],
                source=result.source))
        end
        df |> @vlplot() + @vlplot(
            :line,
            x={:time,axis={title="Time (s)"}},
            y={:level,axis={title="Level"}},
            color=:source
        )
    else
        for result in results
            df = vcat(df,DataFrame(
                time=float(t),
                level=result.probs[1:len],
                source=result.source,
                lower=result.lower[1:len],
                upper=result.upper[1:len]
            ))
        end
        df |> @vlplot() +
            @vlplot(:area,
                x=:time,
                y=:lower,
                y2=:upper,opacity={value=0.3},
                color=:source
            ) + @vlplot(:line,
                encoding={
                    x={:time,axis={title="Time (s)"}},
                    y={:level,axis={title="Level"}}
                },
                color=:source
            )
    end

    plot
end


combine(x,y) =
    isnothing(x) ? y :
    isnothing(y) ? x :
    x + y
combine(x,y,z,more...) = reduce(combine,(x,y,z,more...))

inbound(x,::AllIndices) = true
inbound(x,(lo,hi)::Tuple) = lo <= x <= hi
inbound(x,bounds::Array{<:Tuple}) = inbound.(Ref(x),bounds)

function plotswitches(trial,bounds,stim_events)
    stim_index = stim_events.sound_index[trial]
    direc_file = joinpath(stimulus_dir,"mixtures","testing",
        @sprintf("trial_%02d.direc",stim_index))
    dirs = load_directions(direc_file)
    len = minimum(length.((dirs.dir1,dirs.dir2,dirs.dir3)))
    t = (1:len)./dirs.samplerate

    df = vcat(
        DataFrame(time=t,dir=dirs.dir1,source="male"),
        DataFrame(time=t,dir=dirs.dir2,source="fem1"),
        DataFrame(time=t,dir=dirs.dir3,source="fem2")
    )
    df[inbound.(df.time,Ref(bounds)),:] |> @vlplot(:line,
        height=60,
        x={:time,axis=nothing},
        y={:dir,axis={title="direciton (° Azimuth)"}},color=:source)
end

function plottrial(method,results,stim_info,file;raw=false,bounds=all_indices)
    trial = single(unique(map(r->r.trial,results)),
        "Expected single trial number")
    stim_events, = events_for_eeg(file,stim_info)
    bounds = bounds(stim_events[trial,:])
    attenplot = plotatten(method,results,raw,bounds)

    @vlplot(title=string("Trial ",trial),
        spacing = 15,bounds = "flush") +
    [
        plotswitches(trial,bounds,stim_events);
        combine(
            plotresponse(method,results,stim_events,trial),
            plottarget(stim_events,trial,stim_info),
            attenplot
        );
    ]
end

function channelattend(rows,stim_events,stim_info,fs)
    trial = single(unique(rows.trial),"Expected single trial number")
    sources = ["left","right","left_other"]
    @assert :source in names(rows)
    @assert all(rows.source .∈ Ref(sources))

    target_len = stim_info["target_len"]
    stim_index = stim_events.sound_index[trial]
    stim_fs = stim_info["fs"]

    if stim_events.target_present[trial]
        target_dir = stim_info["test_block_cfg"]["trial_target_dir"][stim_index]
        target_time = stim_info["test_block_cfg"]["target_times"][stim_index]
        start = clamp(floor(Int,target_time * fs),1,length(rows.probs[1]))
        stop = clamp(ceil(Int,(target_time + target_len) * fs),1,length(rows.probs[1]))

        @assert target_dir in ("left","right")
        other_dir = target_dir == "left" ? "right" : "left"

        ti,oi = indexin(rows.source, [target_dir,other_dir])
        @views begin
            t = rows[ti,:probs][start:stop]
            o = rows[oi,:probs][start:stop]
        end
        mean(t .> o)
    else
        0.0
    end
end

function speakerattend(rows,stim_events,stim_info,fs)
    trial = single(unique(rows.trial),"Expected single trial number")
    sources = ["male","fem1","fem2","male_other"]
    @assert :source in names(rows)
    @assert all(rows.source .∈ Ref(sources))

    target_len = stim_info["target_len"]
    stim_index = stim_events.sound_index[trial]
    stim_fs = stim_info["fs"]

    if stim_events.target_present[trial]
        target =
            stim_info["test_block_cfg"]["trial_target_speakers"][stim_index]
        target_time = stim_info["test_block_cfg"]["target_times"][stim_index]
        start = clamp(floor(Int,target_time * fs),1,length(rows.probs[1]))
        stop = clamp(ceil(Int,(target_time + target_len) * fs),1,length(rows.probs[1]))
        others = setdiff(1:3,target)
        ti,o1i,o2i = indexin(rows.source,map(i->sources[i],[target,others...]))
        @views begin
            t = rows[ti,:probs][start:stop]
            o1 = rows[o1i,:probs][start:stop]
            o2 = rows[o2i,:probs][start:stop]
        end
        mean((t .> o1) .& (t .> o2))
    else
        0.0
    end
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

const envelopes = Dict{Any,Vector{Float64}}()

function load_speaker(events,tofs,stim_i,source_i;envelope_method=:rms)
    stim_num = events.sound_index[stim_i]
    load_speaker_(tofs,stim_num,source_i,envelope_method)
end

function load_speaker_(tofs,stim_num,source_i,envelope_method)
    key = (:speaker,tofs,stim_num,source_i,envelope_method)
    if key ∈ keys(envelopes)
        envelopes[key]
    else
        x,fs = load(joinpath(stimulus_dir,"mixtures","testing","mixture_components",
            @sprintf("trial_%02d_%1d.wav",stim_num,source_i)))
        if size(x,2) > 1
            x = sum(x,dims=2)
        end
        result = find_envelope(SampleBuf(x,fs),tofs,method=envelope_method)
        envelopes[key] = result
        result
    end
end

function load_other_speaker(events,tofs,info,stim_i,source_i;
    envelope_method=:rms)

    stim_num = events.sound_index[stim_i]
    stimuli = info["test_block_cfg"]["trial_sentences"]
    sentence_num = stimuli[stim_num][source_i]
    selected = rand(filter(i -> stimuli[i][source_i] != sentence_num,
        1:length(stimuli)))

    load_speaker_(tofs,selected,source_i,envelope_method)
end

function load_channel(events,tofs,stim_i,source_i;envelope_method=:rms)
    stim_num = events.sound_index[stim_i]
    load_channel_(tofs,stim_num,source_i,envelope_method)
end

function load_channel_(tofs,stim_num,source_i,envelope_method)
    @assert source_i ∈ [1,2]
    key = (:channel,tofs,stim_num,source_i,envelope_method)
    if key ∈ keys(envelopes)
        envelopes[key]
    else
        x,fs = load(joinpath(stimulus_dir,"mixtures","testing",
            @sprintf("trial_%02d.wav",stim_num)))
        result = find_envelope(SampleBuf(x[:,source_i],fs),tofs,
            method=envelope_method)
        envelopes[key] = result
        result
    end
end

function load_other_channel(events,tofs,info,stim_i,source_i;
    envelope_method=:rms)

    stim_num = events.sound_index[stim_i]
    n = length(info["test_block_cfg"]["trial_sentences"])
    selected = rand(setdiff(1:n,stim_num))

    load_channel_(tofs,selected,source_i,envelope_method)
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
