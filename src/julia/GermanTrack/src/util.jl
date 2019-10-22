import EEGCoding: AllIndices
export clear_cache!, plottrial, events_for_eeg, alert, only_near,
    not_near, bound, sidfor

function mat2bson(file)
    file
end

function bound(x::AbstractRange;min=nothing,max=nothing)
    from = first(x)
    to = last(x)

    if !isnothing(min)
        from = Base.max(min,from)
        to = Base.max(min,to)
    end

    if !isnothing(max)
        from = Base.min(max,from)
        to = Base.min(max,to)
    end

    if x isa UnitRange
        from:to
    else
        from:step(x):to
    end
end

function clear_cache!()
    cachedir = EEGCoding.cache_dir()
    files = filter(@λ(endswith(_,"bson")),readdir(EEGCoding.cache_dir()))
    for file in files
        file = joinpath(cachedir,file)
        isfile(file) && rm(file)
    end
    @info "$(length(files)) files were removed from $cachedir."
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

# find mean within 1s start window for each speaker
function meanat(indices)
    function(xs)
        isempty(xs[1]) ? missing :
            mean(view(single(xs),clamp.(indices,1,length(xs[1]))))
    end
end

const eventcache = Dict{Int,DataFrame}()
function sound_index(sid,trial)
    events = get!(eventcache,sid) do
        events_for_eeg(sidfile(sid),stim_info)[1]
    end
    events.sound_index[trial]
end

function neartimes(from,to,times)
    function row2switch(sid,trial,norms)
        i = sound_index(sid,trial)
        if ismissing(times[i])
            return missing
        else
            start = clamp(round(Int,(from + times[i]*s)/method.params.window),1,length(norms))
            stop = clamp(round(Int,(to + times[i]*s)/method.params.window),1,length(norms))
            meanat(start:stop)((norms,))
        end
    end
    rows -> map(row2switch,rows.sid,rows.trial,rows.norms)
end

function read_mcca_proj(filename)
    # @info "Reading projected components"
    open(filename) do file
        # number of channels
        nchan = read(file,Int32)
        # channels names
        channels = Vector{String}(undef,nchan)
        for i in 1:nchan
            len = read(file,Int32)
            channels[i] = String(read(file,len))
        end
        # number of components
        ncomp = read(file,Int32)
        # components
        comp = Array{Float64}(undef,ncomp,nchan)
        read!(file,comp)
        # number of trials
        ntrials = read(file,Int32)
        # sample rate
        fs = read(file,Int32)
        # projected trials
        trials = Vector{Array{Float64}}(undef,ntrials)
        for i in 1:ntrials
            # trial size
            row = read(file,Int32)
            col = read(file,Int32)
            # trial
            trial = Array{Float64}(undef,row,col)
            read!(file,trial)
            trials[i] = trial #(trial'comp)'
        end

       EEGData(data=trials,label=channels,fs=fs)
    end
end

const subject_cache = Dict()
function load_subject(file,stim_info;encoding=RawEncoding(),samplerate=missing)
    if !isfile(file)
        error("File '$file' does not exist.")
    end

    stim_events, sid = events_for_eeg(file,stim_info)

    data = get!(subject_cache,(file,encoding,samplerate)) do
        # data = if endswith(file,".mat")
        #     mf = MatFile(file)
        #     get_mvariable(mf,:dat)
        data = if endswith(file,".bson")
            @load file data
            data
        elseif endswith(file,".mcca_proj")
            read_mcca_proj(file)
        else
            pat = match(r"\.(.+)$",file)
            if pat != nothing
                ext = pat[1]
                error("Unsupported data format '.$ext'.")
            else
                error("Unknown file format for $file")
            end
        end

        encode(data,samplerate,encoding)
    end

    (eeg=data, events=stim_events, sid=sid)
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

totuples(::AllIndices,len) = [(0,len)]
totuples((lo,hi)::Tuple,len) = [(lo,min(lo+len,hi))]
totuples(xs::Array{<:Tuple},len) = map(((lo,hi)) -> (lo,min(xs[1][1]+len,hi)),xs)

function plotresponse(method,results,bounds,stim_events,trial)
    len = minimum(map(x -> length(x.probs),results))
    step = ustrip.(uconvert.(s,method.params.window))
    bounds = totuples(bounds,len*step)
    if stim_events.target_present[trial] == stim_events.correct[trial]
        DataFrame(start=map(@λ(_[1]),bounds),stop=map(@λ(_[2]),bounds)) |> @vlplot() +
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
    filter(@λ(inbound(_,x)),range(0,x[2],step=step))[1:len]
totimes(x::Array{<:Tuple},step,len) =
    filter(@λ(inbounds(_,x)),range(0,maximum(getindex.(x,2)),step=step))[1:len]

function plotatten(method,results,raw,bounds)
    len = minimum(map(x -> length(x.probs),results))
    step = ustrip.(uconvert.(s,method.params.window))
    t = totimes(bounds,step,len)

    df = DataFrame()
    plot = if raw
        for result in results
            if !isempty(t)
                df = vcat(df,DataFrame(time=float(t),level=result.norms[1:len],
                    source=result.source))
            end
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
    direc_file = joinpath(stimulus_dir(),"mixtures","testing",
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
        x={:time,scale={domain=t[[1,end]]},axis=nothing},
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
            plotresponse(method,results,bounds,stim_events,trial),
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

timeline(nt;kwds...) = timeline(nt.target_time,nt.target_present,nt.correct;kwds...)
function timeline(target_time,target_present,correct;step=0.05,len=8)
    times = range(0,len,step=step)
    values = Array{Union{Int,Missing}}(missing,length(times))
    if target_present
        values[times .> target_time] .= correct
    end
    DataFrame(value=values, time=times)
end

function sidfor(file)
    pattern = r"eeg_response_([0-9]+)(_[a-z_]+)?([0-9]+)?(_unclean)?\.[a-z_]+$"
    matched = match(pattern,file)
    if isnothing(matched)
        error("Could not find subject id in filename '$file'.")
    end
    parse(Int,matched[1])
end

function events_for_eeg(file,stim_info)
    sid = sidfor(file)
    event_file = joinpath(data_dir(),@sprintf("sound_events_%03d.csv",sid))
    stim_events = DataFrame(CSV.File(event_file))

    target_times = convert(Array{Float64},
        stim_info["test_block_cfg"]["target_times"][stim_events.sound_index])

    # derrived columns
    stim_events[!,:target_source] = convert(Array{Float64},
        stim_info["test_block_cfg"]["trial_target_speakers"][stim_events.sound_index])
    stim_events[!,:target_time] = target_times
    stim_events[!,:target_present] .= target_times .> 0
    stim_events[!,:correct] .= stim_events.target_present .==
        (stim_events.response .== 2)
    if :bad_trial ∈ names(stim_events)
        stim_events[!,:bad_trial] = convert.(Bool,stim_events.bad_trial)
    else
        @warn "Could not find `bad_trial` column in file '$event_file'."
        stim_events[!,:bad_trial] .= false
    end
    stim_events[!,:sid] .= sid

    stim_events, sid
end


only_near(time::Number,max_time;kwds...) =
    only_near((time,),max_time;kwds...)[1]
function only_near(times,max_time;window=(-0.250,0.250))
    result = Array{Tuple{Float64,Float64}}(undef,length(times))

    i = 0
    stop = 0
    for time in times
        new_stop = min(time+window[2],max_time)
        if stop < time+window[1]
            i = i+1
            result[i] = (time+window[1],new_stop)
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

function not_near(times,max_time;window=(0,0.5))
    result = Array{Tuple{Float64,Float64}}(undef,length(times)+1)

    start = 0
    i = 0
    for time in times
        if start < time
            i = i+1
            result[i] = (start,time+window[1])
        end
        start = time+window[2]
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end

    view(result,1:i)
end

function alert(message="Done!")
    if Sys.isapple()
        run(`osascript -e 'display notification "'$message'" with title "Julia"'`)
    elseif Sys.iswindows()
        run(`cmd sg "%username%" $message`)
    else
        @info message
    end
end
