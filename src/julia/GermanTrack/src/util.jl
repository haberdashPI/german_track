import EEGCoding: AllIndices
export clear_cache!, plottrial, events_for_eeg, alert, only_near,
    not_near, bound, sidfor, subdict, padmeanpower, far_from,
    sample_from_ranges, testmodel, ishit, windowtarget, windowbaseline,
    computebands, organize_data_by, optparams, find_powerdiff,
    find_decoder_training_trials

using FFTW
using DataStructures
using PaddedViews
using StatsBase
using Random
using Underscores
using StatsBase
using Lasso
using GLM
using ScikitLearn
using BlackBoxOptim
using Transducers
using BangBang
using IntervalSets
using Random

function mat2bson(file)
    file
end

function find_decoder_training_trials(subject,trial;eeg_sr,final_sr,target_samples)
    result = Empty(DataFrame)
    si = subject.events.sound_index[trial]

    # the labeled segment
    target_time = target_times[si]
    target_stop = target_time + target_samples/final_sr
    condition = subject.events.condition[trial]
    target_source = subject.events.target_source[trial]
    if target_time > 0

        result = append!!(result, DataFrame(
            start = max(1,round(Int,target_time*final_sr)),
            target = true,
            len = target_samples;
            trial = trial,
            (subject.events[trial,[:condition,:sid]])...
        ))
    end

    si = subject.events.sound_index[trial]
    n = size(subject.eeg[trial],2)
    segments = far_from(sort([target_times[si]; switch_times[si]]),n/eeg_sr,
        mindist=0.25,minlength=0.3)
    for (start,stop) in segments
        if target_time == 0 || isempty(intersect(start..stop,target_time..target_stop))
            start_sample = max(1,round(Int,start*final_sr))
            result = append!!(result, DataFrame(
                start = start_sample,
                target = false,
                len = min(n,round(Int,stop * final_sr) - start_sample + 1);
                trial = trial,
                (subject.events[trial,[:condition,:sid]])...
            ))
        end
    end

    result
end

function apply(by::NamedTuple, vals)
    apply_(by::Tuple{}, vals::Tuple{}) = ()
    apply_(by::Tuple, vals::Tuple) =
        (by[1](vals[1]), apply_(Base.tail(by),Base.tail(vals))...)
    NamedTuple{keys(by)}(apply_(Tuple(by), Tuple(vals)))
end
applyer(params, by::NamedTuple) = by
applyer(params, by::Function) = NamedTuple{keys(params)}(fill(by,length(params))...)

function optparams(objective,param_range;by=identity,kwds...)
    by = applyer(param_range, by)

    options = (
        SearchRange = collect(values(param_range)),
        NumDimensions = length(param_range),
        TraceMode = :silent,
        kwds...
    )

    opt = bboptimize(;options...) do params
        objective(apply(by,params))
    end
    best_candidate(opt), best_fitness(opt)
end

function testmodel(sdf,model,idcol,classcol,cols;n_folds=10,kwds...)
    # nprog = 0
    # progress = Progress(n_folds*max_evals)

    xs = term(0)
    for col in propertynames(view(sdf,:,cols))
        xs += term(col)
    end
    formula = term(classcol) ~ xs
    result = Empty(DataFrame)

    for (i, (trainids,testids)) in enumerate(folds(n_folds,unique(sdf[:,idcol])))
        isempty(trainids) && continue
        train = @_ filter(_[idcol] in trainids,sdf)
        test = @_ filter(_[idcol] in testids,sdf)

        f = apply_schema(formula, schema(formula, sdf))
        y,X = modelcols(f, train)

        # if @_ all(first(y) == _,y)
        #     labels = fill(first(y),size(test,1))
        #     append!(result, DataFrame(
        #         label = labels,
        #         correct = labels .== test[:,classcol],
        #         fitness = Inf,
        #         classcol => test[:,classcol],
        #         apply(by,fill(NaN,length(param_range)))...
        #     ))
        # end

        coefs = ScikitLearn.fit!(model,X,vec(y);kwds...)

        y,X = modelcols(f, test)
        level = ScikitLearn.predict(coefs,X)
        _labels = f.lhs.contrasts.levels[round.(Int,level).+1]

        keepcols = propertynames(view(sdf,:,Not(cols)))
        result = append!!(result, DataFrame(
            label = _labels,
            correct = _labels .== test[:,classcol];
            (keepcols .=> eachcol(test[:,keepcols]))...
        ))
    end

    result
end

function computebands(signal,fs;channels=1:30,freqbins=OrderedDict(
        :delta => (1,3),
        :theta => (3,7),
        :alpha => (7,15),
        :beta => (15,30),
        :gamma => (30,100)))

    function freqrange(spect,(from,to))
        freqs = range(0,fs/2,length=size(spect,2))
        view(spect,:,findall(from-step(freqs)*0.51 .≤ freqs .≤ to+step(freqs)*0.51))
    end

    if size(signal,2) < 32
        Empty(DataFrame)
    end
    if size(signal,2) < 128
        newsignal = similar(signal,size(signal,1),128)
        newsignal[:,1:size(signal,2)] = signal
        newsignal[:,(size(signal,2)+1):end] .= 0
        signal = newsignal
    end
    spect = abs.(rfft(signal, 2))
    # totalpower = mean(spect,dims = 2)
    result = mapreduce(hcat,keys(freqbins)) do bin
        mfreq = mean(freqrange(spect, freqbins[bin]), dims = 2) #./ totalpower
        DataFrame(Symbol(bin) => vec(mfreq))
    end
    result[!,:channel] .= channels

    result
end

function windowtarget(trial,event,fs,from,to)
    window = only_near(event.target_time,fs,window=(from,to))

    start = max(1,round(Int,window[1]*fs))
    stop = min(round(Int,window[2]*fs),size(trial,2))
    view(trial,:,start:stop)
end

const baseline_seed = 2017_09_16
function windowbaseline(trial,event,sid,trialnum,fs,from,to;mindist,minlen)
    si = event.sound_index
    times = vcat(switch_times[si], target_times[si]) |> sort!
    ranges = far_from(times, 10, mindist=mindist, minlength=minlen)
    if isempty(ranges)
        error("Could not find any valid region for baseline ",
              "'target'. Times: $(times)")
    end
    at = sample_from_ranges(MersenneTwister(hash((baseline_seed,sid,trialnum))),ranges)
    window = only_near(at,fs,window=(from,to))

    maxlen = size(trial,2)
    ixs = bound_indices(window,fs,maxlen)
    view(trial,:,ixs)
end

function ishit(row)
    if row.condition == "global"
        row.region == "baseline" ? "baseline" :
            row.correct ? "hit" : "miss"
    elseif row.condition == "object"
        row.region == "baseline" ? "baseline" :
            row.target_source == "male" ?
                (row.correct ? "hit" : "miss") :
                (row.correct ? "falsep" : "reject")
    else
        @assert row.condition == "spatial"
        row.region == "baseline" ? "baseline" :
            row.direction == "right" ?
                (row.correct ? "hit" : "miss") :
                (row.correct ? "falsep" : "reject")
    end
end

function find_powerdiff(subjects;kwds...)
    organize_data_by(subjects;kwds...) do windows,timings,counts,fs
        function windowband(signal,timing)
            freqdf = computebands(signal,fs)
            if @_ all(0 ≈ _,signal)
                freqdf[:,Between(:delta,:gamma)] .= 0
            end
            freqdf[!,:window_timing] .= timing
            freqdf
        end
        freqdf = foldl(append!!,MapSplat(windowband), zip(windows,timings))

        if size(freqdf,1) > 0
            powerdf = @_ freqdf |>
                stack(__, Between(:delta,:gamma),
                    variable_name = :freqbin, value_name = :power) |>
                filter(all(!isnan,_.power), __) |>
                unstack(__, :window_timing, :power)

            ε = 1e-8
            logdiff(x,y) = log.(ε .+ x) .- log.(ε .+ y)
            powerdiff = logdiff(powerdf.after,powerdf.before)

            chstr = @_(map(@sprintf("%02d",_),powerdf.channel))
            features = Symbol.("channel_",chstr,"_",powerdf.freqbin)
            DataFrame(weight=minimum(counts);(features .=> powerdiff)...)
        else
            Empty(DataFrame)
        end
    end
end

resolvetime(target,x::Number) = x
resolvetime(target,fn::Function) = fn(target)

function organize_data_by(fn,subjects;groups,windows,hittypes,
    chunk_size=1,before_reflect=true,regions = ["target", "baseline"])

    # @assert Threads.nthreads() > 1 "Run this method with JULIA_NUM_THREADS>1"
    if @_ any(_  ∉ ["target", "baseline"],regions)
        error("Only allowed regions are 'target' and 'baseline'.")
    end

    fs = GermanTrack.framerate(first(values(subjects)).eeg)

    window_timings = ["before", "after"]
    source_names = ["male", "female"]

    N = reduce(*,length.((values(subjects),regions,windows)))
    progress = Progress(N,desc="computing frequency bins")

    med_salience = median(target_salience)
    med_target_time = @_ filter(_ > 0,target_times) |> median

    function assemble_subject(subject)
        events = subject.events
        events.row = 1:size(events,1)
        eeg = subject.eeg

        function build_subject_data(region,window)
            rowdf = @_ filter(_.target_present == 1,events)
            si = rowdf.sound_index
            rowdf.target_source = get.(Ref(source_names),Int.(rowdf.target_source),missing)
            rowdf.salience = @. ifelse(target_salience[si] > med_salience,"high","low")
            rowdf.target_time = @. ifelse(target_times[si] > med_target_time,"early","late")
            rowdf.direction = directions[si]
            rowdf[!,:region] .= region
            rowdf[!,:winlen] .= window.len
            rowdf[!,:winstart] .= window.start
            rowdf[!,:winbefore] .= resolvetime.(target_times[si],window.before)
            rowdf[!,:chunk] = round.(Int,rowdf.sound_index / chunk_size)
            rowdf.hit = ishit.(eachrow(rowdf))
            rowdf = @_ filter(_.hit ∈ hittypes,rowdf)

            categorical!(rowdf,[:region,:condition,:salience,
                :target_time,:direction,:hit],compress=true)

            cols = [:sid,:hit,:condition,:winlen,:winstart,:region,groups...]

            resultdf = combine(groupby(rowdf,cols)) do sdf
                signals_and_counts = map(window_timings) do window_timing
                    ws = map(eachrow(sdf)) do row
                        bounds = window_timing != "after" ?
                            (row.winstart, row.winstart + row.winlen) :
                            (row.winbefore, row.winbefore + row.winlen)

                        rindex = row.row
                        region == "target" ?
                            windowtarget(eeg[rindex],events[rindex,:],fs,bounds...) :
                            windowbaseline(eeg[rindex],events[rindex,:],
                                sdf.sid[1],row.trial,fs,bounds...,mindist=0.2,minlen=0.5)
                    end
                    reduce(hcat,ws), sum(!isempty,ws)
                end
                signals = getindex.(signals_and_counts,1)
                counts = getindex.(signals_and_counts,2)

                result = fn(signals,window_timings,counts,fs)
                @infiltrate any(isinf,Array(result))
                @infiltrate any(isnan,Array(result))
                result
            end

            next!(progress)
            if isempty(resultdf)
                Empty(DataFrame)
            else
                resultdf
            end
        end

        foldl(append!!,MapSplat(build_subject_data),
            Iterators.product(regions,windows))
    end
    foldl(append!!,Map(assemble_subject),values(subjects))
end

function padmeanpower(xs)
    rows = maximum(@λ(size(_,1)), xs)
    cols = maximum(@λ(size(_,2)), xs)
    # power is always postive so we treat -1 as a missing value
    # and compute the mean over non-missing values; `PaddedView`
    # does not support `missing` values.
    padded = map(@λ(PaddedView(-1, _x, (rows, cols))), xs)
    μ = zeros(rows, cols)
    for pad in padded
        μ .= ifelse.(pad .>= 0,μ .+ pad,μ)
    end
    μ
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

function subdict(dict,keys)
    (k => dict[k] for k in keys)
end

function read_eeg_binary(filename)
    open(filename) do file
        # number of channels
        nchan = read(file,Int32)
        # channels names
        channels = Vector{String}(undef,nchan)
        for i in 1:nchan
            len = read(file,Int32)
            channels[i] = String(read(file,len))
        end
        # number of trials
        ntrials = read(file,Int32)
        # sample rate
        fs = read(file,Int32)
        # trials
        trials = Vector{Array{Float64}}(undef,ntrials)
        for i in 1:ntrials
            # trial size
            row = read(file,Int32)
            col = read(file,Int32)
            # trial
            trial = Array{Float64}(undef,row,col)
            read!(file,trial)
            trials[i] = trial
        end

       EEGData(data=trials,label=channels,fs=fs)
    end
end

function read_mcca_proj(filename)
    @info "Reading projected components"
    open(filename) do file
        # number of channels
        nchan = read(file,Int32)
        if nchan > 4096
            error("Unexpected number of channels: $(nchan)")
        end
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
function load_subject(file,stim_info;encoding=RawEncoding(),framerate=missing)
    if !isfile(file)
        error("File '$file' does not exist.")
    end

    stim_events, sid = events_for_eeg(file,stim_info)

    data = get!(subject_cache,(file,encoding,framerate)) do
        # data = if endswith(file,".mat")
        #     mf = MatFile(file)
        #     get_mvariable(mf,:dat)
        data = if endswith(file,".bson")
            @load file data
            data
        elseif endswith(file,".mcca_proj") || endswith(file,".mcca")
            read_mcca_proj(file)
        elseif endswith(file,".eeg")
            read_eeg_binary(file)
        else
            pat = match(r"\.([^\.]+)$",file)
            if pat != nothing
                ext = pat[1]
                error("Unsupported data format '.$ext'.")
            else
                error("Unknown file format for $file")
            end
        end

        encode(data,framerate,encoding)
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


plotcombine(x,y) =
    isnothing(x) ? y :
    isnothing(y) ? x :
    x + y
plotcombine(x,y,z,more...) = reduce(plotcombine,(x,y,z,more...))

inbound(x,::AllIndices) = true
inbound(x,(lo,hi)::Tuple) = lo <= x <= hi
inbound(x,bounds::Array{<:Tuple}) = inbound.(Ref(x),bounds)

function plotswitches(trial,bounds,stim_events)
    stim_index = stim_events.sound_index[trial]
    direc_file = joinpath(stimulus_dir(),"mixtures","testing",
        @sprintf("trial_%02d.direc",stim_index))
    dirs = load_directions(direc_file)
    len = minimum(length.((dirs.dir1,dirs.dir2,dirs.dir3)))
    t = (1:len)./dirs.framerate

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
        plotcombine(
            plotresponse(method,results,bounds,stim_events,trial),
            plottarget(stim_events,trial,stim_info),
            attenplot
        );
    ]
end

function channelattend(rows,stim_events,stim_info,fs)
    trial = single(unique(rows.trial),"Expected single trial number")
    sources = ["left","right","left_other"]
    @assert :source in propertynames(rows)
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
    @assert :source in propertynames(rows)
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
    framerate::Float64
end

function load_directions(file)
    open(file,read=true) do stream
        framerate = read(stream,Float64)
        len1 = read(stream,Int)
        len2 = read(stream,Int)
        len3 = read(stream,Int)

        dir1 = reinterpret(Float64,read(stream,sizeof(Float64)*len1))
        dir2 = reinterpret(Float64,read(stream,sizeof(Float64)*len1))
        dir3 = reinterpret(Float64,read(stream,sizeof(Float64)*len1))

        @assert length(dir1) == len1
        @assert length(dir2) == len2
        @assert length(dir3) == len3

        Directions(dir1,dir2,dir3,framerate)
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

function sidfor(filepath)
    file = splitdir(filepath)[2]
    pattern = r"eeg_response_([0-9]+)(_[a-z_]+)?([0-9]+)?(_unclean)?\.[a-z_]+$"
    matched = match(pattern,file)
    if isnothing(matched)
        pattern = r"([0-9]+).*\.[a-z_]+$"
        matched = match(pattern,file)
        if isnothing(matched)
            error("Could not find subject id in filename '$file'.")
        end
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
    if :bad_trial ∈ propertynames(stim_events)
        stim_events[!,:bad_trial] = convert.(Bool,stim_events.bad_trial)
    else
        @warn "Could not find `bad_trial` column in file '$event_file'."
        stim_events[!,:bad_trial] .= false
    end
    stim_events[!,:sid] .= sid
    stim_events.condition = stim_events.condition

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

function far_from(times,max_time;mindist=0.5,minlength=0.5)
    result = Array{Tuple{Float64,Float64}}(undef,length(times)+1)
    start = 0
    i = 0
    for time in times
        if start < time
            if time-mindist-minlength > start
                i = i+1
                result[i] = (start,time-mindist)
            end
            start = time + mindist
        end
    end
    if start < max_time
        i = i+1
        result[i] = (start,max_time)
    end
    view(result,1:i)
end

sample_from_ranges(ranges) = sample_from_ranges(Random.GLOBAL_RNG,ranges)
function sample_from_ranges(rng,ranges)
    weights = Weights(map(x -> x[2]-x[1],ranges))
    range = StatsBase.sample(rng,ranges,weights)
    rand(Distributions.Uniform(range...))
end
