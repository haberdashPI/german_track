using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack
using AxisArrays
using PlotAxes
using DataFramesMeta
using Random
using Statistics

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(), ASEnvelope())

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end

const speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])
const tindex = Dict("male" => 1, "fem" => 2)
const cond_label = Dict("object" => "object", "global" => "test", "spatial" => "feature")
const direction = convert(Array{String},
    stim_info["test_block_cfg"]["trial_target_dir"])

listen_conds = ["object","global", "spatial"]
targets = ["male","fem"]
labels = ["correct","all"]

train_conditions = map(targets) do target
    target =>
        @λ(_row.correct &&
            speakers[_row.sound_index] == tindex[target] ?
                before_target[_row.sound_index] : no_indices)
end

test_conditions = mapreduce(vcat,listen_conds) do condition
    mapreduce(vcat,targets) do target
        [join((condition,target),"_") =>
            @λ(_row.condition == cond_label[condition] &&
                speakers[_row.sound_index] == tindex[target] ?
                    before_target[_row.sound_index] : no_indices)]
    end
end

# weight based on known "attended" source
function weightfn(row)
    if row.condition == "test" # global (attend all)
        1
    elseif row.condition == "object" # object (attend male)
        repeat([2.0,1.0,1.0],outer=2)'
    elseif row.condition == "feature" # spatial (attend right)
        target_speaker = speakers[row.sound_index]
        target_dir = direction[row.sound_index]
        if (target_speaker == 1 && target_dir == "left") ||
           (target_speaker == 2 && target_dir == "right")
            repeat([1.0,2.0,1.0],outer=2)'
        elseif (target_speaker == 1 && target_dir == "right") ||
               (target_speaker == 2 && target_dir == "left")
            repeat([2.0,1.0,1.0],outer=2)'
        else
            @assert target_speaker == -1
            1 # NOTE: this is not strictly correct, but since
            # we're setting all non-target trials to have `no_indices`
            # (thus excluding them from training), it doesn't matter
            # what we write here
        end
    end
end

function measures(pred,stim)
    (joint_cor = cor(vec(pred),vec(stim)),
     male_cor = cor(vec(pred[:,1:2]),vec(stim[:,1:2])),
     fem1_cor = cor(vec(pred[:,3:4]),vec(stim[:,3:4])),
     fem2_cor = cor(vec(pred[:,5:6]),vec(stim[:,5:6])))
end

df, models = train_test(
    K = 20,
    StaticMethod(NormL2(0.2),measures),
    SpeakerStimMethod(
        encoding=encoding,
        sources=["male-fem1-fem2","male-fem1-fem2_other"]),
    weightfn = weightfn,
    resample = 64,
    eeg_files, stim_info,
    maxlag=0.8,
    return_models = true,
    train = repeat(train_conditions,inner=3),
    test = test_conditions
);
alert()
