
# can we tell a difference between decoding far from target and decoding close
# to targets does this depend on whether the participant responded correctly

using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

during_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(0,1.5))
end
before_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(-1.5,0))
end


df1 = train_stimuli(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(encoding=encoding),
    resample = 64,
    eeg_files, stim_info,
    # train = "all" => all_indices,
    train = "during_target" => row -> during_target[row.sound_index],
    skip_bad_trials = true,
)
alert()

df1[!,:test] = "during_target"

df2 = train_stimuli(
    StaticMethod(NormL2(0.2),cor),
    SpeakerStimMethod(encoding=encoding),
    resample = 65,
    eeg_files, stim_info,
    train = "during_target" => row -> during_target[row.sound_index],
    test = "before_target" => row -> before_target[row.sound_index],
    skip_bad_trials = true,
)
df2[!,:test] = "before_target"
df = vcat(df1,df2)
categorical!(df,:test)
