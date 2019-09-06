
using DrWatson; quickactivate(@__DIR__,"german_track")
using GermanTrack

stim_info = JSON.parsefile(joinpath(stimulus_dir(),"config.json"))
eeg_files = filter(x -> occursin(r"_mcca34\.mcca_proj$",x),readdir(data_dir()))
eeg_files = eeg_files[1:1]

encoding = JointEncoding(PitchSurpriseEncoding(),ASEnvelope())

# gathering stimlus information
target_times =
    convert(Array{Float64},stim_info["test_block_cfg"]["target_times"])

during_target = map(target_times) do time
    iszero(time) ? no_indices : only_near(time,10,window=(0,1.5))
end
non_target = map(target_times) do time
    iszero(time) ? no_indices : not_near(time,10,window=(-0.5,1.5))
end

speakers = convert(Array{Int},
    stim_info["test_block_cfg"]["trial_target_speakers"])


# defining conditions
male_test_target = "male_test_target" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 1 ?
        during_target[_row.sound_index] : no_indices)

male_test_nontarget = "male_test_nontarget" =>
    @λ(_row.condition == "test" && speakers[_row.sound_index] == 1 ?
        non_target[_row.sound_index] : no_indices)

# THOUGHT: can we use z-scored correlation as a way of determining
# a overall, detectability score?

male_object_target = "male_object_target" =>
    @λ(_row.condition == "object" && speakers[_row.sound_index] == 1 ?
        during_target[_row.sound_index] : no_indices)

male_object_nontarget = "male_object_nontarget" =>
    @λ(_row.condition == "object" && speakers[_row.sound_index] == 1 ?
        non_target[_row.sound_index] : no_indices)

