
using DrWatson
@quickactivate("german_track")

using GermanTrack, Statistics, Underscores, DataFrames
import GermanTrack: stim_info, speakers, directions, target_times, switch_times

med_target_time = @_ filter(_ > 0, target_times) |> median

targetdf = DataFrame(time = target_times, switches = switch_times, row = 1:50)
early_switch_targets = @_ filter(_.time > 0, targetdf) |>
    filter(sum(_1.time .> _1.switches) <= 2,__)
early_time_targets = @_ filter(_.time > 0,targetdf) |>
    filter(_.time < med_target_time,__)

