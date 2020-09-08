export select_windows

wmean(x,w) = iszero(sum(w)) ? 0.0 : mean(coalesce.(x,one(eltype(x))/2),weights(w))

"""
    select_windows(conditions, subjects)

Select windows of eeg data defined by a bounding function.

## Arugments

- conditions: a dictionary where each key is a named tuple describing the
specific condition windows will be associated with and each entry is a
function. When applied to the rows of the event data this function will
return the range of times in seconds (as a tuple) to be included or
no_indices (if no data applies for the given case)

- subjects: a dictionary containing the file => data pairs, each data entry
should contain a named tuple of eeg (for EEGData), event (for data frame of
events) and sid (a String) entries.

"""
function select_windows(conditions, subjects)
    reduce(vcat, select_windows_helper.(collect(conditions), Ref(subjects))) |>
        categorical!
end

function select_windows_helper((condition, boundfn), subjects)
    bounds = Dict((file, i) => bounds
        for file in keys(subjects)
        for (i, bounds) in enumerate(boundfn.(eachrow(subjects[file].events))))
    indices = filter(@λ(!isempty(bounds[_])), keys(bounds)) |> collect |> sort!

    if !isempty(indices)
        mapreduce(vcat, indices) do (file, i)
            eeg, events = subjects[file]
            start = bounds[(file, i)][1]
            ixs = bound_indices(bounds[(file, i)], 256, size(eeg[i], 2))

            # power in relevant frequency bins across all channels and times
            DataFrame(
                sid = sidfor(file),
                trial = i;
                condition...,
                window = [view(eeg[i], :, ixs)]
            )
        end
    else
        DataFrame()
    end
end
