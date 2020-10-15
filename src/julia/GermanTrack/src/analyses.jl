export select_windows, shrinktowards, ishit

"""
    wmean(vals, weights, [default = one(eltype(vals))/2])

Compute a weighted mean, treating missing values as `default`.
"""
wmean(x, w, default = one(eltype(x))/2) =
    iszero(sum(w)) ? default : mean(coalesce.(x, default), weights(w))

"""
    shrinktowards([x],mu;by=0.01)

Reduce the value of `x` towards `mu` by (the presumably small value) `by*(x-mu)`.
You can exclude the `x` argument to curry the function.
"""
shrinktowards(mu;by=0.01) = x -> shrinktowards(x,mu,by=by)
shrinktowards(x,mu;by=0.01) = (1-by)*(x-mu) + mu

"""
    spread([x,]scale,npoints,[indices=Colon()])

Create a spread of `npoints` values placed evenly along the Normal distribution
with a standard deviation of scale/2. Leaving out `x` returns curried function.
To select only a subset of points use the `indices` keyword argument.
"""
spread(scale,npoints;indices=Colon()) = x -> spread(x,scale,npoints,indices=indices)
spread(x,scale,npoints;indices=Colon()) =
    quantile.(Normal(x,scale/2),range(0.05,0.95,length=npoints)[indices])

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
    indices = @_ filter(!isempty(bounds[_]), keys(bounds)) |> collect |> sort!

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


"""
    ishit(row; kwds...)

Correctly interprets a given row of the data as a hit, correct rejection, false positive
or miss. Since the directions are different for each condition, how we interpret a an
indication of a detected target depends on the condition.
"""
function ishit(row; kwds...)
    vals = merge(row,kwds)
    if vals.target_present
        if vals.condition == "global"
            vals.reported_target ? "hit" : "miss"
        elseif vals.condition == "object"
            vals.target_source == "male" ?
                (vals.reported_target ? "hit" : "miss") :
                (vals.reported_target ? "falsep" : "reject")
        else
            @assert vals.condition == "spatial"
            vals.direction == "right" ?
                (vals.reported_target ? "hit" : "miss") :
                (vals.reported_target ? "falsep" : "reject")
        end
    else
        vals.reported_target ? "reject" : "falsep"
    end
end
