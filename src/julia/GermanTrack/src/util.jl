
"""
rcat(rarray)

Concatenate an array of arrays along all dimensions.
"""
function rcat(rarray)
    @assert eltype(rarray) <: AbstractArray "Elements must be arrays"

    unsqueeze(x, nd) = reshape(x, size(x)..., ones(Int, nd - ndims(x))...)
    nd = max(ndims(rarray), reduce(max, ndims.(rarray)))
    rarray = unsqueeze(unsqueeze.(rarray, nd), nd)
    dim1(n, nd) = Tuple(n == k ? 1 : 0 for k in 1:nd)

    offsets = zip(
        (cumsum(lag(size.(rarray, d), dim1(d, nd), default = 0), dims = d)
        for d in 1:nd)...
    ) |> collect

    dimsums = [unique(sum(size.(rarray, i), dims = i)) for i in 1:nd]

    @assert all(==(1), length.(dimsums)) "Ragged arrary lengths not permitted."

    result = similar(rarray[1], only.(dimsums)...)

    for I in CartesianIndices(rarray)
        to = rarray[I]
        indices = @_ map(_1 .+ _2, [axes(to, i) for i in 1:nd], offsets[I])
        result[indices...] = to
    end

    result
end
