export stablehash, stableRNG

# These methods make random number generation well defined and reproduciable
# across julia versions

const crc32 = crc(CRC_32)

struct UseWrite; end
hashwrite(io, x, ::UseWrite) = write(io, x)

struct UseIterate; end
function hashwrite(io, x, ::UseIterate)
    for el in x
        hashwrite(io, el)
    end
end

struct UseProperties; end
function hashwrite(io, x, ::UseProperties)
    for key in propertynames(x)
        hashwrite(io, key)
        hashwrite(io, getproperty(x, key))
    end
end

struct UseStringify; end
function hashwrite(io, x, ::UseStringify)
    str = string(x)
    if startswith(str, "#")
        error("Unnamed function objects cannot be hashed to a reliable value")
    end
    hashwrite(io, str)
end

hashmethod(x) = UseWrite()
hashmethod(x::Union{Tuple,Array}) = UseIterate()
hashmethod(x::NamedTuple) = UseProperties()
hashmethod(x::Function) = UseStringify()

hashwrite(io, x) = hashwrite(io, x, hashmethod(x))

function stablehash(obj...)
    io = IOBuffer()
    hashwrite(io,obj)
    crc32(take!(io))
end

function stableRNG(obj...)
    RandomNumbers.Xorshifts.Xoroshiro128Plus(stablehash(obj))
end
