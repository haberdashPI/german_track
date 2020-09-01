export stablehash, stableRNG

# These methods help to make random number generation well defined and reproduciable across
# julia versions. Note: Julia can change the methods by which particular random sequences
# of more complex objects are created, so this is not *perfectly* stable.

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

"""
    stablehash(arg1, arg2, ...)

Create a stable hash of the given objects. You can customize how an object is hashed using
`hashmethod(::MyType)`. There are three methods: `UseWrite`, which writes the object to a
binary format and takes a hash of that, `UseIterate`, which assumes the object is iterable
and finds a hash of all elements, and `UseProperties` which assumed a struct of some type
and uses `propertynames` and `getproperty` to compute a hash of all fields.

"""
function stablehash(obj...)
    io = IOBuffer()
    hashwrite(io,obj)
    crc32(take!(io))
end

"""
    stableRNG(obj1, obj2, ...)

Use stablehash to find a seed value from the given objects and return a fast, reliable
random number generator.
"""

function stableRNG(obj...)
    RandomNumbers.Xorshifts.Xoroshiro128Plus(stablehash(obj))
end
