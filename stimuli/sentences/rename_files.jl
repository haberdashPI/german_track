#!/usr/local/bin/julia
using Printf

# the irregular length of file numbering leads to a confusing sort order for the
# files: change them to all have the same length strings by inserting 0's.

# NOTE: this only *prints* the shell commands to move the files, once these look
# good, pipe them to `sh`
for file in readdir(@__DIR__)
    pattern = match(r"([0-9]+).output.wav",file)
    if !isnothing(pattern)
        number = parse(Int,pattern.captures[1])
        str_number = @sprintf("%04d",number)
        repaired = replace(file,r"([0-9]+).output.wav" => str_number*".output.wav")
        println("mv $file $repaired")
    end
end
