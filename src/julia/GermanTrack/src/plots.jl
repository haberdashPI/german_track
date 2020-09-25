function addpatterns(filename, patterns::Dict{String,Tuple{String,String}}; size=4)

    stripes = prod(patterns) do (name, colors)
        """
        <pattern id="$name" x="0" y="0" width="$(2size)" height="$(2size)" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect x="0" y="0" width="$size" height="$(2size)" style="stroke:none; fill:$(colors[1]);" />
            <rect x="$size" y="0" width="$size" height="$(2size)" style="stroke:none; fill:$(colors[2]);" />
        </pattern>
        """
    end


    stripes_xml = string("<def>",stripes,"</def>") |> parsexml |> __.node |>
        elements |> first |> unlink!
    vgplot = readxml(filename)
    @_ vgplot.root |> elements |> first |> linkprev!(__, stripes)
    open(io -> prettyprint(io, vgplot), joinpath(dir, filename), write = true)
end
