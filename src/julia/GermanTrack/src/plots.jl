export addpatterns, xmlpatterns

colorstr(x::String) = x
colorstr(x::Color) = "#"*hex(x)

"""
    xmlpatterns(patterns; size = 4)

Create a SVG `<defs>` section containing a series of diangonal striped patterns.
Each pattern is defined by a `name => (color1, color2)` pair, and `patterns` should
be an iterable of such pairs (e.g. a `Dict`). Stripes will be `size` pixels wide.
"""
function xmlpatterns(patterns::Dict{String,<:Any}; size=4)
    stripes = prod(patterns) do (name, colors)
        """
        <pattern id="$name" x="0" y="0" width="$(2size)" height="$(2size)"
                patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect x="0" y="0" width="$size" height="$(2size)" style="stroke:none;
                fill:$(colorstr(colors[1]));" />
            <rect x="$size" y="0" width="$size" height="$(2size)" style="stroke:none;
                fill:$(colorstr(colors[2]));" />
        </pattern>
        """
    end

    stripes_xml = @_ string("<defs>",stripes,"</defs>") |> parsexml |> __.node |>
        elements |> first |> unlink!
end

"""
    addpatterns(filename, patterns; size=4)

Use `xmlpatterns` to define some SVG patterns. Inject these patterns into the SVG
file located at `filename`.
"""
function addpatterns(filename, patterns::Dict{String,<:Any}; size=4)
    stripes_xml = xmlpatterns(patterns, size = size)
    vgplot = readxml(filename)
    @_ vgplot.root |> elements |> first |> linkprev!(__, stripes_xml)
    open(io -> prettyprint(io, vgplot), filename, write = true)
end

allcolors = ColorSchemes.lajolla[range(0.1, 0.85, length = 3*3)] |> reverse

"""
    neutral

The neurtral color used for "null" values in our plots
"""
neutral = allcolors[3]

"""
    darkgray

The darkgray used to represent the default condition (Global)
"""
darkgray = allcolors[1]

"""
    colors

The three colors used to distinguish between the three conditions (Global, Object & Spatial)
"""
colors = allcolors[[1, 5, 8]]

lightdark = allcolors[[1, 2, 5, 6, 8, 9]]

"""
    patterns

The patterns used to represent classification between: Global v. Object, Global v. Spatial
and Object v. Spatial.
"""
patterns = begin
    Dict(
        "mix1_2" => colors[1:2],
        "mix1_3"    => colors[[1,3]],
        "mix2_3"  => colors[2:3]
    )
end
