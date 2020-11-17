export addpatterns, xmlpatterns, filereplace, urlcol

colorstr(x::String) = x
colorstr(x::Color) = "#"*hex(x)

"""
    xmlpatterns(patterns; size = 4)

Create a SVG `<defs>` section containing a series of diangonal striped patterns.
Each pattern is defined by a `name => (color1, color2)` pair, and `patterns` should
be an iterable of such pairs (e.g. a `Dict`). Stripes will be `size` pixels wide.
"""
function xmlpatterns(patterns::OrderedDict{String,<:Any}; size=4)
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
function addpatterns(filename, patterns::OrderedDict{String,<:Any}; size=4)
    stripes_xml = xmlpatterns(patterns, size = size)
    vgplot = readxml(filename)
    @_ vgplot.root |> elements |> first |> linkprev!(__, stripes_xml)
    open(io -> prettyprint(io, vgplot), filename, write = true)
end

colorat(i) = ColorSchemes.batlow[range(0.1, 0.9, length = 3*5)][i]
grayify(x) = let c = convert(LCHab, x)
    convert(RGB, LCHab(c.l, 0, 0))
end

"""
    neutral

The neurtral color used for "null" values in our plots
"""
neutral = grayify(colorat(5))

"""
    darkgray

The darkgray used to represent some annotation text
"""
darkgray = grayify(colorat(1))

"""
    colors

The three colors used to distinguish between the three conditions (Global, Object & Spatial)
"""
colors = colorat([1, 7, 12])

lightdark = colorat([1, 3, 7, 9, 12, 14])

"""
    patterns

The patterns used to represent classification between: Global v. Object, Global v. Spatial
and Object v. Spatial.
"""
patterns = begin
    OrderedDict(
        "mix1_2" => colorat([1,7]),
        "mix1_3" => colorat([1,12]),
        "mix2_3" => colorat([7,12])
    )
end

seqpatterns = begin
    OrderedDict(
        "stripe1" => colorat([1,3]),
        "stripe2" => colorat([7,9]),
        "stipre3" => colorat([12,14])
    )
end

urlcol(x) = "url(#$x)"

inpatterns = begin
    OrderedDict(
        "stripe1a" => colorat([1,2]),
        "stripe1b" => colorat([3,4]),
        "stripe2a" => colorat([7,8]),
        "stripe2b" => colorat([9,10]),
        "stripe3a" => colorat([12,13]),
        "stripe3b" => colorat([14,15]),
    )
end

"""
    filereplace(file, pair)

Apply `replace` to all lines of the given file.
"""
function filereplace(file, pairs...)
    lines = readlines(file)
    open(file, write = true) do stream
        for line in lines
            println(stream, replace(line, pairs...))
        end
    end
end

