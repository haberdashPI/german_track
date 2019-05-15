using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
Pkg.instantiate()

using JSON

if !isfile("install.json")
    error("Could not find install.json: see README.md")
elseif !isdir("data")
    data_dir = JSON.parsefile("install.json")["data"]
    symlink(data_dir,"data")
    @info "The folder `data` now links to $data_dir."
else
    @info "Directory `data` has already been created."
end

# generate the workspace file
str = """
{
	"folders": [
		{
			"path": "."
		},
		{
			"path": "$(joinpath(Pkg.devdir(),"EEGCoding"))"
		}
	],
	"settings": {},
	"extensions": {
		"recommendations": [
			"ikuyadeu.r",
			"julialang.language-julia",
			"colinfang.markdown-julia",
			"davidanson.vscode-markdownlint",
            "haberdashPI.terminal-polyglot",
            "gimly81.matlab",
			"haberdashpi.matlab-in-julia"
		]
	}
}
"""
open("german_track.code-workspace",write=true) do f
  write(f,str)
end

