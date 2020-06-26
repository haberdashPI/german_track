# add DrWatson to root environment
using Pkg
pkg"activate"
pkg"add DrWatson"

using DrWatson
cp("Manifest.toml.local","Manifest.toml")
quickactivate(@__DIR__,"german_track")
Pkg.instantiate()

using Conda
Conda.add("r-ggplot2",channel="r")
Conda.add("r-dplyr",channel="r")
Conda.add("r-cowplot",channel="biobuilds")
Conda.add("r-hmisc",channel="r")
