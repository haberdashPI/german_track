# add DrWatson to root environment
using Pkg
pkg"activate"
pkg"add DrWatson"

using DrWatson
ENV["R_HOME"] = "*" # use Conda to install R (comment this line out if you want to use your own installation)
mv("Project.toml","Project.toml.backup")
cp("Project.toml.install","Project.toml")
cp("Manifest.toml.local","Manifest.toml")
quickactivate(@__DIR__,"german_track")

pkg"dev src/julia/GermanTrack"
pkg"dev src/julia/EEGCoding"
Pkg.instantiate()

using Conda
Conda.add("r-ggplot2",channel="r")
Conda.add("r-dplyr",channel="r")
Conda.add("r-cowplot",channel="biobuilds")
Conda.add("r-hmisc",channel="r")
