using Distributed
using ClusterManagers

# TODO: this seems to work well, next step
# use the standard distributed API to run
# multiple EEG analyses
addprocs(SlurmManager(6), partition="CPU", t="00:5:00")

hosts = []
pids = []
for i in workers()
	host, pid = fetch(@spawnat i (gethostname(), getpid()))
	push!(hosts, host)
	push!(pids, pid)
end

# The Slurm resource allocation is released when all the workers have
# exited
for i in workers()
	rmprocs(i)
end
