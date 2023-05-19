using ArgParse

function argparse()
    s=ArgParseSettings()
    @add_arg_table s begin
        "--julia-project", "-j"
            help="The Julia project to run this script with"
            arg_type=String
        "path"
            help="Relative path to the simulation directory to initialise"
    end
    return parse_args(s)
end

args=argparse()

script_dir=pwd()
full_path=joinpath([script_dir, string(args["path"])])
mkdir(full_path)
if !isfile("full_simulation.jl")
    throw(ErrorException("This script should be executed from the same directory as full_simulation.jl and jobscript_slurm.sh"))
end

open(full_path*"/full_simulation.jl", "w") do sim_file
    for line in readlines(script_dir*"/full_simulation.jl")
        if occursin("%%simulation_functions", line)
            println(sim_file,replace(line, "%%simulation_functions" => script_dir*"/initial_distribution.jl"))
        else
            println(sim_file, line)
        end
    end
end

cluster_info=Dict(
    "nodes" => 4,
    "tasks" => 1,
    "time" => "02:00:00",
    "mem" => 3700,
    "cpus" => 48,
)

open(full_path*"/jobscript_slurm.sh", "w") do job_file
    replacements=Dict(
        "%%output" => full_path*"/%j.out",
        "%%fullpath" => full_path,
        "%%project" => haskey(args, "julia-project") ? args["julia-project"] : "",
    )        
    for (k,v) in cluster_info
        push!(replacements, "%%"*k => string(v))
    end
    for line in readlines(script_dir*"/jobscript_slurm.sh")
        println(job_file, replace(line, replacements...))
    end
end

@info "Finished setting up simulation in "*full_path