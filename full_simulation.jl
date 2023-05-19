using AmmoniaSingletsModel
using ArgParse
using DiffEqBase
using Distributed
using JLD2
using Libdl: DL_LOAD_PATH
using NQCDLoadbalancer
using NQCDynamics
using SlurmClusterManager
using Unitful

function argparse()
    s=ArgParseSettings()
    @add_arg_table s begin
        "--julia-project"
            help="The Julia project to run this script with"
            arg_type=String
            "--slurm"
            help="include if running on a slurm cluster"
            action=:store_true
    end
    return parse_args(s)
end
args=argparse()
if args["slurm"]
    # Initialise multiprocessing/multithreading
    slurm_info=SlurmManager()
    addprocs(slurm_info, exeflags="--project="*args["julia-project"])
end
@everywhere include(%%simulation_functions);


# Set fixed parameters
fixed_parameters=Dict(
    "schnet_model" => "path_to_schnet_dir", # SchNet model path
    "starting_structure" => "path_to_file", # Path of input structure for an ase.read call
    "trajectories" => 10, #Trajectories for RPMD
    "runtime" => 5.0u"ps", # Runtime for RPMD
    "timestep" => 0.5u"fs", # Timestep 
    "ensemble_algorithm" => EnsembleThreads(), # Parallelisation method to use
    "saveat" => 10u"fs", # Sampling time interval for RPSH
    "gamma" => 1.0, 
    "equilibration_time" => 50u"fs", # Initial part of a trajectory to discard
    "model_type" => "schnet"
    "adiabatic_state_number" => 2, # Run in the first excited state
    "model_type" => "schnet",
    "task" => "langevin",
    "temperature" => 300u"K",
)


# Set variables

variables=Dict(
    "temperature" => collect(300u"K"),
)
# Total trajectories to run: 1
# Estimated time per trajectory: unknown

job_queue=build_job_queue(fixed_parameters, variables, input_preprocess)

#! Driver function to select simulation task

# This is an attempt to group jobs that will take a similar time together. 
results_unordered=loadbalance_queue(driver_function, sort(job_queue, by=x->x["beads"]); checkpoint_frequency=2)

# Beyond this point, all results exist, so reshape them into the desired format and output. 
results=reshape(results_unordered, Tuple([length(i) for i in values(variables)]))
jldsave("results.jld2"; results)
