python_location = read(`pyenv which python`, String) |> strip
ENV["JULIA_PYTHONCALL_EXE"] = python_location
ENV["JULIA_CONDAPKG_BACKEND"]= "None" # Avoid Conda.jl trying to manage environments and borking the PythonCall setup.

using ArgParse
using Distributed
using BenchmarkTools
using JLD2
using ClusterScripts
using Unitful
using UnitfulAtomic
using NQCDynamics
using NQCModels
using DiffEqBase
using Statistics

# Global parameters
# Base directory containing structures, models and 2TM_profiles folders
base_path = "/store/all/HRu-lasers/"
# Scripts directory (so I can include the correct H2Cu_dynamics.jl file)
scripts_path = "$(base_path)/dynamics-scripts"
# Output file and trajectory dump:
output_dir = "result_files/"
trj_dir = "result_trajectories/"

function argparse()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--worker"
        help = "Julia processes called by parallel should use this option to indicate they need to run something."
        arg_type = Int
        default = 0
        "--params"
        help = "The location of the queue file. "
        default = "simulation_parameters.jld2"
    end
    return parse_args(s)
end
args = argparse()
# Execution block for a worker process. 
if args["worker"] != 0
    if !isdir(output_dir)
        mkpath(output_dir)
    end
    if !isdir(trj_dir)
        mkpath(trj_dir)
    end
    include(scripts_path * "/DynamicsFunctions.jl")
    all_params = jldopen(args["params"])
    run_params = all_params["queue"][args["worker"]][1, 1]
    run_params["jobid"] = args["worker"]
    println("Now running the requested simulation.")
    results = DynamicsFunctions.driver(run_params)
    jldsave("$(output_dir)/trj_$(args["worker"]).jld2"; results=results)
    structure = NQCDynamics.read_extxyz(run_params["starting_structure"]) |> first
    trj_filename = "$(trj_dir)/trajectory_$(args["worker"]).xyz"
    NQCDynamics.write_extxyz(
        trj_filename,
        structure.atoms,
        get_positions.(results[1][1][:OutputDynamicsVariables]),
        structure.cell,
    )
    exit()
end

include(scripts_path * "/DynamicsFunctions.jl")

######### SIMULATION PARAMETER BLOCK #########
fixed_parameters = Dict(
    "task" => "mdef+ttm",
    "pes_type" => "mace",
    "pes_model_path" => "/store/all/HRu-lasers/mace-model.model", # PES
    "trajectories" => 100,
    "runtime" => 3.2u"ps",
    "timestep" => 0.1u"fs", # Timestep
    "saveat" => 0.1u"fs",
    "mace_devicetype" => "cpu",
    "ensemble_algorithm" => EnsembleSerial(), # Parallelisation method to use
    "outputs" => (
        OutputInitial, # Starting structure
        OutputFinal, # Final geometry: ν, J, translational, rotational, vibrational energies, effective temperatures
        OutputDynamicsVariables, 
        OutputNoise,
    ),
    #"Cu_toplayer_indices" => collect(54-8:54), # Defines the top slab layer to measure distance from
    "γ_phonon" => austrip(uconvert(u"ps^-1", 555*u"K" * 1.0u"k" / 1.0u"ħ")) * austrip(63.546u"u"), # From Debye frequency of bulk Cu - Yields ~5K T RMSE for Cu(111)
    "batchsize" => 1, # Split into 128 trajectories to be run in parallel. 
	"run_dynamics_kwargs" => (save_noise=true,),
	#"mace_mobileatoms" => collect(19:56),
    "friction_type" => "LDFA_ACE",
    "friction_model_path" => "/store/all/HRu-lasers/ldfa-density-model-02/ace_model.json",
    # Two-temperature model parameters
    "TTM-file" => "/store/all/HRu-lasers/Ru_TTM_0001_1D.hdf5",
    "TTM-T_ph" => true, # active phonon thermostat
    "TTM-T_el" => true, # active electronic thermostat
    "T_ph_indices" => collect(163:486), # phonon thermostat for four layers is best
)

# Set variables
variables = Dict(
    "structure_info" => [
        Dict( # Combination of structure file and which atoms to apply friction to. 
            "starting_structure" => "/store/all/HRu-lasers/Ru0001_T-600K_6x6x6_lowH_relax.xyz",
            "friction_atoms" => collect(487:494),
            "mace_mobileatoms" => collect(163:486+8), # Everything but the bottom two layers moves
        ),
        Dict( # Combination of structure file and which atoms to apply friction to. 
            "starting_structure" => "/store/all/HRu-lasers/Ru0001_T-600K_9x9x6_10H_relax.xyz",
            "friction_atoms" => collect(487:496),
            "mace_mobileatoms" => collect(163:486+10), # Everything but the bottom two layers moves
        ),
    ],
)

function postprocess_queue(input)
    # Assemble the MC step sizes.
    merge!(input, input["structure_info"])
    return input
end

# Total trajectories to run: 2
# Estimated time per trajectory: unknown

if isfile(args["params"])
    @warn "Simulation queue already exists and will not be regenerated. Delete " * args["params"] * " to regenerate the queue."
    exit()
else
    job_queue = build_job_queue(fixed_parameters, variables, postprocess_queue)
    serialise_queue!(job_queue; filename=args["params"])
    @info "Generated simulation queue at " * args["params"]
end

@info "Testing simulation setup:"
try
    queue = jldopen(args["params"])["queue"]
    sim, atoms, positions, cell = DynamicsFunctions.initialise_simulation(queue[1][])
    println("Simulation: $(sim)")
    println("Atoms: $(atoms)")
    println("Size of system: $(size(positions))")
    @info "Testing potential"
    @time NQCModels.potential(sim.cache.model, positions)
    @info "Testing derivative"
    @time NQCModels.derivative(sim.cache.model, positions)
catch e
    @error "Simulation could not be initialised. The parameters file will be removed. See the error message below for details:"
    rm(args["params"]) #! Dangerous. Default value is set and glob/dir should cause error as long as if isfile(args["params"]) is called before this.
    throw(e)
    exit()
end

