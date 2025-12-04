ENV["JULIA_PYTHONCALL_EXE"] = "@PyCall"
using MACEModels
using ArgParse
using Distributed
using JLD2
using ClusterScripts
using Unitful
using UnitfulAtomic
using NQCDynamics
using DiffEqBase
using Statistics

# Global parameters (directories with a trailing slash)
# Base directory containing structures, models and 2TM_profiles folders
base_path = "/springbrook/share/chemistryrjm/msrkhg/H2onCu/"
# Scripts directory (so I can include the correct H2Cu_dynamics.jl file)
scripts_path = "$(base_path)/EnergyAccountingSimulations/scripts-odf/"
# Initial conditions folder
initial_conditions_path = base_path * "initial_conditions/"
# Temporary file store
output_dir = base_path * "EnergyAccountingSimulations/tmp/"

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
        mkdir(output_dir)
    end
    include(scripts_path * "H2Cu_dynamics.jl")
    all_params = jldopen(args["params"])
    run_params = all_params["queue"][args["worker"]][1, 1]
    run_params["jobid"] = args["worker"]
    kick_params = copy(run_params) # Absolutely DO NOT REMOVE copy() under any circumstances - You will bork your simulation parameters by not copying the dict.
    kick_params["runtime"] = kick_params["timestep"]
    kick_params["saveat"] = kick_params["timestep"]
    @info "Loading ACEds"
    eft_model_ace = ACE1.read_dict(ACE1.load_dict(run_params["eft_model_path"]))
    println("Running a single time step to speed things up.")
    driver(kick_params)
    println("Now running the requested simulation.")
    results = driver(run_params)
    jldsave("$(output_dir)$(ENV["SLURM_JOB_ID"])_$(args["worker"]).jld2"; results=results)
    exit()
end


include(scripts_path * "H2Cu_dynamics.jl")
H2AverageKineticEnergy(sol, i) = mean(last(OutputSubsetKineticEnergy([55, 56])(sol, i), 50))

######### SIMULATION PARAMETER BLOCK #########
friction_atoms = [55, 56]

fixed_parameters = Dict(
    "task" => "mdef+2tm",
    "pes_type" => "mace_nov23",
    "pes_model_path" => base_path * "wojciech-h2cu-repo/models/pes/mace_new/3_best/MACE_model_swa.model", # PES
    "trajectories" => 672,
    "runtime" => 4.7u"ps", # start at -0.4ps from laser maximum
    "timestep" => 0.1u"fs", # Timestep
    "saveat" => 0.1u"fs",
    "ensemble_algorithm" => EnsembleDistributed(), # Parallelisation method to use
    "desorption_min_surface_distance" => 8.0,
    "outputs" => (
        OutputInitial, # Starting structure
        OutputFinal, # Final geometry: ν, J, translational, rotational, vibrational energies, effective temperatures
        OutputFinalTime, # Time of termination (number of desorption steps yields instant desorption time)
        OutputSubsetKineticEnergy([55,56]),
        OutputDynamicsVariables, # Full trajectory output
        OutputDesorptionAngle(friction_atoms), # Angle of desorption if applicable
        OutputDesorptionTrajectory(friction_atoms; surface_distance_threshold=austrip(6.0u"Å"), extra_frames=200), # Desorption trajectory for criterion evaluation, "transition state" geometry
        OutputNoise,
    ),
    "friction_atoms" => friction_atoms,
    "Cu_toplayer_indices" => collect(54-8:54), # Defines the top slab layer to measure distance from
    "γ_phonon" => austrip(uconvert(u"ps^-1", 7.98e13u"rad/s" / 2 / π / u"rad")) * austrip(63.546u"u"), # From Debye frequency of bulk Cu - Yields ~5K T RMSE for Cu(111)
    "2TM-T_ph" => true, # active phonon thermostat
    "T_ph_indices" => collect(19:54), # phonon thermostat for four layers is best
    "batchsize" => 1, # Split into 128 trajectories to be run in parallel. 
	"run_dynamics_kwargs" => (save_noise=true,),
	"mace_mobileatoms" => collect(19:56),
)

# Set variables
variables = Dict(
    "friction_model" => [
        Dict("friction_type" => "LDFA_ACE", "eft_model_path" => base_path * "wojciech-h2cu-repo/models/eft/ldfa/ace/n3_d3_co4_4/h2cu_ace.json"),
        Dict("friction_type" => "ODF_ACE", "eft_model_path" => base_path * "wojciech-h2cu-repo/models/eft/odf/aceds/_old/ac_model_4/eft_ac.model"),
    ],
    "starting_temperature" => [100],
    "fluence" => [120],
    "surface_facet" => ["100", "110", "111", "211"],
)

function postprocess_queue(input)
    merge!(input, input["friction_model"]) # Set parameters for the correct EFT model
    # Set the initial conditions file
    input["initial_conditions_file"] = initial_conditions_path * "initial_Cu$(input["surface_facet"])_T-$(input["starting_temperature"])K_3x3x6_lowH.jld2"
    # Set correct starting structure
    input["starting_structure"] = base_path * "structures/MACE_$(input["starting_temperature"])K/$(input["surface_facet"])/Cu$(input["surface_facet"])_T-$(input["starting_temperature"])K_3x3x6_lowH_relax.in"
    # Set the 2TM profile
    input["2TM-file"] = base_path * "2TM_profiles/2TM1D-toponly-J$(input["fluence"])-$(input["starting_temperature"])K.csv"
    if input["surface_facet"] == "211" # Atomic numbering is different here.
		input["Cu_toplayer_indices"] = collect(Iterators.flatten([16:18, 34:36, 52:54])) # Top layer
        input["T_ph_indices"] = collect(Iterators.flatten([7:18, 25:36, 43:54])) # Top four layers without H atoms
		input["mace_mobileatoms"] = collect(Iterators.flatten([7:18, 25:36, 43:56]))
        # Outputs use max in surface normal component so no renumbering needed.
    end
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
    global eft_model_ace = read_dict(load_dict(queue[763][1,1]["eft_model_path"]))
    sim, atoms, positions, cell = initialise_simulation(queue[763][])
    println("Simulation: $(sim)")
    println("Atoms: $(atoms)")
    println("Size of system: $(size(positions))")
    @info "Testing potential"
    NQCModels.potential(sim.calculator.model, positions)
    @info "Testing derivative"
    NQCModels.derivative(sim.calculator.model, positions)
    @info "Testing friction"
    NQCModels.friction(sim.calculator.model, positions)
catch e
    @error "Simulation could not be initialised. The parameters file will be removed. See the error message below for details:"
    rm(args["params"]) #! Dangerous. Default value is set and glob/dir should cause error as long as if isfile(args["params"]) is called before this.
    throw(e)
    exit()
end
