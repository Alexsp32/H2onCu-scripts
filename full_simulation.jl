using ArgParse
using Distributed
using JLD2
using ClusterScripts
#using PyCall

function argparse()
    s=ArgParseSettings()
    @add_arg_table s begin
        "--worker"
            help="Julia processes called by parallel should use this option to indicate they need to run something."
            arg_type=Int
            default=0
        "--params"
            help="The location of the queue file. "
            default="simulation_parameters.jld2"
    end
    return parse_args(s)
end
args=argparse()
# Execution block for a worker process. 
if args["worker"]!=0
    if !isdir("tmp")
        mkdir("tmp")
    end
    include("H2Cu_dynamics.jl")
    all_params=jldopen(args["params"])
    run_params=all_params["queue"][args["worker"]][1,1]
    run_params["jobid"]=args["worker"]
    # Activate ACE friction if necessary (needs to be initialised in global scope)
    if haskey(run_params, "friction_type") && run_params["friction_type"]=="ODF_ACE"
        # Poke ACEds with a stick!
        @everywhere begin
            model_params = load_object(run_params["ODF_ACE_params"])        
            m_cov = ac_matrixmodel(ACE.EuclideanVector(Float64),model_params["species_friction"],model_params["species_env"];n_rep=model_params["n_rep_cov"], rcut_on = model_params["cutoff"], rcut_off = model_params["cutoff"], maxorder_on=model_params["max_order_on"], maxdeg_on=model_params["max_degree_on"],
                    species_maxorder_dict_on = model_params["species_maxorder_dict_on_cov"], 
                    species_weight_cat_on = model_params["species_weight_cat_on_cov"],
                    species_maxorder_dict_off = model_params["species_maxorder_dict_off_cov"], 
                    species_weight_cat_off = model_params["species_weight_cat_off_cov"],
                    bond_weight = model_params["bond_weight"])
            m_equ = ac_matrixmodel(ACE.EuclideanMatrix(Float64),model_params["species_friction"],model_params["species_env"];n_rep=model_params["n_rep_equ"], rcut_on = model_params["cutoff"], rcut_off = model_params["cutoff"], maxorder_on=model_params["max_order_on"], maxdeg_on=model_params["max_degree_on"],
                    species_maxorder_dict_on = model_params["species_maxorder_dict_on_equ"], 
                    species_weight_cat_on = model_params["species_weight_cat_on_equ"],
                    species_maxorder_dict_off = model_params["species_maxorder_dict_off_equ"], 
                    species_weight_cat_off = model_params["species_weight_cat_off_equ"],
                    bond_weight = model_params["bond_weight"])
            # define a model
            c_fit = load_object(run_params["ODF_ACE_cfit"])
            eft_model_ace = FrictionModel((m_cov,m_equ))
            ACE.set_params!(eft_model_ace, c_fit)
            # ACE friction should now be initialised
        end
    end
    kick_params=run_params
    kick_params["runtime"]=kick_params["timestep"]
    kick_params["saveat"]=kick_params["timestep"]
    println("Running a single time step to speed things up.")
    driver(kick_params)
    println("Now running the requested simulation.")
    results=driver(run_params)
    jldsave("tmp/"*ENV["SLURM_JOBID"]*"_"*string(args["worker"])*".jld2"; results=results)
    exit()
end

using NQCDynamics
using Unitful
using DiffEqBase
include("../scripts/H2Cu_dynamics.jl")

######### SIMULATION PARAMETER BLOCK

# Set fixed parameters
fixed_parameters=Dict(
    "model_path" => "/Users/u5529589/Documents/H2onCu/models/LDD_initial_conditions/h2cu_mace.model", # SchNet model path
    "starting_structure" => "/Users/u5529589/Documents/H2onCu/models/LDD_initial_conditions/h2cu111_0K.in.in",
    "trajectories" => 2, #Trajectories for RPMD
    "runtime" => 0.025u"ps", # Runtime for RPMD
    "timestep" => 0.1u"fs", # Timestep 
    "ensemble_algorithm" => EnsembleDistributed(), # Parallelisation method to use
    "saveat" => 0.1u"fs", # Sampling time interval for RPSH
    "gamma" => 0.375, 
    "pes_type" => "mace",
    #=
    "friction_type" => "ODF_ACE",
    "ODF_ACE_params" => "/Users/u5529589/Documents/H2onCu/models/eft/odf/aceds/wgt_x14_5/h2cu_eft_ace_ds_model_params_m.jld2",
    "ODF_ACE_cfit" => "/Users/u5529589/Documents/H2onCu/models/eft/odf/aceds/wgt_x14_5/h2cu_eft_ace_ds_model_c_fit.jld2",
    # "friction_type" => "ODF_SchNet",
    # "ODF_SchNet_model_path" => "/Users/u5529589/Documents/H2onCu/models/eft/odf/schnet/i5_f128/best_model",
    =#
    "friction_type" => "LDFA_scikit",
    "eft_density_pickle" => "/Users/u5529589/Documents/H2onCu/models/eft/ldfa/scikit/newest/density_soap_d2_a0_001_20230505.pkl",
    "eft_density_scaler" => "/Users/u5529589/Documents/H2onCu/models/eft/ldfa/scikit/newest/scaler_soap_d2_a0_001_20230505.pkl",
    "task" => "mdef+2tm",
    "initial_conditions_file" => "/Users/u5529589/Documents/H2onCu/initial_conditions/Cu111-110-100_3x3x6_2H_initials/Cu111_3x3x6_2H_300K_initial.jld2",
    "T_el_profile" => "/Users/u5529589/Documents/H2onCu/2TM/T_el_300K_J60.txt",
    "desorption_max_hydrogen_bond_distance" => 1.0,
    "desorption_min_surface_distance" => 8.0,
    #"output_type" => "full_trajectory",
    "outputs" => (OutputDynamicsVariables),# , OutputKineticEnergy, OutputPotentialEnergy, OutputHydrogenKineticEnergy([55,56])),
    #"friction_atoms" => [55,56],
    "Cu_toplayer_indices" => collect(54-8:54),
)

# Set variables
variables=Dict(
    "friction_atoms" => [[55,56]],
)
# Total trajectories to run: 1
# Estimated time per trajectory: unknown

if isfile(args["params"])
    @warn "Simulation queue already exists and will not be regenerated. Delete "*args["params"]*" to regenerate the queue."
    exit()
else
    job_queue=build_job_queue(fixed_parameters, variables)
    serialise_queue!(job_queue)
    @info "Generated simulation queue at "*args["params"]
    exit()
end

