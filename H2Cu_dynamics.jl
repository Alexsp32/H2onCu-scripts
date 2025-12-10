using PythonCall
import ACE
import ACE1
using ACEds.FrictionModels
using ACEds.FrictionModels: Gamma
import Random
using BSplineKit
using CSV
using ClusterScripts
using Combinatorics
using DataFrames
using DataInterpolations
using DelimitedFiles
using Dictionaries
using DiffEqBase
using Distributed
using FrictionProviders
using JLD2
using JuLIP
using JuLIP: set_positions!
using LinearAlgebra
using MACEModels
using NQCBase
using NQCDynamics
using NQCModels
using Parameters
using ProgressMeter
using SciMLBase
using Statistics
using Unitful
using UnitfulAtomic

#! Driver function
function driver(parameters)
    if parameters["task"] == "langevin"
        results = langevin_dynamics(parameters)
    elseif parameters["task"] == "mdef+2tm"
        results = mdef_2tm(parameters)
    elseif parameters["task"] == "montecarlo"
        results = thermal_montecarlo(parameters)
    end
    return results
end

#! Utility functions
"""
    cu_surface_z(config, indices)

    Returns the mean z position of the atoms specified. Use with the top layer of Cu atoms in a slab to get a surface z plane.
TBW
"""
function cu_surface_z(config, indices)
    pos = NQCDynamics.get_positions(config)
    return mean([pos[3, i] for i in indices])
end

@with_kw mutable struct StrictDesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    max_hydrogen_bond_distance = 1.0
    min_desorption_distance = 8.0
    desorption_trajectories = 0
    simulation::NQCDynamics.AbstractSimulation
end

@with_kw mutable struct DesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    min_desorption_distance = 8.0
    desorption_trajectories = 0
    simulation::NQCDynamics.AbstractSimulation
end

function (desorption_terminator::StrictDesorptionTerminator)(u, t, integrator)::Bool
    # Find possible hydrogen molecules from all unique distance combinations of H atoms.
    h_combinations = collect(multiset_combinations(desorption_terminator.hydrogen_indices, 2)) # Unique H neighbour list
    distances = [NQCDynamics.Structure.pbc_distance(u, combination...) for combination in h_combinations] # Generate all distances
    h_molecules = findall(x -> x <= desorption_terminator.max_hydrogen_bond_distance, distances) # Find where the distance is sufficiently small.
    surface_z = cu_surface_z(u, desorption_terminator.top_layer_indices)
    for molecule in h_molecules
        if NQCDynamics.Structure.pbc_center_of_mass(u, molecule..., desorption_terminator.simulation)[3] - surface_z >= ang_to_au(desorption_terminator.min_desorption_distance) # Check if H2 COM is far enough away from the surface COM
            desorption_terminator.desorption_trajectories += 1
            return true
        end
    end
    return false
end

"""
    (desorption_terminator::DesorptionTerminator)(u,t,integrator)::Bool

    DesorptionTerminator checks whether any combination of `hydrogen_indices` has a centre of mass distance larger than `min_desorption_distance` from `cu_surface_z`.
    The strict version also enforces a H-H distance maximum (but currently doesn't respect periodicity.)
TBW
"""
function (desorption_terminator::DesorptionTerminator)(u, t, integrator)::Bool
    # Find possible hydrogen molecules from all unique distance combinations of H atoms.
    h_combinations = collect(multiset_combinations(desorption_terminator.hydrogen_indices, 2)) # Unique H neighbour list
    surface_z = cu_surface_z(u, desorption_terminator.top_layer_indices)
    for molecule in h_combinations
        if NQCDynamics.Structure.pbc_center_of_mass(u, molecule..., desorption_terminator.simulation; cutoff=1)[3] - surface_z >= ang_to_au(desorption_terminator.min_desorption_distance) # Check if H2 COM is far enough away from the surface COM
            desorption_terminator.desorption_trajectories += 1
            return true
        end
    end
    return false
end

mutable struct NoiseOutputCallback
    noise::Vector{Matrix{Float64}}
end

NoiseOutputCallback() = NoiseOutputCallback(Float64[])

function (callback::NoiseOutputCallback)(u,t,integrator)
    push!(callback.noise, integrator.W.dW)
    return false
end

SaveCallbacks(sol, i) = sol.prob.callback

function create_langevin_distribution(filename::String, trajectory)
    distribution = DynamicalDistribution(map(x -> get_velocities(x), trajectory), map(x -> NQCDynamics.get_positions(x), trajectory), size(NQCDynamics.get_positions(trajectory[1])))
    jldsave(filename; nqcd_distribution=distribution)
end

function create_langevin_distribution(filename::String, trajectories, equilibration_time::Vector)
    total_dynamicsvariables = []
    for i in eachindex(equilibration_time)
        append!(total_dynamicsvariables, trajectories[i][:OutputDynamicsVariables][trajectories[i][:Time].>=ps_to_au(equilibration_time[i])])
    end
    create_langevin_distribution(filename, total_dynamicsvariables)
end

#! MLIP interfaces

"""
    set_potential_energy_surface(ase_structure, parameters)

    Add calls to new ML model loading functions here.
TBW
"""
function set_potential_energy_surface(ase_structure, parameters)
    if haskey(parameters, "model_type")
        method = parameters["model_type"]
    elseif haskey(parameters, "pes_type")
        method = parameters["pes_type"]
    else
        error("No PES method supplied.")
    end
    if method == "schnet" #keeping model_type for backward compatibility
        load_schnet_model!(ase_structure, parameters)
    elseif method == "mace"
        load_mace_model!(ase_structure, parameters)
    elseif method == "mace_nov23"
        load_nov23_mace_model!(ase_structure, parameters)
    elseif method == "ace"
        load_ace_model!(ase_structure, parameters)
    elseif method == "macemodels-mpi"
        load_macemodels_remotemodel(ase_structure, parameters)
    end
end

"""
    load_macemodels_remotemodel(ase_structure, parameters)

Loads a RemoteModel from MACEModels for batched evaluation of MACE. You are required to launch evaluator processes yourself. 
"""
function load_macemodels_remotemodel(ase_structure, parameters)
    mpiconfig = parameters["MACEModels_MPI_config"]
    ensemble_config = parameters["ensemble_algorithm"]
    atoms, positions, cell = NQCBase.convert_from_ase_atoms(ase_structure)
    return MACEModels.Ensemble.RemoteModel(mpiconfig, positions)
end

"""
    load_ace_model!(ase_structure, parameters)

    Initialises an ACE PES model. Returns a JuLIPModel to be further used in a simulation.
TBW
"""
function load_ace_model!(ase_structure, parameters)
    model_path = haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"]
    julip_atoms = JuLIP.Atoms(ASE.ASEAtoms(ase_structure)) # Create JuLIP atoms.
    ace_potential = ACEpotentials.read_dict(load_dict(model_path)["IP"])
    JuLIP.set_calculator!(julip_atoms, ace_potential)
    if haskey(parameters, "fixed_atoms")
        # We have a freeze atoms constraints to apply
        movement_matrix = Matrix{Float64}(undef, size(ase_structure.get_positions()')...)
        fill!(movement_matrix, 1.0)
        movement_matrix[[CartesianIndex(i...) for i in Iterators.product(collect(1:size(ase_structure.get_positions())[end]), collect([parameters["fixed_atoms"]]))]] .= 0
        return AdiabaticModels.JuLIPModel(julip_atoms, movement_matrix)
    elseif length(ase_structure.constraints) != 0
        # Assume a freeze atoms constraint needs to be copied from ase
        movement_matrix = Matrix{Float64}(undef, size(ase_structure.get_positions()')...)
        fill!(movement_matrix, 1.0)
        movement_matrix[[CartesianIndex(i...) for i in Iterators.product(collect(1:size(ase_structure.get_positions())[end]), ase_structure.constraints[1].get_indices() .+ 1)]] .= 0
        return AdiabaticModels.JuLIPModel(julip_atoms, movement_matrix)
    else
        return AdiabaticModels.JuLIPModel(julip_atoms)
    end
end

"""
    load_schnet_model!(ase_structure, parameters)

    SchNet model loader.
TBW
"""
function load_schnet_model!(ase_structure, parameters)
    spk_utils = pyimport("schnetpack.utils")
    spk_interfaces = pyimport("schnetpack.interfaces")
    # ToDo: Find a way of integrating parameters["model_path"] for SchNet to make parameters more consistent.
    schnet_model = spk_utils.load_model(parameters["schnet_model"] * "/best_model"; map_location="cpu")
    schnet_args = spk_utils.read_from_json(parameters["schnet_model"] * "/args.json")
    schnet_environment = spk_utils.script_utils.settings.get_environment_provider(schnet_args, device="cpu")
    schnet_calculator = spk_interfaces.SpkCalculator(schnet_model, energy="energy", forces="forces", environment_provider=schnet_environment)
    ase_structure.set_calculator(schnet_calculator)
    return nothing
end

"""
    load_mace_model!(ase_structure, parameters)

    MACE 0.2.0 as of Nov 2023 model loader
TBW
"""
function load_nov23_mace_model!(ase_structure, parameters)
    atoms, positions, cell = NQCBase.convert_from_ase_atoms(ase_structure)
    model_paths = parameters["pes_model_path"]
    if isa(model_paths, String) # Ensure model path is always a vector
        model_paths = [model_paths]
    end
    model = MACEModel(
        atoms,
        cell,
        model_paths;
        device = get!(parameters, "mace_devicetype", "cpu"),
        default_dtype = Float32,
        mobile_atoms = get!(parameters, "mace_mobileatoms", 19:56 |> collect),
        batch_size = get!(parameters, "mace_batchsize", 1),
    )
    return model
end

function EFT_LDFA_cube_model(ase_structure, parameters)
    nqcd_atoms, nqcd_positions, nqcd_cell = NQCBase.convert_from_ase_atoms(ase_structure)
    density = CubeDensity(parameters["LDFA_cube_file"], nqcd_cell)
    return LDFAFriction(density, nqcd_atoms; friction_atoms=parameters["friction_atoms"])
end


function EFT_LDFA_scikit(ase_structure, parameters)
    nqcd_atoms, nqcd_positions, nqcd_cell = NQCBase.convert_from_ase_atoms(ase_structure)
    descriptors = pyimport("dscribe.descriptors")
    pd = pyimport("pandas")
    density = SciKitDensity(
        descriptors.SOAP(
            species = ase_structure.symbols |> unique |> PyList,
            periodic=true,
            r_cut = get!(parameters, "EFT_SOAP_rcut", 7.0),
            n_max = get!(parameters, "EFT_SOAP_nmax", 12),
            l_max = get!(parameters, "EFT_SOAP_lmax", 8),
            average="off",
        ),
        pd.read_pickle(parameters["eft_density_pickle"]),
        ase_structure;
        density_unit=u"Å^-3",
        scaler=pd.read_pickle(parameters["eft_density_scaler"]))
    return LDFAFriction(density, nqcd_atoms; friction_atoms=parameters["friction_atoms"])
end

function EFT_LDFA_ACE(ase_structure, parameters)
    ase_io = pyimport("ase.io")
    # Construct independent H atom model of surface.
    single_H_structure = ase_io.read(parameters["starting_structure"])
    atoms, positions, cell = NQCBase.convert_from_ase_atoms(single_H_structure)
    H_indices = sort(parameters["friction_atoms"]; rev=true)
    if length(H_indices) ≥ 2
        for i in H_indices[2:end]
            single_H_structure.pop(i - 1)
        end
    end
    tmp_filename = Random.randstring(32)
    ase_io.write("$(tmp_filename).xyz", single_H_structure)
    julip_atoms = ACEpotentials.ACEbase.read_extxyz("$(tmp_filename).xyz")[1]
    rm("$(tmp_filename).xyz", force=true)
    eft_model = ACEpotentials.read_dict(load_dict(parameters["eft_model_path"])["IP"])
    JuLIP.set_calculator!(julip_atoms, eft_model)
    density_model = AceLDFA(AdiabaticModels.JuLIPModel(julip_atoms); density_unit=u"Å^-3")
    return LDFAFriction(density_model, atoms; friction_atoms=parameters["friction_atoms"])
end

function EFT_ODF_ACE(ase_structure, parameters)
    eft_unit = u"ps^-1"
    # Skip importing JuLIP atoms from the ase structure and read directly from an input XYZ. 
    julip_atoms = JuLIP.read_extxyz(parameters["starting_structure"]) |> first
    # create an ACEds model object
    #* eft_model_ace has to be defined in global scope in the running file because ACEds...
    eft_model = FrictionProviders.ACEdsODF(eft_model_ace, Gamma, julip_atoms; friction_unit=eft_unit)
    return ODFriction(eft_model; friction_atoms=parameters["friction_atoms"])
end

function EFT_ODF_SchNet(ase_structure, parameters)
    schnet_friction_tensor = pyimport("friction_tensor")
    schnet_environment = pyimport("schnetpack.environment")
    torch = pyimport("torch")
    eft = SchNetODF(
        schnet_friction_tensor.FrictionCalculator(
            torch.load(parameters["ODF_SchNet_model_path"], map_location=get!(parameters, "ODF_SchNet_device", "cpu")),
            device=parameters["ODF_SchNet_device"],
            cutoff=get!(parameters, "ODF_SchNet_cutoff", 5.0),
            friction_tensor=u"ps^-1",
            friction_indices=torch.Tensor(parameters["friction_atoms"] .- 1), # Need to subtract 1 for python indices.
            environment_provider=schnet_environment.AseEnvironmentProvider(parameters["ODF_SchNet_cutoff"])
        ),
        ase_structure;
        friction_unit=u"ps^-1"
    )
    return ODFriction(eft; friction_atoms=parameters["friction_atoms"])
end

function load_EFT_model(ase_structure, parameters)
    # If no friction_atoms are specified, assume all hydrogens are meant to experience friction.
    if haskey(parameters, "friction_atoms") == false
        parameters["friction_atoms"] = findall(x -> x == "H", ase_structure.symbols) # These indices are Julian, so from 1
    end
    # Decide which model to apply
    if parameters["friction_type"] == "ODF_SchNet"
        return EFT_ODF_SchNet(ase_structure, parameters)
    elseif parameters["friction_type"] == "ODF_ACE"
        return EFT_ODF_ACE(ase_structure, parameters)
    elseif parameters["friction_type"] == "LDFA_ACE"
        return EFT_LDFA_ACE(ase_structure, parameters)
    elseif parameters["friction_type"] == "LDFA_scikit"
        return EFT_LDFA_scikit(ase_structure, parameters)
    elseif parameters["friction_type"] == "LDFA_cube_file"
        return EFT_LDFA_cube_model(ase_structure, parameters)
    end
    error("No friction method was specified")
end


#! Dynamics functions
function langevin_dynamics(parameters::Dict{String,Any})
    # Simulation initialisation
    sim_kwargs = Dict{Symbol,Any}(
        :γ => get!(parameters, "gamma", 0.5)
    )
    simulation, atoms, positions, cell = initialise_simulation(parameters; method=Langevin, sim_kwargs=sim_kwargs)
    # Let's try freezing atoms (lowest layers of Cu) by setting their sigma=0 in addition to the 0 forces from the AdiabaticASEModel
    if haskey(parameters, "fixed_atoms")
        # Manually set frozen atoms (separate constraint to ase)
        simulation.method.σ[[CartesianIndex(i) for i in Iterators.product(collect(NQCModels.dofs(simulation.calculator.model)), parameters["fixed_atoms"])]] .= 0.0
    else
        # Try to automatically set frozen atoms (copy ase constraint)
        if length(simulation.calculator.model.atoms.constraints) != 0
            simulation.method.σ[[CartesianIndex(i...) for i in Iterators.product(collect(NQCModels.dofs(simulation.calculator.model)), simulation.calculator.model.atoms.constraints[1].get_indices() .+ 1)]] .= 0
        end
    end
    # Starting conditions: initial positions and zero velocity
    u = DynamicsVariables(simulation, zeros(size(simulation)), positions)
    # Decide when run_dynamics should save, respecting eq time and saveat
    # set saveat=timestep if unset, force timestep if unset
    if !haskey(parameters, "saveat")
        parameters["saveat"] = get!(parameters, "timestep", 0.1u"fs")
    end
    # Assuming no eq time is specified, saveat can be passed as value of parameters["saveat"]
    if get!(parameters, "equilibration_time", 0.0) == 0.0
        saveat_arg = parameters["saveat"]
    else
        saveat_arg = map(austrip, collect(parameters["equilibration_time"]:parameters["timestep"]:parameters["runtime"]))
    end
    if haskey(parameters, "run_dynamics_kwargs")
        dynamics_kwargs = parameters["run_dynamics_kwargs"]
    else
        dynamics_kwargs = ()
    end
    # Now run dynamics
    traj = run_dynamics(
        simulation,
        (0.0u"fs", get!(parameters, "runtime", 0.3u"ps")),
        u;
        dt=get!(parameters, "timestep", 0.1u"fs"),
        trajectories=get!(parameters, "trajectories", 1),
        saveat=saveat_arg, # This might slow down calculations, check if actually useful.
        output=haskey(parameters, "outputs") ? parameters["outputs"] : (OutputDynamicsVariables, OutputPotentialEnergy), # Positions, velocities and
        ensemble_algorithm=get!(parameters, "ensemble_algorithm", EnsembleSerial()),
        callbacks=DynamicsUtils.CellBoundaryCallback(),
        dynamics_kwargs...
    )
    # Pack single trajectory into Array to ensure similarity with >1 trajectory
    if parameters["trajectories"] == 1
        results = [traj]
    else
        results = traj
    end
    return (results, parameters)
end

function thermal_montecarlo(parameters)
    sim_kwargs = Dict{Symbol,Any}(:temperature => get!(parameters, "temperature", 300u"K"))
    simulation, atoms, positions, cell = initialise_simulation(parameters; method=Classical, sim_kwargs=sim_kwargs)
    # Check if constraints are active. If so, create a copy of Atoms, where frozen atoms are of type :X
    if haskey(parameters, "fixed_atoms")
        # Manually set frozen atoms (separate constraint to ase)
        simulation.atoms.types[parameters["fixed_atoms"]] .= :X
    else
        # Try to automatically set frozen atoms (copy ase constraint)
        if length(simulation.calculator.model.atoms.constraints) != 0
            simulation.atoms.types[simulation.calculator.model.atoms.constraints[1].get_indices().+1] .= :X
        end
    end
    step_sizes = Dict(
        :H => get!(parameters, "MC_stepsize_H", 0.05),
        :Cu => get!(parameters, "MC_stepsize_Cu", 0.1),
        :X => 0.0,
    )
    # Now start MC sampling
    mc_chain = InitialConditions.ThermalMonteCarlo.run_advancedmh_sampling(
        simulation, # Simulation to run HMC sampling with
        positions, # Initial configuration
        get!(parameters, "MC_steps", 10), # Number of MonteCarlo steps to do.
        step_sizes; # Step sizes per atom species.
        get!(parameters, "MC_kwargs", Dict{Symbol,Any}())... # Additional kwargs to pass to the MC sampling function.
    )
    return (mc_chain, parameters)
end

function mdef_2tm(parameters)
    simulation, atoms, positions, cell = initialise_simulation(parameters; method=MDEF)
    # Load initial distribution
    if haskey(parameters, "initial_conditions_file")
        nqcd_distribution = jldopen(parameters["initial_conditions_file"])["nqcd_distribution"]
    else
        nqcd_distribution = DynamicsVariables(simulation, zeros(size(simulation)), positions)
    end
    desorption_callback = DesorptionTerminator(parameters["friction_atoms"], parameters["Cu_toplayer_indices"], parameters["desorption_min_surface_distance"], 0, simulation)
    terminate_callback = DynamicsUtils.TerminatingCallback(desorption_callback)
    # Run desorption simulations
    if haskey(parameters, "outputs")
        outputs = parameters["outputs"]
    elseif get!(parameters, "output_type", "last") == "full_trajectory"
        outputs = (OutputDynamicsVariables, OutputPotentialEnergy, OutputKineticEnergy)
    elseif parameters["output_type"] == "last"
        outputs = OutputFinal
    end
    if !haskey(parameters, "saveat")
        parameters["saveat"] = get!(parameters, "timestep", 0.1u"fs")
    end
    # Assuming no eq time is specified, saveat can be passed as value of parameters["saveat"]
    if get!(parameters, "equilibration_time", 0.0) == 0.0
        saveat_arg = parameters["saveat"]
    else
        saveat_arg = map(austrip, collect(parameters["equilibration_time"]:parameters["timestep"]:parameters["runtime"]))
    end
    if haskey(parameters, "run_dynamics_kwargs")
        run_dynamics_kwargs = parameters["run_dynamics_kwargs"]
    else
        run_dynamics_kwargs = ()
    end
    GC.gc()
    traj = run_dynamics(
        simulation,
        (0.0u"fs", get!(parameters, "runtime", 0.3u"ps")),
        nqcd_distribution;
        dt=get!(parameters, "timestep", 0.1u"fs"),
        trajectories=get!(parameters, "trajectories", 1),
        output=outputs,
        ensemble_algorithm=get!(parameters, "ensemble_algorithm", EnsembleSerial()),
        callback=CallbackSet(DynamicsUtils.CellBoundaryCallback(), terminate_callback),
        saveat=saveat_arg,
        run_dynamics_kwargs...
    )
    # Pack single trajectory into Array to ensure similarity with >1 trajectory
    if parameters["trajectories"] == 1
        results = [traj]
    else
        results = traj
    end
    return (results, parameters)
end

function evaluate_energies_forces_friction(position_trajectory, parameters::Dict{String,Any})
    sim, atoms, positions, cell = initialise_simulation(parameters; method=haskey(parameters, "friction_atoms") ? MDEF : Classical)
    energies = Float64[]
    forces = Matrix{Float64}[]
    @showprogress "Energy & Forces (Process $(myid()))" for configuration in position_trajectory
        push!(energies, NQCModels.potential(sim.calculator.model, configuration))
        push!(forces, NQCModels.derivative(sim.calculator.model, configuration))
    end
    friction = []
    if haskey(parameters, "friction_atoms")
        @debug "Calculating friction"
        @showprogress "Friction" for configuration in position_trajectory
            friction_conf = zeros(NQCModels.ndofs(sim.calculator.model) * length(sim.atoms), NQCModels.ndofs(sim.calculator.model) * length(sim.atoms))
            NQCModels.friction!(sim.calculator.model, friction_conf, configuration)
            push!(friction, friction_conf)
        end
    end
    outputs = Dict(
        "positions" => convert.(Matrix{Float64}, position_trajectory),
        "energy" => energies,
        "forces" => forces,
        "friction" => length(friction) > 0 ? friction : nothing,
    )
    return outputs
end

function initialise_simulation(parameters::Dict{String,Any}; method::Type=Classical, sim_kwargs=Dict{Symbol,Any}())
    @info "Loading ase (Python)"
    ase_io = pyimport("ase.io")
    ase = pyimport("ase")
    ase_structure = ase_io.read(parameters["starting_structure"]; index=0)
    atoms, initial_positions, cell = NQCDynamics.convert_from_ase_atoms(ase_structure)
    # Initialise PES model
    @info "Loading PES model"
    nqcd_model = nothing
    nqcd_model = set_potential_energy_surface(ase_structure, parameters) # Choose the desired ML model type and attach its calculator to ase_structure
    nqcd_model = typeof(nqcd_model) == Nothing ? AdiabaticASEModel(ase_structure) : nqcd_model
    @info "Assigning friction model if friction_atoms is set"
    # Assign the chosen EFT model if friction_type is provided. If no friction_atoms are specified, apply to all H-atoms. 
    EFT_model = haskey(parameters, "friction_type") ? load_EFT_model(ase_structure, parameters) : nothing
    """
    T_function_from_file(file::String, index::Int=2)

    [Henry's 2TM code](https://github.com/Snowd1n/Two-Temperature-Model---Extended-2TM) on commit 4c0e03b2d8c1e3554492d21b5e59d45ede469dfc currently generates CSVs
    for the 1D 2TM model with the time in ps, T_el in K, T_ph in K. This function reads the CSV files and
    returns a function that can be used to interpolate the temperature at any time.
    """
    function T_function_from_file(file::String, index::Int=2)
        TTM_file = CSV.read(file, DataFrame)
        T_spline = interpolate(TTM_file.Time, TTM_file[:, index], BSplineOrder(4)) # is a cubic spline
        T_extrapolation = extrapolate(T_spline, Smooth()) #! Don't use to go earlier than the first point!
        T_function(time_ps) = T_extrapolation(ustrip(u"ps", time_ps)) * u"K"
        return T_function
    end
    @info "Generating model and thermostat for simulation type"
    if !haskey(parameters, "2TM-file")
        # No 2TM file --> Constant temperature case
        thermostats = parameters["temperature"]
        @info "Combining PES and friction models if necessary"
        complete_model = isa(EFT_model, Nothing) ? nqcd_model : CompositeFrictionModel(nqcd_model, EFT_model)
    else haskey(parameters, "2TM-file")
        thermostats = TemperatureSetting[]
        subsystems = Subsystem[Subsystem(nqcd_model, 1:size(initial_positions,2))]
        # Apply phonon thermostat to selected atoms if specified
        if get(parameters, "2TM-T_ph", false)
            T_ph_function = T_function_from_file(parameters["2TM-file"], 3)
            phonon_indices = !haskey(parameters, "T_ph_indices") ? throw(ValueError("Please specify atoms to apply T_ph to by setting T_ph_indices")) : parameters["T_ph_indices"]
            push!(thermostats, TemperatureSetting(T_ph_function, phonon_indices)) # Apply T_ph profile
            push!(subsystems, Subsystem(ConstantFriction(size(initial_positions,1), parameters["γ_phonon"]), phonon_indices)) # Apply phononic friction
        end
        if get(parameters, "2TM-T_el", true) # Default is to apply T_el in order to maintain backwards compat with previous configs. 
            T_el_function = T_function_from_file(parameters["2TM-file"], 2)
            atom_indices = get!(parameters, "T_el_indices", parameters["friction_atoms"])
            push!(thermostats, TemperatureSetting(T_el_function, atom_indices)) # Assign T_el profile
            push!(subsystems, Subsystem(EFT_model, atom_indices)) # Apply electronic friction
        end
        remaining_atoms = symdiff(1:size(initial_positions)[2], [t.indices for t in thermostats]...)
        if remaining_atoms != [] # If any atoms aren't assigned to a thermostat, make them classical.
            push!(thermostats, TemperatureSetting(0.0, remaining_atoms))
        end
        complete_model = CompositeModel(subsystems...) # Build model for simulations
    end
    @info "Generated PES (+Friction) model"
    # Setup simulation and termination condition.
    # Initialise NCQD Simulation
    set_periodicity!(cell, [true, true, true]) #? Is this actually necessary or already set?
    sim_kwargs[:cell] = cell
    sim_kwargs[:temperature] = thermostats
    simulation = Simulation{method}(
        atoms,
        isa(complete_model, Nothing) ? nqcd_model : complete_model;
        sim_kwargs...
    )
    return simulation, atoms, initial_positions, cell
end
