module DynamicsFunctions

using PythonCall
import ACEpotentials
import ACEfriction
import Random
using CSV
using ClusterScripts
using Combinatorics
using DataFrames
using DataInterpolations
using DelimitedFiles
using Dictionaries
using Distributed
using FrictionProviders
using JLD2
using HDF5
import JuLIP
using LinearAlgebra
using MACEModels
using NQCBase
using NQCDynamics
using NQCModels
using NQCDInterfASE
using SciMLBase
using Statistics
using Unitful
using UnitfulAtomic

#! Driver function
function driver(parameters)
    if parameters["task"] == "langevin"
        results = langevin_dynamics(parameters)
    elseif parameters["task"] == "mdef+ttm"
        results = mdef_2tm(parameters)
    elseif parameters["task"] == "montecarlo"
        results = thermal_montecarlo(parameters)
    end
    return results
end

export driver

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

"""
Terminating callback which ends a simulation when a H2 desorption was detected. 

"""
mutable struct StrictDesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    max_hydrogen_bond_distance::Float64
    min_desorption_distance::Float64
    desorption_trajectories::Int
    simulation::NQCDynamics.AbstractSimulation
end
function StrictDesorptionTerminator(
    hydrogen_indices,
    top_layer_indices::Vector{Int},
    simulation::NQCDynamics.AbstractSimulation,
    max_hydrogen_bond_distance::Float64=1.0,
    min_desorption_distance::Float64=8.0,
    desorption_trajectories::Int=0,
)
    return StrictDesorptionTerminator(hydrogen_indices, top_layer_indices, max_hydrogen_bond_distance, min_desorption_distance, desorption_trajectories, simulation)
end

"""
Terminating callback which ends a simulation when a H2 desorption was detected. 

"""
mutable struct DesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    min_desorption_distance::Float64
    desorption_trajectories::Int
    simulation::NQCDynamics.AbstractSimulation
end
function DesorptionTerminator(
    hydrogen_indices,
    top_layer_indices::Vector{Int},
    simulation::NQCDynamics.AbstractSimulation,
    min_desorption_distance::Float64=8.0,
    desorption_trajectories::Int=0,
)
    return DesorptionTerminator(hydrogen_indices, top_layer_indices, min_desorption_distance, desorption_trajectories, simulation)
end

"""
    (desorption_terminator::StrictDesorptionTerminator)(u,t,integrator)::Bool

    StrictDesorptionTerminator checks whether any combination of `hydrogen_indices` has a centre of mass distance larger than `min_desorption_distance` from `cu_surface_z`, while also enforcing a maximum H-H distance of `max_hydrogen_bond_distance`.
"""
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

"""
    create_langevin_distribution(filename::String, trajectory)

Generates a `DynamicalDistribution` from a Langevin trajectory. 
"""
function create_langevin_distribution(filename::String, trajectory)
    distribution = DynamicalDistribution(map(x -> get_velocities(x), trajectory), map(x -> NQCDynamics.get_positions(x), trajectory), size(NQCDynamics.get_positions(trajectory[1])))
    jldsave(filename; nqcd_distribution=distribution)
end

"""
    create_langevin_distribution(filename::String, trajectories, equilibration_time::Vector)

Generates a `DynamicalDistribution` from a Langevin trajectory. Only considers structures after an initial `equilibration_time`. 
"""
function create_langevin_distribution(filename::String, trajectories, equilibration_time::Vector)
    total_dynamicsvariables = []
    for i in eachindex(equilibration_time)
        append!(total_dynamicsvariables, trajectories[i][:OutputDynamicsVariables][trajectories[i][:Time].>=ps_to_au(equilibration_time[i])])
    end
    create_langevin_distribution(filename, total_dynamicsvariables)
end


"""
    set_potential_energy_surface(ase_structure, parameters)

    Decides which potential energy surface interface to use .
"""
function set_potential_energy_surface(nqcd_structure, parameters)
    if haskey(parameters, "model_type")
        method = parameters["model_type"]
    elseif haskey(parameters, "pes_type")
        method = parameters["pes_type"]
    else
        error("No PES method supplied.")
    end
    if method == "schnet" #keeping model_type for backward compatibility
        @error "Don't bother using SchNet, MACE is better. "
    elseif method in ["mace_nov23", "mace"] # Edge case where I distinguished between different versions of MACE. 
        load_mace_model!(nqcd_structure, parameters)
    elseif method == "ace"
        load_ace_model!(nqcd_structure, parameters)
    elseif method == "macemodels-mpi"
        load_macemodels_remotemodel(nqcd_structure, parameters)
    end
end

"""
    load_EFT_model(ase_structure, parameters)

Decides which electronic friction tensor interface to use. 
"""
function load_EFT_model(ase_structure, parameters)
    # If no friction_atoms are specified, assume all hydrogens are meant to experience friction.
    if haskey(parameters, "friction_atoms") == false
        parameters["friction_atoms"] = findall(x -> x == "H", ase_structure.symbols) # These indices are Julian, so from 1
    end
    # Decide which model to apply
    if parameters["friction_type"] == "ODF_SchNet"
        @error "Currently not implemented"
    elseif parameters["friction_type"] == "ODF_ACE"
        return EFT_ODF_ACE(ase_structure, parameters)
    elseif parameters["friction_type"] == "LDFA_ACE"
        return EFT_LDFA_ACE(ase_structure, parameters)
    elseif parameters["friction_type"] == "LDFA_scikit"
        @error "Currently not implemented"
    elseif parameters["friction_type"] == "LDFA_cube_file"
        return EFT_LDFA_cube_model(ase_structure, parameters)
    end
    error("No friction method was specified")
end

# MLIP interfaces
"""
    load_macemodels_remotemodel(ase_structure, parameters)

Loads a RemoteModel from MACEModels for batched evaluation of MACE. You are required to launch evaluator processes yourself. 
"""
function load_macemodels_remotemodel(nqcd_structure, parameters)
    mpiconfig = parameters["MACEModels_MPI_config"]
    return MACEModels.Ensemble.RemoteModel(mpiconfig, nqcd_structure.positions)
end

"""
    load_ace_model!(ase_structure, parameters)

Initialises an ACEpotentials model. Tested working as of 04.03.2026.
"""
function load_ace_model!(nqcd_structure, parameters)
    model_path = haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"]
    ace_model, _ = ACEpotentials.load_model(model_path)
    return NQCModels.ClassicalModels.AtomsCalculatorsModel(
        ace_model,
        nqcd_structure
    )
end

"""
    load_mace_model!(ase_structure, parameters)

Load a MACEModels.jl version of a MACE model. (Requires models to be saved in mace ≥ v0.3.3)
"""
function load_mace_model!(nqcd_structure, parameters)
    model_paths = parameters["pes_model_path"]
    if isa(model_paths, String) # Ensure model path is always a vector
        model_paths = [model_paths]
    end
    model = MACEModels.MACEModel(
        nqcd_structure.atoms,
        nqcd_structure.cell,
        model_paths;
        device = get!(parameters, "mace_devicetype", "cpu"),
        default_dtype = Float32,
        mobile_atoms = get!(parameters, "mace_mobileatoms", 1:length(nqcd_structure.atoms) |> collect),
        batch_size = get!(parameters, "mace_batchsize", 1),
    )
    return model
end

function EFT_LDFA_cube_model(nqcd_structure, parameters)
    @error "Cube density LDFA models are deprecated. "
end


function EFT_LDFA_scikit(nqcd_structure, parameters)
    descriptors = pyimport("dscribe.descriptors")
    pd = pyimport("pandas")
    ase_structure = convert_to_ase_atoms(nqcd_structure)
    density = SciKitDensity(
        descriptors.SOAP(
            species = ase_structure.types |> unique |> PyList,
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
    return LDFAFriction(density, nqcd_structure.atoms; friction_atoms=parameters["friction_atoms"])
end

function EFT_LDFA_ACE(nqcd_structure, parameters)
    model_path = haskey(parameters, "eft_model_path") ? parameters["eft_model_path"] : parameters["friction_model_path"]
    ace_model, _ = ACEpotentials.load_model(model_path)
    H_indices = findall(nqcd_structure.atoms.types .== :H)
    atoms_to_copy = symdiff(1:length(nqcd_structure.atoms), H_indices[2:end]) # Remove H atoms if more than one is present. 
    one_H_structure = NQCBase.Structure(
        NQCBase.Atoms(nqcd_structure.atoms.types[atoms_to_copy]),
        nqcd_structure.positions[:, atoms_to_copy],
        nqcd_structure.cell
    )
    ace_density = NQCModels.ClassicalModels.AtomsCalculatorsModel(
        ace_model,
        one_H_structure
    )
    return LDFAFriction(FrictionProviders.AceLDFA(ace_density, u"Å^-3"), nqcd_structure.atoms; friction_atoms=parameters["friction_atoms"])
end

global eft_model_ace = nothing

function EFT_ODF_ACE(nqcd_structure, parameters)
    eft_unit = u"ps^-1"
    # convert ase atoms to julip atoms
    julip_atoms = JuLIP.read_extxyz(parameters["starting_structure"]) |> first
    # create an ACEds model object
    # `eft_model_ace` must be defined at runtime because Julia will throw a hissy fit about world ages otherwise. 
    eft_model=FrictionProviders.ACEdsODF(eft_model_ace, ACEfriction.Gamma, julip_atoms; friction_unit=eft_unit, friction_atoms = parameters["friction_atoms"])
    return eft_model
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
    end
    base_step_sizes = Dict(
        :H => 0.05,
        :Cu => 0.1,
        :X => 0.0,
    )
    if haskey(parameters, "MC_step_sizes")
        step_sizes = merge(base_step_sizes, parameters["MC_step_sizes"])
    else
        step_sizes = base_step_sizes
        parameters["MC_step_sizes"] = base_step_sizes
    end
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
    # desorption_callback = DesorptionTerminator(parameters["friction_atoms"], parameters["Cu_toplayer_indices"], parameters["desorption_min_surface_distance"], 0, simulation)
    # terminate_callback = DynamicsUtils.TerminatingCallback(desorption_callback)
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
        callback=CallbackSet(DynamicsUtils.CellBoundaryCallback(), ),
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
    for configuration in position_trajectory
        push!(energies, NQCModels.potential(sim.calculator.model, configuration))
        push!(forces, NQCModels.derivative(sim.calculator.model, configuration))
    end
    friction = []
    if haskey(parameters, "friction_atoms")
        @debug "Calculating friction"
        for configuration in position_trajectory
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
    nqcd_structure = NQCBase.read_extxyz(parameters["starting_structure"]) |> first
    # Pass this on to potential interfaces
    # Initialise PES model
    @info "Loading PES model"
    nqcd_model = set_potential_energy_surface(nqcd_structure, parameters) # Choose the desired ML model type and attach its calculator to ase_structure
    @info "Assigning friction model if friction_atoms is set"
    # Assign the chosen EFT model if friction_type is provided. If no friction_atoms are specified, apply to all H-atoms. 
    EFT_model = haskey(parameters, "friction_type") ? load_EFT_model(nqcd_structure, parameters) : nothing
    """
    T_function_from_file(file::String, index::Int=2)

    [Henry's 2TM code](https://github.com/Snowd1n/Two-Temperature-Model---Extended-2TM) on commit 4c0e03b2d8c1e3554492d21b5e59d45ede469dfc currently generates CSVs
    for the 1D 2TM model with the time in ps, T_el in K, T_ph in K. This function reads the CSV files and
    returns a function that can be used to interpolate the temperature at any time.
    """
    function T_function_from_file(file::String, index::String="Electronic")
        TTM_file = HDF5.h5open(file, "r")
        # Read times and convert them to au. 
        ttm_times = TTM_file["Time Span"][:]
        ttm_times = ttm_times .- ttm_times[1] # Set time=0 to the start of the simulation
        ttm_times_ps = uconvert.(u"ps", ttm_times .* u"fs") .|> ustrip
        # Take the temperature at the top of the surface
        ttm_temperature_K = TTM_file["$(index) Temperature"]["Temperature"][:,1]
        T_spline = BSplineInterpolation(
            ttm_temperature_K, 
            ttm_times_ps,
            3, 
            :ArcLen, 
            :Average; 
            extrapolation = ExtrapolationType.Linear, # Extrapolate linearly (but please don't go there)
        ) 
        T_function(t) = T_spline(ustrip(u"ps", t)) .* u"K" # Convert input time in ps to au, output temperature in K.
        return T_function # The temperature spline should take a time in ps and return a temperature in K.
    end
    @info "Generating model and thermostat for simulation type"
    if !haskey(parameters, "TTM-file")
        # No TTM file --> Constant temperature case
        thermostats = parameters["temperature"]
        @info "Combining PES and friction models if necessary"
        complete_model = isa(EFT_model, Nothing) ? nqcd_model : CompositeFrictionModel(nqcd_model, EFT_model)
    else haskey(parameters, "TTM-file")
        thermostats = TemperatureSetting[]
        subsystems = Subsystem[Subsystem(nqcd_model, 1:size(nqcd_structure.positions,2))]
        # Apply phonon thermostat to selected atoms if specified
        if get(parameters, "TTM-T_ph", false)
            T_ph_function = T_function_from_file(parameters["TTM-file"], "Phononic")
            phonon_indices = !haskey(parameters, "T_ph_indices") ? throw(ArgumentError("Please specify atoms to apply T_ph to by setting T_ph_indices")) : parameters["T_ph_indices"]
            push!(thermostats, TemperatureSetting(T_ph_function, phonon_indices)) # Apply T_ph profile
            push!(subsystems, Subsystem(ConstantFriction(size(nqcd_structure.positions,1), parameters["γ_phonon"]), phonon_indices)) # Apply phononic friction
        end
        if get(parameters, "TTM-T_el", true) # Default is to apply T_el in order to maintain backwards compat with previous configs. 
            T_el_function = T_function_from_file(parameters["TTM-file"], "Electronic")
            atom_indices = get!(parameters, "T_el_indices", parameters["friction_atoms"])
            push!(thermostats, TemperatureSetting(T_el_function, atom_indices)) # Assign T_el profile
            push!(subsystems, Subsystem(EFT_model, atom_indices)) # Apply electronic friction
        end
        remaining_atoms = symdiff(1:size(nqcd_structure.positions)[2], [t.indices for t in thermostats]...)
        if remaining_atoms != [] # If any atoms aren't assigned to a thermostat, make them classical.
            push!(thermostats, TemperatureSetting(0.0, remaining_atoms))
        end
        complete_model = CompositeModel(subsystems...) # Build model for simulations
    end
    @info "Generated PES (+Friction) model"
    # Setup simulation and termination condition.
    # Initialise NCQD Simulation
    set_periodicity!(nqcd_structure.cell, [true, true, false]) # Ensure we are not doing PBC along the slab Z. 
    sim_kwargs[:cell] = nqcd_structure.cell
    sim_kwargs[:temperature] = thermostats
    simulation = Simulation{method}(
        nqcd_structure.atoms,
        isa(complete_model, Nothing) ? nqcd_model : complete_model;
        sim_kwargs...
    )
    return simulation, nqcd_structure.atoms, nqcd_structure.positions, nqcd_structure.cell
end

end #Module
