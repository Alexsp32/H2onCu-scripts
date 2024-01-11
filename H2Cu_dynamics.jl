using ACE
using ACEds.FrictionModels
using ACEds.FrictionModels: Gamma
using ACEds: ac_matrixmodel
using ASE
using BSplineKit
using ClusterScripts
using Combinatorics
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
using NNInterfaces
using NPZ
using NQCDynamics
using NQCBase: NQCBase
using NQCModels
using Parameters
using ProgressMeter
using PyCall
using Unitful
using UnitfulAtomic
using SciMLBase
using Statistics

#! Driver function
function driver(parameters)
    if parameters["task"]=="langevin"
        results=langevin_dynamics(parameters)
    elseif parameters["task"]=="mdef+2tm"
        results=mdef_2tm(parameters)
    elseif parameters["task"]=="montecarlo"
        results=thermal_montecarlo(parameters)
    end
    return results
end

#! Utility functions
function generate_angle_triplets(config::Any, sim::Simulation, species1, species2, species3;args...)
    # We need to distinguish four cases: 
    if species1==species2==species3
        # Here we are interested in three unique permutations per combination of three indices. 
        combinations=collect(combinations(findall(sim.atoms.types .== species1), 3))
        permutations=collect.(permutations.(combinations, 3)) # This includes reversions
        permutations=map(x->x[begin:3], permutations)
        triplets=map(x->vcat(x...), permutations)
        return triplets
    elseif species1!=species2!=species3
        # In this case, we need to look at all combinations of species1-species2-species3 triplets
        triplets=collect(Base.product(findall(sim.atoms.types .== species1), findall(sim.atoms.types .== species2), findall(sim.atoms.types .== species3)))
        return vec(map(x->vcat(x...), triplets))
    elseif species1==species2
        # Now we need to consider permutations of species1 & 2 combined with species3
        species1_permutations=permutations(findall(sim.atoms.types .== species1), 2)
        combinations=Base.product(species1_permutations, findall(sim.atoms.types .== species3))
        triplets=map(x->vcat(x...), collect(combinations))
        return vec(triplets)
    elseif species2==species3
        # Same as above, but with different indices
        species2_permutations=permutations(findall(sim.atoms.types .== species2), 2)
        combinations=Base.product(findall(sim.atoms.types .== species1), species2_permutations)
        triplets=map(x->vcat(x...), collect(combinations))
        return vec(triplets)
    elseif species1==species3
        # Same as above
        species1_permutations=permutations(findall(sim.atoms.types .== species1), 2)
        combinations=Base.product(species1_permutations, findall(sim.atoms.types .== species2))
        triplets=map(x->[x[1][1], x[2], x[1][2]], collect(combinations))
        return vec(triplets)
    end
end

"""
    all_distances(config::Any, sim::AbstractSimulation, species1, species2; args...)

Iterate through all (PBC compliant) distances
"""
function all_distances(config::Any, sim::Simulation, species1, species2; r_cut=10.0u"Å", args...)
    # Generate unique combinations of atomic distances. 
    if species1==species2
        combinations=Combinatorics.combinations(findall(sim.atoms.types .== species1), 2) # Find all non-redundant 2 combinations of species1==species2 indices.
    else
        combinations=Iterators.product(findall(sim.atoms.types .== species1), findall(sim.atoms.types .== species2)) # Find all non-redundant combinations of species1 != species2 indices. 
    end
    # Distance iterator exists now. 
    pbc_distances=[]
    for combination in combinations
        dist=Structure.pbc_distance(config, combination..., sim; args...)
        if dist<r_cut
            push!(pbc_distances, dist)
        end
    end
    return pbc_distances
end

function all_distances(trj::Vector, sim::Simulation, species1, species2; args...)
    pbc_distances=[]
    for config in trj
        pbc_distances=vcat(pbc_distances, all_distances(config, sim, species1, species2; args...))
    end
    return pbc_distances
end

function pbc_neighbour_positions(positions::Any, index::Int, distance_in_au, simulation::NQCDynamics.AbstractSimulation)
    distances=map(x->Structure.pbc_distance(positions, index, x, simulation), collect(1:length(simulation.atoms)))
    neighbours=findall(x->x≤auconvert(u"Å",distance_in_au) && x!=0.0u"Å", distances)
    positions_matrix=NQCDynamics.get_positions(positions)
    translations=[[i,j,k] for i in -1:1 for j in -1:1 for k in -1:1]
    relative_positions_pbc=zeros((NQCModels.dofs(simulation), length(neighbours)))
    for j in eachindex(neighbours)
        which_translation=argmin(map(x->norm(positions_matrix[:,index]-positions_matrix[:,neighbours[j]]+simulation.cell.vectors*x), translations))
        relative_positions_pbc[:,j].=positions_matrix[:,index]-positions_matrix[:,neighbours[j]]+simulation.cell.vectors*translations[which_translation]
    end
    return (relative_positions_pbc, simulation.atoms.types[neighbours])
end

function extract_from_distribution!(parameters)
    nqcd_distribution=jldopen(parameters["initial_conditions_file"])["nqcd_distribution"]
    parameters["DynamicsVariables"]=rand(nqcd_distribution)
    return parameters
end

"""
    export_trajectory(fname, trj, sim)
    
    Exports a single trajectory as the filename given in `fname` from the trajectory `trj` with the simulation parameters `sim`
TBW
"""
function export_trajectory(fname, trj, sim; to_export=:OutputDynamicsVariables)
    ase_io=pyimport("ase.io")
    positions=map(NQCDynamics.get_positions, trj[to_export])
    ase_trj=NQCBase.convert_to_ase_atoms(sim.atoms, positions, sim.cell)
    ase_io.write(fname, ase_trj)
end

"""
    cu_surface_z(config, indices)
    
    Returns the mean z position of the atoms specified. Use with the top layer of Cu atoms in a slab to get a surface z plane. 
TBW
"""
function cu_surface_z(config, indices)
    pos=NQCDynamics.get_positions(config)
    return mean([pos[3,i] for i in indices])
end

struct OutputHydrogenKineticEnergy
    hydrogen_indices::Vector{Int}
end

function (hydrogen_kinetic_energy::OutputHydrogenKineticEnergy)(sol, i)
    return map(x->DynamicsUtils.classical_kinetic_energy([1837.4715941070515 for i in hydrogen_kinetic_energy.hydrogen_indices],getindex(x, :, hydrogen_kinetic_energy.hydrogen_indices)), [DynamicsUtils.get_velocities(i) for i in sol.u])
end

@with_kw mutable struct StrictDesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    max_hydrogen_bond_distance=1.0
    min_desorption_distance=8.0
end

@with_kw mutable struct DesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    min_desorption_distance=8.0
end

function (desorption_terminator::StrictDesorptionTerminator)(u,t,integrator)::Bool
    # Find possible hydrogen molecules from all unique distance combinations of H atoms. 
    h_combinations=collect(multiset_combinations(desorption_terminator.hydrogen_indices, 2)) # Unique H neighbour list
    distances=[Structure.pbc_distance(u, combination...) for combination in h_combinations] # Generate all distances
    h_molecules=findall(x->x<=desorption_terminator.max_hydrogen_bond_distance, distances) # Find where the distance is sufficiently small. 
    surface_z=cu_surface_z(u, desorption_terminator.top_layer_indices)
    for molecule in h_molecules
        if Structure.pbc_center_of_mass(u, molecule..., integrator.p)[3] - surface_z >= ang_to_au(desorption_terminator.min_desorption_distance) # Check if H2 COM is far enough away from the surface COM
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
function (desorption_terminator::DesorptionTerminator)(u,t,integrator)::Bool
    # Find possible hydrogen molecules from all unique distance combinations of H atoms. 
    h_combinations=collect(multiset_combinations(desorption_terminator.hydrogen_indices, 2)) # Unique H neighbour list
    surface_z=cu_surface_z(u, desorption_terminator.top_layer_indices)
    for molecule in h_combinations
        if Structure.pbc_center_of_mass(u, molecule..., desorption_terminator.simulation;cutoff=1)[3]-surface_z>=ang_to_au(desorption_terminator.min_desorption_distance) # Check if H2 COM is far enough away from the surface COM
            return true
        end
    end
    return false
end


function create_langevin_distribution(filename::String, trajectory)
    distribution=DynamicalDistribution(map(x->get_velocities(x), trajectory), map(x->NQCDynamics.get_positions(x), trajectory), size(NQCDynamics.get_positions(trajectory[1])))
    jldsave(filename; nqcd_distribution=distribution)
end

function create_langevin_distribution(filename::String, trajectories, equilibration_time::Vector)
    total_dynamicsvariables=[]
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
        method=parameters["model_type"]
    elseif haskey(parameters, "pes_type")
        method=parameters["pes_type"]
    else
        error("No PES method supplied.")
    end
    if method=="schnet" #keeping model_type for backward compatibility
        load_schnet_model!(ase_structure, parameters)
    elseif method=="mace"
        load_mace_model!(ase_structure, parameters)
    elseif method=="mace_nov23"
        load_nov23_mace_model!(ase_structure, parameters)
    elseif method=="mace_nov23_committee"
        load_nov23_mace_models!(ase_structure, parameters)
    elseif method=="ace"
        load_ace_model!(ase_structure, parameters)
    elseif method=="painn"
        load_painn_model!(ase_structure, parameters)
    elseif method=="reann"
        load_reann_model!(ase_structure, parameters)
    end
end

"""
    load_ace_model!(ase_structure, parameters)

    Initialises an ACE PES model. Returns a JuLIPModel to be further used in a simulation. 
TBW
"""
function load_ace_model!(ase_structure, parameters)
    model_path=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"]
    julip_atoms=JuLIP.Atoms(ASE.ASEAtoms(ase_structure)) # Create JuLIP atoms.
    ace_potential=ACE1.read_dict(load_dict(model_path)["IP"])
    JuLIP.set_calculator!(julip_atoms, ace_potential)
    if haskey(parameters, "fixed_atoms")
        # We have a freeze atoms constraints to apply
        movement_matrix=Matrix{Float64}(undef, size(ase_structure.get_positions()')...)
        fill!(movement_matrix, 1.0)
        movement_matrix[[CartesianIndex(i...) for i in Iterators.product(collect(1:size(ase_structure.get_positions())[end]), collect([parameters["fixed_atoms"]]))]].=0
        return AdiabaticModels.JuLIPModel(julip_atoms, movement_matrix)
    elseif length(ase_structure.constraints)!=0
        # Assume a freeze atoms constraint needs to be copied from ase
        movement_matrix=Matrix{Float64}(undef, size(ase_structure.get_positions()')...)
        fill!(movement_matrix, 1.0)
        movement_matrix[[CartesianIndex(i...) for i in Iterators.product(collect(1:size(ase_structure.get_positions())[end]), ase_structure.constraints[1].get_indices().+1)]].=0
        return AdiabaticModels.JuLIPModel(julip_atoms, movement_matrix)
    else
        return AdiabaticModels.JuLIPModel(julip_atoms)
    end
end

"""
    load_painn_model!(ase_structure, parameters)

    Loads a PaiNN PES model by applying the ase calculator to ase_structure. 
TBW
"""
function load_painn_model!(ase_structure, parameters)
    torch=pyimport("torch")
    schnet_interfaces=pyimport("schnetpack.interfaces")
    schnet_transform=pyimport("schnetpack.transform")
    model_path=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"]
    painn_model=torch.load(model_path, map_location=get!(parameters, "painn_device", "cpu")).to(parameters["painn_device"]) # PaiNN-device sounds like I want to torture someone
    painn_calculator=schnet_interfaces.SpkCalculator(
        model=painn_model,
        converter=schnet_interfaces.AtomsConverter(neighbor_list=schnet_transform.ASENeighborList(cutoff=get!(parameters, "PES_PaiNN_neighbourlist_cutoff", 4.0)), dtype=torch.float32),
        energy_units="eV",
        forces_units="ev/Angstrom"
    )
    ase_structure.set_calculator(painn_calculator)
    return nothing
end

"""
    load_reann_model!(ase_structure, parameters)

    Loads a REANN ML PES and sets the calculator for the ASE structure specified. 
    Returns nothing since loading some models requires actually returning something. 
TBW
"""
function load_reann_model!(ase_structure, parameters)
    reann=pyimport("ase.calculators.reann")
    pbc=[1,1,1] # Periodic boundary conditions
    model_path=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"]
    reann_calc=reann.REANN(device="cpu", atomtype=parameters["REANN_atom_types"], period=pbc, nn=model_path)
    ase_structure.set_calculator(reann_calc)
    return nothing
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
    schnet_model=spk_utils.load_model(parameters["schnet_model"]*"/best_model";map_location="cpu")
    schnet_args=spk_utils.read_from_json(parameters["schnet_model"]*"/args.json")
    schnet_environment=spk_utils.script_utils.settings.get_environment_provider(schnet_args,device="cpu")
    schnet_calculator=spk_interfaces.SpkCalculator(schnet_model, energy="energy", forces="forces", environment_provider=schnet_environment)
    ase_structure.set_calculator(schnet_calculator)
    return nothing
end

"""
    load_mace_model!(ase_structure, parameters)
    
    MACE 0.1.0 model loader
TBW
"""
function load_mace_model!(ase_structure, parameters)
    mace_calc=pyimport("mace.calculators")
    mace_calculator=mace_calc.MACECalculator(model_path=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"], device=get!(parameters, "mace_devicetype", "cpu"))
    ase_structure.set_calculator(mace_calculator)
    return nothing
end

"""
    load_mace_model!(ase_structure, parameters)
    
    MACE 0.2.0 as of Nov 2023 model loader
TBW
"""
function load_nov23_mace_model!(ase_structure, parameters)
    mace_calc=pyimport("mace.calculators")
    mace_calculator=mace_calc.MACECalculator(model_paths=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"], device=get!(parameters, "mace_devicetype", "cpu"), default_dtype="float32")
    ase_structure.set_calculator(mace_calculator)
    return nothing
end

"""
    load_mace_models!(ase_structure, parameters)
    
    MACE 0.2.0 as of Nov 2023 model loader with support for model committees in newer versions of mace. 

"""
function load_nov23_mace_models!(ase_structure, parameters)
    mace_calc=pyimport("mace.calculators")
    mace_calculator=mace_calc.MACECalculator(model_paths=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"], device=get!(parameters, "mace_devicetype", "cpu"), default_dtype="float32")
    ase_structure.set_calculator(mace_calculator)
    return nothing
end
    
function EFT_LDFA_cube_model(ase_structure, parameters)
    nqcd_atoms, nqcd_positions, nqcd_cell=NQCBase.convert_from_ase_atoms(ase_structure)
    density=CubeDensity(parameters["LDFA_cube_file"], nqcd_cell)
    return LDFAFriction(density, nqcd_atoms;friction_atoms=parameters["friction_atoms"])
end


function EFT_LDFA_scikit(ase_structure, parameters)
    nqcd_atoms, nqcd_positions, nqcd_cell=NQCBase.convert_from_ase_atoms(ase_structure)
    descriptors=pyimport("dscribe.descriptors")
    pd=pyimport("pandas")
    density=SciKitDensity(
        descriptors.SOAP(
            species=unique(ase_structure.symbols),
            periodic=true,
            r_cut=get!(parameters, "EFT_SOAP_rcut", 7.0),
            n_max=get!(parameters, "EFT_SOAP_nmax", 12),
            l_max=get!(parameters, "EFT_SOAP_lmax", 8),
            average="off",
        ),
        pd.read_pickle(parameters["eft_density_pickle"]),
        ase_structure;
        density_unit=u"Å^-3",
        scaler=pd.read_pickle(parameters["eft_density_scaler"]))
    return LDFAFriction(density, nqcd_atoms; friction_atoms=parameters["friction_atoms"])
end


function EFT_ODF_ACE(ase_structure, parameters)
    eft_unit = u"ps^-1"
    # convert ase atoms to julip atoms
    ase_jl = ASE.ASEAtoms(ase_structure)
    julip_atoms = JuLIP.Atoms(ase_jl)
    # create an ACEds model object
    eft_model=FrictionProviders.ACEdsODF(eft_model_ace, Gamma, julip_atoms; friction_unit=eft_unit)
    return ODFriction(eft_model; friction_atoms=parameters["friction_atoms"])
end

function EFT_ODF_SchNet(ase_structure, parameters)
    schnet_friction_tensor=pyimport("friction_tensor")
    schnet_environment=pyimport("schnetpack.environment")
    torch=pyimport("torch")
    eft=SchNetODF(
        schnet_friction_tensor.FrictionCalculator(
            torch.load(parameters["ODF_SchNet_model_path"], map_location=get!(parameters, "ODF_SchNet_device", "cpu")),
            device=parameters["ODF_SchNet_device"],
            cutoff=get!(parameters, "ODF_SchNet_cutoff", 5.0),
            friction_tensor=u"ps^-1",
            friction_indices=torch.Tensor(parameters["friction_atoms"].-1), # Need to subtract 1 for python indices. 
            environment_provider=schnet_environment.AseEnvironmentProvider(parameters["ODF_SchNet_cutoff"])
        ),
        ase_structure; 
        friction_unit=u"ps^-1"
    )
    return ODFriction(eft; friction_atoms=parameters["friction_atoms"])
end

function set_friction_model(ase_structure, parameters)
    # If no friction_atoms are specified, assume all hydrogens are meant to experience friction.
    if haskey(parameters, "friction_atoms")==false
        parameters["friction_atoms"]=findall(x->x=="H", ase_structure.symbols) # These indices are Julian, so from 1
    end
    # Decide which model to apply
    if parameters["friction_type"]=="ODF_SchNet"
        return EFT_ODF_SchNet(ase_structure, parameters)
    elseif parameters["friction_type"]=="ODF_ACE"
        # ToDo: Commented out for testing 
        return EFT_ODF_ACE(ase_structure, parameters)
    elseif parameters["friction_type"]=="LDFA_scikit"
        return EFT_LDFA_scikit(ase_structure, parameters)
    elseif parameters["friction_type"]=="LDFA_cube_file"
        return EFT_LDFA_cube_model(ase_structure, parameters)
    end
    error("No friction method was specified")
end


#! Dynamics functions
function langevin_dynamics(parameters::Dict{String,Any})
    # Simulation initialisation
    sim_kwargs=Dict{Symbol, Any}(
        :γ => get!(parameters, "gamma", 0.5)
    )
    simulation, atoms, positions, cell=initialise_simulation(parameters; method=Langevin, sim_kwargs=sim_kwargs)
    # Let's try freezing atoms (lowest layers of Cu) by setting their sigma=0 in addition to the 0 forces from the AdiabaticASEModel
    if haskey(parameters, "fixed_atoms")
        # Manually set frozen atoms (separate constraint to ase)
        simulation.method.σ[[CartesianIndex(i) for i in Iterators.product(collect(NQCModels.dofs(simulation.calculator.model)), parameters["fixed_atoms"])]].=0.0
    else
        # Try to automatically set frozen atoms (copy ase constraint)
        if length(simulation.calculator.model.atoms.constraints)!=0
            simulation.method.σ[[CartesianIndex(i...) for i in Iterators.product(collect(NQCModels.dofs(simulation.calculator.model)), simulation.calculator.model.atoms.constraints[1].get_indices().+1)]].=0
        end
    end
    # Starting conditions: initial positions and zero velocity
    u=DynamicsVariables(simulation, zeros(size(simulation)), positions)
    # Decide when run_dynamics should save, respecting eq time and saveat
    # set saveat=timestep if unset, force timestep if unset
    if !haskey(parameters, "saveat")
        parameters["saveat"]=get!(parameters, "timestep", 0.1u"fs")
    end
    # Assuming no eq time is specified, saveat can be passed as value of parameters["saveat"]
    if get!(parameters, "equilibration_time", 0.0)==0.0
        saveat_arg=parameters["saveat"]
    else
        saveat_arg=map(austrip, collect(parameters["equilibration_time"]:parameters["timestep"]:parameters["runtime"]))
    end
    # Now run dynamics
    traj=run_dynamics(
        simulation,
        (0.0u"fs", get!(parameters, "runtime", 0.3u"ps")),
        u;
        dt=get!(parameters, "timestep", 0.1u"fs"),
        trajectories=get!(parameters, "trajectories", 1),
        saveat=saveat_arg, # This might slow down calculations, check if actually useful. 
        output=haskey(parameters, "outputs") ? parameters["outputs"] : (OutputDynamicsVariables, OutputPotentialEnergy), # Positions, velocities and 
        ensemble_algorithm=get!(parameters, "ensemble_algorithm", EnsembleSerial()),
        callbacks=DynamicsUtils.CellBoundaryCallback()
    )
    # Pack single trajectory into Array to ensure similarity with >1 trajectory
    if parameters["trajectories"]==1
        results=[traj]
    else
        results=traj
    end
    return (results, parameters)
end

function thermal_montecarlo(parameters)
    sim_kwargs=Dict{Symbol, Any}(:temperature => get!(parameters, "temperature", 300u"K"))
    simulation, atoms, positions, cell=initialise_simulation(parameters; method=Classical, sim_kwargs=sim_kwargs)
    # Check if constraints are active. If so, create a copy of Atoms, where frozen atoms are of type :X
    if haskey(parameters, "fixed_atoms")
        # Manually set frozen atoms (separate constraint to ase)
        simulation.atoms.types[parameters["fixed_atoms"]].=:X
    else
        # Try to automatically set frozen atoms (copy ase constraint)
        if length(simulation.calculator.model.atoms.constraints)!=0
            simulation.atoms.types[simulation.calculator.model.atoms.constraints[1].get_indices().+1].=:X
        end
    end
    step_sizes=Dict(
        :H => get!(parameters, "MC_stepsize_H", 0.05),
        :Cu => get!(parameters, "MC_stepsize_Cu", 0.1),
        :X => 0.0,
    )
    # Now start MC sampling
    mc_chain=InitialConditions.ThermalMonteCarlo.run_advancedmh_sampling(
        simulation, # Simulation to run HMC sampling with
        positions, # Initial configuration
        get!(parameters, "MC_steps", 10), # Number of MonteCarlo steps to do. 
        step_sizes; # Step sizes per atom species. 
        get!(parameters, "MC_kwargs", Dict{Symbol, Any}())... # Additional kwargs to pass to the MC sampling function. 
    )
    return(mc_chain, parameters)
end

function mdef_2tm(parameters)
    # Load initial distribution
    if haskey(parameters, "DynamicsVariables")
        nqcd_distribution=parameters["DynamicsVariables"]
    else
        nqcd_distribution=jldopen(parameters["initial_conditions_file"])["nqcd_distribution"]
    end
    
    simulation, atoms, positions, cell=initialise_simulation(parameters; method=MDEF)
    desorption_callback=DesorptionTerminator(friction_model.friction_atoms, parameters["Cu_toplayer_indices"], parameters["desorption_min_surface_distance"], 0, simulation)
    terminate_callback=DynamicsUtils.TerminatingCallback(desorption_callback)
    # Run desorption simulations
    if haskey(parameters, "outputs")
        outputs=parameters["outputs"]
    elseif get!(parameters, "output_type", "last")=="full_trajectory"
        outputs=(OutputDynamicsVariables, OutputPotentialEnergy, OutputKineticEnergy)
    elseif parameters["output_type"]=="last"
        outputs=OutputFinal
    end
    if !haskey(parameters, "saveat")
        parameters["saveat"]=get!(parameters, "timestep", 0.1u"fs")
    end
    # Assuming no eq time is specified, saveat can be passed as value of parameters["saveat"]
    if get!(parameters, "equilibration_time", 0.0)==0.0
        saveat_arg=parameters["saveat"]
    else
        saveat_arg=map(austrip, collect(parameters["equilibration_time"]:parameters["timestep"]:parameters["runtime"]))
    end
    GC.gc()
    traj=run_dynamics(
        simulation,
        (0.0u"fs", get!(parameters, "runtime", 0.3u"ps")),
        nqcd_distribution;
        dt=get!(parameters, "timestep", 0.1u"fs"),
        trajectories=get!(parameters, "trajectories", 1),
        output=outputs,
        ensemble_algorithm=get!(parameters, "ensemble_algorithm", EnsembleSerial()),
        callback=CallbackSet(DynamicsUtils.CellBoundaryCallback(),terminate_callback),
        saveat=saveat_arg,
    )
    # Pack single trajectory into Array to ensure similarity with >1 trajectory
    if parameters["trajectories"]==1
        results=[traj]
    else
        results=traj
    end
    return (results, parameters)
end

function evaluate_energies_forces_friction(position_trajectory, parameters::Dict{String, Any})
    sim, atoms, positions, cell=initialise_simulation(parameters; method=haskey(parameters, "friction_atoms") ? MDEF : Classical)
    energies=Float64[]
    forces=Matrix{Float64}[]
    @showprogress "Energy & Forces (Process $(myid()))" for configuration in position_trajectory
        push!(energies, NQCModels.potential(sim.calculator.model, configuration))
        push!(forces, NQCModels.derivative(sim.calculator.model, configuration))
    end
    friction=[]
    if haskey(parameters, "friction_atoms")
        @debug "Calculating friction"
        @showprogress "Friction" for configuration in position_trajectory
            friction_conf=zeros(NQCModels.ndofs(sim.calculator.model)*length(sim.atoms), NQCModels.ndofs(sim.calculator.model)*length(sim.atoms))
            NQCModels.friction!(sim.calculator.model, friction_conf, configuration)
            push!(friction, friction_conf)
        end
    end
    outputs=Dict(
        "positions" => convert.(Matrix{Float64}, position_trajectory),
        "energy" => energies,
        "forces" => forces,
        "friction" => length(friction)>0 ? friction : nothing,
    )
    return outputs
end

function initialise_simulation(parameters::Dict{String, Any}; method::Type=Classical, sim_kwargs=Dict{Symbol, Any}())
    @debug "Loading ase (Python)"
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    ase_structure=ase_io.read(parameters["starting_structure"]; index=0)
    # Initialise PES model
    @debug "Loading PES model"
    nqcd_model=nothing
    nqcd_model=set_potential_energy_surface(ase_structure, parameters) # Choose the desired ML model type and attach its calculator to ase_structure
    if typeof(nqcd_model)==Nothing
        # In this case, the ML method has modified ase_structure by attaching a calculator to it. 
        nqcd_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    end
    @debug "Assigning friction model if friction_atoms is set"
    # Assign the chosen EFT model if friction_atoms are set. 
    friction_model=haskey(parameters, "friction_atoms") ? set_friction_model(ase_structure, parameters) : nothing
    @debug "Combining PES+Friction models if necessary"
    # Combine models to CompositeFrictionModel
    nqcd_model=isa(friction_model, Nothing) ? nqcd_model : CompositeFrictionModel(nqcd_model, friction_model)
    @debug "Reading T_el profile if necessary"
    # Generate temperature profile function
    if haskey(parameters, "T_el_profile")
        T_el_profile=readdlm(parameters["T_el_profile"])
        T_el_function(time_ps)=interpolate(T_el_profile[:,1], T_el_profile[:,2], BSplineOrder(3))(ustrip(u"ps", time_ps))*u"K" # Put T_el function in the right format for NQCD
        parameters["T_el_function"]=T_el_function # Give the temperature profile in parameters for nice plotting. 
    end
    # Setup simulation and termination condition. 
    # Initialise NCQD Simulation
    atoms,initial_positions,cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(cell,[true,true,true]) #? Is this actually necessary or already set?
    sim_kwargs[:cell]=cell
    sim_kwargs[:temperature]=haskey(parameters, "temperature") ? parameters["temperature"] : T_el_function
    simulation=Simulation{method}(
        atoms,
        nqcd_model;
        sim_kwargs...
    )
    return simulation, atoms, initial_positions, cell
end
    

