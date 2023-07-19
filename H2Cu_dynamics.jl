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
    end
    return results
end

#! Utility functions
function angle_between(v1::Vector, v2::Vector)
    return rad2deg(acos(dot(v1,v2)/prod(norm, [v1, v2])))
end

function desorption_angle(trajectory::Vector, indices::Vector{Int}, simulation::NQCDynamics.AbstractSimulation; surface_normal=[0,0,1], distance_threshold=1.0u"Å")
    # First determine where the reaction occurred on the surface. 
    desorption_frame=get_desorption_frame(trajectory, indices, simulation; distance_threshold=distance_threshold)
    println("Desorption frame: ", desorption_frame)
    # Determine the average centre of mass velocity to decrease error due to vibration and rotation orthogonal to true translational component. 
    com_velocities=map(x->velocity_com(x, indices[1], indices[2], simulation), trajectory[desorption_frame:end])
    average_velocity=mean(cat(com_velocities...;dims=2);dims=2)
    # Now convert into an angle by arccos((a•b)/(|a|*|b|))
    return angle_between(vec(average_velocity), surface_normal)
end

struct OutputDesorptionTrajectory{S,I,N,D,X}
    sim::S
    indices::I
    surface_normal::N
    distance_threshold::D
    extra_frames::X
end
OutputDesorptionTrajectory(sim, indices; surface_normal=[0,0,1], distance_threshold=0.9u"Å", extra_frames=0)=OutputDesorptionTrajectory(sim, indices, surface_normal, distance_threshold, extra_frames)
"""
    (output::OutputDesorptionTrajectory)(sol, i)

    Only output parts of the simulation where desorption is occurring
TBW
"""
function (output::OutputDesorptionTrajectory)(sol, i)
    desorption_frame=get_desorption_frame(sol.u, output.indices..., output.sim; distance_threshold=output.distance_threshold)
    return sol.u[desorption_frame-output.extra_frames:end]
end

function get_desorption_frame(trajectory::Vector, indices::Vector{Int}, simulation::NQCDynamics.AbstractSimulation;distance_threshold=1.4u"Å")
    return findfirst(x->pbc_distance(x, indices..., simulation)<distance_threshold && velocity_com(x, indices..., simulation)[3]>0,trajectory)
end

"""
    pbc_distance(config::Any, ind1::Int, ind2::Int, simulation::NQCDynamics.AbstractSimulation)

    **Returns in Angstom, not in Bohr - Check units.**
    Calculates the distance between two atoms, including a check if the copy of the second atom in any neighbouring unit cell is closer. 
    This variant is designed for trajectories where cell boundary wrapping has been used. 
TBW
"""
function pbc_distance(config::Any, ind1::Int, ind2::Int, simulation::NQCDynamics.AbstractSimulation)
    positions=NQCDynamics.get_positions(config)
    translations=[[i,j,k] for i in -1:1 for j in -1:1 for k in -1:1]
    return min(map(x->au_to_ang(norm(positions[:,ind1].-positions[:,ind2]+simulation.cell.vectors*x)), translations)...).*u"Å"
end

function pbc_center_of_mass(config::Matrix, ind1::Int, ind2::Int, simulation::NQCDynamics.AbstractSimulation)
    translations=[[i,j,k] for i in -1:1 for j in -1:1 for k in -1:1]
    which_translation=argmin(map(x->au_to_ang(norm(config[:,ind1].-config[:,ind2]+simulation.cell.vectors*x)), translations))
    return (simulation.atoms.masses[ind1]*config[:,ind1]+simulation.atoms.masses[ind2]*config[:,ind2]+simulation.cell.vectors*translations[which_translation])/sum(simulation.atoms.masses[[ind1,ind2]])
end

function pbc_center_of_mass(config::Any, ind1::Int, ind2::Int, simulation::NQCDynamics.AbstractSimulation)
    positions=NQCDynamics.get_positions(config)
    return pbc_center_of_mass(positions, ind1, ind2, simulation)
end

function velocity_com(config::Any, ind1::Int, ind2::Int, simulation::NQCDynamics.AbstractSimulation)
    positions=NQCDynamics.get_velocities(config)
    return pbc_center_of_mass(positions, ind1, ind2, simulation)
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
    distance(config::Matrix, i1, i2)
    Interatomic distance for a position Matrix. 
TBW
"""
function distance(config::Matrix, i1, i2)
    return au_to_ang(norm(config[:,i1].-config[:,i2]))
end

"""
    distance(config::Any, i1, i2)
    Interatomic distance for DynamicsVariables. 
TBW
"""
function distance(config::Any, i1, i2)
    pos=NQCDynamics.get_positions(config)
    return au_to_ang(norm(pos[:,i1].-pos[:,i2]))
end

"""
    h2_com(config::Any, sim::NQCDynamics.Simulation, i1, i2)
    Returns centre of mass of two atoms based on their masses.
    This variant of the function is meant to work on NQCD DynamicsVariables. 
    The simulation is only accessed for atomic masses.   
TBW
"""
function h2_com(config::Any, sim::NQCDynamics.Simulation, i1::Int, i2::Int)
    pos=NQCDynamics.get_positions(config)
    masses=sim.atoms.masses
    return h2_com(pos, masses, i1, i2)
end


"""
    h2_com(config::Matrix, masses::Vector, i1::Int, i2::Int)

    Determines the centre of mass between slices `[:,i1]` and `[:,i2]` of the Matrix `config`. 
TBW
"""
function h2_com(config::Matrix, masses::Vector, i1::Int, i2::Int)
    return sum([masses[i]*config[:,i] for i in [i1, i2]])/sum([masses[i] for i in [i1, i2]])
end

"""
    cu_surface_z(config, indices)
    
    Returns the mean z position of the atoms specified. Use with the top layer of Cu atoms in a slab to get a surface z plane. 
TBW
"""
function cu_surface_z(config, indices)
    pos=NQCDynamics.get_positions(config)
    return mean([pos[end,i] for i in indices])
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
    desorption_trajectories=0
    simulation::NQCDynamics.AbstractSimulation
end

@with_kw mutable struct DesorptionTerminator
    hydrogen_indices
    top_layer_indices::Vector{Int}
    min_desorption_distance=8.0
    desorption_trajectories=0
    simulation::NQCDynamics.AbstractSimulation
end

function (desorption_terminator::StrictDesorptionTerminator)(u,t,integrator)::Bool
    # Find possible hydrogen molecules from all unique distance combinations of H atoms. 
    h_combinations=collect(multiset_combinations(desorption_terminator.hydrogen_indices, 2)) # Unique H neighbour list
    distances=[distance(u, combination...) for combination in h_combinations] # Generate all distances
    h_molecules=findall(x->x<=desorption_terminator.max_hydrogen_bond_distance, distances) # Find where the distance is sufficiently small. 
    surface_z=cu_surface_z(u, desorption_terminator.top_layer_indices)
    for molecule in h_molecules
        if h2_com(u, desorption_terminator.simulation, molecule...)[3]-surface_z>=ang_to_au(desorption_terminator.min_desorption_distance) # Check if H2 COM is far enough away from the surface COM
            desorption_terminator.desorption_trajectories+=1
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
        if h2_com(u, desorption_terminator.simulation, molecule...)[3]-surface_z>=ang_to_au(desorption_terminator.min_desorption_distance) # Check if H2 COM is far enough away from the surface COM
            desorption_terminator.desorption_trajectories+=1
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
    reann=pyimport("ase.calculators.reann") # ToDo: I still need to install this!
    pbc=[1,1,1] # Periodic boundary conditions
    model_path=haskey(parameters, "pes_model_path") ? parameters["pes_model_path"] : parameters["model_path"]
    reann_calc=reann.REANN(device="cpu", atomtype=unique(ase_structure.get_chemical_symbols()), period=pbc, nn=model_path)
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
    
    MACE model loader
TBW
"""
function load_mace_model!(ase_structure, parameters)
    mace_calc=pyimport("mace.calculators")
    mace_calculator=mace_calc.MACECalculator(model_path=parameters["model_path"], device=get!(parameters, "mace_devicetype", "cpu"))
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
    # Prepare Python packages
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    # Load structure
    ase_structure=ase_io.read(parameters["starting_structure"]) # Assuming ASE will recognise the format.
    # Initialise PES model
    nqcd_model=nothing
    nqcd_model=set_potential_energy_surface(ase_structure, parameters) # Choose the desired ML model type and attach its calculator to ase_structure
    if typeof(nqcd_model)==Nothing
        # In this case, the ML method has modified ase_structure by attaching a calculator to it. 
        nqcd_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    end
    # Initialise NCQD Simulation
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(nqcd_cell,[true,true,true]) #? Is this actually necessary or already set?
    simulation=Simulation{Langevin}(
        nqcd_atoms, 
        nqcd_model; 
        temperature=get!(parameters, "temperature", 300u"K"),
        γ=get!(parameters, "gamma", 0.5),
        cell=nqcd_cell,
        #extra_args...
    )
    # Let's try freezing atoms (lowest layers of Cu) by setting their sigma=0 in addition to the 0 forces from the AdiabaticASEModel
    if haskey(parameters, "fixed_atoms")
        # Manually set frozen atoms (separate constraint to ase)
        simulation.method.σ[[CartesianIndex(i) for i in Iterators.product(collect(NQCModels.dofs(nqcd_model)), parameters["fixed_atoms"])]].=0.0
    else
        # Try to automatically set frozen atoms (copy ase constraint)
        if length(ase_structure.constraints)!=0
            simulation.method.σ[[CartesianIndex(i...) for i in Iterators.product(collect(NQCModels.dofs(nqcd_model)), ase_structure.constraints[1].get_indices().+1)]].=0
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
    # Hand over an ase object for the initial structure and a simulation object for further evaluation
    parameters["nqcd_simulation"]=simulation
    parameters["ase_object"]=ase_structure
    return (results, parameters)
end

function mh_montecarlo(parameters)
    # Prepare Python packages
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    # Load structure
    ase_structure=ase_io.read(parameters["starting_structure"]) # Assuming ASE will recognise the format.
    # Choose type of ML model (SchNet only for now) and load it
    # ToDo: Handle PES models that aren't an ASE calculator
    set_potential_energy_surface(ase_structure, parameters) # Choose the desired ML model type and attach its calculator to ase_structure
    nqcd_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(nqcd_cell,[true,true,true]) #? Is this actually necessary or already set?
    simulation=Simulation{Classical}(
        nqcd_atoms, 
        nqcd_model, 
        temperature=get!(parameters, "temperature", 300u"K"),
        cell=nqcd_cell
    )
    #=
    # ToDo: Find out if I can use a newer method or if model constraints won't allow that.  
    mc_chain=InitialConditions.ThermalMonteCarlo.run_advancedmh_sampling(
        simulation,
        positions,
        get!(parameters, "mc_steps", 10000), 
        get!(parameters, "mc_stepsize", Dict(:H => 0.6, :Cu => 0.4));

    )
    =#
    mc_chain=InitialConditions.MetropolisHastings.run_monte_carlo_sampling(
        simulation,
        positions,
        get!(parameters, "mc_stepsize", Dict(:H => 0.6, :Cu => 0.4)),
        get!(parameters, "mc_steps", 10000);
        fix=parameters["mc_fixed"]
    )
    parameters["nqcd_simulation"]=simulation
    parameters["ase_object"]=ase_structure
    return(mc_chain, parameters)
end

function mdef_2tm(parameters)
    # Load initial distribution
    if haskey(parameters, "DynamicsVariables")
        nqcd_distribution=parameters["DynamicsVariables"]
    else
        nqcd_distribution=jldopen(parameters["initial_conditions_file"])["nqcd_distribution"]
    end
    # Create an AdiabaticASEModel for the base slab
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    ase_structure=ase_io.read(parameters["starting_structure"])
    # Assign the chosen PES model
    # ToDo: Handle PES models that aren't an ASE calculator.
    # Initialise PES model
    nqcd_model=nothing
    nqcd_model=set_potential_energy_surface(ase_structure, parameters) # Choose the desired ML model type and attach its calculator to ase_structure
    if typeof(nqcd_model)==Nothing
        # In this case, the ML method has modified ase_structure by attaching a calculator to it. 
        nqcd_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    end
    # Assign the chosen EFT model
    friction_model=set_friction_model(ase_structure, parameters)
    # Combine models to CompositeFrictionModel
    nqcd_model=CompositeFrictionModel(nqcd_model, friction_model)
    # Generate temperature profile function
    if haskey(parameters, "T_el_profile")
        T_el_profile=readdlm(parameters["T_el_profile"])
        T_el_function(time_ps)=interpolate(T_el_profile[:,1], T_el_profile[:,2], BSplineOrder(3))(ustrip(u"ps", time_ps))*u"K" # Put T_el function in the right format for NQCD
        parameters["T_el_function"]=T_el_function # Give the temperature profile in parameters for nice plotting. 
    end
    # Setup simulation and termination condition. 
    # Initialise NCQD Simulation
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(nqcd_cell,[true,true,true]) #? Is this actually necessary or already set?
    simulation=Simulation{MDEF}(
        nqcd_atoms,
        nqcd_model,
        cell=nqcd_cell,
        temperature=haskey(parameters, "temperature") ? parameters["temperature"] : T_el_function
    )
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
    # Hand over an ase object for the initial structure and a simulation object for further evaluation
    # parameters["nqcd_simulation"]=simulation
    # parameters["ase_object"]=ase_structure
    # Add the number of times the termination callback was true for the number of desorption events. 
    # parameters["desorption_trajectories"]=desorption_callback.desorption_trajectories
    return (results, parameters)
end



    