using DiffEqBase
using Distributed
using JLD2
using ClusterScripts
using LinearAlgebra
using NQCDynamics
using PyCall
using Unitful
using UnitfulAtomic

#! Driver function
function driver(params)
    if params["task"]=="langevin"
        results=langevin_dynamics(params)
    end
    return results
end

#! Utility functions
"""
    export_trajectory(fname, trj, sim)
    
    Exports a single trajectory as the filename given in `fname` from the trajectory `trj` with the simulation parameters `sim`
TBW
"""
function export_trajectory(fname, trj, sim)
    ase_io=pyimport("ase.io")
    positions=map(get_positions, trj[:OutputDynamicsVariables])
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
    pos=get_positions(config)
    return au_to_ang(norm(pos[:,i1].-pos[:,i2]))
end

"""
    h2_com(config::Any, sim::NQCDynamics.Simulation, i1, i2)

    Returns centre of mass of two atoms based on their masses. 
TBW
"""
function h2_com(config::Any, sim::NQCDynamics.Simulation, i1, i2)
    mass_ratio=sim.masses[i2]/sim.masses[i1]
    pos=get_positions(config)
    return pos[:,i1].+mass_ratio*(pos[:,i2]-pos[:,i1])
end

"""
    cu_surface_z(config, indices)
    
    Returns the mean z position of the atoms specified. Use with the top layer of Cu atoms in a slab to get a surface z plane. 
TBW
"""
function cu_surface_z(config, indices)
    pos=get_positions(config)
    return mean([pos[end,i] for i in indices])
end

"""
    set_calculator(ase_structure, params)

    Add calls to new ML model loading functions here. 
TBW
"""
function set_calculator(ase_structure, params)
    if params["model_type"]=="schnet"
        load_schnet_model!(ase_structure, params)
    elseif params["model_type"]=="mace"
        load_mace_model!(ase_structure, params)
    end
end

"""
    load_schnet_model!(ase_structure, params)
    
    SchNet model loader. 
TBW
"""
function load_schnet_model!(ase_structure, params)
    spk_utils = pyimport("schnetpack.utils")
    spk_interfaces = pyimport("schnetpack.interfaces")
    # ToDo: Find a way of integrating params["model_path"] for SchNet to make params more consistent. 
    schnet_model=spk_utils.load_model(params["schnet_model"]*"/best_model";map_location="cpu")
    schnet_args=spk_utils.read_from_json(params["schnet_model"]*"/args.json")
    schnet_environment=spk_utils.script_utils.settings.get_environment_provider(schnet_args,device="cpu")
    schnet_calculator=spk_interfaces.SpkCalculator(schnet_model, energy="energy", forces="forces", environment_provider=schnet_environment)
    ase_structure.set_calculator(schnet_calculator)
end

"""
    load_mace_model!(ase_structure, params)
    
    MACE model loader
TBW
"""
function load_mace_model!(ase_structure, params)
    mace_calc=pyimport("mace.calculators")
    mace_calculator=mace_calc.MACECalculator(model_path=params["model_path"], device=get!(params, "mace_devicetype", "cpu"))
    ase_structure.set_calculator(mace_calculator)
end
    


#! Dynamics functions
function langevin_dynamics(params::Dict{String,Any})
    # Prepare Python packages
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    # Load structure
    ase_structure=ase_io.read(params["starting_structure"]) # Assuming ASE will recognise the format.
    # Choose type of ML model (SchNet only for now) and load it
    set_calculator(ase_structure, params) # Choose the desired ML model type and attach its calculator to ase_structure
    nqcd_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    # Initialise NCQD Simulation
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(nqcd_cell,[true,true,true]) #? Is this actually necessary or already set?
    simulation=Simulation{Langevin}(
        nqcd_atoms, 
        nqcd_model; 
        temperature=get!(params, "temperature", 300u"K"),
        γ=get!(params, "gamma", 0.5),
        cell=nqcd_cell,
        #extra_args...
    )
    # Let's try freezing atoms (lowest layers of Cu) by setting their sigma=0 in addition to the 0 forces from the AdiabaticASEModel
    if haskey(params, "fixed_atoms")
        simulation.method.σ[CartesianIndices(NQCModels.dofs(nqcd_model)),params["fixed_atoms"]].=0.0
    end
    # Starting conditions: initial positions and zero velocity
    u=DynamicsVariables(simulation, zeros(size(simulation)), positions)
    # Decide when run_dynamics should save, respecting eq time and saveat
    # set saveat=timestep if unset, force timestep if unset
    if !haskey(params, "saveat")
        params["saveat"]=get!(params, "timestep", 0.1u"fs")
    end
    # Assuming no eq time is specified, saveat can be passed as value of params["saveat"]
    if get!(params, "equilibration_time", 0.0)==0.0
        saveat_arg=params["saveat"]
    else
        saveat_arg=map(austrip, collect(params["equilibration_time"]:params["timestep"]:params["runtime"]))
    end
    # Now run dynamics
    traj=run_dynamics(
        simulation,
        (0.0u"fs", get!(params, "runtime", 0.3u"ps")),
        u;
        dt=get!(params, "timestep", 0.1u"fs"),
        trajectories=get!(params, "trajectories", 1),
        saveat=saveat_arg, # This might slow down calculations, check if actually useful. 
        output=(OutputDynamicsVariables, OutputPotentialEnergy), # Positions, velocities and 
        ensemble_algorithm=get!(params, "ensemble_algorithm", EnsembleSerial())
    )
    # Pack single trajectory into Array to ensure similarity with >1 trajectory
    if params["trajectories"]==1
        results=[traj]
    else
        results=traj
    end
    # Hand over an ase object for the initial structure and a simulation object for further evaluation
    params["nqcd_simulation"]=simulation
    params["ase_object"]=ase_structure
    return (results, params)
end

function mh_montecarlo(params)
    # Prepare Python packages
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    # Load structure
    ase_structure=ase_io.read(params["starting_structure"]) # Assuming ASE will recognise the format.
    # Choose type of ML model (SchNet only for now) and load it
    set_calculator(ase_structure, params) # Choose the desired ML model type and attach its calculator to ase_structure
    nqcd_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(nqcd_cell,[true,true,true]) #? Is this actually necessary or already set?
    simulation=Simulation{Classical}(
        nqcd_atoms, 
        nqcd_model, 
        temperature=get!(params, "temperature", 300u"K"),
        cell=nqcd_cell
    )
    #=
    # ToDo: Find out if I can use a newer method or if model constraints won't allow that.  
    mc_chain=InitialConditions.ThermalMonteCarlo.run_advancedmh_sampling(
        simulation,
        positions,
        get!(params, "mc_steps", 10000), 
        get!(params, "mc_stepsize", Dict(:H => 0.6, :Cu => 0.4));

    )
    =#
    mc_chain=InitialConditions.MetropolisHastings.run_monte_carlo_sampling(
        simulation,
        positions,
        get!(params, "mc_stepsize", Dict(:H => 0.6, :Cu => 0.4)),
        get!(params, "mc_steps", 10000);
        fix=params["mc_fixed"]
    )
    params["nqcd_simulation"]=simulation
    params["ase_object"]=ase_structure
    return(mc_chain, params)
end


    