using DiffEqBase
using Distributed
using JLD2
using NQCDynamics
using UnitfulAtomic
using Unitful
using NQCDLoadbalancer
using PyCall

#! Driver function
function driver(params)
    if params["task"]=="langevin"
        results=langevin_dynamics(params)
    end
    return results
end

#! Utility functions


#! Dynamics functions
function langevin_dynamics(params::Dict{String,Any})
    # Prepare Python packages
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    spk_utils = pyimport("schnetpack.utils")
    spk_interfaces = pyimport("schnetpack.interfaces")
    # Load structure
    ase_structure=ase_io.read(params["starting_structure"]) # Assuming ASE will recognise the format.
    # Choose type of ML model (SchNet only for now) and load it
    if params["model_type"]=="schnet"
        schnet_model=spk_utils.load_model(params["schnet_model"]*"/best_model";map_location="cpu")
        schnet_args=spk_utils.read_from_json(params["schnet_model"]*"/args.json")
        schnet_environment=spk_utils.script_utils.settings.get_environment_provider(schnet_args,device="cpu")
        schnet_calculator=spk_interfaces.SpkCalculator(schnet_model, energy="energy", forces="forces", environment_provider=schnet_environment)
        ase_structure.set_calculator(schnet_calculator)
        nqcd_model=AdiabaticASEModel(ase_structure) # Use this as the main output of a ML model loading block
    end
    # Initialise NCQD Simulation
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    simulation=Simulation{Langevin}(
        nqcd_atoms, 
        nqcd_model, 
        temperature=get!(params, "temperature", 300u"K"),
        γ=get!(params, "gamma", 0.5),
        cell=nqcd_cell
    )
    # Starting conditions: initial positions and zero velocity
    u=DynamicsVariables(simulation, zeros(size(simulation)), positions)
    # Decide when run_dynamics should save, respecting eq time and saveat
    # set saveat=timestep if unset, force timestep if unset
    if not haskey(params, "saveat")
        params["saveat"]=get!(params, "timestep", 0.1"fs")
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
        output=(OutputDynamicsVariables, OutputPotentialEnergy) # Positions, velocities and 
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

    