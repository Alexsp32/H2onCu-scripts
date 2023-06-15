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
using NQCModels
using Pandas
using Parameters
using PyCall
using Unitful
using UnitfulAtomic

#! Driver function
function driver(params)
    if params["task"]=="langevin"
        results=langevin_dynamics(params)
    elseif params["task"]=="mdef+2tm"
        results=mdef_2tm(params)
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
    h2_com(config:Any, mass_ratio::Float64, i1, i2)

    Returns centre of mass of two atoms without requiring a Simulation object
TBW
"""
function h2_com(config::Any, mass_ratio::Float64, i1, i2)
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


@with_kw mutable struct DesorptionTerminator
    hydrogen_indices
    top_layer_indices
    max_hydrogen_bond_distance=1.0
    min_desorption_distance=8.0
    desorption_trajectories=0
end

function (desorption_terminator::DesorptionTerminator)(u,t,integrator)::Bool
    # Find possible hydrogen molecules from all unique distance combinations of H atoms. 
    h_combinations=collect(multiset_combinations(desorption_terminator.hydrogen_indices, 2)) # Unique H neighbour list
    distances=[distance(u, combination...) for combination in h_combinations] # Generate all distances
    h_molecules=findall(x->x<=desorption_terminator.max_hydrogen_bond_distance, distances) # Find where the distance is sufficiently small. 
    surface_z=cu_surface_z(u, desorption_terminator.top_layer_indices)
    for molecule in h_molecules
        if h2_com(u, 1.0, h_combinations[molecule]...)-surface_z>=desorption_terminator.min_desorption_distance # Check if H2 COM is far enough away from the surface COM
            desorption_terminator.desorption_trajectories+=1
            return true
        end
    end
    return false
end

mutable struct ElectronicTemperatureFunction
    temperature_profile
end


function create_langevin_distribution(filename::String, trajectory)
    distribution=DynamicalDistribution(map(x->get_velocities(x), trajectory), map(x->get_positions(x), trajectory), size(get_positions(trajectory[1])))
    jldsave(filename; nqcd_distribution=distribution)
end

#! MLIP interfaces

"""
    set_potential_energy_surface(ase_structure, params)

    Add calls to new ML model loading functions here. 
TBW
"""
function set_potential_energy_surface(ase_structure, params)
    if haskey(params, "model_type")
        method=params["model_type"]
    elseif haskey(params, "pes_type")
        method=params["pes_type"]
    else
        error("No PES method supplied.")
    end
    if method=="schnet" #keeping model_type for backward compatibility
        load_schnet_model!(ase_structure, params)
    elseif method=="mace"
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
    
function EFT_LDFA_cube_model(ase_structure, params)
    nqcd_atoms, nqcd_positions, nqcd_cell=NQCBase.convert_from_ase_atoms(ase_structure)
    density=CubeDensity(params["LDFA_cube_file"], nqcd_cell)
    return LDFAFriction(density, nqcd_atoms;friction_atoms=params["friction_atoms"])
end

function EFT_LDFA_scikit(ase_structure, params)
    nqcd_atoms, nqcd_positions, nqcd_cell=NQCBase.convert_from_ase_atoms(ase_structure)
    descriptors=pyimport("dscribe.descriptors")
    density=SciKitDensity(
        descriptors.SOAP(
            species=unique(ase_structure.symbols),
            periodic=true,
            r_cut=get!(params, "EFT_SOAP_rcut", 7.0),
            n_max=get!(params, "EFT_SOAP_nmax", 7.0),
            l_max=get!(params, "EFT_SOAP_lmax", 7.0),
            average="off",
            )
    )
    return LDFAFriction(density, nqcd_atoms; friction_atoms=params["friction_atoms"])
end

function EFT_ODF_ACE(ase_structure, params)
    eft_unit = u"ps^-1"
    model_params = load_object(params["ODF_ACE_params"])        
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
    c_fit = load_object(params["ODF_ACE_cfit"])
    eft_model_ace = FrictionModel((m_cov,m_equ))
    ACE.set_params!(eft_model_ace, c_fit)
    # convert ase atoms to julip atoms
    ase_jl = ASE.ASEAtoms(ase_structure)
    julip_atoms = JuLIP.Atoms(ase_jl)
    # create an ACEds model object
    eft_model=FrictionProviders.ACEdsODF(eft_model_ace, Gamma, julip_atoms; friction_unit=eft_unit)
    return eft_model
    # ODFriction(eft_model; friction_atoms=params["friction_atoms"])
end

function EFT_ODF_SchNet(ase_structure, params)
    schnet_friction_tensor=pyimport("friction_tensor")
    schnet_environment=pyimport("schnetpack.environment")
    torch=pyimport("torch")
    eft=SchNetODF(
        schnet_friction_tensor.FrictionCalculator(
            torch.load(params["ODF_SchNet_model_path"], map_location=get!(params, "ODF_SchNet_device", "cpu")),
            device=params["ODF_SchNet_device"],
            cutoff=get!(params, "ODF_SchNet_cutoff", 5.0),
            friction_tensor=u"ps^-1",
            friction_indices=torch.Tensor(params["friction_atoms"].-1), # Need to subtract 1 for python indices. 
            environment_provider=schnet_environment.AseEnvironmentProvider(params["ODF_SchNet_cutoff"])
        ),
        ase_structure; 
        friction_unit=u"ps^-1"
    )
    return ODFriction(eft; friction_atoms=params["friction_atoms"])
end

function set_friction_model(ase_structure, params)
    # If no friction_atoms are specified, assume all hydrogens are meant to experience friction.
    if haskey(params, "friction_atoms")==false
        params["friction_atoms"]=findall(x->x=="H", ase_structure.symbols) # These indices are Julian, so from 1
    end
    # Decide which model to apply
    if params["friction_type"]=="ODF_SchNet"
        return EFT_ODF_SchNet(ase_structure, params)
    elseif params["friction_type"]=="ODF_ACE"
        eft=EFT_ODF_ACE(ase_structure, params)
        return ODFriction(eft; friction_atoms=params["friction_atoms"])
    elseif params["friction_type"]=="LDFA_scikit"
        return EFT_LDFA_scikit(ase_structure, params)
    elseif params["friction_type"]=="LDFA_cube_file"
        return EFT_LDFA_cube_model(ase_structure, params)
    end
    error("No friction method was specified")
end


#! Dynamics functions
function langevin_dynamics(params::Dict{String,Any})
    # Prepare Python packages
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    # Load structure
    ase_structure=ase_io.read(params["starting_structure"]) # Assuming ASE will recognise the format.
    # Choose type of ML model (SchNet only for now) and load it
    set_potential_energy_surface(ase_structure, params) # Choose the desired ML model type and attach its calculator to ase_structure
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
        # Manually set frozen atoms (separate constraint to ase)
        simulation.method.σ[CartesianIndices(NQCModels.dofs(nqcd_model)),params["fixed_atoms"]].=0.0
    else
        # Try to automatically set frozen atoms (copy ase constraint)
        if length(ase_structure.constraints)!=0
            simulation.method.σ[CartesianIndices(NQCModels.dofs(nqcd_model)),CartesianIndices(ase_structure.constraints[1].get_indices())].=0.0
        end
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
    set_potential_energy_surface(ase_structure, params) # Choose the desired ML model type and attach its calculator to ase_structure
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

function mdef_2tm(params)
    # Load initial distribution
    nqcd_distribution=jldopen(params["initial_conditions_file"])["nqcd_distribution"]
    # Create an AdiabaticASEModel for the base slab
    ase_io=pyimport("ase.io")
    ase=pyimport("ase")
    ase_structure=ase_io.read(params["starting_structure"])
    # Assign the chosen PES model
    set_potential_energy_surface(ase_structure, params)
    pes_model=AdiabaticASEModel(ase_structure) # Connect ase_structure calculator to NQCD
    # Assign the chosen EFT model
    friction_model=set_friction_model(ase_structure, params)
    # Combine models to CompositeFrictionModel
    nqcd_model=CompositeFrictionModel(pes_model, friction_model)
    # Generate temperature profile function
    T_el_profile=readdlm(params["T_el_profile"])
    T_el_function=interpolate(T_el_profile[:,1], T_el_profile[:,2], BSplineOrder(3))
    params["T_el_function"]=T_el_function # Give the temperature profile in parameters for nice plotting. 
    # Setup simulation and termination condition. 
    # Initialise NCQD Simulation
    nqcd_atoms,positions,nqcd_cell=NQCDynamics.convert_from_ase_atoms(ase_structure)
    set_periodicity!(nqcd_cell,[true,true,true]) #? Is this actually necessary or already set?
    simulation=Simulation{MDEF}(
        nqcd_atoms,
        nqcd_model,
        cell=nqcd_cell,
        temperature=T_el_function
    )
    desorption_callback=DesorptionTerminator(friction_model.friction_atoms, params["Cu_toplayer_indices"], params["desorption_max_hydrogen_bond_distance"], params["desorption_min_surface_distance"], 0)
    terminate_callback=DynamicsUtils.TerminatingCallback(desorption_callback)
    # Run desorption simulations
    if get!(params, "output_type", "last")=="full_trajectory"
        outputs=(OutputDynamicsVariables, OutputPotentialEnergy, OutputKineticEnergy)
    elseif params["output_type"]=="last"
        outputs=OutputFinal
    end
    traj=run_dynamics(
        simulation,
        (0.0u"fs", get!(params, "runtime", 0.3u"ps")),
        nqcd_distribution;
        dt=get!(params, "timestep", 0.1u"fs"),
        trajectories=get!(params, "trajectories", 1),
        output=outputs,
        ensemble_algorithm=get!(params, "ensemble_algorithm", EnsembleSerial()),
        callback=terminate_callback
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
    # Add the number of times the termination callback was true for the number of desorption events. 
    params["desorption_trajectories"]=desorption_callback.desorption_trajectories
    return (results, params)
end



    