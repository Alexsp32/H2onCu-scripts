using Distributed
using ProgressMeter
using JLD2
using NQCDynamics
using Statistics

# Load trajectory file
trajectory_file=jldopen("$1")["results"]

# Take simulation parameters from the results file. 
simulation_parameters=trajectory_file[2]

# Add local overrides, e.g. starting structure to enable work on other machines:
overrides=Dict(
    "starting_structure" => "/Users/u5529589/Documents/H2onCu/structures/111/Cu111_3x3x6_lowH_relax.in",
    "pes_type" => "mace_nov23",
    "T_el_profile" => "T_el_100K_J60.txt"
)
simulation_parameters=merge(simulation_parameters, overrides)
pop!(simulation_parameters, "friction_atoms") # Since we don't want friction

model_paths=[
    "/Users/u5529589/Documents/H2onCu/models/pes/mace_0.2.0_Nov23/model01.model",
    "/Users/u5529589/Documents/H2onCu/models/pes/mace_0.2.0_Nov23/model02.model",
    "/Users/u5529589/Documents/H2onCu/models/pes/mace_0.2.0_Nov23/model03-best.model",
    "/Users/u5529589/Documents/H2onCu/models/pes/mace_0.2.0_Nov23/model04.model",
    "/Users/u5529589/Documents/H2onCu/models/pes/mace_0.2.0_Nov23/model05.model",
]

# Push all positions into a single array
positions=mapreduce(x->NQCDynamics.get_positions.(x[:OutputDynamicsVariables]), vcat, trajectory_file[1])

@everywhere begin
    include("H2Cu_dynamics.jl")
    using SharedArrays
end

positions_shared=convert.(SharedArray{Float64}, positions)

# Load models and evaluate potentials
n_tasks=parse(Int, ARGS[1]) # testing version
#n_tasks=ENV["SLURM_NTASKS"] # Prod version
# Split positions into equal size views
view_size=ceil(length(positions_shared)/n_tasks) # Chunk size for n n_tasks
position_views=[view(positions_shared, Int.(range(i*view_size+1,min(i*view_size+view_size, length(positions_shared))))) for i in 0:n_tasks-1] # Split into chunks starting from index 1 until all positions are covered (min check)

all_parameters=[]
outputs=[]

@showprogress for model in model_paths
    parameters=merge(simulation_parameters, Dict("model_path" => model))
    push!(all_parameters, parameters)
    model_results=pmap(x->evaluate_energies_forces_friction(x, parameters), position_views)
    model_results_flat=mergewith(vcat, model_results...) # Merge dictionaries with vcat for identical keys along the n_tasks dimension of the results Matrix
    push!(outputs, model_results_flat) # Move to output vector. 
end

jldsave("test_output.jld2", results=outputs, parameters=all_parameters) # Save here in case of stack overflow for large vectors. 

# Energy errors
energy=hcat([output["energy"] for output in outputs]...)
energy_var=var(energy; dims=2)
forces_full=[cat([output["forces"][i] for output in outputs]...; dims=3) for i in eachindex(outputs[1]["forces"])]
forces_var=map(x->dropdims(var(x; dims=3);dims=3), forces_full)


outputs["forces_var"]=forces_var
outputs["energy_var"]=energy_var
jldsave("test_output.jld2", results=outputs, parameters=all_parameters)