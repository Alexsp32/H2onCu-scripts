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
    include("/Users/u5529589/Documents/H2onCu/scripts/H2Cu_dynamics.jl")
    using SharedArrays
end

positions_shared=convert.(SharedArray{Float64}, positions)
parameters=[merge(simulation_parameters, Dict("model_path" => path)) for path in model_paths]

# Load models and evaluate potentials
outputs=@showprogress pmap(x->evaluate_energies_forces_friction(positions_shared, x), parameters)

# Energy errors
energy=hcat([output["energy"] for output in outputs])
energy_var=var(energy; dims=2)
forces_full=[cat([output["forces"][i] for output in outputs]...; dims=3) for i in eachindex(outputs[1]["forces"])]
forces_var=map(x->dropdims(var(x; dims=3);dims=3), forces_full)


outputs["forces_var"]=forces_var
outputs["energy_var"]=energy_var
jldsave("test_output.jld2", results=outputs)