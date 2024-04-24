
#? Run this script as extract_top1percent_errors.jl trajectory_file.jld2 output_base_filename
include("H2Cu_dynamics.jl")

input_file=jldopen(ARGS[1], "r+")
output_filename=ARGS[2]

threshold=0.01 # Pick the top 1% of errors
num_structures=Int(floor(length(input_file["results"][1]["positions"])*threshold))

trajectory=input_file["results"]
# Give simulation parameters here and make any overrides if you're running this on a different machine than the original calculation was performed on. 
parameters=input_file["parameters"][3]
println(parameters)

"""
    top_i(positions, sortby, i; rev=true)

Returns positions associated with the top i values ranked in descending order (rev=true) by default. 

e.g. top 20 RMSE snapshots. 
"""
function top_i(positions, sortby, i; rev=true)
    perms=sortperm(sortby; rev=true)
    positions_sorted=@view positions[perms]
    return positions_sorted[1:i]
end

if !haskey(input_file, "energy_var")
    @info "Building energy variance since it wasn't present in $(ARGS[1])"
    energy=hcat([output["energy"] for output in trajectory]...) # Combine energies for all trajectories into a single Matrix
    energy_var=dropdims(var(energy; dims=2);dims=2) # Determine variance across energies for the same snapshots
    println(size(energy_var))
    input_file["energy_var"]=energy_var
elseif length(size(input_file["energy_var"]))==2 # Flatten if 2D
    @info "Flattening energy variance since it was multidimensional."
    new_energy_var=input_file["energy_var"][:]
    jldsave(replace(ARGS[1], ".jld2" => "_new.jld2"); results=input_file["results"], parameters=input_file["parameters"], energy_var=new_energy_var, forces_var=input_file["forces_var"])
    input_file=jldopen(replace(ARGS[1], ".jld2" => "_new.jld2"), "a")
    @warn "Flattened energy variance and wrote a new output file to $(replace(ARGS[1], ".jld2" => "_new.jld2"))"
end

println(input_file["energy_var"][1])

if !haskey(input_file, "forces_var")
    @info "Building force variance since it wasn't present in $(ARGS[1])"
    forces_full=[cat([output["forces"][i] for output in trajectory]...; dims=3) for i in eachindex(trajectory[1]["forces"])]
    forces_var=map(x->dropdims(var(x; dims=3);dims=3), forces_full)
    input_file["forces_var"]=forces_var
end

println(sum(input_file["forces_var"][1][:,[55,56]]))

top1percent_energy=Dict(
    "label" => "Top 1% (total energy variance)",
    "positions" => top_i(input_file["results"][1]["positions"], input_file["energy_var"], num_structures),
    "energy_errors" => top_i(input_file["energy_var"], input_file["energy_var"], num_structures),
    "force_errors" => top_i(input_file["forces_var"], input_file["energy_var"], num_structures),
    "parameters" => input_file["parameters"],
)

top1percent_forces=Dict(
    "positions" => top_i(input_file["results"][1]["positions"], map(sum, input_file["forces_var"][:]), num_structures),
    "force_errors" => top_i(map(sum, input_file["forces_var"][:]), map(sum, input_file["forces_var"][:]), num_structures),
    "energy_errors" => top_i(input_file["energy_var"], map(sum, input_file["forces_var"][:]), num_structures),
    "parameters" => input_file["parameters"],
    "label" => "Top 1% (total force variance)",
)

top1percent_H2only=Dict(
    "positions" => top_i(input_file["results"][1]["positions"], map(x->sum(x[:,[55,56]]), input_file["forces_var"][:]), num_structures),
    "force_errors" => top_i(map(x->sum(x[:,[55,56]]), input_file["forces_var"][:]), map(x->sum(x[:,[55,56]]), input_file["forces_var"][:]), num_structures),
    "energy_errors" => top_i(input_file["energy_var"], map(x->sum(x[:,[55,56]]), input_file["forces_var"][:]), num_structures),
    "parameters" => input_file["parameters"],
    "label" => "Top 1% (H2 force variance)",
)

using PyCall
ase_io=pyimport("ase.io")
base_Atoms, base_positions, base_Cell=NQCBase.convert_from_ase_atoms(ase_io.read(input_file["parameters"][3]["starting_structure"]))

@info "Building ase trajectories"
output_trj1=NQCBase.convert_to_ase_atoms(base_Atoms, top1percent_energy["positions"], base_Cell)
output_trj2=NQCBase.convert_to_ase_atoms(base_Atoms, top1percent_forces["positions"], base_Cell)
output_trj3=NQCBase.convert_to_ase_atoms(base_Atoms, top1percent_H2only["positions"], base_Cell)

@info "Adding additional properties and writing output"
databases=[top1percent_energy, top1percent_forces, top1percent_H2only]
outputs=[output_trj1, output_trj2, output_trj3]
names=["$(output_filename)_top1percent_energy_var.xyz", "$(output_filename)_top1percent_forces_var.xyz","$(output_filename)_top1percent_h2forces_var.xyz"]
for set in zip(databases, outputs, names)
    for i in eachindex(set[2])
        set[2][i].info=PyDict(Dict("energy_var" => set[1]["energy_errors"][i],"forces_var" => sum(set[1]["force_errors"][i])))
    end
    ase_io.write(set[3], set[2])
end
