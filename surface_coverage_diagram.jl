using CairoMakie,NQCDynamics,UnitfulAtomic,Unitful,JLD2, KernelDensity,PyCall,ArgParse

arg_opts=ArgParseSettings()
@add_arg_table arg_opts begin
	"--structure", "-s"
		help = "Starting structure file in a format readable by `ase.io`."
		arg_type = String
	"--trajectory", "-t"
		help = "Trajectory file (JLD2 format) to use. Must have a `results` key containing a (Vector,Dict) Tuple containing the MC positions."
		arg_type = String
	"--output", "-o"
		help = "Output filename to save the plot to."
		arg_type = String
end

run_args=parse_args(arg_opts)



function surface_coverage_diagram(structure::String, trajectory::String, h_atoms::Union{Vector, CartesianIndices, UnitRange}=55:56, output_file::String="Figure.pdf"; 
	atoms_to_draw::Union{Vector, CartesianIndices, UnitRange}=(54-18):54, 
	padding=1.5, 
	cu_marker_size=100, 
	n_atoms_per_layer::Int=9,
	alpha_max=0.75
	)
	ase_io=pyimport("ase.io")
	starting_structure=ase_io.read(structure)
	_,starting_positions,cell=NQCBase.convert_from_ase_atoms(starting_structure)
	starting_positions=ustrip.(u"Å", auconvert.(u"Å", starting_positions))
	cell_ang=au_to_ang.(cell.vectors)
	
	# Monte Carlo trajectory file
	trajectory_file=jldopen(trajectory)["results"]
	positions=trajectory_file[1]

	# Load positions file and wrap around cell boundaries
	for i in eachindex(positions)
	    NQCBase.apply_cell_boundaries!(cell, positions[i])
	end
	positions_ang=[ustrip.(u"Å", auconvert.(u"Å", config)) for config in positions]
	println("File loaded")

	# Position scatter plot	
	# Create H2 position kde
	xy_kde=kde(vcat([config[1:2, h_atoms]' for config in positions_ang]...); boundary=((-padding,sum(cell_ang[1,1:2])+padding),(-padding, max(cell_ang[2,:]...)+padding)))

	# Plot the figure
	f1=Figure(;size=28.3 .* 2 .* (sum(cell_ang[1,1:2]),max(cell_ang[2,:]...)))
	atoms_ax=Axis(f1[1,1], xlabel="X coordinate / Å", ylabel="Y coordinate / Å", xminorticksvisible=true, yminorticksvisible=true)
	H_density=heatmap!(xy_kde.x, xy_kde.y, xy_kde.density; colormap=:inferno, alpha=1.0)
	layers_to_draw=ceil(length(atoms_to_draw)/n_atoms_per_layer) # Number of layers we need. 
	alpha=reverse(collect(0.0:(alpha_max/layers_to_draw):alpha_max)) # Fading intensity
	# Draw surface atoms in background
	atoms_to_draw=reverse(collect(atoms_to_draw))
	for i in 0:layers_to_draw-2
		scatter!(starting_positions[1:2,atoms_to_draw[Int(n_atoms_per_layer*i+1):Int(n_atoms_per_layer*(i+1))]], color=("darkorange3",alpha[Int(i+1)]), marker=:circle, markersize=cu_marker_size, strokewidth=0.25, strokecolor="chocolate4")
	end
	# Draw cell boundaries
	poly!(Point2f[(0,0), (cell_ang[1,1],0), (sum(cell_ang[1,1:2]), cell_ang[2,2]), (cell_ang[1,2], cell_ang[2,2])], strokecolor="white" ,strokewidth=1, color=("black", 0.0))
	# Centre view on cell
	xlims!(-padding,sum(cell_ang[1,1:2])+padding)
	ylims!(-padding,max(cell_ang[2,:]...)+padding)
	# Add color bar
	Colorbar(f1[1,2], H_density, label="H atom position density")
	# Save output
	save(output_file, f1)
end

