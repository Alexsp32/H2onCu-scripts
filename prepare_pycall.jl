using Pkg
# ? No, you can't skip the build step by checking if PyCall.python matches with `which python`, since `using PyCall` will segfault if loaded Python version differs 🙄 
# This script assumes that the desired python environment is now loaded and a python executable is available as "python"
ENV["PYTHON"]=read(`which python`, String)[begin:end-1] # Set the python path (remove trailing newline)
Pkg.build("PyCall")
Pkg.precompile()
using PyCall
println("PyCall redirected to python executable located at:")
printstyled(ENV["PYTHON"]; color=:green)
exit()
