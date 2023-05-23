using Pkg
using PyCall

# This script assumes that the desired python environment is now loaded and a python executable is available as "python"
ENV["PYTHON"]=read(`which python`, String)[begin:end-1] # Set the python path (remove trailing newline)
if ENV["PYTHON"]!=PyCall.python
    Pkg.build("PyCall")
    Pkg.precompile()
    using PyCall
    println("PyCall redirected to python executable located at:")
    printstyled(ENV["PYTHON"]; color=:green)
else
    println("PyCall already passing function calls to python executable located at:")
    printstyled(PyCall.python; color=:green)
end
exit()
