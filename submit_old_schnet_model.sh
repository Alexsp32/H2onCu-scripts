#!/bin/bash
#SBATCH --job-name="oldschnet"
#SBATCH --nodes=%%nodes
#SBATCH --ntasks-per-node=%%tasks
#SBATCH --time=%%time
#SBATCH --mem-per-cpu=%%mem
#SBATCH --cpus-per-task=%%cpus
#SBATCH --output=%%output
#SBATCH --chdir=%%fullpath

# Path to executable
x=/home/chem/msrkhg/bin/julia

# Path to input script
f=full_simulation.jl

# Path to output file
o=%%file_output


# Any further definitions
export JULIA_PROJECT=%%project
pycall_script=/storage/mssgwp_grp/msrkhg/H2onCu/scripts/prepare_pycall.jl


# Pre-execution step to load correct python and link to Julia's PyCall
module purge;
module load GCCcore/11.2.0 GCC/11.2.0;
module load OpenMPI/4.1.1 Python/3.9.6;
echo "(1/3) After module loading, python executable in scope is:";
echo | which python;
echo "(2/3) Rebuilding PyCall to use correct python:";
$x $pycall_script

echo "(3/3) Running full_simulation.jl:";
# Execute job
$x $f --julia-project $JULIA_PROJECT --slurm


