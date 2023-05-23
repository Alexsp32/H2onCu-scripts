#!/bin/bash
#SBATCH --job-name="python3.8.6-mace0.1.0WS"
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
module load GCCcore/10.2.0 Python/3.8.6; # Necessary due to requirements for torch and MACE0.1.0
echo "Modules loaded, activating virtual environment";
source /storage/mssgwp_grp/msrkhg/H2onCu/python3.8.6_mace0.1.0/bin/activate
echo "(1/3) Loading complete, Python executable in scope is:";
echo | which python;
echo "(2/3) Rebuilding PyCall to use correct python if necessary:";
$x $pycall_script

echo "(3/3) Running full_simulation.jl:";
# Execute job
$x $f --julia-project $JULIA_PROJECT --slurm


