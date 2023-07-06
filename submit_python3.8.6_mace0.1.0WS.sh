#!/bin/bash
#SBATCH --job-name="python3.8.6-mace0.1.0WS"
#SBATCH --nodes=%%nodes
#SBATCH --ntasks-per-node=%%tasks
#SBATCH --time=%%time
#SBATCH --mem-per-cpu=%%mem
#SBATCH --cpus-per-task=%%cpus
#SBATCH --partition=%%somewhere
#SBATCH --account=%%blingbling
#SBATCH --output=%%output
#SBATCH --chdir=%%fullpath

# Executable to call
x="julia +release"

# Path to input script
f=full_simulation.jl

# Path to output file
o=%%file_output

# Parallelisation options
SRUN_OPTIONS="-N 1 -n 1 --exclusive"
PARALLEL_OPTIONS="-N 1 --delay=2 -j ${SLURM_NTASKS} --joblog parallel-${SLURM_JOBID}.log"
PARALLEL_EXEC="$x $f --worker"


# Any further definitions
PYTHON_ENV=/home/chem/msrkhg/H2onCu/python3.8.6_mace0.1.0ws_torchcpu/bin/activate
export JULIA_PROJECT=/home/chem/msrkhg/H2onCu/julia1.9
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
pycall_script=/home/chem/msrkhg/H2onCu/scripts/prepare_pycall.jl

# Pre-execution step to load correct python and link to Julia's PyCall
echo "Slurm job id is: ${SLURM_JOB_ID}"
module purge;
module load GCCcore/10.2.0 Python/3.8.6 parallel/20210322; # Necessary due to requirements for torch and MACE0.1.0
echo "Modules loaded, activating virtual environment";
source $PYTHON_ENV
echo "(1/4) Python executable in scope is:";
echo | which python;
echo "
(2/4) Rebuilding PyCall with Python executable in scope:";
$x --project=$JULIA_PROJECT $pycall_script

echo "
\
(3/4) Creating job queue from $f:";
$x --project=$JULIA_PROJECT $f 

echo "(4/4) Running parallel jobs from $f:"
parallel $PARALLEL_OPTIONS srun $SRUN_OPTIONS $PARALLEL_EXEC ::: {1..2}

