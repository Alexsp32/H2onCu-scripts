# H2onCu_scripts
This is a collection of scripts I use to simulate reactive dynamics of H on Cu surfaces. 
More information goes here once I spend the time to write proper documentation. 

I've moved several scripts which might be important for analysis into separate packages, see [[#Julia Environment]] for more details. 

Job submission scripts are designed for one of multiple cluster systems at the University of Warwick - you will probably need to modify them. 

`H2Cu_dynamics.jl` contains all the functions necessary to load model potential energy surfaces, friction models and run MDEF+2TM simulations, while `full_simulation.jl` defines the parameters of the simulations to run, as well as taskfarming statistical sampling of trajectories. 

My scripts heavily rely upon my [ClusterScripts](https://github.com/NQCD/ClusterScripts) package to keep a record of input parameters and taskfarm large batches of simulations with >1000 trajectories per data point. 


## Python environment
**Set this up first, since setting up the Julia environment requires Python to be available!**

Used to run `ase`, as well as to load MACE ML models. 

See [group docs](https://maurergroup.github.io/MaurerGroupDocs/machinelearning/mace0.1.0_wojciech/) for detailed instructions. 

MACE 0.1.0 is known working with `torch==1.8.2` on Python 3.8.6 and 3.9.6. 

## Julia environment
This will work if you have access to all the necessary repositories and an SSH key set up for GitHub:

**Julia must be <1.10, otherwise dependencies will be screwed up!**

```julia
] registry add https://github.com/NQCD/NQCRegistry
registry add https://github.com/ACEsuit/ACEregistry.git
activate .
instantiate
precompile
```



