# Updates

- (July, 2023) Update the python-julia interface with the SOTA version of [JuliaCall & PythonCall](https://github.com/JuliaPy/PythonCall.jl)


# Recast Code of the Paper ['Hybrid Imitation Learning for Real-Time Service Restoration in Resilient Distribution Systems'](https://ieeexplore.ieee.org/document/9424985)

Please use this bibtex if you want to cite this repository in your publications:

    @misc{IL_Self_Healing,
      author = {Ran Zhu},
      title = {Recast Code of the Paper 'Hybrid Imitation Learning for Real-Time Service Restoration in Resilient Distribution Systems'},
      year = {2023},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/RanZhu1989/IL_Self_Healing}},
    }

## Thanks
Thanks to the authors of the original paper for their valuable research and [code](https://github.com/whoiszyc/IntelliHealer) on Github.

## Abstract
This is the reproduced code for the paper titled "Hybrid Imitation Learning for Real-Time Service Restoration in Resilient Distribution Systems". Compared to the [original code](https://github.com/whoiszyc/IntelliHealer), this recast code has the following updates:

|              | This repository      | Original     
|--------------|--------------|--------------| 
| **Optimization environment**    | JuMP and Gurobipy (Matrix programming) | Pymoo ('For loop' programming)| 
| **Gymnasium standard environment**    | &#10003; | &#10007;  |
| **Deep learning framework**    | Pytorch | Tensorflow |
| **Running speed**    | ~500% UP | 100% |

The so-called behavior cloning (BC) algorithm in origin paper is actually the data augmentation (DAgger) algorithm. I have corrected this in the recast code.

## Reproduction Results
In general, the recast code reproduces the results in the original paper. The following figures show the success ratio of the proposed algorithm in different scenarios. There are some points slight larger than 1, which is due to numerical issues.
### N-1 Test (Fig. 5 and 11)
![N_1](/pics/N-1_Success_Ratio.png)

### N-2 Test (Fig. 12)
![N_2](/pics/N-2_Success_Ratio.png)

### N-5 Test (Fig. 6)
![N_5](/pics/N-5_Success_Ratio.png)

## Getting started
- Install python environment, `cd` into the gym environment directory `./gym_SelfHealing/` and type `pip install -e .`
- Install Julia environment, add the following packages: `JuMP`, `PythonCall`
- Configure your python-julia interface according to the [documentation](https://juliapy.github.io/PythonCall.jl/stable/pythoncall/), especially add your python path into the julia environment variable `PYTHON`.
- Install solvers:
    - Gurobi: Install Gurobi in your OS, and run `pip install gurobipy` in python or run `Pkg.add("Gurobi")` in Julia (see [documentations](https://github.com/jump-dev/Gurobi.jl))
    - CPLEX: Install CPLEX in your OS, and run `Pkg.add("CPLEX")` in Julia (see [documentations](https://github.com/jump-dev/CPLEX.jl))
    - GLPK: Just run `Pkg.add("GLPK")` in Julia
- You can test your environment configuration by running the scripts in `./test/` directory.
- To run the DAgger algorithm, set hyperparameters according to `configs.py`

- You can edit the system file in `./gym_SelfHealing/selfhealing_env/envs/case_data/*.xlsx`.

- Known dependencies: 
  - **Python:** Python (3.8.19), PyTorch (2.3.0), Gymnasium (0.28.1), CUDA (12.1), Gurobipy (11.0), juliacall (0.9.20)
  - **Julia:** Julia (1.10.3), JuMP (1.22.1), PythonCall (0.9.20)
  - **Solvers:** Gurobi (v10 and v11), CPLEX (12.10), GLPK (5.0)

