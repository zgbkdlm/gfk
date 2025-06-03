# Generative diffusion posterior sampling for informative likelihoods
This implementation is associated with the paper xxx. 
In the paper we develop a new approach for conditional sampling of generative diffusion models with sequential Monte Carlo methods.

<img src="./experiments/post.svg" style="width: 80%; height: auto; display: block; margin-left: auto; margin-right: auto">

# Installation
Install the package via a standard procedure:

```bash
git clone git@github.com:zgbkdlm/gfk.git
cd gfk
pip install -e .
```

Depending on whether you need to run in a CPU/GPU, you may want to uninstall `jax`and `jaxlib` and then reinstall.

# Reproduce experiments
To exactly reproduce the numbers and figures in the paper, first run experiments:

```bash
cd experiments
python runs_gms/bash_aux.sh --dx=256 --nparticles=16384
python runs_gms/bash_aux_noiseless.sh --dx=256 --nparticles=16384
python runs_gms/bash_mcgdiff.sh --dx=256 --nparticles=16384
python runs_gms/bash_wu.sh --dx=256 --nparticles=16384
```

Then, run the scripts in `./summary` to produce the tables and figures, e.g.,

```bash
cd experiements
python ./summary/tabulate_gms.py
```

will produce the table. 

# Citation
@article{Zhao2025b0smc, 
    author = {Zhao, Zheng}, 
    title = {Generative diffusion posterior sampling for informative likelihoods},
    journal = {arXiv preprint arXiv:2506.01083},
    year = {2025},
}

# Contact
Zheng Zhao, Linköping University, https://zz.zabemon.com.
