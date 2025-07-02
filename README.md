# Parametric PDE discovery using UBIC 
Using the uncertainty-penalized Bayesian information criterion (UBIC) originally proposed in [Adaptive Uncertainty-Penalized Model Selection for Data-Driven PDE Discovery](https://ieeexplore.ieee.org/document/10401233) to discover parametric PDEs automatically.

- Please first install our modified pysindy package archived at [this OneDrive link](https://1drv.ms/u/c/39cecf604f8b30de/Ed4wi09gz84ggDl7AAAAAAABgZ89ebMdSRESd2a8jiF01w?e=6m0EXl) (Password: UBIC). The more updated version is avaiable at [this repository](https://github.com/Pongpisit-Thanasutives/pysindy).
- To use the L0BnB best-subset solver, please install [the package](https://github.com/Pongpisit-Thanasutives/l0bnb).

The three major files for reproducing the results in [arXiv:2408.08106](https://www.arxiv.org/abs/2408.08106) (accepted as a workshop paper in [the Machine Learning and the Physical Sciences workshop, NeurIPS 2024](https://ml4physicalsciences.github.io/2024/)) are in the following.

- Burgers' PDE: `Parametric_Burgers-DEV3.ipynb`
- Advection-diffusion PDE: `Spatial_Advection_Diffusion_WeakLib_KRR-DEV3.ipynb`
- Kuramoto-Sivashinsky PDE: `Spatial_KS_Equation_Chaotic-DEV2.ipynb`

# Citing UBIC

```
@article{thanasutives2023adaptive,
  author={Thanasutives, Pongpisit and Morita, Takashi and Numao, Masayuki and Fukui, Ken-ichi},
  journal={IEEE Access},
  title={Adaptive Uncertainty-Penalized Model Selection for Data-Driven PDE Discovery},
  year={2024},
  volume={12},
  pages={13165-13182},
  doi={10.1109/ACCESS.2024.3354819}
}

@misc{thanasutives2024adaptation,
  title={Adaptation of uncertainty-penalized Bayesian information criterion for parametric partial differential equation discovery}, 
  author={Thanasutives, Pongpisit and Fukui, Ken-ichi},
  year={2024},
  eprint={2408.08106},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2408.08106}, 
}
```

