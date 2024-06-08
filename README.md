# PyHippSims - A Python package for analyzing galaxy simulations

[![Last commit](https://img.shields.io/github/last-commit/ChenYangyao/pyhipp-sims/master)](https://github.com/ChenYangyao/pyhipp-sims/commits/master)
[![Workflow Status](https://img.shields.io/github/actions/workflow/status/ChenYangyao/pyhipp-sims/run-test.yml)](https://github.com/ChenYangyao/pyhipp-sims/actions/workflows/run-test.yml)
[![MIT License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/ChenYangyao/pyhipp-sims/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyhipp-sims)](https://pypi.org/project/pyhipp-sims/)

The package provides a set of unified interface to load and analyze N-body and hydrodynamical simulations, mainly for 
galaxy formation.

To install, run:
```bash
pip install pyhipp-sims
```
Alternatively, you can clone the repository and install the package locally via `pip install -e /path/to/the/repo`.


## Usage 

See the Jupyter notebooks under `docs/`:
- `load_simulation_info.ipynb`: Load the basic information of a simulation run (e.g. box size, mass table, cosmology, etc.). A set of commonly used suites of simulations are predefined (e.g. TNG, EAGLE, etc. No need to download the actual simulation data).

## Contributors

- Yangyao Chen (USTC, [yangyaochen.astro@foxmail.com](mailto:yangyaochen.astro@foxmail.com)).
