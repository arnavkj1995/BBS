# BringBackShapes
Codebase for BringBackShapes environment introduced in [Learning Robust Dynamics Through Variational Sparse Gating](https://github.com/arnavkj1995/VSG).

## Setup

```bash
conda env create -f env.yml
conda activate bringbackshapes
conda env update --file env.yml --prune
pip install -e .
```

Make sure you have installed mujoco and mujoco_py. [Download](https://mujoco.org/download) and extract `mujoco210` into your `~/.mujoco` directory.

```bash
pip install 'mujoco-py<2.2,>=2.1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

If you find this code useful, please reference in your paper:
```
@InProceedings{Jain22,
    author    = "Jain, A.~K. and Sujit, S. and Joshi, S. and Michalski, V. and Hafner, D. and Kahou, S.~E.",
    title     = "Learning Robust Dynamics through Variational Sparse Gating",
    booktitle = {Advances in Neural Information Processing Systems},
    month     = {December},
    year      = {2022}
  }
```
