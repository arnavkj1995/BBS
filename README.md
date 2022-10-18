# BringBackShapes Environment
Codebase for the BringBackShapes (BBS) environment introduced in [Learning Robust Dynamics through Variational Sparse Gating](https://arnavkj1995.github.io/pubs/Jain22.pdf).

![](assets/bbs.gif)

## Setup

```bash
conda env create -f env.yml
conda activate bringbackshapes
conda env update --file env.yml --prune
pip install -e .
```

To run a demo of the environment:
```bash
python demos/demo_2d_env.py
```

## Bibtex
If you find this code useful, please cite:

```
@InProceedings{Jain22,
    author    = "Jain, Arnav Kumar and Sujit, Shivakanth and Joshi, Shruti and Michalski, Vincent and Hafner, Danijar and Kahou, Samira Ebrahimi",
    title     = "Learning Robust Dynamics through Variational Sparse Gating",
    booktitle = {Advances in Neural Information Processing Systems},
    month     = {December},
    year      = {2022}
  }
```
