# BringBackShapes Environment

## Setup

```bash
conda env create -f env.yml
conda activate bringbackshapes
conda env update --file env.yml --prune
pip install -e .
```

last command to update the env after making mods to the `requirements.txt`

Make sure you have installed mujoco and mujoco_py. [Download](https://mujoco.org/download) and extract `mujoco210` into your `~/.mujoco` directory.

```bash
pip install 'mujoco-py<2.2,>=2.1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sjoshi/.mujoco/mujoco210/bin
```

## Envs

```bash
python demos/demo_2d_playground.py
python demos/demo_2d_env.py
```

to run the current state of the 2d env.
