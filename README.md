# INTUIT-VLNCE: Autonomous Navigation through Vision-and-Language

This repository contains code for the master's thesis of Kirill Rodionov. 

The code is based on the projects [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and [LAW-VLNCE](https://github.com/3dlg-hcvc/LAW-VLNCE)

## Setup
The setup mimics the installation process of VLN-CE. The code requires [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) both of versions 0.1.7. Install them beforehand.

This project is created with Python 3.6:

```bash
conda create -n vlnce python=3.6
conda activate vlnce
```
[Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) installation:

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) installation:

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
```

You should update requirements.txt to include upper boundaries for legacy versions:
```
gym==0.18.0
numpy>=1.16.1
yacs>=0.1.5
numpy-quaternion>=2019.3.18.14.33.20
attrs>=19.1.0
opencv-python>=3.3.0, <4.7.0.72
pickle5; python_version < '3.8'
# visualization optional dependencies
imageio>=2.2.0
imageio-ffmpeg>=0.2.0
scipy>=1.0.0
tqdm>=4.0.0
numba>=0.44.0
Pillow<=7.2.0
```

```bash
# installs both habitat and habitat_baselines
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

Now you can download this project.
```bash
git clone git@github.com:SirKrovlood/INTUIT-VLNCE.git
cd INTUIT-VLNCE
python -m pip install -r requirements.txt
```

## Issues
Taken from [LAW-VLNCE](https://github.com/3dlg-hcvc/LAW-VLNCE):

If you find an issue installing torch-scatter, use the following and replace {cuda-version} with your cuda version and {torch-version} with your installed torch version: 
pip install torch-scatter==latest+{cuda-version} -f https://pytorch-geometric.com/whl/torch-{torch-version}.html

[Refer torch-scatter](https://github.com/rusty1s/pytorch_scatter)

## Datasets
This projects requires two datasets

#### Scenes: Matterport3D
You will need to sign Terms of Service in order to acquire the download script (`download_mp.py`). Refere to the official site for instructions: [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.
#### Scenes: Episodes: Room-to-Room (R2R)
This project uses **R2R_VLNCE_v1-3 dataset**. Refere to [VLN-CE](https://github.com/jacobkrantz/VLN-CE) page for installation procedures.

## Usage
Code execution is performed through `run.py` script. Provide a configuration file and run type as such:
```bash
python run.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
```


## Training example
The following command will initiate training of a CMA intuition model on a *joint_train_envdrop* split of **R2R_VLNCE_v1-3 dataset**.

```bash
python run.py \
  --exp-config vlnce_baselines/config/paper_configs/intuition_config/cma_pm_aug.yaml \
  --run-type train
```
Note: check the following parameters to be set for training:
```
ENVIRONMENT: MAX_EPISODE_STEPS: 7500
TASK: TYPE: Nav-dual

ENV_NAME: "VLNCEDaggerIntuitionEnv"
```

## Evaluation example
The following command will initiate evaluation of a CMA intuition checkpoints from a directory `EVAL_CKPT_PATH_DIR`  on a `EVAL:SPLIT` specified split of **R2R_VLNCE_v1-3 dataset**.

```bash
python run.py \
  --exp-config vlnce_baselines/config/paper_configs/intuition_config/cma_pm_aug.yaml \
  --run-type eval
```
Note: check the following parameters to be set for training:
```
ENVIRONMENT: MAX_EPISODE_STEPS: 500
TASK: TYPE: VLN-v0

ENV_NAME: "VLNCEDaggerEnv"
```

## Acknowledgements

A huge gratitude is expressed to Jacob Krantz for the [VLN-CE](https://github.com/jacobkrantz/VLN-CE) codebase, as well as the work of the team behind [Language-Aligned Waypoint (LAW) Supervision for Vision-and-Language Navigation in Continuous Environments](https://github.com/3dlg-hcvc/LAW-VLNCE) [[Project Website]](https://github.com/3dlg-hcvc/LAW-VLNCE).

