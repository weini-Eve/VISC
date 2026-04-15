# MmWave Radar Perception Learning using Pervasive Visual-Inertial Supervision

**[Yiwen Zhou](#)**, **[Shengkai Zhang](#)**  
*Wuhan University of technology*

---

This repository is an official PyTorch implementation of the paper *"MmWave Radar Perception Learning using Pervasive Visual-Inertial Supervision"*, which is a radar perception learning framework guided by a pervasive visual-inertial(VI) sensor suite, tracking  moving objects in the lack of 3D observations.

---




## Overview

![Overview](images/overview.jpg)

VIMP consists of two designs. First, we propose a recursive sensor fusion method that recursively uses the IMU measurements to compensate for the temporal drift of VIO. Second, we propose a feature-selection cross-modal learning framework. It first selects background visual features to supervise the radar’s point cloud reliably.


---




## Qualitative evaluation of novel scene flow estimation on synthetic dataset.

![low](images/flow1.jpg)

![low](images/flow2.jpg)

## Qualitative evaluation of motion segmentation and pose estimation on synthetic and real-world dataset.

| motion segmentation                  | pose estimation                   |
|-------------------------|-------------------------|
| ![motion segmentation图片](images/detect.jpg) | ![pose estimation图片](images/odometry.jpg) |

# 1. Installation

> **Note:** our code has been tested on Ubuntu 20.04 with Python 3.7, CUDA 11.1/11.0, PyTorch 1.7. 

Before you run our code, please follow the steps below to build up your environment.

a. Clone the repository to local

```bash
git clone git@github.com:weini-Eve/VISC.git
```

b. Set up a new environment (Python 3.7) with Anaconda

```bash
conda create -n $ENV_NAME$ python=3.7
source activate $ENV_NAME$
```

c. Install common dependencies and pytorch

```bash
pip install -r requirements.txt
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

d. Install [PointNet++](https://github.com/sshaoshuai/Pointnet2.PyTorch) library for basic point cloud operation

```bash
cd lib
python setup.py install
cd ..
```
# 2. Model Training

To train our models, please run:

```bash
python main.py --dataset_path $DATA_PATH$ --exp_name $EXP_NAME$ --model visc
```

Here, `$DATA_PATH$` is the path where you save your preprocessed scene flow samples. `$EXP_NAME$` is the name of the current experiment defined by yourself. Training logs and results will be saved under `checkpoints/$EXP_NAME$/`. Besides, you can also modify training args, such as batch size, learning rate and number of epochs, by editing the configuration file `configs.yaml`.
# 3. Model Evaluation

We provide our trained models under folders in `checkpoints/`.

To evaluate the trained models on the test set, please run:

```bash
python main.py --eval --dataset_path $DATA_PATH$ --exp_name visc --model visc
```

Once the evaluation is completed, the results on different metrics will be printed. If you want to save the model outputs, please add `--save_res` in the command. The results will be saved at `checkpoints/$EXP_NAME$/results/`. To enable the visualization of the estimated scene flow and motion segmentation in BEV, please add `--vis` in the command. The visualization figures will be saved under `checkpoints/$EXP_NAME$/test_vis_flow` and `../test_vis_seg`. For scene flow visualization, the corresponding color wheel is `checkpoints/flow_encoding.png`. In the motion segmentation visualization figures, orange indicates moving points while blue is static.
