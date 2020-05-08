# Semi-supervised Brain Segmentation

This repository contains a simplistic code for barin MRI segmentation task in the semi-supervised setting.
[The dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation) that is used here contains only 110 patients and is obtained from The Cancer Genome Atlas (TCGA).
Also, some parts of this code are building on top of [this notebook](https://www.kaggle.com/mateuszbuda/brain-segmentation-pytorch).


## Installation
We recommend using python 3.7 and pytorch 1.4.

```
# Create environment
conda create -n brain-segmentation python=3.7

# Install pytorch
# conda install pytorch torchvision -c pytorch # no gpu
# OR
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch # cuda >= 10.1 gpu

# Additional libraries
pip install -r requirements.txt
```


## Quick Start
For intractive running of code follow the [notebook](notebook.ipynb).


## Training and Evaluation in Command Line
To train and evaluate a model run:

```
CUDA_VISIBLE_DEVICES=0 python main.py --alg semi-supervised \
    --search_params '{"num_unlabeled_patients": [95, 50, 5]}' \
    --run run0 
```
