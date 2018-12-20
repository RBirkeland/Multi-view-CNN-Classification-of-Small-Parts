# Multi-view-CNN-Classification-of-Small-Parts

## Setup
```
mkdir checkpoint
```

## Make virtualenv for Python 2.7 and install packages
```
virtualenv --python=/usr/bin/python2.7 --no-site-packages env
source env/bin/activate

pip install -r requirements.txt 
```

## Settings (inside controller.py)
```
modelnet_dataset = '../modelnet'  # Only needed if use_modelnet is True
dataset_path = '../all'           # Only needed if use_modelnet is False

use_modelnet = False              # Use dataloader that supports the ModelNet structure

resume = False                    # Resume from checkpoint
only_test = False                 # Evaluation, only run on test set

lr = 0.0001    
n_epochs = 25
batch_size = 4

architecture = 'resnet'  # Architecture (either alexnet or resnet)
```

## Run network
```
python controller.py
```
