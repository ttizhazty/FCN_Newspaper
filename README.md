# Fully Convolutional Neural Network for Newspaper Article Segementation
This is a reporduce of [Fully Convolutional Neural Network for Newspaper Article Segementation](https://pd.zhaw.ch/publikation/upload/212962.pdf) with Pytorch 1.0. The model is trained by [UCI Newspaper and Magazine Images Segmentation Dataset ](https://archive.ics.uci.edu/ml/datasets/Newspaper+and+magazine+images+segmentation+dataset)

### Requirements
```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```
When you're done working on the project, deactivate the virtual environment with ```deactivate```

### Downloading Dataset
Download dataset [here](https://archive.ics.uci.edu/ml/datasets/Newspaper+and+magazine+images+segmentation+dataset). It contains the original newspaper and magazine images and corresponding masks. The structure of data:
```
dataset_segmentation/
        1_m.png
        1.jpg
        ...
```
### WorkSpace
```
FCN_Newspaper\
    data\
    experiment\
        base_model\
        finetune_model\
    model\
        data_loader.py
        feature_extract.py
        net.py
    build_dataset.py
    train.py
    utils.py
```
This is the structure of the workspace, where data will save the training, testing and validation data, the experiment will save configuration files and weights for the Neural Network of models, and model contains the framework of Neural Network.
### Quick Start
##### 1. To build the dataset for training progress, run
```
python3 build_dataset.py --data_dir data/dataset_segmentation --output_dir data/FCN_dataset
```
##### 2. To train the model, run
```
python3 train.py --data_dir ata/dataset_segmentation --model_dir experiments/base_model
```