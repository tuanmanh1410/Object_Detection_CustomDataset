# A Detection Faster RCNN PyTorch implementation for detecting object with custom dataset

The project refer to **[FasterRCNN Pytorch Training Pipeline](https://github.com/sovit-123/fasterrcnn-pytorch-training-pipeline#Train-on-Custom-Dataset)**

Train PyTorch FasterRCNN models easily on any custom dataset. Choose between official PyTorch models trained on COCO dataset, or choose any backbone from Torchvision classification models, or even write your own custom backbones. 

***In my experimental results, I run with one GPU 2080 8GB with maximum batch size 8***.

## Get Started


* [Setup for Ubuntu](#Setup-for-Ubuntu)
* [Train on Custom Dataset](#Train-on-Custom-Dataset)
* [Inference](#Inference)

## Setup for Ubuntu

1. Clone the repository.


2. Install requirements.

   1. **Method 1**: If you have CUDA and cuDNN set up already, do this in your environment of choice.

      ```
      pip install -r requirements.txt
      ```

   2. **Method 2**: If you want to install PyTorch with CUDA Toolkit in your environment of choice.

      Please install the version with CUDA support as per your choice from Pytorch GetStarted **[here](https://pytorch.org/get-started/locally/)**.

      Then install the remaining requirements file.
   
   3. **Method 3**(Recommended): If your machine already had anaconda , you can using virtual_env.yml file to create env.

      ```
      conda env create -f virtual_env.yml
      ```

## Train on Custom Dataset

The `egg_mono.yaml` is in the `data_configs` directory. Assuming, we store the egg data in the `data` directory

```
├── data
│   ├── Detection_Data
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── README.md
├── data_configs
│   |── egg_mono.yaml
|   ...
├── models
│   ├── create_fasterrcnn_model.py
│   ...
│   └── __init__.py
├── outputs
│   ├── inference
│   └── training
│       ...
├── readme_images
│   ...
├── torch_utils
│   ├── coco_eval.py
│   ...
├── utils
│   ├── annotations.py
│   ...
├── datasets.py
├── inference.py
├── inference_video.py
├── plot_AP.py
├── README.md
├── requirements.txt
├── train.py
└── validate.py
```

The content of the `egg_mono.yaml` should be the following: (Just example, depend on your dataset. The data path should be absoluted) 

```yaml
# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES: ../../xml_data/smoke_pascal_voc/archive/train/images
TRAIN_DIR_LABELS: ../../xml_data/smoke_pascal_voc/archive/train/annotations
# VALID_DIR should be relative to train.py
VALID_DIR_IMAGES: ../../xml_data/smoke_pascal_voc/archive/valid/images
VALID_DIR_LABELS: ../../xml_data/smoke_pascal_voc/archive/valid/annotations

# Class names.
CLASSES: [
    '__background__',
    'Normal',
    'Blood Spot'
    'Crack',
    'Bleached'
    'Impurity'
    'Deformity'
]

# Number of classes (object classes + 1 for background class in Faster RCNN).
NC: 7

# Whether to save the predictions of the validation set while training.
SAVE_VALID_PREDICTION_IMAGES: True
```

***Note that*** *the data and annotations can be in the same directory as well. In that case, the TRAIN_DIR_IMAGES and TRAIN_DIR_LABELS will save the same path. Similarly for VALID images and labels. The `datasets.py` will take care of that*.

Next, to start the training, you can use the following command.

**Command format:**

```
python train.py --config <path to the data config YAML file> --epochs 100 --model <model name (defaults to fasterrcnn_resnet50)> --project-name <folder name inside output/training/> --batch-size 16
```

**In this case, the exact command would be:**

```
python train.py --config data_configs/egg_mono.yaml --epochs 50 --model fasterrcnn_resnet50_fpn --project-name Egg_Detection --batch-size 8
```

## Inference

### Image Inference on Pretrained Model

By default using Faster RCNN ResNet50 FPN model.

```
python inference.py
```

Use model of your choice with an image input.

```
python inference.py --model fasterrcnn_mobilenetv3_large_fpn --input example_test_data/image_1.jpg
```

### Image Inference in Custom Trained Model

In this case you only need to give the weights file path and input file path. The config file and the model name will be automatically inferred from the weights file.

```
python inference.py --input data/inference_data/image_1.jpg --weights outputs/training/Egg_Detection/best_model.pth
```

