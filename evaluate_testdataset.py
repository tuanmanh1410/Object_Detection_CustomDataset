import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms

from torch_utils.engine import (
    train_one_epoch, evaluate, valid_one_epoch
)
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from utils.general import (
    set_training_dir, Averager, 
    save_model, save_train_loss_plot, save_epoch_loss_plot,
    show_tranformed_image,
    save_mAP, save_model_state, SaveBestModel
)

def parse_opt():
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '-c', '--config', 
        default='data_configs/egg.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-w', '--weights', default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-th', '--threshold', default=0.3, type=float,
        help='detection threshold'
    )
    parser.add_argument(
        '-si', '--show-image', dest='show_image', action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '-mpl', '--mpl-show', dest='mpl_show', action='store_true',
        help='visualize using matplotlib, helpful in notebooks'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-ims', '--img-size', dest='img_size', default=640, type=int, 
        help='image size to feed to the network'
    )
    args = vars(parser.parse_args())
    return args

def main(args):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations
    with open(args['config']) as file:
        data_configs = yaml.safe_load(file)

    # Inference settings and constants.
    TEST_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
    TEST_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    DEVICE = args['device']
    COLORS = np.random.uniform(0, 255, size=(5, 3))
    OUT_DIR = set_infer_dir()
  
    # Model configurations
    IMAGE_WIDTH = args['img_size']
    IMAGE_HEIGHT = args['img_size']

    # Load the pretrained model
    if args['weights'] is None:
        build_model = create_model[args['model']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        NUM_CLASSES = checkpoint['config']['NC']
        CLASSES = checkpoint['config']['CLASSES']
        model_name = checkpoint['model_name']
        build_model = create_model[str(model_name)]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # Load dataset and dataloader
    test_dataset = create_valid_dataset(
        TEST_DIR_IMAGES, TEST_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )
    test_loader = create_valid_loader(test_dataset, 8, 4) # Batch size, num_workers
    print(f"Number of validation samples: {len(test_dataset)}\n")

    # Load the best model.
    # Load the pre-trained model
    if args['weights'] is None:
        build_model = create_model[args['model']]
        model = build_model(num_classes=NUM_CLASSES, coco_model=True)
    # Load weights if path provided.
    if args['weights'] is not None:
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        NUM_CLASSES = checkpoint['config']['NC']
        CLASSES = checkpoint['config']['CLASSES']
        model_name = checkpoint['model_name']
        build_model = create_model[str(model_name)]
        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    
    coco_evaluator, stats, val_pred_image = evaluate(model, test_loader, device=DEVICE, save_valid_preds=False, out_dir=OUT_DIR, classes=CLASSES, colors=COLORS)
    mAP = stats[0]
    print(f"mAP: {mAP}")

    mAP50 = stats[1]
    print(f"mAP50: {mAP50}")
    

if __name__ == '__main__':
    args = parse_opt()
    main(args)