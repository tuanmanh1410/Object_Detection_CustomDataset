'''
This is script to compare the mAP and mAP50 results between two different models
Here we have chosen the Faster-RCNN and the DETR model in this case.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Get the value mAP and mAP50 from csv file path
def get_value(csv_file):
    """
    Gets the mAP values from the csv file.

    :param csv_file: Path to the csv file.
    :return: Two List containing the mAP and mAP50 values.
    """
    F_map_05 = []
    F_map = []
    D_map = []
    D_map_05 = []

    with open(csv_file, 'r') as f:
        for line in f:
            if line.startswith('epoch'):
                continue
            line = line.strip().split(',')
            F_map.append(float(line[1]))
            F_map_05.append(float(line[2]))
            D_map.append(float(line[4]))
            D_map_05.append(float(line[3]))
    return F_map_05, F_map, D_map_05, D_map


def save_mAP(F_map_05, F_map, D_map_05, D_map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.

    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        F_map_05, color='tab:blue', linestyle='-', 
        label='mAP@0.5-FasterRCNN'
    )
    ax.plot(
        F_map, color='tab:red', linestyle='-', 
        label='mAP@0.5:0.95-FasterRCNN'
    )
    ax.plot(
        D_map_05, color='tab:green', linestyle='dashdot',
        label='mAP@0.5-DETR'
    )
    ax.plot(
        D_map, color='tab:orange', linestyle='dashdot', 
        label='mAP@0.5:0.95-DETR'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig("./map_combine.png")
    plt.close('all')

# Main function
def main():
    """
    Main function.
    """
    csv_file = './results_old.csv'
    map_05, map, a, b = get_value(csv_file)
    save_mAP(map_05, map, a, b)

# Run main function
if __name__ == '__main__':
    main()



