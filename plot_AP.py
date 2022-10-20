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
    map_05 = []
    map = []
    with open(csv_file, 'r') as f:
        for line in f:
            if line.startswith('epoch'):
                continue
            line = line.strip().split(',')
            map.append(float(line[1]))
            map_05.append(float(line[2]))
    return map_05, map


def save_mAP(map_05, map):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.

    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:blue', linestyle='-', 
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-', 
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig("./map.png")
    plt.close('all')

# Main function
def main():
    """
    Main function.
    """
    csv_file = './results.csv'
    map_05, map = get_value(csv_file)
    save_mAP(map_05, map)

# Run main function
if __name__ == '__main__':
    main()



