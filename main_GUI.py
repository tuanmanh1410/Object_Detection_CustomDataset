# Build the Object Dectection GUI using Tkinter with Pytorch
# https://www.youtube.com/watch?v=QX4Z3Z1wYqU
# 
# Path: main_GUI.py
# Compare this snippet from inference.py:
#     # Define the detection threshold any detection having
#     # score below this will be discarded.
#     detection_threshold = args['threshold']

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import cv2
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms

# Define the device to be used.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the classes to be detected.
CLASSES = [
    '__background__',
    'Normal',
    'Blood Spot',
    'Crack',
    'Bleached',
    'Impurity',
    'Deformity',
]

# Define the colors for the bounding boxes.
COLORS = np.array([[220,220,220], [30,144,255], [151,255,255], [238,18,137], [180,238,180], [255,165,0], [191,62,255]])


# Define the model.
checkpoint = torch.load('./outputs/training/Egg_1119_6classes/best_model.pth', map_location=DEVICE)
NUM_CLASSES = 7
model_name = checkpoint['model_name']
build_model = create_model[str(model_name)]
model = build_model(num_classes=NUM_CLASSES, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# Set the model to the device.
model.to(DEVICE)

# Define the detection threshold any detection having
# score below this will be discarded.
detection_threshold = 0.5

# Define the function to be called when the user clicks on the
# "Select Image" button.
def select_image(panel, panel_1, panel_2):
    # Allow user to select a single image.
    path = filedialog.askopenfilename()
    # Check if the user selected an image.
    if len(path) > 0:
        # Load the image.
        image = cv2.imread(path)
        # Resize the image.
        image = cv2.resize(image, (640, 640))
        orig_image = image.copy()
        # Apply the transforms on the image.
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        # Convert the image to PIL format.
        image_1 = Image.fromarray((image).astype(np.uint8))
        # Convert the image to ImageTk format.
        image_1 = ImageTk.PhotoImage(image_1)
        # If the panel is not None, we need to initialize it.
        if panel_1 is None:
            # The first panel will store our original image.
            panel_1 = tk.Label(image=image_1)
            panel_1.image = image_1
            panel_1.pack(side="right", padx=10, pady=10)
        # Otherwise, simply update the panel.
        else:
            # Update the panel.
            panel_1.configure(image=image_1)
            panel_1.image = image_1

        image = infer_transforms(image)
        # Add batch dimension.
        image = torch.unsqueeze(image, 0)
        # Carry out inference.
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
        # Annotate the image.
        orig_image = inference_annotations(
            outputs, detection_threshold, CLASSES,
            COLORS, orig_image
        )
        # Convert the image to PIL format.
        image = Image.fromarray((orig_image).astype(np.uint8))
        # Convert the image to ImageTk format.
        image = ImageTk.PhotoImage(image)
        # If the panel is not None, we need to initialize it.
        if panel is None:
            # The first panel will store our original image.
            panel = tk.Label(image=image)
            panel.image = image
            panel.pack(side="right", padx=10, pady=10)
        # Otherwise, simply update the panel.
        else:
            # Update the panel.
            panel.configure(image=image)
            panel.image = image
        
        # Get all bounding boxes.
        boxes = outputs[0]['boxes']
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        # Crop the orriginal image according to the bounding boxes.
        for box in boxes:
            # Get the coordinates of the bounding box.
            x1, y1, x2, y2 = box
            # Crop the image.
            crop = image_1[y1:y2, x1:x2]
            # Resize the cropped image to 240
            # Save the image.
            cv2.imwrite('./outputs/inference/crops/' + str(time.time()) + '.jpg', crop)
            # Show the cropped images in bottom panel.
            # Convert the image to PIL format.
            crop = Image.fromarray((crop).astype(np.uint8))
            # Convert the image to ImageTk format.
            crop = ImageTk.PhotoImage(crop)
            # If the panel is not None, we need to initialize it.
            if panel_2 is None:
                # The first panel will store our original image.
                panel_2 = tk.Label(image=crop)
                panel_2.image = crop
                panel_2.pack(side="bottom", padx=10, pady=10)
            # Otherwise, simply update the panel.
            else:
                # Update the panel.
                panel_2.configure(image=crop)
                panel_2.image = crop
            


# Initialize the window toolkit along with the two buttons and panel.
root = tk.Tk()
# Change the title of the window.
root.title("Egg Quality Assurance")
# Change the size of the window.
root.geometry("800x600")
# Add some message to the window.
message = tk.Label(
    root, text="Select input images to carry out inference.",
    font=('Helvetica', 14)
)
# Create the option menu.
option = tk.StringVar(root)
option.set("Select detection model")
option_menu = tk.OptionMenu(root, option, "Select an option", "Binary Detection", "Multi-class Detection")

'''
# Intialize the panel.
panel = None
# Intialize the panel showing the original image.
panel_original = None
'''

# Show the original image on the left side of the window.
panel_original = tk.Label(image=None)
panel_original.pack(side="left", padx=10, pady=10)

# Show the annotated image on the right side of the window.
panel = tk.Label(image=None)
panel.pack(side="right", padx=10, pady=10)

# Show the cropped images on the bottom side of the window.
panel_2 = tk.Label(image=None)
panel_2.pack(side="bottom", padx=10, pady=10)

# Create a button that lets the user select an image.
btn_select_image = tk.Button(
    root, text="Select Image", command=lambda: select_image(panel, panel_original, panel_2)
)

# Pack the buttons and panel.
message.pack(side="top", padx=10, pady=10)
btn_select_image.pack(side="bottom", padx=10, pady=10)

# Start the GUI.
root.mainloop()

