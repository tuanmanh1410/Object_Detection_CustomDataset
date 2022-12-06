import numpy as np
import cv2

def inference_annotations(
    outputs, detection_threshold, classes,
    colors, orig_image
):
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()
    # Filter out boxes according to `detection_threshold`.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    draw_boxes = boxes.copy()
    # Get all the predicited class names.
    pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]
    # Compute the predicted class scores.
    pred_scores = outputs[0]['scores'].cpu().numpy()
    # Keep the highest score for each box.
    #pred_scores = pred_scores[pred_scores >= 0.5]    
    # Draw the bounding boxes and write the class name with score.
    for i, box in enumerate(draw_boxes):
        color = colors[classes.index(pred_classes[i])]
        color = color.astype(np.int32).tolist()
        # Draw bounding box.
        if pred_scores[i] >= 0.5:
            cv2.rectangle(
                orig_image, (box[0], box[1]), (box[2], box[3]),
                color, thickness=2
            )
            # Write the class name and score in two lines.
            # Write two lines of text.
            text1 = f"{pred_classes[i]}"
            text2 = f"{pred_scores[i]:.2f}"
            # Combine text1 and text2 with two lines
            #text = f"{pred_classes[i]}: {pred_scores[i]:.2f}"
            cv2.putText(
                orig_image, text1, (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            cv2.putText(
                orig_image, text2, (box[0], box[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    return orig_image