import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np
from torchvision.ops import nms, box_convert
from torchvision.ops import box_iou


# Load model
model_path = "/home/ss6928/E6691_assign2/runs/detect/train11/weights/best.pt"  
model = YOLO(model_path)

# Define image and label paths
image_folder_path = "/home/ss6928/E6691_assign2/Data/val/images"
label_folder_path = "/home/ss6928/E6691_assign2/Data/val/labels"

# Function to read labels in YOLO format
def read_labels(label_path):
    labels = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
    return labels

# Convert YOLO format (center x, center y, width, height) to (x1, y1, x2, y2)
def yolo_to_x1y1x2y2(labels, image_shape):
    # Convert center coordinates (x_center, y_center) and size (width, height) to (x1, y1, x2, y2)
    new_labels = np.zeros((labels.shape[0], 4))
    new_labels[:, 0] = labels[:, 1] - labels[:, 3] / 2  # x1 = x_center - width / 2
    new_labels[:, 1] = labels[:, 2] - labels[:, 4] / 2  # y1 = y_center - height / 2
    new_labels[:, 2] = labels[:, 1] + labels[:, 3] / 2  # x2 = x_center + width / 2
    new_labels[:, 3] = labels[:, 2] + labels[:, 4] / 2  # y2 = y_center + height / 2

    # Scale the coordinates to the image dimensions
    new_labels[:, [0, 2]] *= image_shape[1]  # Scale x coordinates to image width
    new_labels[:, [1, 3]] *= image_shape[0]  # Scale y coordinates to image height

    return new_labels

# Process images and compute metrics
iou_threshold = 0.2
conf_threshold = 0.3

def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.3):
    
    if pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
        return 0.0, 0.0  # Precision and recall
    if not isinstance(pred_boxes, torch.Tensor):
        pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
    if not isinstance(true_boxes, torch.Tensor):
        true_boxes = torch.tensor(true_boxes, dtype=torch.float32)
    
    device = pred_boxes.device
    true_boxes = true_boxes.to(device)

    # Check for empty inputs
    if pred_boxes.shape[0] == 0 or true_boxes.shape[0] == 0:
        return 0.0, 0.0  # Precision and recall
    
    # Calculate IoU
    ious = box_iou(pred_boxes, true_boxes)
    max_iou, max_idx = ious.max(1)
    
    # Determine True Positives (TPs)
    TP = max_iou >= iou_threshold
    num_TP = TP.sum().item()
    num_FP = TP.shape[0] - num_TP  # False Positives (FP) are detections that didn't match any ground truth
    num_FN = true_boxes.shape[0] - num_TP  # False Negatives (FN) are ground truths not detected
    
    # Calculate precision and recall
    precision = num_TP / (num_TP + num_FP) if num_TP + num_FP > 0 else 0
    recall = num_TP / (num_TP + num_FN) if num_TP + num_FN > 0 else 0
    
    return precision, recall



for image_path in Path(image_folder_path).glob('*.jpg'):
    # Load image
    image = Image.open(image_path).convert("RGB")
    results = model(image)
    boxes_object = results[0].boxes

    # Extract the bounding boxes, confidences, and class IDs
    boxes = results[0].boxes.data    # Tensor of shape [num_detections, 6]
    confidences = boxes[:, 4]  # Confidence scores
    class_ids = boxes[:, 5]    # Class IDs
  
    # Filter detections by confidence score
    conf_threshold = 0.5
    keep = confidences >= conf_threshold
    filtered_boxes = boxes[keep]
    boxes_for_nms = filtered_boxes[:, :4]
    filtered_confidences = confidences[keep]
    filtered_class_ids = class_ids[keep]
    
    # Apply NMS
    keep_nms_indices = nms(boxes_for_nms, filtered_confidences, iou_threshold)
    final_detections = filtered_boxes[keep_nms_indices]
    
    final_boxes = filtered_boxes[keep_nms_indices]
    final_boxes = final_boxes[:, :4]
    final_confidences = filtered_confidences[keep_nms_indices]
    final_class_ids = filtered_class_ids[keep_nms_indices]

    # Load labels
    label_path = label_folder_path + '/' + image_path.name.replace('.jpg', '.txt')
    labels = read_labels(label_path)
    true_boxes = yolo_to_x1y1x2y2(labels, image.size)

    # Convert to tensors
    true_boxes_tensor = torch.tensor(true_boxes[:, :4], dtype=torch.float32, device=final_boxes.device)
    
    # Calculate IoU for each prediction and true box pair
    i = 1
    print("final_boxes",final_boxes)
    print("true_boxes_tensor",true_boxes_tensor)
    ious = box_iou(final_boxes, true_boxes_tensor)
    i +=1
    if i >10:
        break
    #print(ious)
    precision, recall = calculate_precision_recall(final_boxes, true_boxes_tensor, iou_threshold)
    print(f"Precision: {precision}, Recall: {recall}")


print("Metrics calculated.")
