"""
Object Detector Utility using YOLOv8
Provides meat detection with bounding box coordinates.
"""
from ultralytics import YOLO
import numpy as np
import cv2

# Singleton pattern for model loading
_detector_model = None

def initialize_detector(model_path=None):
    """
    Initialize YOLO detector (call once at startup).
    
    Args:
        model_path: Path to detection_model.pt (if None, uses default location)
    """
    global _detector_model
    
    if _detector_model is not None:
        return  # Already initialized
    
    if model_path is None:
        import os
        from pathlib import Path
        script_dir = Path(__file__).parent.parent
        model_path = script_dir / "models" / "detection_model.pt"
    
    print(f"[Object Detector] Loading model from {model_path}...")
    _detector_model = YOLO(str(model_path))
    print("[Object Detector] Initialized successfully")


def detect_meat(image, conf_threshold=0.5):
    """
    Detect meat in image and return bounding boxes.
    
    Args:
        image: numpy array (H, W, 3) in RGB format
        conf_threshold: Minimum confidence threshold
        
    Returns:
        List of bboxes: [(x1, y1, x2, y2, confidence), ...]
        Returns empty list if no detection
    """
    global _detector_model
    
    if _detector_model is None:
        raise RuntimeError("Detector not initialized. Call initialize_detector() first.")
    
    # Run detection
    results = _detector_model(image, verbose=False)
    
    bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()
            if conf < conf_threshold:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bboxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    
    return bboxes


def get_largest_bbox(bboxes):
    """
    Get the largest bbox by area.
    
    Args:
        bboxes: List of (x1, y1, x2, y2, conf)
        
    Returns:
        (x1, y1, x2, y2, conf) or None if empty
    """
    if not bboxes:
        return None
    
    def area(bbox):
        x1, y1, x2, y2, _ = bbox
        return (x2 - x1) * (y2 - y1)
    
    return max(bboxes, key=area)


def crop_to_bbox(image, bbox, padding=10):
    """
    Crop image to bounding box with padding.
    
    Args:
        image: numpy array
        bbox: (x1, y1, x2, y2, conf)
        padding: pixels to add around bbox
        
    Returns:
        Cropped image
    """
    if bbox is None:
        return image  # Return full image if no bbox
    
    x1, y1, x2, y2, _ = bbox
    h, w = image.shape[:2]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image.
    
    Args:
        image: numpy array (will be modified in-place)
        bboxes: List of (x1, y1, x2, y2, conf)
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with drawn bboxes
    """
    for bbox in bboxes:
        x1, y1, x2, y2, conf = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw confidence score
        label = f"{conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image


def is_initialized():
    """Check if detector is initialized."""
    return _detector_model is not None
