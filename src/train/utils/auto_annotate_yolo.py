"""
Automatic Bbox Annotation using Pretrained YOLOv8
Generates YOLO format annotations for meat detection dataset.
"""
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def auto_annotate_dataset():
    """
    Use pretrained YOLOv8 to auto-generate bbox annotations.
    Detects 'food' category from COCO and saves as YOLO format.
    """
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    source_dataset = project_root / "meat_freshness_dataset" / "Meat Freshness.v1-new-dataset.multiclass"
    output_dataset = project_root / "meat_detection_dataset"
    
    # Load pretrained YOLOv8
    print("[Auto-Annotate] Loading YOLOv8 (COCO weights)...")
    model = YOLO('yolov8n.pt')  # Nano model (fastest)
    
    # COCO class IDs for food-related items
    # 46: banana, 47: apple, 48: sandwich, 49: orange, 50: broccoli, 51: carrot, 52: hot dog, 53: pizza, etc.
    # We'll use ALL detections and filter by confidence
    
    for split in ['train', 'valid']:
        print(f"\n[Auto-Annotate] Processing {split} split...")
        
        source_dir = source_dataset / split
        if not source_dir.exists():
            print(f"Skipping {split} (not found)")
            continue
        
        # Create output directories
        images_dir = output_dataset / "images" / split
        labels_dir = output_dataset / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each class folder
        total_images = 0
        total_detections = 0
        
        for class_folder in source_dir.iterdir():
            if not class_folder.is_dir():
                continue
            
            print(f"  Processing class: {class_folder.name}")
            
            for img_path in class_folder.glob("*.jpg"):
                # Run detection
                results = model(str(img_path), verbose=False)
                
                # Get image dimensions
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                
                # Copy image to new location
                dest_img_path = images_dir / img_path.name
                cv2.imwrite(str(dest_img_path), img)
                
                # Save YOLO format annotations
                label_path = labels_dir / f"{img_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            # Filter: only high confidence detections
                            if box.conf[0] < 0.3:
                                continue
                            
                            # Convert to YOLO format (class_id x_center y_center width height)
                            # All detections -> class 0 (meat)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            total_detections += 1
                
                total_images += 1
                
                if total_images % 100 == 0:
                    print(f"    Processed {total_images} images...")
        
        print(f"  Total images: {total_images}, Total detections: {total_detections}")
    
    # Create data.yaml
    data_yaml_path = output_dataset / "data.yaml"
    with open(data_yaml_path, 'w') as f:
        f.write(f"path: {output_dataset.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/valid\n")
        f.write("\nnames:\n")
        f.write("  0: meat\n")
    
    print("\n[Auto-Annotate] Complete!")
    print(f"Dataset saved to: {output_dataset}")
    print(f"Config: {data_yaml_path}")

if __name__ == "__main__":
    auto_annotate_dataset()
