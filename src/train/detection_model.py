"""
YOLOv8 Object Detection Training for Meat Detection
Fine-tunes YOLOv8 on auto-annotated dataset.
"""
import os
from pathlib import Path
from ultralytics import YOLO

def train_detection_model():
    """
    Train YOLOv8 for meat detection.
    """
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    dataset_yaml = project_root / "meat_detection_dataset" / "data.yaml"
    
    if not dataset_yaml.exists():
        print(f"Error: Dataset not found at {dataset_yaml}")
        print("Please run 'auto_annotate_yolo.py' first.")
        return
    
    # Load pretrained YOLOv8
    print("[Training] Loading YOLOv8n...")
    model = YOLO('yolov8n.pt')  # Nano model
    
    # Training configuration
    print(f"[Training] Starting training on {dataset_yaml}...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=416,  # Match dataset size
        batch=16,
        patience=10,  # Early stopping
        save=True,
        project=str(project_root / "runs" / "detect"),
        name="meat_detector",
        exist_ok=True,
        verbose=True
    )
    
    # Save best model to models directory
    best_model_path = project_root / "runs" / "detect" / "meat_detector" / "weights" / "best.pt"
    output_model_path = project_root / "src" / "models" / "detection_model.pt"
    
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, output_model_path)
        print(f"\n[Training] Model saved to: {output_model_path}")
    else:
        print("[Training] Warning: Best model not found!")
    
    print("[Training] Complete!")
    print(f"Results: {project_root / 'runs' / 'detect' / 'meat_detector'}")

if __name__ == "__main__":
    train_detection_model()
