import os
import cv2
import torch
import numpy as np
import urllib.request

# Fix for SSL certificate errors on some systems
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Configuration
    # Determine absolute paths based on script location
    # Script is in src/train/utils/
    # Project root is 3 levels up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    
    input_root = os.path.join(project_root, "meat_freshness_dataset")
    output_root = os.path.join(project_root, "meat_freshness_dataset_depth")
    
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    # model_type = "DPT_Large"     # MiDaS v3.0 - Large     (highest accuracy, slowest inference speed)
    
    if not os.path.exists(input_root):
        print(f"Error: Input dataset not found at {input_root}")
        return

    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    
    print(f"Loading {model_type}...")
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Using device: {device}")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Process Dataset
    print(f"Processing images from {input_root} to {output_root}...")
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                
                # Determine relative path to maintain structure
                rel_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, rel_path)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_path = os.path.join(output_dir, file)
                
                if os.path.exists(output_path):
                    continue # Skip existing
                
                print(f"Processing: {file}")
                
                # Load Image
                img = cv2.imread(input_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Transform
                input_batch = transform(img).to(device)

                # Predict
                with torch.no_grad():
                    prediction = midas(input_batch)

                    # Resize to original resolution
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                # Post-process
                depth_map = prediction.cpu().numpy()
                
                # Normalize to 0-255
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                
                if depth_max - depth_min > 1e-6:
                    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                else:
                    depth_map = np.zeros_like(depth_map)
                    
                depth_map = (depth_map * 255).astype(np.uint8)
                
                # Apply ColorMap? No, save as Grayscale for training
                # The model expects 3-channel input usually if using transfer learning models (like DenseNet),
                # so we might want to save it as a 3-channel grayscale (R=G=B) or just grayscale.
                # cv2.imwrite saves 1 channel if 2D array.
                # To be safe for various loaders, let's keep it simple grayscale.
                # But wait, DenseNet expects 3 channels usually.
                # Let's save as single channel, and let the loader handle duplicating channels (or use cv2.IMREAD_GRAYSCALE)
                # Correction: The current depth_model.py uses standard ImageDataGenerator which loads RGB. 
                # So if we save grayscale, cv2/keras might load it as 3-channel (copying) or we might need to be explicit.
                # Let's simple write it.
                
                cv2.imwrite(output_path, depth_map)

    print("Depth generation complete.")

if __name__ == "__main__":
    main()
