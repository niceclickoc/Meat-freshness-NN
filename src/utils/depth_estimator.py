"""
Depth Estimation Utility using MiDaS
Provides on-the-fly depth map generation for inference.
"""
import torch
import numpy as np
import cv2

# Singleton pattern for model loading
_midas_model = None
_midas_transform = None
_device = None

def initialize_midas(model_type="MiDaS_small"):
    """
    Initialize MiDaS model (call once at startup).
    
    Args:
        model_type: "MiDaS_small" (fast) or "DPT_Large" (accurate)
    """
    global _midas_model, _midas_transform, _device
    
    if _midas_model is not None:
        return  # Already initialized
    
    print(f"[Depth Estimator] Loading {model_type}...")
    
    # Load model
    _midas_model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    
    # Determine device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    
    print(f"[Depth Estimator] Using device: {_device}")
    
    _midas_model.to(_device)
    _midas_model.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        _midas_transform = midas_transforms.dpt_transform
    else:
        _midas_transform = midas_transforms.small_transform
    
    print("[Depth Estimator] Initialized successfully")


def estimate_depth(rgb_image):
    """
    Convert RGB image to depth map.
    
    Args:
        rgb_image: numpy array (H, W, 3) in RGB format, values 0-255
        
    Returns:
        depth_map: numpy array (H, W) grayscale, values 0-255
    """
    global _midas_model, _midas_transform, _device
    
    if _midas_model is None:
        raise RuntimeError("MiDaS not initialized. Call initialize_midas() first.")
    
    # Ensure RGB format
    if rgb_image.dtype != np.uint8:
        rgb_image = rgb_image.astype(np.uint8)
    
    # Transform
    input_batch = _midas_transform(rgb_image).to(_device)
    
    # Predict
    with torch.no_grad():
        prediction = _midas_model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Post-process to 0-255
    depth_map = prediction.cpu().numpy()
    
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    
    if depth_max - depth_min > 1e-6:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_map = np.zeros_like(depth_map)
    
    depth_map = (depth_map * 255).astype(np.uint8)
    
    return depth_map


def is_initialized():
    """Check if MiDaS is initialized."""
    return _midas_model is not None
