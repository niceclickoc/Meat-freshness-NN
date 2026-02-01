import cv2
import numpy as np
from skimage.feature import hog

def preprocess_chromatic(image):
    return image / 255.0

def preprocess_hog(image):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(image, (128, 128))
    features = hog(resized_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
    return features

def preprocess_depth_map(image):
    """
    Preprocess image for depth model.
    Converts RGB image to depth map using MiDaS, then normalizes.
    
    Args:
        image: RGB numpy array (H, W, 3), values 0-255
        
    Returns:
        Normalized depth map (H, W, 3) for model input
    """
    from src.utils.depth_estimator import estimate_depth, is_initialized
    
    if not is_initialized():
        raise RuntimeError(
            "Depth estimator not initialized. "
            "Call depth_estimator.initialize_midas() before using depth model."
        )
    
    # Convert RGB to grayscale depth map (0-255)
    depth_map = estimate_depth(image)
    
    # Convert grayscale to 3-channel (DenseNet expects 3 channels)
    depth_map_3ch = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0, 1]
    return depth_map_3ch / 255.0

