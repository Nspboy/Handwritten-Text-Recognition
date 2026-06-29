"""
Image Preprocessing Module for Handwritten Text Recognition

This module handles image preprocessing operations including:
- Reading and grayscale conversion
- Gaussian blur for noise reduction
- Binary thresholding with Otsu's method
- Optional morphological operations
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for HTR tasks."""
    
    def __init__(self, 
                 blur_kernel: Tuple[int, int] = (5, 5),
                 morphology_enabled: bool = False):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            blur_kernel: Gaussian blur kernel size (must be odd numbers)
            morphology_enabled: Whether to apply morphological operations
        """
        self.blur_kernel = blur_kernel
        self.morphology_enabled = morphology_enabled
        
        if blur_kernel[0] % 2 == 0 or blur_kernel[1] % 2 == 0:
            raise ValueError("Blur kernel dimensions must be odd numbers")
    
    def preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image for handwritten text recognition.
        
        Args:
            img_path: Path to the input image
            
        Returns:
            Preprocessed binary image as numpy array, or None if processing fails
        """
        try:
            # Validate path
            img_path = Path(img_path)
            if not img_path.exists():
                logger.error(f"Image not found: {img_path}")
                return None
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            
            # Apply Otsu's thresholding to perfectly preserve text strokes without fragmentation
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Optional: morphological operations to improve connectivity
            if self.morphology_enabled:
                binary = self._apply_morphology(binary)
            
            logger.info(f"Successfully preprocessed: {img_path}")
            return binary
            
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            return None

    def resize_with_padding(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio, applying padding.
        
        Args:
            img: Input image
            target_size: (height, width)
            
        Returns:
            Padded and resized image
        """
        target_h, target_w = target_size
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas (padding with white/255 for binary images)
        canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
        
        # Center horizontally and vertically
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas

    def augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation to binary image.
        
        Args:
            img: Binary image
            
        Returns:
            Augmented image
        """
        # 1. Random Rotation (small angles for text)
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        
        # 2. Random Translation (shift)
        tx = random.uniform(-5, 5)
        ty = random.uniform(-2, 2)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        
        # 3. Random Zoom
        zoom = random.uniform(0.9, 1.1)
        new_w, new_h = int(w * zoom), int(h * zoom)
        img_zoomed = cv2.resize(img, (new_w, new_h))
        
        if zoom > 1: # Crop
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            img = img_zoomed[start_y:start_y+h, start_x:start_x+w]
        else: # Pad
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            img = np.full((h, w), 255, dtype=np.uint8)
            img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_zoomed
            
        return img
    
    def _apply_morphology(self, binary_img: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to improve text connectivity.
        
        Args:
            binary_img: Binary image
            
        Returns:
            Morphologically processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Close small holes in foreground
        closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opened
    
    def batch_preprocess(self, 
                        input_dir: str, 
                        output_dir: str,
                        file_extension: str = "*.png") -> int:
        """
        Preprocess all images in a directory.
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save preprocessed images
            file_extension: File pattern to match (default: *.png)
            
        Returns:
            Number of successfully preprocessed images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 0
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_file in input_path.glob(file_extension):
            processed = self.preprocess_image(str(img_file))
            if processed is not None:
                output_file = output_path / img_file.name
                cv2.imwrite(str(output_file), processed)
                count += 1
                logger.info(f"Saved: {output_file}")
        
        logger.info(f"Batch preprocessing complete: {count} images processed")
        return count


def preprocess_image(img_path: str) -> Optional[np.ndarray]:
    """
    Convenience function for preprocessing a single image.
    
    Args:
        img_path: Path to the input image
        
    Returns:
        Preprocessed binary image
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_image(img_path)

def detect_language(image_array: np.ndarray) -> str:
    """
    Simple pixel density check - Kannada has more 
    connected strokes than English letters.
    Assumes image is scaled 0-1 or 0-255.
    """
    if image_array.max() > 1.0:
        # Assuming 0-255 range where text is 0 and background is 255
        density = np.mean(image_array < 128)
    else:
        # Assuming 0-1 range where text is 0 and background is 1
        density = np.mean(image_array < 0.5)
        
    if density > 0.35:
        return "kannada"
    return "english"

def get_charset_for_language(lang: str) -> str:
    """Return the character set for the specified language."""
    ENGLISH_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}\"' "
    KANNADA_CHARS = "".join([chr(c) for c in range(0x0C80, 0x0CFF)])
    
    if lang == "kannada":
        return KANNADA_CHARS
    return ENGLISH_CHARS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Single image preprocessing
    # processed = preprocessor.preprocess_image("sample.png")
    
    # Batch preprocessing
    # count = preprocessor.batch_preprocess("raw_images/", "enhanced_images/")
