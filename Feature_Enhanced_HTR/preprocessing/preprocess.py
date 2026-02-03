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
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing for HTR tasks."""
    
    def __init__(self, 
                 blur_kernel: Tuple[int, int] = (5, 5),
                 morphology_enabled: bool = True):
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
            
            # Apply Otsu's binary thresholding
            _, binary = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Optional: morphological operations to improve connectivity
            if self.morphology_enabled:
                binary = self._apply_morphology(binary)
            
            logger.info(f"Successfully preprocessed: {img_path}")
            return binary
            
        except Exception as e:
            logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            return None
    
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Single image preprocessing
    # processed = preprocessor.preprocess_image("sample.png")
    
    # Batch preprocessing
    # count = preprocessor.batch_preprocess("raw_images/", "enhanced_images/")
