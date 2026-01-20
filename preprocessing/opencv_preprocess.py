"""
Image Preprocessing Module - Step 2 of the Feature Enhancement Pipeline

OpenCV-based preprocessing for handwritten text:
- Convert to grayscale
- Noise removal (Gaussian / Median filtering)
- Binarize using Otsu threshold
- Correct skew
- Resize to fixed dimensions

Goal: Remove background noise and standardize input before feature extraction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


class OpenCVPreprocessor:
    """
    Handwriting image preprocessor using OpenCV.
    Applies: grayscale → blur → Otsu binarization → skew correction → resize.
    """

    def __init__(
        self,
        target_height: int = 128,
        target_width: Optional[int] = None,
        blur_kernel: Tuple[int, int] = (5, 5),
        blur_type: str = 'gaussian',  # 'gaussian' or 'median'
        deskew: bool = True,
        invert: bool = True,
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.blur_kernel = blur_kernel
        self.blur_type = blur_type.lower()
        self.deskew = deskew
        self.invert = invert

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        if self.blur_type == 'median':
            k = max(3, self.blur_kernel[0] | 1)
            return cv2.medianBlur(gray, k)
        return cv2.GaussianBlur(gray, self.blur_kernel, 0)

    def _binarize_otsu(self, blur: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img > 0))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _resize(self, img: np.ndarray, aspect_ratio: Optional[float] = None) -> np.ndarray:
        h, w = img.shape[:2]
        if self.target_width is not None:
            out_h, out_w = self.target_height, self.target_width
        else:
            scale = self.target_height / float(h)
            out_w = int(round(w * scale))
            out_w = max(out_w, 1)
            out_h = self.target_height
        resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return resized

    def __call__(self, img: Union[np.ndarray, str, Path]) -> np.ndarray:
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img}")
        img = np.asarray(img)
        gray = self._to_grayscale(img)
        blur = self._denoise(gray)
        binary = self._binarize_otsu(blur)
        if self.deskew:
            binary = self._deskew(binary)
        out = self._resize(binary)
        if self.invert:
            out = cv2.bitwise_not(out)
        return out

    def to_float_normalized(self, img: np.ndarray) -> np.ndarray:
        """Convert to float in [0,1] with white-on-black convention for the model."""
        x = np.asarray(img, dtype=np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        return 1.0 - x


def preprocess_image(
    img: Union[np.ndarray, str, Path],
    target_height: int = 128,
    blur_kernel: Tuple[int, int] = (5, 5),
    blur_type: str = 'gaussian',
    deskew: bool = True,
    invert: bool = True,
) -> np.ndarray:
    """
    One-shot preprocessing for a single image.

    Args:
        img: numpy array (H,W) or (H,W,3), or path to image
        target_height: output height
        blur_kernel: (kx, ky) for Gaussian, or single k for Median
        blur_type: 'gaussian' or 'median'
        deskew: whether to correct skew
        invert: whether to invert (black text on white) for model input

    Returns:
        Preprocessed image (uint8 or float depending on usage).
        Use OpenCVPreprocessor.to_float_normalized for model input.
    """
    p = OpenCVPreprocessor(
        target_height=target_height,
        blur_kernel=blur_kernel,
        blur_type=blur_type,
        deskew=deskew,
        invert=invert,
    )
    return p(img)


def preprocess_batch(
    images: list,
    target_height: int = 128,
    **kwargs,
) -> list:
    """Preprocess a list of images (paths or arrays)."""
    p = OpenCVPreprocessor(target_height=target_height, **kwargs)
    return [p(im) for im in images]


if __name__ == '__main__':
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sample = root / 'samples' / 'example.png'
    if sample.exists():
        out = preprocess_image(str(sample), target_height=128)
        print('Shape:', out.shape, 'dtype:', out.dtype)
        cv2.imwrite(str(root / 'samples' / 'preprocessed_example.png'), out)
