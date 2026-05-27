"""
HTR Evaluation and Demonstration Pipeline
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import sys

# Add root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from engine.preprocessing.preprocess import ImagePreprocessor
from engine.nlp.postprocess import TextCorrector, TextNormalizer

# Metrics
try:
    from jiwer import cer, wer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("WARNING: jiwer not installed. Install with: pip install jiwer")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HTRDemoEvaluation:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.text_corrector = TextCorrector(use_transformers=False)
        self.text_normalizer = TextNormalizer()
        self.results = {
            "preprocessing": {},
            "inference": [],
            "metrics": {},
            "comparison": {}
        }
    
    def preprocess_images(self, dataset_path: str = "data/benchmark") -> Tuple[np.ndarray, List[str]]:
        logger.info("="*60)
        logger.info("PREPROCESSING IMAGES")
        logger.info("="*60)
        
        dataset_path = Path(dataset_path)
        raw_images_dir = dataset_path / "raw_samples"
        labels_path = Path("data/labels/test_labels.json")
        
        if not labels_path.exists():
            logger.error(f"Labels not found at {labels_path}")
            return np.array([]), []

        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        images = []
        texts = []
        
        for i, label_info in enumerate(labels_data[:10]):
            img_path = raw_images_dir / label_info['image']
            text = label_info['text']
            processed_img = self.preprocessor.preprocess_image(str(img_path))
            if processed_img is not None:
                processed_img = cv2.resize(processed_img, (128, 128))
                images.append(processed_img)
                texts.append(text)
        
        images = np.array(images)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        images = images.astype(np.float32) / 255.0
        return images, texts

    def run_full_demo(self):
        logger.info("Running HTR Demo...")
        # (Simplified demo logic)
        test_images, test_texts = self.preprocess_images()
        if len(test_images) == 0:
            logger.warning("No images found for demo.")
            return
        logger.info(f"Demo complete on {len(test_images)} samples.")

if __name__ == "__main__":
    demo = HTRDemoEvaluation()
    demo.run_full_demo()
