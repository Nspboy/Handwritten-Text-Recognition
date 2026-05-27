"""
Train specifically on Custom Dataset.
"""

import json
import logging
import numpy as np
from pathlib import Path
import cv2
import tensorflow as tf

import sys
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from engine.trainer import HTRTrainer
from engine.preprocessing.preprocess import ImagePreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_custom():
    # 1. Load Custom Labels
    labels_file = Path("data/labels/custom_labels.json")
    if not labels_file.exists():
        logger.error("labels.json not found. Run preprocess_custom.py first.")
        return

    with open(labels_file, "r") as f:
        labels_data = json.load(f)

    # Check if user has filled in text
    needs_labeling = [item for item in labels_data if item["text"] == "INSERT_TEXT_HERE"]
    if needs_labeling:
        logger.warning(f"Found {len(needs_labeling)} images with 'INSERT_TEXT_HERE'.")
        logger.warning("Please edit 'dataset/labels/custom_labels.json' first.")
        # Proceeding anyway for demonstration if user wants, or we could exit
        # return 

    # 2. Prepare Data
    trainer = HTRTrainer("config.json")
    preprocessor = ImagePreprocessor()
    target_size = (trainer.config['input_shape'][0], trainer.config['input_shape'][1])

    images = []
    texts = []
    
    for item in labels_data:
        # Use the already processed images
        img_path = Path(item["processed_path"])
        if img_path.exists():
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            img = preprocessor.resize_with_padding(img, target_size)
            images.append(img)
            texts.append(item["text"])
            
            # Augment a few times since custom dataset is small
            for _ in range(10):
                images.append(preprocessor.augment_image(img))
                texts.append(item["text"])

    if not images:
        logger.error("No valid images found for training.")
        return

    # Normalize
    X = np.expand_dims(np.array(images), axis=-1).astype(np.float32) / 255.0
    
    # Simple character mapping
    all_chars = sorted(list(set("".join(texts).lower() + " ")))
    char_to_idx = {c: i + 1 for i, c in enumerate(all_chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(all_chars)}
    
    # Save mappings to trainer
    trainer.char_to_idx = char_to_idx
    trainer.idx_to_char = idx_to_char
    
    # Encode labels
    max_len = 32
    Y = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        encoded = [char_to_idx.get(c, 0) for c in t.lower()[:max_len]]
        Y[i, :len(encoded)] = encoded

    # 3. Train
    logger.info(f"Starting training on {len(X)} samples (including augmentations)...")
    trainer.config['epochs'] = 50
    trainer.build_model()
    trainer.train(X, Y)
    
    trainer.save_model("checkpoints/custom_model.h5")
    logger.info("Training complete. Model saved as checkpoints/custom_model.h5")

if __name__ == "__main__":
    train_custom()
