import json
import logging
import numpy as np
import random
import cv2
import pandas as pd
from pathlib import Path
from engine.trainer import HTRTrainer
from engine.preprocessing.preprocess import ImagePreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_array(preprocessor, img_array):
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array
    blur = cv2.GaussianBlur(gray, preprocessor.blur_kernel, 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def load_parquet_dataset(parquet_path, preprocessor, target_size, limit=None):
    logger.info(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    if limit:
        df = df.head(limit)
    
    images, texts = [], []
    for idx, row in df.iterrows():
        try:
            # Extract bytes and decode
            img_bytes = row['image']['bytes']
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Preprocess
                binary = preprocess_array(preprocessor, img)
                std = preprocessor.resize_with_padding(binary, target_size)
                images.append(std)
                texts.append(str(row['text']))
        except Exception as e:
            logger.warning(f"Failed to process image at index {idx}: {e}")
            
    images = np.expand_dims(np.array(images), axis=-1).astype(np.float32) / 255.0
    logger.info(f"Loaded {len(images)} samples from {parquet_path}.")
    return images, texts

def train_on_iam(epochs=5, train_limit=200):
    preprocessor = ImagePreprocessor()
    trainer = HTRTrainer("config.json")
    target_size = (32, 256) # Fix to prevent NoneType error since config is [32, null, 1]
    
    # Load dataset
    x_train, y_train = load_parquet_dataset("data/IAM dataset/train/train.parquet", preprocessor, target_size, limit=train_limit)
    x_test, y_test = load_parquet_dataset("data/IAM dataset/test/test.parquet", preprocessor, target_size, limit=max(20, int(train_limit*0.1)))
    
    if len(x_train) == 0:
        logger.error("No training data found. Exiting.")
        return
        
    # Build Character Vocabulary matching the new dataset
    chars = sorted(list(set("".join(y_train + y_test))))
    if ' ' not in chars:
        chars.append(' ')
    chars = sorted(chars)
    
    trainer.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    trainer.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    trainer.config['num_classes'] = len(chars) + 1 # +1 for blank
    
    # Save the mapping so `app.py` can load it
    with open('checkpoints/best_model_mapping.json', 'w') as f:
        json.dump(trainer.idx_to_char, f)
    
    logger.info("TRAINING...")
    trainer.config['epochs'] = epochs
    trainer.build_model()
    
    y_train_encoded = trainer.encode_labels(y_train)
    
    trainer.train(x_train, y_train_encoded)
    trainer.save_model()
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    import sys
    limit = 200
    epochs = 2
    if len(sys.argv) > 1: limit = int(sys.argv[1])
    if len(sys.argv) > 2: epochs = int(sys.argv[2])
    train_on_iam(epochs=epochs, train_limit=limit)
