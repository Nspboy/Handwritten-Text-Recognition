"""
Quick Training Script for HTR with Sample Dataset

Loads the sample dataset and trains the model.
"""

import json
import logging
import argparse
import numpy as np
from pathlib import Path
from train import HTRTrainer
from preprocessing.preprocess import ImagePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str = "dataset"):
    """Load images and labels from dataset."""
    logger.info("Loading dataset...")
    
    dataset_path = Path(dataset_path)
    raw_images_dir = dataset_path / "raw_images"
    labels_path = dataset_path / "labels" / "train_labels.json"
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    logger.info(f"Found {len(labels_data)} training samples")
    
    # Load and preprocess images
    preprocessor = ImagePreprocessor()
    images = []
    texts = []
    
    for i, label_info in enumerate(labels_data):
        img_path = raw_images_dir / label_info['image']
        text = label_info['text']
        
        # Preprocess image
        processed_img = preprocessor.preprocess_image(str(img_path))
        if processed_img is not None:
            images.append(processed_img)
            texts.append(text)
        
        if (i + 1) % 20 == 0:
            logger.info(f"Loaded {i + 1} images")
    
    logger.info(f"Successfully loaded {len(images)} preprocessed images")
    
    # Expand dimensions to add channel dimension if needed
    images = np.array(images)
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    
    logger.info(f"Image array shape: {images.shape}")
    
    return images, texts


def create_label_sequences(texts: list, max_length: int = 32, num_classes: int = 80) -> np.ndarray:
    """Convert text labels to numeric sequences."""
    logger.info("Creating label sequences...")
    
    # Create a simple character-to-index mapping
    char_set = set()
    for text in texts:
        char_set.update(text)
    
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(char_set))}
    char_to_idx[' '] = 0  # Space is blank character
    
    logger.info(f"Character set size: {len(char_to_idx)}")
    
    # Convert texts to sequences
    sequences = []
    for text in texts:
        seq = [char_to_idx.get(char, 0) for char in text[:max_length]]
        # Pad to max_length
        seq = seq + [0] * (max_length - len(seq))
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.int32)


def train_model(epochs: int = 5, batch_size: int = 16, config_path: str = "config.json"):
    """Train the HTR model."""
    
    # Load dataset
    x_train, texts = load_dataset()
    
    # Create label sequences
    y_train = create_label_sequences(texts, max_length=80)  # Match output layer size (num_classes)
    logger.info(f"Training labels shape: {y_train.shape}")
    
    # Normalize labels to [0, 1] range for MSE loss
    y_train = y_train / 80.0  # num_classes = 80
    
    # Initialize trainer
    trainer = HTRTrainer(config_path)
    
    # Update config with command-line arguments
    trainer.config['epochs'] = epochs
    trainer.config['batch_size'] = batch_size
    
    # Build model
    logger.info("Building model...")
    trainer.build_model()
    
    # Train
    logger.info(f"Starting training with {epochs} epochs and batch size {batch_size}...")
    try:
        history = trainer.train(x_train, y_train)
        logger.info("Training completed successfully!")
        
        # Save model
        trainer.save_model()
        
        return history
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HTR Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--config", type=str, default="config.json", help="Config file path")
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, config_path=args.config)
