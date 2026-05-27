
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2
import sys
import json

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from engine.trainer import HTRTrainer
from engine.preprocessing.preprocess import ImagePreprocessor

def debug_prediction():
    config_path = "config.json"
    trainer = HTRTrainer(config_path)
    model_path = "checkpoints/best_model.h5"
    if not Path(model_path).exists():
        print("Model not found")
        return
    
    trainer.load_model(model_path)
    
    # Load some data to get mappings (replicate the mapping logic)
    dataset_path = Path("data/labels/train_labels.json")
    with open(dataset_path, 'r') as f:
        labels_data = json.load(f)
    texts = [item['text'] for item in labels_data]
    
    # Replicate mapping logic from train_full_pipeline.py
    # NOTE: This is fragile if the training used different data!
    chars = sorted(list(set("".join(texts).lower()) | {' '}))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    
    trainer.char_to_idx = char_to_idx
    trainer.idx_to_char = idx_to_char
    
    preprocessor = ImagePreprocessor()
    img_path = "data/benchmark/raw_samples/sample_0056.png" # Ground truth: "Photography visual art"
    
    img = preprocessor.preprocess_image(img_path)
    std = cv2.resize(img, (128, 128))
    x = np.expand_dims(std, axis=-1).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    
    preds = trainer.model.predict(x)
    print(f"Preds shape: {preds.shape}")
    
    # Check max probabilities
    softmax_preds = tf.nn.softmax(preds[0], axis=-1).numpy()
    max_indices = np.argmax(softmax_preds, axis=-1)
    max_probs = np.max(softmax_preds, axis=-1)
    
    non_blank = np.where(max_indices != 0)[0]
    print(f"Non-blank indices: {non_blank}")
    if len(non_blank) > 0:
        print(f"Indices: {max_indices[non_blank]}")
        print(f"Probs: {max_probs[non_blank]}")
        text = "".join([idx_to_char.get(i, "?") for i in max_indices[non_blank]])
        print(f"Raw Decode: {text}")
    else:
        print("All predictions are blank!")

if __name__ == "__main__":
    debug_prediction()
