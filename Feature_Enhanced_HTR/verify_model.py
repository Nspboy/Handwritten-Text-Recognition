
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2

# Add root to path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

from engine.preprocessing.preprocess import ImagePreprocessor
from engine.model.decoder_ctc import CTCDecoder

def load_char_mapping():
    # Attempt to load mapping from labels
    import json
    labels_path = Path("data/labels/labels.json")
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        texts = [item['text'] for item in labels]
        chars = sorted(list(set("".join(texts).lower()) | {' '}))
        char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        return char_to_idx, idx_to_char
    return None, None

def verify():
    model_path = "checkpoints/best_model.h5"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # Load without compilation to avoid custom loss requirement issues
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    char_to_idx, idx_to_char = load_char_mapping()
    if not char_to_idx:
        print("Could not load character mapping.")
        return

    preprocessor = ImagePreprocessor()
    test_img_path = "data/custom/image1.png" # Using custom image as test
    
    img = preprocessor.preprocess_image(test_img_path)
    if img is None:
        print(f"Failed to preprocess {test_img_path}")
        return
        
    img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0) # Batch dim

    print(f"Predicting for {test_img_path}...")
    preds = model.predict(img)
    
    # Simple greedy decoding
    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(preds, perm=[1, 0, 2]),
        sequence_length=[preds.shape[1]]
    )
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    
    result_text = "".join([idx_to_char.get(i, "") for i in decoded_dense[0] if i != -1])
    print(f"Prediction: '{result_text}'")

if __name__ == "__main__":
    verify()
