import os
from pathlib import Path
import json
import cv2
import numpy as np
import base64
import time
from flask import Flask, request, jsonify, render_template

from engine.preprocessing.preprocess import ImagePreprocessor
from engine.trainer import HTRTrainer
from engine.nlp.postprocess import TextCorrector, TextNormalizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global instances
preprocessor = ImagePreprocessor()
text_corrector = TextCorrector(use_transformers=False)
text_normalizer = TextNormalizer()

# Load Model
repo_root = Path(__file__).resolve().parent
trainer = HTRTrainer(config_path=str(repo_root / 'config.json'))
model = trainer.build_model()
model_path = repo_root / 'checkpoints' / 'best_model.h5'

try:
    trainer.load_model(str(model_path))
    model = trainer.model
except Exception:
    try:
        model.load_weights(str(model_path))
        print("Loaded weights successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")

# Build char mapping
labels_path = repo_root / 'data' / 'labels' / 'test_labels.json'
if not labels_path.exists():
    labels_path = repo_root / 'dataset' / 'labels' / 'labels.json'

def build_char_mapping():
    chars = set()
    if labels_path.exists():
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            chars.update(item.get('text', '').lower())
    chars.add(' ')
    char_to_idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    char_to_idx['<blank>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return idx_to_char

idx_to_char = build_char_mapping()

def decode_predictions(pred: np.ndarray, idx_to_char: dict) -> str:
    seq = np.argmax(pred, axis=-1)
    text_chars = []
    prev = -1
    for idx in seq:
        if int(idx) == prev:
            continue
        if int(idx) == 0:
            prev = int(idx)
            continue
        text_chars.append(idx_to_char.get(int(idx), ''))
        prev = int(idx)
    return ''.join(text_chars)

def image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples')
def get_samples():
    custom_dir = repo_root / 'data' / 'custom'
    if not custom_dir.exists():
        custom_dir = repo_root / 'dataset' / 'coustem images'
        
    samples = []
    if custom_dir.exists():
        for filename in sorted(os.listdir(custom_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append(filename)
    return jsonify(samples)

@app.route('/api/sample/<filename>')
def get_sample(filename):
    custom_dir = repo_root / 'data' / 'custom'
    if not custom_dir.exists():
        custom_dir = repo_root / 'dataset' / 'coustem images'
        
    filename = os.path.basename(filename)
    file_path = custom_dir / filename
    if not file_path.exists():
        return jsonify({'error': 'Sample image not found'}), 404
        
    img = cv2.imread(str(file_path))
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 500
        
    _, buffer = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        'name': filename,
        'image': f"data:image/png;base64,{img_b64}"
    })

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    nlp_method = request.form.get('nlp_method', 'simple')
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # 1. Preprocessing
    img = cv2.imread(file_path)
    processed_img = preprocessor.preprocess_image(file_path)
    
    # Store processed image as base64 for frontend
    if processed_img is not None:
        processed_b64 = image_to_base64(processed_img)
    else:
        return jsonify({'error': 'Failed to preprocess image'}), 500

    # 2. Inference Preparation
    input_shape = tuple(trainer.config['input_shape'])
    resized = cv2.resize(processed_img, (input_shape[1], input_shape[0]))
    if resized.ndim == 2:
        resized = np.expand_dims(resized, axis=-1)
    img_batch = resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_batch, axis=0)

    # 3. Model Prediction (CNN -> BiLSTM -> HRNN -> CTC)
    start_time = time.time()
    preds = model.predict(img_batch)
    inference_time = time.time() - start_time

    pred0 = preds[0]
    if pred0.ndim == 1:
        pred0 = np.expand_dims(pred0, axis=0)

    # 4. Decoding & NLP Correction
    raw_text = decode_predictions(pred0, idx_to_char)
    corrected_text = text_corrector.correct_text(raw_text, method=nlp_method)
    
    return jsonify({
        'raw_text': raw_text,
        'corrected_text': corrected_text,
        'inference_time': round(inference_time, 3),
        'processed_image': f"data:image/png;base64,{processed_b64}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
