import cv2
import numpy as np
from pathlib import Path
from engine.trainer import HTRTrainer
from engine.preprocessing.preprocess import ImagePreprocessor
import json

trainer = HTRTrainer('config.json')
model = trainer.build_model()
model.load_weights('checkpoints/best_model.h5')

mapping_path = 'checkpoints/best_model_mapping.json'
with open(mapping_path, 'r') as f:
    data = json.load(f)
idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}

preprocessor = ImagePreprocessor()
img = preprocessor.preprocess_image('static/samples/image1.png')

# Dummy segment
line_crop = img[0:128, 0:512]
target_h = 32
h, w = line_crop.shape[:2]
aspect_ratio = w / max(1, h)
target_w = int(target_h * aspect_ratio)
target_w = max(128, target_w)
target_w = ((target_w + 3) // 4) * 4

resized = preprocessor.resize_with_padding(line_crop, (target_h, target_w))
if resized.ndim == 2:
    resized = np.expand_dims(resized, axis=-1)
img_batch = resized.astype(np.float32) / 255.0
img_batch = np.expand_dims(img_batch, axis=0)

pred = model.predict(img_batch, verbose=0)
seq = np.argmax(pred[0], axis=-1)
text_chars = []
prev = -1
for idx in seq:
    if int(idx) == prev: continue
    if int(idx) == 0:
        prev = int(idx)
        continue
    text_chars.append(idx_to_char.get(int(idx), ''))
    prev = int(idx)

print("PREDICTION:", ''.join(text_chars))
