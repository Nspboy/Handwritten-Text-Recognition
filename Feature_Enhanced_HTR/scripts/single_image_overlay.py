"""
Single image inference and overlay utility for HTR project.

Usage:
  python single_image_overlay.py --image "path/to/image.png" \
      --model checkpoints/best_model.h5 --output out.png

If no image is provided the script will use the first image found in
`dataset/coustem images/`.
"""
import argparse
from pathlib import Path
import json
import cv2
import numpy as np
import logging
import sys

from preprocessing.preprocess import ImagePreprocessor
from train import HTRTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_char_mapping_from_labels(labels_path: Path) -> dict:
    if not labels_path.exists():
        return {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chars = set()
    for item in data:
        text = item.get('text', '')
        chars.update(text.lower())
    chars.add(' ')
    # reserve 0 for CTC blank
    char_to_idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
    char_to_idx['<blank>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return idx_to_char


def decode_predictions(pred: np.ndarray, idx_to_char: dict) -> str:
    # pred: (time_steps, num_classes) probabilities
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


def overlay_text_on_image(orig_img_path: Path, text: str, out_path: Path) -> None:
    img = cv2.imread(str(orig_img_path))
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {orig_img_path}")

    # Prepare text box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    margin = 10

    # Compute text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = margin, margin + text_h

    # Draw filled rectangle for readability
    rect_pt1 = (x - 5, y - text_h - 5)
    rect_pt2 = (x + text_w + 5, y + 5)
    overlay = img.copy()
    cv2.rectangle(overlay, rect_pt1, rect_pt2, (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Put white text
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Save
    cv2.imwrite(str(out_path), img)
    logger.info(f"Saved overlay image to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Single image HTR + overlay")
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.h5', help='Path to model file')
    parser.add_argument('--output', type=str, default='overlay_result.png', help='Output image path')
    parser.add_argument('--labels', type=str, default='dataset/labels/labels.json', help='Labels json to build char map')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    # Determine input image
    if args.image:
        img_path = Path(args.image)
    else:
        custom_dir = repo_root / 'dataset' / 'coustem images'
        candidates = list(custom_dir.glob('*.*'))
        if not candidates:
            logger.error('No images found in dataset/coustem images. Provide --image.')
            sys.exit(1)
        img_path = candidates[0]

    # Preprocess
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_image(str(img_path))
    if processed is None:
        logger.error('Preprocessing failed')
        sys.exit(1)

    # Build trainer and model
    trainer = HTRTrainer(config_path=str(repo_root / 'config.json'))
    # Ensure model architecture exists
    model = trainer.build_model()

    model_path = Path(args.model)
    try:
        # Try loading entire model first
        trainer.load_model(str(model_path))
        model = trainer.model
    except Exception:
        try:
            model.load_weights(str(model_path))
            logger.info('Loaded weights into built model')
        except Exception as e:
            logger.error(f'Failed to load model or weights: {e}')
            sys.exit(1)

    # Resize processed image to input shape
    input_shape = tuple(trainer.config['input_shape'])
    resized = cv2.resize(processed, (input_shape[1], input_shape[0]))
    if resized.ndim == 2:
        resized = np.expand_dims(resized, axis=-1)
    img_batch = resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_batch, axis=0)

    # Predict
    preds = model.predict(img_batch)

    # Build idx_to_char mapping from provided labels file
    idx_to_char = build_char_mapping_from_labels(repo_root / args.labels)
    if not idx_to_char:
        logger.warning('Failed to build character mapping from labels; predictions may not map to readable text')

    # If model output is batched, take first
    pred0 = preds[0]
    # If model outputs 2D (num_classes,) expand dims to (time_steps, num_classes)
    if pred0.ndim == 1:
        pred0 = np.expand_dims(pred0, axis=0)

    text = decode_predictions(pred0, idx_to_char)
    logger.info(f'Predicted text: "{text}"')

    # Overlay onto original (color) image
    out_path = Path(args.output)
    overlay_text_on_image(img_path, text, out_path)


if __name__ == '__main__':
    main()
