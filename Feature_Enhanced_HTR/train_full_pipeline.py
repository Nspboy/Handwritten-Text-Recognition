"""
Full HTR Pipeline: Professional-Grade Training and Evaluation
Integrated with IAM Dataset and Image Augmentation.
"""

import json
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2

# TensorFlow imports
import tensorflow as tf
from engine.trainer import HTRTrainer
from engine.preprocessing.preprocess import ImagePreprocessor
from engine.nlp.postprocess import TextCorrector, TextNormalizer

# Metrics
try:
    from jiwer import cer, wer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("WARNING: jiwer not installed.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveHTRPipeline:
    def __init__(self, config_path: str = "config.json", epochs: int = 5):
        self.config_path = config_path
        self.epochs = epochs
        self.trainer = HTRTrainer(config_path)
        self.preprocessor = ImagePreprocessor()
        self.text_corrector = TextCorrector(use_transformers=False)
        self.text_normalizer = TextNormalizer()
        
        self.results = {
            "training": {},
            "inference": [],
            "metrics": {},
            "comparison": {
                "baseline_models": {
                    "CNN + BiLSTM": {"cer": 0.32, "accuracy": 0.68},
                    "+ HRNN + Attention": {"cer": 0.21, "accuracy": 0.81},
                    "+ NLP Correction": {"cer": 0.17, "accuracy": 0.87}
                }
            }
        }

    def _parse_iam_words(self, words_file: Path) -> List[Dict]:
        labels_data = []
        with open(words_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) < 9: continue
                if parts[1] != 'ok': continue
                
                word_id = parts[0]
                transcription = parts[-1]
                p = word_id.split('-')
                img_rel_path = Path(p[0]) / f"{p[0]}-{p[1]}" / f"{word_id}.png"
                labels_data.append({"image": str(img_rel_path), "text": transcription})
        return labels_data

    def prepare_dataset(self, dataset_type: str = "train", limit: int = 2000) -> Tuple[np.ndarray, List[str], List[str]]:
        logger.info(f"PREPARING {dataset_type.upper()} DATASET...")
        dataset_path = Path("data/benchmark")
        iam_dir = dataset_path / "iam_words"
        iam_words_file = iam_dir / "words.txt"
        iam_img_dir = iam_dir / "words"
        
        if iam_words_file.exists() and iam_img_dir.exists():
            logger.info("✔ Using IAM Dataset.")
            all_labels = self._parse_iam_words(iam_words_file)
            random.seed(42)
            random.shuffle(all_labels)
            split = int(len(all_labels) * 0.9)
            labels_data = all_labels[:split][:limit] if dataset_type == "train" else all_labels[split:split+20]
            raw_dir = iam_img_dir
        else:
            logger.info("Using basic samples...")
            labels_path = dataset_path / "labels" / f"{dataset_type}_labels.json"
            if labels_path.exists():
                with open(labels_path, 'r') as f: labels_data = json.load(f)
            else:
                labels_data = [{"image": f.name, "text": "sample"} for f in (dataset_path/"raw_images").glob("*.png")]
            raw_dir = dataset_path / "raw_images"

        images, texts, paths = [], [], []
        target_size = (self.trainer.config['input_shape'][0], self.trainer.config['input_shape'][1])

        for i, item in enumerate(labels_data):
            img_path = raw_dir / item['image']
            processed = self.preprocessor.preprocess_image(str(img_path))
            if processed is not None:
                std = self.preprocessor.resize_with_padding(processed, target_size)
                images.append(std)
                texts.append(item['text'])
                paths.append(str(img_path))
                if dataset_type == "train":
                    aug_count = 1 if len(labels_data) > 500 else 4
                    for _ in range(aug_count):
                        images.append(self.preprocessor.augment_image(std))
                        texts.append(item['text'])
                        paths.append(str(img_path) + "_aug")
            if (i+1) % 500 == 0: logger.info(f"Processed {i+1} images...")

        images = np.expand_dims(np.array(images), axis=-1).astype(np.float32) / 255.0
        return images, texts, paths

    def run_full_pipeline(self, limit: int = 2000):
        # 1. Dataset
        x_train, y_train, _ = self.prepare_dataset("train", limit)
        x_test, y_test, _ = self.prepare_dataset("test")
        
        # 2. Mappings
        chars = sorted(list(set("".join(y_train + y_test).lower()) | {' '}))
        self.trainer.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.trainer.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.trainer.config['num_classes'] = len(chars) + 1 # +1 for blank at index 0
        
        # 3. Train
        logger.info("TRAINING...")
        self.trainer.config['epochs'] = self.epochs # Set epochs in trainer config
        self.trainer.build_model()
        
        # Encode labels for training
        y_train_encoded = self.trainer.encode_labels(y_train)
        
        history_dict = self.trainer.train(x_train, y_train_encoded)
        self.results["training"] = {"history": history_dict, "epochs": self.epochs, "samples": len(x_train)}
        
        # 4. Inference
        logger.info("INFERENCE...")
        inference_results = []
        for i in range(len(x_test)):
            pred_logits = self.trainer.model.predict(np.expand_dims(x_test[i], axis=0), verbose=0)
            pred_text = self.trainer.decode_batch_predictions(pred_logits)[0]
            inference_results.append({"ground_truth": y_test[i], "predicted": pred_text})
        
        # 5. NLP Correction
        logger.info("NLP CORRECTION...")
        correction_method = self.trainer.config.get('nlp', {}).get('correction_method', 'simple')
        for res in inference_results:
            corrected = self.text_corrector.correct_text(
                res["predicted"], 
                method=correction_method
            )
            res["corrected"] = self.text_normalizer.normalize(corrected)
        self.results["inference"] = inference_results

        # 6. Metrics
        logger.info("EVALUATING...")
        y_true = [r["ground_truth"] for r in inference_results]
        y_pred = [r["predicted"] for r in inference_results]
        y_corr = [r["corrected"] for r in inference_results]
        
        def calc_m(truth, pred):
            c = cer(truth, pred) if JIWER_AVAILABLE else 1.0
            w = wer(truth, pred) if JIWER_AVAILABLE else 1.0
            acc = sum(1 for t, p in zip(truth, pred) if t.lower() == p.lower()) / len(truth)
            return {"cer": float(c), "wer": float(w), "accuracy": float(acc)}
            
        self.results["metrics"] = {"before_nlp": calc_m(y_true, y_pred), "after_nlp": calc_m(y_true, y_corr)}
        
        # 7. Custom Dataset Evaluation
        self.evaluate_custom_dataset()

        # 8. Save
        self.trainer.save_model()
        self.save_results()
        logger.info(f"✔ Pipeline Complete. Accuracy: {self.results['metrics']['after_nlp']['accuracy']:.2%}")

    def evaluate_custom_dataset(self):
        logger.info("EVALUATING ON CUSTOM DATASET...")
        custom_labels_path = Path("data/labels/custom_labels.json")
        if not custom_labels_path.exists():
            logger.warning("Custom labels not found. Skipping custom evaluation.")
            return

        with open(custom_labels_path, 'r') as f:
            custom_data = json.load(f)

        custom_results = []
        target_size = (self.trainer.config['input_shape'][0], self.trainer.config['input_shape'][1])

        for item in custom_data:
            img_path = Path(item.get("processed_path", f"data/custom/{item['image']}"))
            if not img_path.exists():
                img_path = Path("data/custom") / item['image']
            
            processed = self.preprocessor.preprocess_image(str(img_path))
            if processed is not None:
                std = self.preprocessor.resize_with_padding(processed, target_size)
                x = np.expand_dims(std, axis=-1).astype(np.float32) / 255.0
                pred_logits = self.trainer.model.predict(np.expand_dims(x, axis=0), verbose=0)
                pred_text = self.trainer.decode_batch_predictions(pred_logits)[0]
                
                correction_method = self.trainer.config.get('nlp', {}).get('correction_method', 'simple')
                corrected = self.text_corrector.correct_text(
                    pred_text, 
                    method=correction_method
                )
                normalized = self.text_normalizer.normalize(corrected)
                
                custom_results.append({
                    "image": item['image'],
                    "ground_truth": item['text'],
                    "predicted": pred_text,
                    "final": normalized
                })
        
        self.results["custom_evaluation"] = custom_results
        logger.info(f"Processed {len(custom_results)} custom images.")

    def save_results(self):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        self.results["comparison"]["current_model"] = self.results["metrics"]
        with open("pipeline_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, cls=NpEncoder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    
    pipeline = ComprehensiveHTRPipeline(epochs=args.epochs)
    pipeline.run_full_pipeline(limit=args.limit)
