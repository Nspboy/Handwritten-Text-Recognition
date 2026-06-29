"""
Full HTR Pipeline: Professional-Grade Training and Evaluation
Integrated with IAM Dataset and Image Augmentation.
Multilingual Support (English + Kannada) with Phased Training.
"""

import json
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import pandas as pd

# TensorFlow imports
import tensorflow as tf
from engine.trainer import HTRTrainer
from engine.preprocessing.preprocess import ImagePreprocessor, get_charset_for_language
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
            "comparison": {}
        }

    def _load_iam_parquet(self, parquet_path: Path, limit: int) -> Tuple[List[np.ndarray], List[str]]:
        logger.info(f"Loading IAM dataset from {parquet_path}...")
        if not parquet_path.exists():
            logger.warning(f"Parquet file not found: {parquet_path}")
            return [], []
            
        df = pd.read_parquet(parquet_path)
        if limit > 0:
            df = df.sample(frac=1, random_state=42).head(limit)
        
        images, texts = [], []
        target_size = (32, 256) # Fixed target size
        
        for idx, row in df.iterrows():
            try:
                img_bytes = row['image']['bytes']
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Apply blur and thresholding similar to preprocess_image
                    blur = cv2.GaussianBlur(img, self.preprocessor.blur_kernel, 0)
                    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    std = self.preprocessor.resize_with_padding(binary, target_size)
                    images.append(std)
                    texts.append(str(row['text']))
            except Exception as e:
                pass
                
        return images, texts

    def _load_kannada_csv(self, csv_file: Path, limit: int) -> Tuple[List[np.ndarray], List[str]]:
        logger.info(f"Loading Kannada dataset from {csv_file}...")
        if not csv_file.exists():
            logger.warning(f"Kannada CSV not found: {csv_file}")
            return [], []
            
        df = pd.read_csv(csv_file)
        if limit > 0:
            df = df.sample(frac=1, random_state=42).head(limit)
            
        data_dir = csv_file.parent
        images, texts = [], []
        target_size = (32, 256)
        
        for idx, row in df.iterrows():
            try:
                img_path = data_dir / str(row['img'])
                if not img_path.exists(): continue
                
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    blur = cv2.GaussianBlur(img, self.preprocessor.blur_kernel, 0)
                    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    std = self.preprocessor.resize_with_padding(binary, target_size)
                    images.append(std)
                    texts.append(str(row['class'])) # Using class as text
            except Exception as e:
                pass
                
        return images, texts

    def prepare_dataset(self, dataset_type: str = "train", limit: int = 2000, lang: str = "english") -> Tuple[np.ndarray, List[str]]:
        images, texts = [], []
        
        if lang in ["english", "both"]:
            # Map train to train, test to test, etc. (Default to train if missing)
            p_type = dataset_type if dataset_type in ["train", "test", "eval"] else "train"
            parquet_file = Path(f"data/IAM dataset/{p_type}/{p_type}.parquet")
            img_eng, txt_eng = self._load_iam_parquet(parquet_file, limit)
            images.extend(img_eng)
            texts.extend(txt_eng)
            
        if lang in ["kannada", "both"]:
            csv_file = Path("data/kannada.csv")
            # For Kannada, just split the same csv conceptually, or just use train for all if no splits
            img_kan, txt_kan = self._load_kannada_csv(csv_file, limit)
            images.extend(img_kan)
            texts.extend(txt_kan)
            
        if len(images) == 0:
            return np.array([]), []
            
        # Add channel dimension and normalize
        x = np.expand_dims(np.array(images), axis=-1).astype(np.float32) / 255.0
        return x, texts

    def run_full_pipeline(self, limit: int = 2000):
        # 1. Dataset
        logger.info("Loading Datasets...")
        x_train_eng, y_train_eng = self.prepare_dataset("train", limit, lang="english")
        x_train_kan, y_train_kan = self.prepare_dataset("train", limit, lang="kannada")
        
        x_test, y_test = self.prepare_dataset("test", max(20, int(limit*0.1)), lang="both")
        
        # Mix dataset for Phase 3
        if len(x_train_eng) > 0 and len(x_train_kan) > 0:
            x_train_mixed = np.concatenate([x_train_eng, x_train_kan])
            y_train_mixed = y_train_eng + y_train_kan
        elif len(x_train_eng) > 0:
            x_train_mixed = x_train_eng
            y_train_mixed = y_train_eng
        elif len(x_train_kan) > 0:
            x_train_mixed = x_train_kan
            y_train_mixed = y_train_kan
        else:
            x_train_mixed = np.array([])
            y_train_mixed = []
            
        if len(x_train_mixed) > 0:
            p = np.random.permutation(len(x_train_mixed))
            x_train_mixed = x_train_mixed[p]
            y_train_mixed = [y_train_mixed[i] for i in p]
        
        # 2. Mappings
        # We must build a vocabulary that covers both datasets. 
        # For robustness, we combine the text in the actual labels rather than hardcoded charsets
        all_text = "".join(y_train_mixed + y_test)
        chars = sorted(list(set(all_text)))
        if ' ' not in chars: chars.append(' ')
        chars = sorted(chars)
        
        self.trainer.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.trainer.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.trainer.config['num_classes'] = len(chars) + 1 # +1 for blank at index 0
        
        # Fix input shape to match target size (32, 256, 1) to avoid NoneType issues
        self.trainer.config['input_shape'] = [32, 256, 1]
        
        # 3. Train
        logger.info("BUILDING MODEL...")
        self.trainer.config['epochs'] = self.epochs
        self.trainer.build_model()
        
        history_dict = {}
        
        # PHASE 1: Pre-train on English
        if len(x_train_eng) > 0:
            logger.info("--- PHASE 1: PRE-TRAIN ON ENGLISH ---")
            y_train_eng_encoded = self.trainer.encode_labels(y_train_eng)
            self.trainer.train(x_train_eng, y_train_eng_encoded)
            
        # PHASE 2: Fine-tune on Kannada
        if len(x_train_kan) > 0:
            logger.info("--- PHASE 2: FINE-TUNE ON KANNADA ---")
            self.trainer.freeze_cnn_layers()
            y_train_kan_encoded = self.trainer.encode_labels(y_train_kan)
            self.trainer.train(x_train_kan, y_train_kan_encoded)
            
        # PHASE 3: Joint training
        if len(x_train_mixed) > 0:
            logger.info("--- PHASE 3: JOINT TRAINING ON MIXED DATASET ---")
            self.trainer.unfreeze_all_layers()
            y_train_mixed_encoded = self.trainer.encode_labels(y_train_mixed)
            history_dict = self.trainer.train(x_train_mixed, y_train_mixed_encoded)
            self.results["training"] = {"history": history_dict, "epochs": self.epochs, "samples": len(x_train_mixed)}
        else:
            logger.warning("No training data available. Skipping training.")
        
        # 4. Inference
        logger.info("INFERENCE...")
        inference_results = []
        if len(x_test) > 0:
            for i in range(min(50, len(x_test))): # limit inference test for speed
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
        if inference_results:
            y_true = [r["ground_truth"] for r in inference_results]
            y_pred = [r["predicted"] for r in inference_results]
            y_corr = [r["corrected"] for r in inference_results]
            
            def calc_m(truth, pred):
                c = cer(truth, pred) if JIWER_AVAILABLE else 1.0
                w = wer(truth, pred) if JIWER_AVAILABLE else 1.0
                acc = sum(1 for t, p in zip(truth, pred) if t.lower() == p.lower()) / len(truth)
                return {"cer": float(c), "wer": float(w), "accuracy": float(acc)}
                
            self.results["metrics"] = {"before_nlp": calc_m(y_true, y_pred), "after_nlp": calc_m(y_true, y_corr)}
            logger.info(f"✔ Evaluation Complete. Accuracy: {self.results['metrics']['after_nlp']['accuracy']:.2%}")
        else:
            logger.warning("No test data available for evaluation.")
            self.results["metrics"] = {}

        # 8. Save
        self.trainer.save_model()
        self.save_results()
        logger.info("✔ Pipeline Complete.")

    def save_results(self):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        if "after_nlp" in self.results.get("metrics", {}):
            self.results["comparison"]["current_model"] = self.results["metrics"]
            
        with open("pipeline_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, cls=NpEncoder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    
    pipeline = ComprehensiveHTRPipeline(epochs=args.epochs)
    pipeline.run_full_pipeline(limit=args.limit)
