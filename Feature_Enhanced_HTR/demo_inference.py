"""
Sample Inference and Demo Script for HTR

Demonstrates how to use the HTR pipeline for text recognition.
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HTRDemo:
    """Demo class for HTR inference and visualization."""
    
    def __init__(self):
        """Initialize demo components."""
        from preprocessing.preprocess import ImagePreprocessor
        from nlp.postprocess import TextNormalizer
        
        self.preprocessor = ImagePreprocessor()
        self.normalizer = TextNormalizer()
        self.dataset_path = Path("dataset")
        
        logger.info("HTR Demo initialized successfully")
    
    def load_sample_images(self, num_samples: int = 5) -> List[Dict]:
        """
        Load sample images and labels from dataset.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            List of sample data dictionaries
        """
        samples = []
        
        try:
            # Load labels
            labels_file = self.dataset_path / "labels" / "labels.json"
            with open(labels_file, 'r') as f:
                all_labels = json.load(f)
            
            # Select random samples
            import random
            selected_labels = random.sample(all_labels, min(num_samples, len(all_labels)))
            
            for label_info in selected_labels:
                img_path = self.dataset_path / "raw_images" / label_info["image"]
                
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    
                    sample = {
                        "image_path": str(img_path),
                        "image": img,
                        "ground_truth": label_info["text"],
                        "id": label_info["id"]
                    }
                    samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} sample images")
            return samples
        
        except Exception as e:
            logger.error(f"Error loading samples: {str(e)}")
            return []
    
    def preprocess_sample(self, sample: Dict) -> Dict:
        """
        Preprocess a single sample.
        
        Args:
            sample: Sample dictionary with image
            
        Returns:
            Updated sample with preprocessed image
        """
        try:
            preprocessed = self.preprocessor.preprocess_image(sample["image_path"])
            sample["preprocessed"] = preprocessed
            return sample
        except Exception as e:
            logger.error(f"Error preprocessing: {str(e)}")
            return sample
    
    def simulate_recognition(self, preprocessed_img: np.ndarray) -> str:
        """
        Simulate text recognition (mock prediction).
        
        Args:
            preprocessed_img: Preprocessed image array
            
        Returns:
            Simulated recognized text
        """
        # In a real scenario, this would use the trained model
        # For now, we'll return a mock prediction based on image characteristics
        
        # Calculate image statistics as mock features
        mean_pixel = preprocessed_img.mean()
        std_pixel = preprocessed_img.std()
        
        # Mock character predictions
        mock_texts = [
            "Computer Vision",
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
            "Text Recognition",
            "Feature Extraction",
            "Pattern Detection",
            "Image Processing"
        ]
        
        # Select based on image features (deterministic for reproducibility)
        idx = int((mean_pixel + std_pixel) * 10) % len(mock_texts)
        return mock_texts[idx]
    
    def run_demo(self, num_samples: int = 5) -> None:
        """
        Run complete demo pipeline.
        
        Args:
            num_samples: Number of samples to process
        """
        logger.info("\n" + "="*70)
        logger.info("Handwritten Text Recognition - Demo Pipeline")
        logger.info("="*70)
        
        # Load samples
        logger.info(f"\nStep 1: Loading {num_samples} sample images...")
        samples = self.load_sample_images(num_samples)
        
        if not samples:
            logger.error("No samples loaded. Exiting.")
            return
        
        # Process each sample
        logger.info(f"\nStep 2: Processing samples through pipeline...")
        results = []
        
        for i, sample in enumerate(samples):
            logger.info(f"\n--- Sample {i+1}/{len(samples)} ---")
            
            # Preprocess
            logger.info("a) Preprocessing image...")
            sample = self.preprocess_sample(sample)
            
            if "preprocessed" in sample:
                logger.info(f"   ✓ Preprocessed shape: {sample['preprocessed'].shape}")
            
            # Simulate recognition
            logger.info("b) Recognizing text (mock prediction)...")
            predicted_text = self.simulate_recognition(sample.get("preprocessed", sample["image"]))
            sample["predicted"] = predicted_text
            
            # Normalize
            logger.info("c) Normalizing recognized text...")
            normalized_text = self.normalizer.normalize(predicted_text)
            sample["normalized"] = normalized_text
            
            # Display results
            logger.info(f"\n   Ground Truth: '{sample['ground_truth']}'")
            logger.info(f"   Predicted:    '{sample['predicted']}'")
            logger.info(f"   Normalized:   '{sample['normalized']}'")
            
            results.append({
                "id": sample["id"],
                "ground_truth": sample["ground_truth"],
                "predicted": sample["predicted"],
                "normalized": sample["normalized"]
            })
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("Demo Summary")
        logger.info("="*70)
        
        logger.info(f"\nProcessed {len(results)} samples successfully")
        logger.info("\nDetailed Results:")
        
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. Sample ID: {result['id']}")
            logger.info(f"   Ground Truth:  {result['ground_truth']}")
            logger.info(f"   Predicted:     {result['predicted']}")
            logger.info(f"   Normalized:    {result['normalized']}")
        
        # Save results to file
        results_file = Path("demo_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Demo results saved to: {results_file}")


def demonstrate_data_loading():
    """Demonstrate data loading and statistics."""
    logger.info("\n" + "="*70)
    logger.info("Dataset Statistics and Information")
    logger.info("="*70)
    
    try:
        dataset_path = Path("dataset")
        
        # Load labels
        labels_file = dataset_path / "labels" / "labels.json"
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        logger.info(f"\n✓ Total samples: {len(labels)}")
        
        # Analyze text lengths
        text_lengths = [len(label['text'].split()) for label in labels]
        logger.info(f"✓ Average text length: {np.mean(text_lengths):.1f} words")
        logger.info(f"✓ Min text length: {min(text_lengths)} words")
        logger.info(f"✓ Max text length: {max(text_lengths)} words")
        
        # Show sample texts
        logger.info(f"\nSample texts from dataset:")
        for i, label in enumerate(labels[:10]):
            logger.info(f"  {i+1}. {label['text']}")
        
        # Check train/test split
        train_file = dataset_path / "labels" / "train_labels.json"
        test_file = dataset_path / "labels" / "test_labels.json"
        
        with open(train_file, 'r') as f:
            train_labels = json.load(f)
        with open(test_file, 'r') as f:
            test_labels = json.load(f)
        
        logger.info(f"\n✓ Train samples: {len(train_labels)}")
        logger.info(f"✓ Test samples: {len(test_labels)}")
        
        # Image statistics
        img_path = dataset_path / "raw_images" / labels[0]['image']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        logger.info(f"\n✓ Image dimensions: {img.shape}")
        logger.info(f"✓ Image dtype: {img.dtype}")
        logger.info(f"✓ Pixel range: [{img.min()}, {img.max()}]")
        logger.info(f"✓ Mean pixel value: {img.mean():.1f}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")


def main():
    """Main execution."""
    
    # Show dataset info
    demonstrate_data_loading()
    
    # Run demo
    demo = HTRDemo()
    demo.run_demo(num_samples=5)
    
    logger.info("\n" + "="*70)
    logger.info("✓ Demo completed successfully!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
