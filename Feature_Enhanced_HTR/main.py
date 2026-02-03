"""
Main Execution Module for Enhanced Handwritten Text Recognition

Complete pipeline for image preprocessing, text recognition, and correction.
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2

from preprocessing.preprocess import ImagePreprocessor
from model.cnn_feature_extractor import CNNFeatureExtractor
from model.sequence_model import BiLSTMSequenceModel
from model.enhancement_hrnn import HierarchicalRNNEnhancer
from model.decoder_ctc import CTCDecoder
from nlp.postprocess import TextCorrector, TextNormalizer
from train import HTRTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HTRPipeline:
    """End-to-end pipeline for handwritten text recognition."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize HTR Pipeline.
        
        Args:
            model_path: Path to pre-trained model
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.model_path = model_path
        self.preprocessor = ImagePreprocessor()
        self.text_corrector = TextCorrector(use_transformers=False)
        self.text_normalizer = TextNormalizer()
        self.trainer = None
        self.model = None
        
        if model_path:
            self._load_model()
    
    def _load_model(self) -> None:
        """Load pre-trained model."""
        try:
            self.trainer = HTRTrainer(self.config_path)
            self.trainer.load_model(self.model_path)
            self.model = self.trainer.model
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def preprocess_image(self, img_path: str) -> Optional[np.ndarray]:
        """
        Preprocess a single image.
        
        Args:
            img_path: Path to input image
            
        Returns:
            Preprocessed image array
        """
        return self.preprocessor.preprocess_image(img_path)
    
    def recognize_text(self, 
                      img_array: np.ndarray,
                      apply_correction: bool = True) -> str:
        """
        Recognize text from preprocessed image.
        
        Args:
            img_array: Preprocessed image array
            apply_correction: Whether to apply NLP correction
            
        Returns:
            Recognized and optionally corrected text
        """
        if self.model is None:
            logger.error("No model loaded for recognition")
            return ""
        
        try:
            # Prepare input
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.expand_dims(img_array, axis=-1)
            
            # Normalize to [0, 1]
            if img_array.max() > 1:
                img_array = img_array / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)
            
            # Decode predictions (placeholder - actual CTC decoding needed)
            text = self._decode_predictions(predictions)
            
            # Apply correction
            if apply_correction:
                text = self.text_corrector.correct_text(text, method='simple')
            
            # Normalize
            text = self.text_normalizer.normalize(text)
            
            logger.info(f"Recognized text: {text}")
            return text
            
        except Exception as e:
            logger.error(f"Error in text recognition: {str(e)}")
            return ""
    
    @staticmethod
    def _decode_predictions(predictions: np.ndarray) -> str:
        """
        Decode model predictions to text.
        
        Args:
            predictions: Model output predictions
            
        Returns:
            Decoded text string
        """
        # Placeholder implementation
        # In actual implementation, use CTC decoding
        # This is a simple argmax approach for demonstration
        
        if len(predictions.shape) == 3:
            # Get argmax for each time step
            char_indices = np.argmax(predictions[0], axis=1)
        else:
            char_indices = np.argmax(predictions, axis=1)
        
        # Create character map (placeholder)
        # In production, load actual character set
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        text = ''.join([alphabet[idx] if idx < len(alphabet) else '' 
                       for idx in char_indices if idx > 0])
        
        return text
    
    def process_image_file(self, 
                          img_path: str,
                          apply_correction: bool = True) -> str:
        """
        Complete pipeline: load -> preprocess -> recognize -> correct.
        
        Args:
            img_path: Path to input image
            apply_correction: Whether to apply NLP correction
            
        Returns:
            Recognized text
        """
        try:
            # Preprocess
            processed = self.preprocess_image(img_path)
            if processed is None:
                logger.error(f"Failed to preprocess {img_path}")
                return ""
            
            # Recognize
            text = self.recognize_text(processed, apply_correction)
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            return ""
    
    def process_batch(self,
                     input_dir: str,
                     output_file: Optional[str] = None,
                     apply_correction: bool = True) -> List[str]:
        """
        Process multiple images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_file: Optional file to save results
            apply_correction: Whether to apply correction
            
        Returns:
            List of recognized texts
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Directory not found: {input_dir}")
            return []
        
        results = []
        
        for img_file in input_path.glob("*.png"):
            logger.info(f"Processing: {img_file.name}")
            text = self.process_image_file(str(img_file), apply_correction)
            results.append({
                'image': img_file.name,
                'text': text
            })
        
        # Save results if requested
        if output_file:
            self._save_results(results, output_file)
        
        return results
    
    @staticmethod
    def _save_results(results: List[dict], output_file: str) -> None:
        """Save recognition results to file."""
        try:
            with open(output_file, 'w') as f:
                for item in results:
                    f.write(f"{item['image']}\t{item['text']}\n")
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Enhanced Handwritten Text Recognition Pipeline'
    )
    
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], 
                       default='predict', help='Mode of operation')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--input-dir', type=str, help='Input directory for batch processing')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--model', type=str, help='Path to pre-trained model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--no-correction', action='store_true', 
                       help='Disable text correction')
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        # Initialize pipeline
        pipeline = HTRPipeline(model_path=args.model, config_path=args.config)
        
        if args.image:
            # Single image
            text = pipeline.process_image_file(args.image, 
                                              apply_correction=not args.no_correction)
            print(f"Recognized text: {text}")
        
        elif args.input_dir:
            # Batch processing
            results = pipeline.process_batch(args.input_dir, 
                                            output_file=args.output,
                                            apply_correction=not args.no_correction)
            
            for item in results:
                print(f"{item['image']}: {item['text']}")
        
        else:
            parser.print_help()
    
    elif args.mode == 'train':
        # Training mode
        trainer = HTRTrainer(config_path=args.config)
        model = trainer.build_model()
        
        # Note: Actual training requires dataset
        logger.info("Trainer initialized. Load dataset and call trainer.train()")


if __name__ == "__main__":
    main()
