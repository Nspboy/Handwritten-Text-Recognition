"""
Utility functions and helpers for Enhanced HTR.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to config.json
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Config loaded: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config not found: {config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {str(e)}")
            return {}
    
    @staticmethod
    def save(config: Dict[str, Any], config_path: str) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save config
            
        Returns:
            Success status
        """
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Config saved: {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False


class DataUtil:
    """Data utility functions."""
    
    @staticmethod
    def normalize_image(img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1].
        
        Args:
            img: Image array
            
        Returns:
            Normalized image
        """
        if img.max() > 1:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)
    
    @staticmethod
    def denormalize_image(img: np.ndarray) -> np.ndarray:
        """
        Denormalize image from [0, 1] to [0, 255].
        
        Args:
            img: Normalized image array
            
        Returns:
            Denormalized image (0-255)
        """
        return (img * 255).astype(np.uint8)
    
    @staticmethod
    def pad_sequence(sequence: List, max_length: int, 
                     pad_value: int = 0) -> np.ndarray:
        """
        Pad sequence to fixed length.
        
        Args:
            sequence: Input sequence
            max_length: Target length
            pad_value: Padding value
            
        Returns:
            Padded sequence
        """
        if len(sequence) >= max_length:
            return np.array(sequence[:max_length])
        
        padded = np.full(max_length, pad_value, dtype=np.int32)
        padded[:len(sequence)] = sequence
        return padded


class FileUtil:
    """File utility functions."""
    
    @staticmethod
    def list_images(directory: str, extensions: List[str] = None) -> List[str]:
        """
        List all images in directory.
        
        Args:
            directory: Directory path
            extensions: Allowed extensions (default: .png, .jpg, .jpeg)
            
        Returns:
            List of image file paths
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg']
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        images = []
        for ext in extensions:
            images.extend(list(dir_path.glob(f"*{ext}")))
        
        return [str(img) for img in images]
    
    @staticmethod
    def create_directory(path: str) -> bool:
        """
        Create directory if not exists.
        
        Args:
            path: Directory path
            
        Returns:
            Success status
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory: {str(e)}")
            return False


class MetricsUtil:
    """Metrics calculation utilities."""
    
    @staticmethod
    def character_error_rate(predictions: List[str], 
                            references: List[str]) -> float:
        """
        Calculate Character Error Rate (CER).
        
        Args:
            predictions: Predicted text list
            references: Reference text list
            
        Returns:
            CER score (lower is better)
        """
        if len(predictions) != len(references):
            logger.error("Lengths of predictions and references don't match")
            return 1.0
        
        total_errors = 0
        total_chars = 0
        
        for pred, ref in zip(predictions, references):
            errors = MetricsUtil._edit_distance(pred, ref)
            total_errors += errors
            total_chars += len(ref)
        
        return total_errors / total_chars if total_chars > 0 else 0.0
    
    @staticmethod
    def word_error_rate(predictions: List[str], 
                       references: List[str]) -> float:
        """
        Calculate Word Error Rate (WER).
        
        Args:
            predictions: Predicted text list
            references: Reference text list
            
        Returns:
            WER score (lower is better)
        """
        if len(predictions) != len(references):
            logger.error("Lengths of predictions and references don't match")
            return 1.0
        
        total_errors = 0
        total_words = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            errors = MetricsUtil._edit_distance(pred_words, ref_words)
            total_errors += errors
            total_words += len(ref_words)
        
        return total_errors / total_words if total_words > 0 else 0.0
    
    @staticmethod
    def _edit_distance(seq1, seq2) -> int:
        """
        Calculate Levenshtein (edit) distance.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Edit distance
        """
        if len(seq1) < len(seq2):
            return MetricsUtil._edit_distance(seq2, seq1)
        
        if len(seq2) == 0:
            return len(seq1)
        
        previous_row = range(len(seq2) + 1)
        
        for i, c1 in enumerate(seq1):
            current_row = [i + 1]
            
            for j, c2 in enumerate(seq2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            
            previous_row = current_row
        
        return previous_row[-1]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config = Config.load("config.json")
    print("Config keys:", list(config.keys()))
    
    # Test data utilities
    img = np.random.randint(0, 256, (128, 128))
    normalized = DataUtil.normalize_image(img)
    print("Normalized shape:", normalized.shape)
    
    # Test metrics
    cer = MetricsUtil.character_error_rate(
        ["hello"], ["helo"]
    )
    print(f"CER: {cer:.2%}")
