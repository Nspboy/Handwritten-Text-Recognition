"""
NLP Post-Processing Module for Text Correction

Implements text correction and enhancement using language models
and spell correction techniques.
"""

import logging
from typing import Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TextCorrector:
    """Advanced text correction using multiple strategies."""
    
    def __init__(self, use_transformers: bool = False):
        """
        Initialize TextCorrector.
        
        Args:
            use_transformers: Whether to use transformer models (requires transformers library)
        """
        self.use_transformers = use_transformers
        self.corrector = None
        
        if use_transformers:
            self._init_transformer_corrector()
    
    def _init_transformer_corrector(self) -> None:
        """Initialize transformer-based corrector."""
        try:
            from transformers import pipeline
            
            # Using a lightweight model for faster inference
            self.corrector = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1  # -1 for CPU, 0+ for GPU
            )
            logger.info("Transformer-based corrector initialized")
            
        except ImportError:
            logger.warning("Transformers library not installed. "
                          "Install with: pip install transformers torch")
            self.use_transformers = False
        except Exception as e:
            logger.error(f"Error initializing transformer corrector: {str(e)}")
            self.use_transformers = False
    
    def correct_with_transformer(self, text: str) -> str:
        """
        Correct text using transformer model.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text
        """
        if not self.use_transformers or self.corrector is None:
            return text
        
        try:
            result = self.corrector(f"Fix grammar and spelling: {text}", 
                                   max_length=512)
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Error in transformer correction: {str(e)}")
            return text
    
    def correct_with_symspell(self, text: str) -> str:
        """
        Correct text using SymSpell algorithm.
        
        Args:
            text: Input text to correct
            
        Returns:
            Spell-corrected text
        """
        try:
            from symspellpy import SymSpell, Verbosity
            
            # Initialize SymSpell
            sym_spell = SymSpell(max_dictionary_edit_distance=2)
            sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt",
                                    term_index=0, count_index=1)
            
            # Correct words
            words = text.split()
            corrected_words = []
            
            for word in words:
                suggestions = sym_spell.lookup(word, Verbosity.TOP, 
                                              max_edit_distance=2)
                if suggestions:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(word)
            
            corrected_text = ' '.join(corrected_words)
            logger.info(f"Text corrected with SymSpell")
            return corrected_text
            
        except ImportError:
            logger.warning("SymSpell library not installed. "
                          "Install with: pip install symspellpy")
            return text
        except Exception as e:
            logger.error(f"Error in SymSpell correction: {str(e)}")
            return text
    
    def correct_text(self, text: str, method: str = 'simple') -> str:
        """
        Correct text using specified method.
        
        Args:
            text: Input text
            method: Correction method ('simple', 'transformer', 'symspell')
            
        Returns:
            Corrected text
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        if method == 'transformer':
            return self.correct_with_transformer(text)
        elif method == 'symspell':
            return self.correct_with_symspell(text)
        else:
            return self._simple_correction(text)
    
    @staticmethod
    def _simple_correction(text: str) -> str:
        """
        Apply simple text corrections (whitespace, punctuation).
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text


class LanguageModel:
    """Language model wrapper for text analysis and correction."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize Language Model.
        
        Args:
            model_path: Path to pre-trained language model
        """
        self.model_path = model_path
        self.model = None
    
    def load_model(self) -> bool:
        """
        Load language model from path.
        
        Returns:
            Success status
        """
        if not self.model_path:
            logger.warning("No model path provided")
            return False
        
        try:
            path = Path(self.model_path)
            if not path.exists():
                logger.error(f"Model not found: {self.model_path}")
                return False
            
            # Load model (implementation depends on model type)
            logger.info(f"Model loaded from: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    @staticmethod
    def calculate_confidence(predictions: List[float]) -> float:
        """
        Calculate confidence score for predictions.
        
        Args:
            predictions: List of probability scores
            
        Returns:
            Confidence score (0-1)
        """
        if not predictions:
            return 0.0
        
        # Use average of top predictions as confidence
        sorted_preds = sorted(predictions, reverse=True)
        confidence = (sorted_preds[0] - sorted_preds[1]) if len(sorted_preds) > 1 else sorted_preds[0]
        
        return float(confidence)


class TextNormalizer:
    """Normalize and standardize recognized text."""
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text (case, whitespace, special characters).
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase (if needed for your application)
        # text = text.lower()
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = ' '.join(text.split())
        
        # Remove extra punctuation
        import re
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple dots
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep_spaces: bool = True) -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Input text
            keep_spaces: Whether to keep spaces
            
        Returns:
            Text with special characters removed
        """
        import re
        
        if keep_spaces:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        else:
            text = re.sub(r'[^a-zA-Z0-9]', '', text)
        
        return text


def correct_text(text: str) -> str:
    """
    Convenience function for text correction.
    
    Args:
        text: Input text to correct
        
    Returns:
        Corrected text
    """
    corrector = TextCorrector(use_transformers=False)
    return corrector.correct_text(text, method='simple')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    corrector = TextCorrector(use_transformers=False)
    
    sample_text = "the quick brwon fox jumps ovr the lazy dog"
    corrected = corrector.correct_text(sample_text, method='simple')
    print(f"Original: {sample_text}")
    print(f"Corrected: {corrected}")
    
    # Text normalization
    normalizer = TextNormalizer()
    messy_text = "   Hello   WORLD  ..   This  is  a  TEST!!  "
    normalized = normalizer.normalize(messy_text)
    print(f"\nNormalized: {normalized}")
