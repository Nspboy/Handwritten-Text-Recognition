"""
CNN Feature Extractor Module for Handwritten Text Recognition

Implements a convolutional neural network for extracting visual features
from preprocessed handwritten text images.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, Dropout, Input, Reshape
)
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CNNFeatureExtractor:
    """CNN-based feature extractor for handwritten text images."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (128, 128, 1),
                 dropout_rate: float = 0.3):
        """
        Initialize CNN feature extractor.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.model = None
    
    def build_cnn(self) -> Sequential:
        """
        Build CNN model for feature extraction.
        
        Architecture:
        - Conv2D (32 filters, 3x3) + BatchNorm + MaxPool
        - Conv2D (64 filters, 3x3) + BatchNorm + MaxPool
        - Conv2D (128 filters, 3x3) + BatchNorm + MaxPool
        - Dropout for regularization
        
        Returns:
            Compiled Keras Sequential model
        """
        try:
            model = Sequential([
                # Block 1
                Conv2D(32, (3, 3), activation='relu', 
                       padding='same', input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(self.dropout_rate),
                
                # Block 2
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(self.dropout_rate),
                
                # Block 3
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(self.dropout_rate),
            ])
            
            self.model = model
            logger.info("CNN model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building CNN model: {str(e)}")
            raise
    
    def build_cnn_functional(self) -> Model:
        """
        Build CNN using Functional API for more flexibility.
        
        Returns:
            Keras Model instance
        """
        try:
            inputs = Input(shape=self.input_shape)
            
            # Block 1
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Block 2
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(self.dropout_rate)(x)
            
            # Block 3
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(self.dropout_rate)(x)
            
            model = Model(inputs=inputs, outputs=x)
            logger.info("Functional CNN model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building functional CNN: {str(e)}")
            raise
    
    def get_feature_dimension(self) -> Tuple[int, ...]:
        """
        Calculate output feature dimension after CNN processing.
        
        Returns:
            Output shape (height, width, channels)
        """
        if self.model is None:
            self.build_cnn()
        
        # Create dummy input to infer shape
        dummy_input = tf.zeros((1, *self.input_shape))
        output = self.model(dummy_input)
        return output.shape[1:]
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            self.build_cnn()
        self.model.summary()


def build_cnn(input_shape: Tuple[int, int, int] = (128, 128, 1)) -> Sequential:
    """
    Convenience function to build a CNN feature extractor.
    
    Args:
        input_shape: Input image shape
        
    Returns:
        Compiled CNN model
    """
    extractor = CNNFeatureExtractor(input_shape=input_shape)
    return extractor.build_cnn()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    extractor = CNNFeatureExtractor()
    model = extractor.build_cnn()
    
    print("\nFeature output shape:", extractor.get_feature_dimension())
    extractor.summary()
