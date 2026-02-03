"""
Feature Enhancement Module (HRNN with Attention)

Implements hierarchical RNN with attention mechanisms for feature enhancement
in handwritten text recognition.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Attention, MultiHeadAttention, Dense, Dropout, LayerNormalization, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AttentionEnhancer:
    """Feature enhancement using attention mechanisms."""
    
    @staticmethod
    def self_attention(features: tf.Tensor) -> tf.Tensor:
        """
        Apply self-attention to enhance features.
        
        Args:
            features: Input feature tensor (batch, sequence_length, features)
            
        Returns:
            Attention-enhanced features
        """
        try:
            # Self-attention: query=key=value = features
            attention = Attention()([features, features])
            logger.info("Self-attention applied successfully")
            return attention
            
        except Exception as e:
            logger.error(f"Error in self-attention: {str(e)}")
            raise
    
    @staticmethod
    def multi_head_attention(features: tf.Tensor, 
                            num_heads: int = 4) -> tf.Tensor:
        """
        Apply multi-head attention for richer feature interactions.
        
        Args:
            features: Input feature tensor
            num_heads: Number of attention heads
            
        Returns:
            Multi-head attention enhanced features
        """
        try:
            mha = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=features.shape[-1] // num_heads,
                dropout=0.1
            )
            
            # Self-attention
            attention_out = mha(features, features)
            logger.info(f"Multi-head attention ({num_heads} heads) applied")
            return attention_out
            
        except Exception as e:
            logger.error(f"Error in multi-head attention: {str(e)}")
            raise


class HierarchicalRNNEnhancer:
    """Hierarchical RNN with attention for feature enhancement."""
    
    def __init__(self,
                 feature_dim: int = 256,
                 num_heads: int = 4,
                 dropout_rate: float = 0.1):
        """
        Initialize Hierarchical RNN Enhancer.
        
        Args:
            feature_dim: Feature dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
    
    def build_attention_block(self, x: tf.Tensor) -> tf.Tensor:
        """
        Build a single attention block with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Enhanced features
        """
        # Store input for residual connection
        residual = x
        
        # Multi-head attention
        mha = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.feature_dim // self.num_heads,
            dropout=self.dropout_rate
        )
        attn_out = mha(x, x)
        
        # Residual connection
        x = Add()([residual, attn_out])
        
        # Layer normalization
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        x_residual = x
        x = Dense(self.feature_dim * 2, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.feature_dim)(x)
        
        # Residual connection + normalization
        x = Add()([x_residual, x])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        return x
    
    def build_enhancement_model(self, 
                               input_shape: Tuple[int, ...],
                               num_blocks: int = 2) -> Model:
        """
        Build complete feature enhancement model.
        
        Args:
            input_shape: Input feature shape
            num_blocks: Number of attention blocks to stack
            
        Returns:
            Keras Model for feature enhancement
        """
        try:
            inputs = Input(shape=input_shape, name='features_input')
            x = inputs
            
            # Project to feature_dim if needed
            if input_shape[-1] != self.feature_dim:
                x = Dense(self.feature_dim, activation='relu', 
                         name='feature_projection')(x)
            
            # Stack attention blocks
            for i in range(num_blocks):
                x = self.build_attention_block(x)
            
            model = Model(inputs=inputs, outputs=x, 
                         name='hierarchical_rnn_enhancer')
            logger.info(f"HRNN enhancement model built with {num_blocks} blocks")
            return model
            
        except Exception as e:
            logger.error(f"Error building HRNN model: {str(e)}")
            raise


class CrossModalAttention:
    """Cross-modal attention for combining different feature types."""
    
    @staticmethod
    def apply_cross_modal_attention(visual_features: tf.Tensor,
                                    linguistic_features: tf.Tensor) -> tf.Tensor:
        """
        Apply cross-modal attention between visual and linguistic features.
        
        Args:
            visual_features: CNN/LSTM extracted visual features
            linguistic_features: Language model features
            
        Returns:
            Fused features
        """
        try:
            # Cross-attention: visual as query, linguistic as key/value
            cross_attn = MultiHeadAttention(
                num_heads=4,
                key_dim=64,
                dropout=0.1
            )
            
            fused = cross_attn(visual_features, linguistic_features)
            logger.info("Cross-modal attention applied")
            return fused
            
        except Exception as e:
            logger.error(f"Error in cross-modal attention: {str(e)}")
            raise


def enhance_features(features: tf.Tensor, 
                    num_blocks: int = 2) -> tf.Tensor:
    """
    Convenience function to enhance features using attention.
    
    Args:
        features: Input features
        num_blocks: Number of enhancement blocks
        
    Returns:
        Enhanced features
    """
    enhancer = HierarchicalRNNEnhancer(feature_dim=features.shape[-1])
    
    # For direct tensor enhancement
    x = features
    for _ in range(num_blocks):
        x = enhancer.build_attention_block(x)
    
    return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    enhancer = HierarchicalRNNEnhancer(feature_dim=256, num_heads=4)
    
    # Build enhancement model
    input_shape = (32, 256)  # sequence_length=32, features=256
    model = enhancer.build_enhancement_model(input_shape, num_blocks=2)
    
    model.summary()
