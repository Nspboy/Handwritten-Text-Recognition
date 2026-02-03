"""
Sequence Model Module (BiLSTM) for Handwritten Text Recognition

Implements bidirectional LSTM layers for temporal sequence modeling
of CNN-extracted features.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional, LSTM, Dense, Dropout, Reshape, TimeDistributed
)
from tensorflow.keras.models import Model
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BiLSTMSequenceModel:
    """Bidirectional LSTM sequence model for feature sequence encoding."""
    
    def __init__(self,
                 lstm_units: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.3,
                 return_sequences: bool = True):
        """
        Initialize BiLSTM Sequence Model.
        
        Args:
            lstm_units: Number of LSTM units per layer
            num_layers: Number of stacked LSTM layers
            dropout_rate: Dropout rate between layers
            return_sequences: Whether to return sequences
        """
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
    
    def add_bilstm(self, x, layer_idx: int = 0) -> tf.Tensor:
        """
        Add a bidirectional LSTM layer to the model.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (for naming)
            
        Returns:
            Output tensor from BiLSTM layer
        """
        is_last_layer = (layer_idx == self.num_layers - 1)
        
        # BiLSTM layer
        x = Bidirectional(
            LSTM(self.lstm_units, 
                 return_sequences=(not is_last_layer or self.return_sequences),
                 name=f'lstm_{layer_idx}'),
            name=f'bidirectional_{layer_idx}'
        )(x)
        
        # Dropout for regularization
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate, name=f'dropout_{layer_idx}')(x)
        
        return x
    
    def build_sequence_model(self, input_shape: Tuple[int, ...]) -> Model:
        """
        Build complete BiLSTM sequence model.
        
        Args:
            input_shape: Input tensor shape (sequence_length, features)
            
        Returns:
            Keras Model
        """
        try:
            from tensorflow.keras.layers import Input
            
            inputs = Input(shape=input_shape, name='sequence_input')
            x = inputs
            
            # Stack multiple BiLSTM layers
            for i in range(self.num_layers):
                x = self.add_bilstm(x, layer_idx=i)
            
            model = Model(inputs=inputs, outputs=x, name='bilstm_sequence_model')
            logger.info(f"BiLSTM sequence model built with {self.num_layers} layers")
            return model
            
        except Exception as e:
            logger.error(f"Error building BiLSTM model: {str(e)}")
            raise
    
    def build_with_cnn_output(self, 
                             cnn_feature_shape: Tuple[int, ...]) -> Model:
        """
        Build BiLSTM model that processes CNN feature maps.
        
        Reshapes CNN output (H, W, C) to sequence (H*W, C) for LSTM processing.
        
        Args:
            cnn_feature_shape: Shape of CNN output (height, width, channels)
            
        Returns:
            Keras Model
        """
        try:
            from tensorflow.keras.layers import Input
            
            inputs = Input(shape=cnn_feature_shape, name='cnn_features')
            
            # Reshape CNN features to sequence format
            # From (height, width, channels) to (height*width, channels)
            height, width, channels = cnn_feature_shape
            x = Reshape((height * width, channels), name='reshape_to_sequence')(inputs)
            
            # Apply BiLSTM layers
            for i in range(self.num_layers):
                x = self.add_bilstm(x, layer_idx=i)
            
            model = Model(inputs=inputs, outputs=x, name='cnn_bilstm_model')
            logger.info("CNN-BiLSTM integration model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building CNN-BiLSTM model: {str(e)}")
            raise


class LSTMAttentionLayer:
    """LSTM layer with attention mechanism."""
    
    @staticmethod
    def add_lstm_with_attention(x, lstm_units: int = 128) -> tf.Tensor:
        """
        Add LSTM with self-attention.
        
        Args:
            x: Input tensor
            lstm_units: Number of LSTM units
            
        Returns:
            Output tensor with attention applied
        """
        from tensorflow.keras.layers import Attention, Dense
        
        lstm_out = Bidirectional(
            LSTM(lstm_units, return_sequences=True)
        )(x)
        
        # Self-attention: compare sequence against itself
        attention_out = Attention()([lstm_out, lstm_out])
        
        # Dense layer for attention output
        output = Dense(lstm_units, activation='relu')(attention_out)
        
        return output


def add_bilstm(x, lstm_units: int = 128, dropout: float = 0.3) -> tf.Tensor:
    """
    Convenience function to add BiLSTM layer.
    
    Args:
        x: Input tensor
        lstm_units: Number of LSTM units
        dropout: Dropout rate
        
    Returns:
        Output tensor
    """
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    seq_model = BiLSTMSequenceModel(lstm_units=128, num_layers=2)
    
    # Build model from CNN features
    cnn_feature_shape = (16, 16, 128)  # After pooling
    model = seq_model.build_with_cnn_output(cnn_feature_shape)
    
    model.summary()
