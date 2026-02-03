"""
CTC Decoder Module for Handwritten Text Recognition

Implements Connectionist Temporal Classification (CTC) loss and decoding
for character-level text recognition without explicit alignment.
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class CTCDecoder:
    """CTC loss and decoding implementation."""
    
    def __init__(self, num_classes: int, blank_index: int = 0):
        """
        Initialize CTC Decoder.
        
        Args:
            num_classes: Number of character classes (including blank)
            blank_index: Index of blank character (default: 0)
        """
        self.num_classes = num_classes
        self.blank_index = blank_index
    
    @staticmethod
    def ctc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute CTC loss for batch of sequences.
        
        Args:
            y_true: Ground truth labels (batch_size, seq_length)
            y_pred: Predicted logits (batch_size, seq_length, num_classes)
            
        Returns:
            CTC loss value
        """
        try:
            # y_pred shape: (batch_size, time_steps, num_classes)
            # y_true shape: (batch_size, label_length)
            
            # Get sequence lengths (assuming all sequences use full length)
            batch_size = tf.shape(y_pred)[0]
            input_length = tf.shape(y_pred)[1]
            label_length = tf.shape(y_true)[1]
            
            # Create length tensors
            input_length = tf.fill([batch_size], input_length)
            label_length = tf.fill([batch_size], label_length)
            
            # Compute CTC loss
            loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
            
            return loss
            
        except Exception as e:
            logger.error(f"Error computing CTC loss: {str(e)}")
            raise
    
    @staticmethod
    def ctc_decode(y_pred: tf.Tensor, 
                   input_length: Optional[tf.Tensor] = None,
                   greedy: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Decode CTC predictions to character sequences.
        
        Args:
            y_pred: Predicted logits (batch_size, time_steps, num_classes)
            input_length: Length of each sequence in batch
            greedy: Use greedy decoding (True) or beam search (False)
            
        Returns:
            Decoded sequences and probabilities
        """
        try:
            batch_size = tf.shape(y_pred)[0]
            time_steps = tf.shape(y_pred)[1]
            
            if input_length is None:
                input_length = tf.fill([batch_size], time_steps)
            
            if greedy:
                # Greedy decoding: select highest probability at each step
                decoded, log_prob = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(y_pred, perm=[1, 0, 2]),  # time_major
                    sequence_length=input_length
                )
            else:
                # Beam search decoding (more accurate but slower)
                decoded, log_prob = tf.nn.ctc_beam_search_decoder(
                    inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                    sequence_length=input_length,
                    beam_width=50,
                    top_paths=1
                )
            
            # Sparse tensor to dense
            decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1)
            
            return decoded_dense, log_prob
            
        except Exception as e:
            logger.error(f"Error in CTC decoding: {str(e)}")
            raise
    
    @staticmethod
    def predictions_to_text(predictions: tf.Tensor, 
                           char_map: dict) -> List[str]:
        """
        Convert prediction indices to text strings.
        
        Args:
            predictions: Predicted character indices
            char_map: Mapping from index to character
            
        Returns:
            List of decoded text strings
        """
        texts = []
        for pred in predictions:
            # Remove blanks and invalid indices
            pred = pred[pred >= 0]  # Remove padding
            text = ''.join([char_map.get(int(idx), '') for idx in pred])
            texts.append(text)
        
        return texts


class CTCLossLayer(tf.keras.layers.Layer):
    """Custom Keras layer for CTC loss computation."""
    
    def __init__(self, **kwargs):
        super(CTCLossLayer, self).__init__(**kwargs)
    
    def call(self, args):
        """
        Compute CTC loss.
        
        Args:
            args: Tuple of (y_true, y_pred)
            
        Returns:
            CTC loss
        """
        y_true, y_pred = args
        batch_size = tf.shape(y_pred)[0]
        input_length = tf.shape(y_pred)[1]
        label_length = tf.shape(y_true)[1]
        
        input_length = tf.fill([batch_size], input_length)
        label_length = tf.fill([batch_size], label_length)
        
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        
        return y_pred


def build_ctc_model(feature_shape: Tuple[int, ...],
                   num_classes: int) -> Model:
    """
    Build complete model with CTC output layer.
    
    Args:
        feature_shape: Shape of input features
        num_classes: Number of character classes
        
    Returns:
        Keras Model with CTC output
    """
    try:
        inputs = Input(shape=feature_shape, name='features')
        
        # Output layer: dense layer for character probabilities
        outputs = Dense(num_classes, activation='softmax', 
                       name='ctc_output')(inputs)
        
        model = Model(inputs=inputs, outputs=outputs, name='ctc_model')
        logger.info(f"CTC model built with {num_classes} classes")
        return model
        
    except Exception as e:
        logger.error(f"Error building CTC model: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    decoder = CTCDecoder(num_classes=80)
    
    # Build example model
    feature_shape = (32, 256)  # sequence_length=32, features=256
    model = build_ctc_model(feature_shape, num_classes=80)
    
    model.summary()
    
    # Example: Create sample predictions
    # batch_size = 2, sequence_length = 32, num_classes = 80
    # y_pred = tf.random.normal((2, 32, 80))
    # y_true = tf.constant([[1, 2, 3, 0, 0], [5, 6, 0, 0, 0]])
    # loss = CTCDecoder.ctc_loss(y_true, y_pred)
