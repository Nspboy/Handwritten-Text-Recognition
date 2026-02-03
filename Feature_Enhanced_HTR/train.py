"""
Training Module for Enhanced Handwritten Text Recognition

Implements training pipeline with proper error handling, validation,
and checkpoint management.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from preprocessing.preprocess import ImagePreprocessor
from model.cnn_feature_extractor import CNNFeatureExtractor
from model.sequence_model import BiLSTMSequenceModel
from model.enhancement_hrnn import HierarchicalRNNEnhancer
from model.decoder_ctc import CTCDecoder, build_ctc_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HTRTrainer:
    """Trainer class for Enhanced Handwritten Text Recognition."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize HTR Trainer.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.history = None
        self._setup_directories()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from JSON file."""
        default_config = {
            "dataset_path": "dataset/",
            "model_save_dir": "checkpoints/",
            "log_dir": "logs/",
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "input_shape": [128, 128, 1],
            "num_classes": 80,
            "lstm_units": 128,
            "num_lstm_layers": 2,
            "validation_split": 0.1,
            "early_stopping_patience": 10
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Config loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Error loading config: {str(e)}. Using defaults.")
        
        return default_config
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.config['model_save_dir'],
            self.config['log_dir'],
            Path(self.config['dataset_path']) / 'enhanced_images'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {dir_path}")
    
    def build_model(self) -> tf.keras.models.Model:
        """
        Build complete HTR model architecture.
        
        Returns:
            Compiled Keras model
        """
        try:
            input_shape = tuple(self.config['input_shape'])
            num_classes = self.config['num_classes']
            
            # Input layer
            inputs = tf.keras.layers.Input(shape=input_shape, name='image_input')
            
            # CNN Feature Extraction
            logger.info("Building CNN feature extractor...")
            cnn_extractor = CNNFeatureExtractor(input_shape=input_shape)
            cnn_model = cnn_extractor.build_cnn()
            
            # Remove pooling-free CNN and rebuild with functional API
            cnn_features = cnn_model(inputs)
            
            # Reshape for sequence processing
            # Get feature map dimensions
            feature_shape = cnn_extractor.get_feature_dimension()
            height, width, channels = feature_shape
            
            x = tf.keras.layers.Reshape(
                (height * width, channels), 
                name='reshape_for_sequence'
            )(cnn_features)
            
            # BiLSTM Sequence Modeling
            logger.info("Building BiLSTM sequence model...")
            seq_model = BiLSTMSequenceModel(
                lstm_units=self.config['lstm_units'],
                num_layers=self.config['num_lstm_layers'],
                dropout_rate=0.3
            )
            
            lstm_layers = seq_model.num_layers
            for i in range(lstm_layers):
                is_last = (i == lstm_layers - 1)
                x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.config['lstm_units'],
                        return_sequences=not is_last,
                        name=f'lstm_{i}'
                    ),
                    name=f'bilstm_{i}'
                )(x)
                x = tf.keras.layers.Dropout(0.3)(x)
            
            # Feature Enhancement with HRNN
            logger.info("Building Feature Enhancement (HRNN)...")
            if not is_last:  # Only if output is sequence
                enhancer = HierarchicalRNNEnhancer(
                    feature_dim=self.config['lstm_units'] * 2
                )
                x = enhancer.build_attention_block(x)
            
            # CTC Output Layer
            logger.info("Adding CTC output layer...")
            outputs = tf.keras.layers.Dense(
                num_classes,
                activation='softmax',
                name='ctc_output'
            )(x)
            
            # Create model
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
            
            self.model = model
            logger.info("Model built and compiled successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    @staticmethod
    def _ctc_loss(y_true, y_pred):
        """CTC loss function."""
        # y_pred shape: (batch_size, num_classes) or (batch_size, sequence_length, num_classes)
        # y_true shape: (batch_size, label_length)
        
        batch_size = tf.shape(y_pred)[0]
        
        # Check if y_pred is 2D or 3D
        y_pred_shape = tf.shape(y_pred)
        is_2d = tf.equal(tf.rank(y_pred), 2)
        
        # For 2D, expand to 3D for CTC loss
        def expand_y_pred():
            # Add sequence dimension
            return tf.expand_dims(y_pred, axis=1)
        
        def use_y_pred_as_is():
            return y_pred
        
        y_pred_expanded = tf.cond(is_2d, expand_y_pred, use_y_pred_as_is)
        
        max_time = tf.shape(y_pred_expanded)[1]
        
        # Create input length for all samples (assuming full sequence length)
        input_length = tf.ones([batch_size], dtype=tf.int32) * max_time
        
        # Create label length from y_true (count non-zero elements)
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
        label_length = tf.maximum(label_length, 1)  # At least 1 to avoid issues
        
        return tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred_expanded,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=0
        )
    
    def train(self, x_train, y_train, 
              x_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the model.
        
        Args:
            x_train: Training images
            y_train: Training labels
            x_val: Validation images (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(
                    self.config['model_save_dir'], 
                    'best_model.h5'
                ),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            TensorBoard(
                log_dir=self.config['log_dir'],
                histogram_freq=1
            )
        ]
        
        try:
            logger.info("Starting training...")
            self.history = self.model.fit(
                x_train, y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(x_val, y_val) if x_val is not None else None,
                validation_split=self.config['validation_split'] if x_val is None else 0,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Training completed successfully")
            return self.history.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save trained model."""
        if self.model is None:
            logger.error("No model to save")
            return
        
        save_path = path or os.path.join(
            self.config['model_save_dir'], 
            'final_model.h5'
        )
        
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """Load trained model."""
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            logger.error("No model to summarize")
            return
        
        self.model.summary()


if __name__ == "__main__":
    # Example usage
    trainer = HTRTrainer()
    
    # Build model
    model = trainer.build_model()
    model.summary()
    
    # To train with actual data:
    # x_train = np.random.randn(100, 128, 128, 1)
    # y_train = np.random.randint(0, 80, (100, 32))
    # history = trainer.train(x_train, y_train)
