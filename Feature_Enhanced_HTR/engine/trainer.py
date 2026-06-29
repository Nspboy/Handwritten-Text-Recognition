"""
Training Module for Enhanced Handwritten Text Recognition

Implements training pipeline with proper error handling, validation,
and checkpoint management.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from engine.preprocessing.preprocess import ImagePreprocessor
from engine.model.cnn_feature_extractor import CNNFeatureExtractor
from engine.model.sequence_model import BiLSTMSequenceModel
from engine.model.enhancement_hrnn import HierarchicalRNNEnhancer
from engine.model.decoder_ctc import CTCDecoder, build_ctc_model

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
            Path(self.config['dataset_path']) / 'processed'
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
            # cnn_features shape: (batch, height, width, channels)
            # We want: (batch, width, height * channels)
            
            x = tf.keras.layers.Permute((2, 1, 3), name='permute_for_sequence')(cnn_features)
            
            # Get feature map dimensions after permutation
            height, width, channels = cnn_extractor.get_feature_dimension()
            # After permutation (2, 1, 3): 
            # height becomes new width, width becomes new height
            
            x = tf.keras.layers.Reshape(
                (-1, height * channels), 
                name='reshape_for_sequence'
            )(x)
            
            # BiLSTM Sequence Modeling
            logger.info("Building BiLSTM sequence model...")
            seq_model = BiLSTMSequenceModel(
                lstm_units=self.config['lstm_units'],
                num_layers=self.config['num_lstm_layers'],
                dropout_rate=0.3
            )
            
            lstm_layers = seq_model.num_layers
            for i in range(lstm_layers):
                # FOR HTR/CTC: All BiLSTM layers must return sequences
                # especially the last one so CTC can decode the time steps
                x = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(
                        self.config['lstm_units'],
                        return_sequences=True,
                        name=f'lstm_{i}'
                    ),
                    name=f'bilstm_{i}'
                )(x)
                x = tf.keras.layers.Dropout(0.3)(x)
            
            # Feature Enhancement with HRNN (as seen in Step E of architecture)
            logger.info("Building Feature Enhancement (HRNN/Attention)...")
            enhancer = HierarchicalRNNEnhancer(
                feature_dim=self.config['lstm_units'] * 2
            )
            # Apply attention-based enhancement to the sequential features
            x = enhancer.build_attention_block(x)
            
            # CTC Output Layer (as seen in Step F of architecture)
            logger.info("Adding CTC output layer...")
            num_classes = self.config['num_classes']
            # NOTE: Use activation=None for ctc_loss (expects logits)
            outputs = tf.keras.layers.Dense(
                num_classes,
                activation=None,
                name='ctc_output'
            )(x)
            
            # Create model
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            
            # Compile model with CTC loss
            # NOTE: Custom CTC loss is required for HTR
            model.compile(
                optimizer=Adam(learning_rate=self.config['learning_rate']),
                loss=self._ctc_loss
            )
            
            self.model = model
            logger.info("Model built and compiled with CTC loss successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    @staticmethod
    def _ctc_loss(y_true, y_pred):
        """CTC loss function."""
        # y_pred shape: (batch_size, sequence_length, num_classes)
        # y_true shape: (batch_size, label_length)
        
        batch_size = tf.shape(y_pred)[0]
        max_time = tf.shape(y_pred)[1]
        
        # Ensure y_pred is 3D (batch, time, classes)
        # Our architecture guaranteed this, so we simplify to avoid cond-op gradient errors
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Create input length for all samples (full sequence length)
        input_length = tf.ones([batch_size], dtype=tf.int32) * max_time
        
        # Create label length from y_true (count non-zero elements)
        # Assuming 0 is the CTC blank/padding index
        label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
        label_length = tf.maximum(label_length, 1)  # Minimum length of 1
        
        return tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
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
    
    def freeze_cnn_layers(self) -> None:
        """Freeze CNN layers for Phase 2 training (Fine-tune RNN only)."""
        if self.model is None:
            logger.error("Model not built. Cannot freeze layers.")
            return
            
        logger.info("Freezing CNN layers...")
        for layer in self.model.layers:
            # We assume functional API or layers starting with certain names.
            # Typical functional model has layers like 'conv2d', 'batch_normalization'
            if 'conv2d' in layer.name or 'batch_normalization' in layer.name or 'max_pooling2d' in layer.name:
                layer.trainable = False
        
        # Recompile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate']),
            loss=self._ctc_loss
        )
        logger.info("CNN layers frozen and model recompiled.")
        
    def unfreeze_all_layers(self) -> None:
        """Unfreeze all layers for Phase 3 joint training."""
        if self.model is None:
            return
            
        logger.info("Unfreezing all layers...")
        for layer in self.model.layers:
            layer.trainable = True
            
        # Recompile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['learning_rate'] * 0.1), # lower LR for fine tuning
            loss=self._ctc_loss
        )
        logger.info("All layers unfrozen and model recompiled.")
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save trained model and mappings."""
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
            
            # Save mappings
            if hasattr(self, 'char_to_idx'):
                mapping_path = save_path.replace('.h5', '_mapping.json')
                with open(mapping_path, 'w') as f:
                    json.dump({
                        'char_to_idx': self.char_to_idx,
                        'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
                    }, f)
                logger.info(f"Mappings saved to {mapping_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, path: str, compile: bool = False) -> None:
        """Load trained model and mappings."""
        try:
            self.model = tf.keras.models.load_model(path, compile=compile)
            logger.info(f"Model loaded from {path} (compile={compile})")
            
            # Load mappings
            mapping_path = path.replace('.h5', '_mapping.json')
            if Path(mapping_path).exists():
                with open(mapping_path, 'r') as f:
                    data = json.load(f)
                    self.char_to_idx = data['char_to_idx']
                    self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
                logger.info(f"Mappings loaded from {mapping_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def model_summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            logger.error("No model to summarize")
            return
        
        self.model.summary()

    def encode_labels(self, texts: List[str], max_len: int = 32) -> np.ndarray:
        """Encode text labels to numerical indices."""
        if not hasattr(self, 'char_to_idx'):
            logger.error("char_to_idx mapping not set")
            return np.zeros((len(texts), max_len))
            
        Y = np.zeros((len(texts), max_len), dtype=np.int32)
        for i, t in enumerate(texts):
            encoded = [self.char_to_idx.get(c, 0) for c in t.lower()[:max_len]]
            Y[i, :len(encoded)] = encoded
        return Y

    def decode_batch_predictions(self, pred: np.ndarray) -> List[str]:
        """Decode batch of predictions to text."""
        # Using the logic from CTCDecoder
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Greedy decoding
        decode, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(pred, perm=[1, 0, 2]),
            sequence_length=input_len.astype(np.int32),
            blank_index=0
        )
        
        # Convert sparse to dense
        decoded_dense = tf.sparse.to_dense(decode[0], default_value=-1).numpy()
        
        # Mapping to chars
        results = []
        for res in decoded_dense:
            text = "".join([self.idx_to_char.get(i, "") for i in res if i != -1])
            results.append(text)
        return results


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
