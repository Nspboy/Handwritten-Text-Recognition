# API Reference and Module Relationships

## Module Dependency Graph

```
main.py (Entry Point)
  ├── train.py (HTRTrainer)
  │   ├── model/cnn_feature_extractor.py
  │   ├── model/sequence_model.py
  │   ├── model/enhancement_hrnn.py
  │   ├── model/decoder_ctc.py
  │   └── preprocessing/preprocess.py
  │
  ├── preprocessing/preprocess.py (ImagePreprocessor)
  │   └── cv2 (OpenCV)
  │
  ├── model/ (All model components)
  │   ├── cnn_feature_extractor.py (CNNFeatureExtractor)
  │   ├── sequence_model.py (BiLSTMSequenceModel)
  │   ├── enhancement_hrnn.py (HierarchicalRNNEnhancer)
  │   └── decoder_ctc.py (CTCDecoder)
  │
  ├── nlp/postprocess.py (TextCorrector, TextNormalizer)
  │   ├── symspellpy (optional)
  │   └── transformers (optional)
  │
  └── utils.py (Utility Functions)
      ├── Config, DataUtil, FileUtil, MetricsUtil
      └── Path, json, numpy
```

---

## Complete API Reference

### preprocessing.preprocess

#### **ImagePreprocessor**

```python
class ImagePreprocessor:
    def __init__(blur_kernel=(5,5), morphology_enabled=True)
    def preprocess_image(img_path: str) → np.ndarray
    def batch_preprocess(input_dir: str, output_dir: str) → int
    def _apply_morphology(binary_img: np.ndarray) → np.ndarray
```

#### Functions

```python
preprocess_image(img_path: str) → Optional[np.ndarray]
```

---

### model.cnn_feature_extractor

#### **CNNFeatureExtractor**

```python
class CNNFeatureExtractor:
    def __init__(input_shape=(128,128,1), dropout_rate=0.3)
    def build_cnn() → Sequential
    def build_cnn_functional() → Model
    def get_feature_dimension() → Tuple
    def summary() → None
```

#### Functions

```python
build_cnn(input_shape=(128,128,1)) → Sequential
```

---

### model.sequence_model

#### **BiLSTMSequenceModel**

```python
class BiLSTMSequenceModel:
    def __init__(lstm_units=128, num_layers=2, dropout_rate=0.3)
    def add_bilstm(x: tf.Tensor, layer_idx=0) → tf.Tensor
    def build_sequence_model(input_shape: Tuple) → Model
    def build_with_cnn_output(cnn_feature_shape: Tuple) → Model
```

#### **LSTMAttentionLayer**

```python
class LSTMAttentionLayer:
    @staticmethod
    def add_lstm_with_attention(x, lstm_units=128) → tf.Tensor
```

#### Functions

```python
add_bilstm(x, lstm_units=128, dropout=0.3) → tf.Tensor
```

---

### model.enhancement_hrnn

#### **AttentionEnhancer**

```python
class AttentionEnhancer:
    @staticmethod
    def self_attention(features: tf.Tensor) → tf.Tensor
    @staticmethod
    def multi_head_attention(features: tf.Tensor, num_heads=4) → tf.Tensor
```

#### **HierarchicalRNNEnhancer**

```python
class HierarchicalRNNEnhancer:
    def __init__(feature_dim=256, num_heads=4, dropout_rate=0.1)
    def build_attention_block(x: tf.Tensor) → tf.Tensor
    def build_enhancement_model(input_shape: Tuple, num_blocks=2) → Model
```

#### **CrossModalAttention**

```python
class CrossModalAttention:
    @staticmethod
    def apply_cross_modal_attention(visual_features, linguistic_features) → tf.Tensor
```

#### Functions

```python
enhance_features(features: tf.Tensor, num_blocks=2) → tf.Tensor
```

---

### model.decoder_ctc

#### **CTCDecoder**

```python
class CTCDecoder:
    def __init__(num_classes: int, blank_index=0)
    @staticmethod
    def ctc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) → tf.Tensor
    @staticmethod
    def ctc_decode(y_pred, input_length=None, greedy=True) → Tuple
    @staticmethod
    def predictions_to_text(predictions, char_map: dict) → List[str]
```

#### **CTCLossLayer**

```python
class CTCLossLayer(tf.keras.layers.Layer):
    def call(args) → tf.Tensor
```

#### Functions

```python
build_ctc_model(feature_shape: Tuple, num_classes: int) → Model
```

---

### nlp.postprocess

#### **TextCorrector**

```python
class TextCorrector:
    def __init__(use_transformers: bool = False)
    def correct_text(text: str, method='simple') → str
    def correct_with_transformer(text: str) → str
    def correct_with_symspell(text: str) → str
    def _simple_correction(text: str) → str
```

#### **LanguageModel**

```python
class LanguageModel:
    def __init__(model_path: str)
    def load_model() → bool
    @staticmethod
    def calculate_confidence(predictions: List[float]) → float
```

#### **TextNormalizer**

```python
class TextNormalizer:
    @staticmethod
    def normalize(text: str) → str
    @staticmethod
    def remove_special_chars(text: str, keep_spaces=True) → str
```

#### Functions

```python
correct_text(text: str) → str
```

---

### train

#### **HTRTrainer**

```python
class HTRTrainer:
    def __init__(config_path: Optional[str] = None)
    def build_model() → tf.keras.models.Model
    def train(x_train, y_train, x_val=None, y_val=None) → Dict
    def save_model(path: Optional[str] = None) → None
    def load_model(path: str) → None
    def model_summary() → None
    def _load_config(config_path: Optional[str]) → Dict
    def _setup_directories() → None
    @staticmethod
    def _ctc_loss(y_true, y_pred) → tf.Tensor
```

---

### main

#### **HTRPipeline**

```python
class HTRPipeline:
    def __init__(model_path: Optional[str] = None, config_path: Optional[str] = None)
    def preprocess_image(img_path: str) → Optional[np.ndarray]
    def recognize_text(img_array: np.ndarray, apply_correction=True) → str
    def process_image_file(img_path: str, apply_correction=True) → str
    def process_batch(input_dir: str, output_file=None, apply_correction=True) → List[dict]
    def _load_model() → None
    @staticmethod
    def _decode_predictions(predictions: np.ndarray) → str
    @staticmethod
    def _save_results(results: List[dict], output_file: str) → None
```

#### Functions

```python
main() → None
```

---

### utils

#### **Config**

```python
class Config:
    @staticmethod
    def load(config_path: str) → Dict
    @staticmethod
    def save(config: Dict, config_path: str) → bool
```

#### **DataUtil**

```python
class DataUtil:
    @staticmethod
    def normalize_image(img: np.ndarray) → np.ndarray
    @staticmethod
    def denormalize_image(img: np.ndarray) → np.ndarray
    @staticmethod
    def pad_sequence(sequence: List, max_length: int, pad_value=0) → np.ndarray
```

#### **FileUtil**

```python
class FileUtil:
    @staticmethod
    def list_images(directory: str, extensions=None) → List[str]
    @staticmethod
    def create_directory(path: str) → bool
```

#### **MetricsUtil**

```python
class MetricsUtil:
    @staticmethod
    def character_error_rate(predictions: List[str], references: List[str]) → float
    @staticmethod
    def word_error_rate(predictions: List[str], references: List[str]) → float
    @staticmethod
    def _edit_distance(seq1, seq2) → int
```

---

## Data Flow Examples

### Training Workflow

```python
from train import HTRTrainer
import numpy as np

# Load config
trainer = HTRTrainer(config_path='config.json')

# Build model
model = trainer.build_model()

# Prepare data (your dataset)
x_train = np.random.randn(1000, 128, 128, 1)
y_train = np.random.randint(0, 80, (1000, 32))

# Train
history = trainer.train(x_train, y_train)

# Save
trainer.save_model('best_model.h5')
```

### Prediction Workflow

```python
from main import HTRPipeline

# Initialize pipeline with trained model
pipeline = HTRPipeline(model_path='best_model.h5')

# Single image
text = pipeline.process_image_file('sample.png', apply_correction=True)
print(f"Recognized: {text}")

# Batch
results = pipeline.process_batch('dataset/raw_images/', output_file='results.txt')
for item in results:
    print(f"{item['image']}: {item['text']}")
```

### Image Preprocessing Workflow

```python
from preprocessing.preprocess import ImagePreprocessor

preprocessor = ImagePreprocessor(blur_kernel=(5,5), morphology_enabled=True)

# Single image
processed = preprocessor.preprocess_image('image.png')

# Save preprocessed
import cv2
cv2.imwrite('preprocessed.png', processed)

# Batch
count = preprocessor.batch_preprocess('raw/', 'enhanced/')
```

### Feature Enhancement Workflow

```python
from model.enhancement_hrnn import HierarchicalRNNEnhancer
import tensorflow as tf

enhancer = HierarchicalRNNEnhancer(feature_dim=256, num_heads=4)
model = enhancer.build_enhancement_model(input_shape=(32, 256), num_blocks=2)

# Use in training
features = tf.random.normal((32, 32, 256))  # batch_size=32, seq_len=32, features=256
enhanced = model(features)
```

### Text Correction Workflow

```python
from nlp.postprocess import TextCorrector, TextNormalizer

corrector = TextCorrector(use_transformers=False)
normalizer = TextNormalizer()

text = "the quik brwon fox"
corrected = corrector.correct_text(text, method='simple')
normalized = normalizer.normalize(corrected)
print(normalized)
```

---

## Configuration Schema

```json
{
  "dataset_path": "string", // Data directory
  "model_save_dir": "string", // Model checkpoint directory
  "log_dir": "string", // TensorBoard logs
  "batch_size": "integer", // Training batch size
  "epochs": "integer", // Training epochs
  "learning_rate": "float", // Initial learning rate
  "input_shape": [128, 128, 1], // Image input shape
  "num_classes": 80, // Character classes
  "lstm_units": 128, // LSTM hidden units
  "num_lstm_layers": 2, // Number of LSTM layers
  "dropout_rate": 0.3, // Dropout rate
  "validation_split": 0.1, // Validation split
  "early_stopping_patience": 10, // EarlyStopping patience
  "preprocessing": {
    "blur_kernel": [5, 5], // Gaussian blur kernel
    "morphology_enabled": true // Enable morphology
  },
  "cnn": {
    "dropout_rate": 0.3 // CNN dropout
  },
  "lstm": {
    "units": 128,
    "num_layers": 2,
    "return_sequences": true
  },
  "enhancement": {
    "num_heads": 4, // Attention heads
    "dropout_rate": 0.1,
    "num_blocks": 2 // Enhancement blocks
  },
  "nlp": {
    "use_transformers": false, // Use transformer models
    "correction_method": "simple" // Correction strategy
  }
}
```

---

## Error Handling

All modules implement consistent error handling:

```python
try:
    # Operation
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    return None
except ValueError as e:
    logger.error(f"Invalid value: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")
    return None
```

---

## Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

**Log Levels Used**:

- `INFO`: Operation start/completion, progress
- `ERROR`: Failures, exceptions, validation errors
- `WARNING`: Fallbacks, missing optional dependencies

---

## Testing Utilities

```python
from utils import MetricsUtil

# Calculate CER
cer = MetricsUtil.character_error_rate(
    predictions=['hello'],
    references=['helo']
)
print(f"CER: {cer:.2%}")  # Output: CER: 50.00%

# Calculate WER
wer = MetricsUtil.word_error_rate(
    predictions=['hello world'],
    references=['helo world']
)
print(f"WER: {wer:.2%}")  # Output: WER: 100.00%
```

---

**Last Updated**: February 2026  
**Version**: 1.0.0
